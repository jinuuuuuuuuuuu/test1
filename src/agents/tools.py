"""LangGraph 에이전트(②정보 Agent, ③상품 Agent)에 바인딩할 툴 9개.

src/rules/, src/storage/의 순수 Python 함수를 LangChain 호환 @tool로 감싼다. 각 툴은
데이터클래스 결과를 JSON 직렬화 가능한 dict로 변환해 반환한다 (Enum은 .value 문자열로 변환).

- ②정보 Agent 툴: calculate_tax_credit, calculate_pension_withdrawal,
  check_early_withdrawal, check_default_option, check_in_kind_transfer, search_pension_docs
- ③상품 Agent 툴: check_product_pension_eligibility, search_funds, get_fund_detail
"""

import functools
from dataclasses import asdict, is_dataclass
from datetime import date
from enum import Enum
from typing import Any, Literal, Optional

from langchain_core.tools import tool

from src.rules.comprehensive_tax import determine_comprehensive_tax
from src.rules.default_option import check_optin_eligibility, get_auto_purchase_schedule
from src.rules.early_withdrawal import (
    WITHDRAWAL_DEADLINE_RULES,
    PlanType,
    calculate_deadline,
    check_disaster_eligibility,
    check_home_purchase_eligibility,
    check_medical_treatment_eligibility,
    check_rehabilitation_bankruptcy_eligibility,
    check_rental_deposit_eligibility,
)
from src.rules.in_kind_transfer import TRANSFER_BLOCK_CODES, check_transfer_eligibility
from src.rules.investment_limit import check_product_eligibility
from src.rules.pension_withdrawal import calculate_pension_withdrawal as _calculate_pension_withdrawal
from src.rules.retirement_tax_reduction import get_deferred_retirement_tax_rate  # noqa: F401 (재수출용)
from src.rules.tax_credit import calculate_tax_credit as _calculate_tax_credit
from src.rules.withdrawal_limit import calculate_withdrawal_limit  # noqa: F401 (재수출용)
from src.storage.queries import get_fund_detail as _get_fund_detail
from src.storage.queries import search_funds as _search_funds
from src.storage.queries import search_pension_docs as _search_pension_docs
from src.storage.queries import search_prospectus_text as _search_prospectus_text


def _to_jsonable(obj: Any) -> Any:
    """데이터클래스(중첩 포함)를 JSON 직렬화 가능한 dict로, Enum은 .value로, date는 ISO 문자열로 변환한다."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _parse_date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    return date.fromisoformat(value)


def _safe_tool(fn):
    """툴 실행 중 예외(잘못된 enum 문자열, 알 수 없는 상품유형 등)가 그래프 전체를 죽이지 않도록
    감싼다. LLM이 파라미터를 잘못 채워도(예: 존재하지 않는 product_type) 그래프가 멈추지 않고
    에이전트가 오류 내용을 보고 스스로 회복(재시도 또는 사용자에게 재질문)할 수 있게 한다."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    return wrapper


# ── 1. calculate_tax_credit ──────────────────────────────────────────────

@tool
@_safe_tool
def calculate_tax_credit(
    pension_savings_paid: int,
    irp_paid: int,
    total_salary: Optional[int] = None,
    comprehensive_income: Optional[int] = None,
) -> dict:
    """연금저축·IRP 납입액으로 연말정산/종합소득세 세액공제액을 계산한다.

    total_salary(직장인 총급여) 또는 comprehensive_income(종합소득자 종합소득금액) 중
    최소 하나는 반드시 제공해야 한다. 둘 다 있으면 total_salary를 우선 적용한다.
    연금저축 단독 세액공제 한도는 600만원, 연금저축+IRP 합산 한도는 900만원이다.
    """
    result = _calculate_tax_credit(pension_savings_paid, irp_paid, total_salary, comprehensive_income)
    return _to_jsonable(result)


# ── 2. calculate_pension_withdrawal ──────────────────────────────────────

@tool
@_safe_tool
def calculate_pension_withdrawal(
    account_value: int,
    pension_payment_year: int,
    actual_receipt_year: int,
    age: int,
    tax_credited_principal_and_gains_withdrawn: int,
    is_lifetime_annuity: bool = False,
) -> dict:
    """연금계좌 인출 시나리오의 한도·세금을 종합 계산한다 (연금수령한도 + 퇴직소득세 감면율 +
    세액공제 재원 종합과세 여부, 세 가지를 한 번에 반환).

    account_value: 연금계좌 평가액(원).
    pension_payment_year: 연금수령연차 — 개시 가능 시점부터 실제 인출 여부와 무관하게 매년
      자동 누적되는 값 (연금수령한도 산정용).
    actual_receipt_year: 연금실제수령연차 — 실제로 인출한 해만 카운트되는 값, 위와는 다른 개념
      (퇴직소득세 감면율 산정용).
    age: 만 나이 (세액공제 재원의 종합과세 세율 산정용).
    tax_credited_principal_and_gains_withdrawn: 세액공제받은 납입금+운용수익 재원에서
      올해 인출하는 금액 (비과세 원금·퇴직금 인출액은 제외하고 이 재원만).
    is_lifetime_annuity: 종신연금으로 수령하는지 여부 (해당 시 연령 무관 3.3% 적용).
    """
    result = _calculate_pension_withdrawal(
        account_value=account_value,
        pension_payment_year=pension_payment_year,
        actual_receipt_year=actual_receipt_year,
        age=age,
        tax_credited_principal_and_gains_withdrawn=tax_credited_principal_and_gains_withdrawn,
        is_lifetime_annuity=is_lifetime_annuity,
    )
    return _to_jsonable(result)


# ── 3. check_early_withdrawal ────────────────────────────────────────────

EarlyWithdrawalReason = Literal["요양", "개인회생파산", "무주택전월세", "무주택주택구입", "재난피해"]


def _transfer_code_detail(code: str) -> dict:
    """실물이전 불가사유 코드를 doc34 원문 설명과 함께 돌려준다."""
    entry = TRANSFER_BLOCK_CODES.get(code, {})
    return {
        "code": code,
        "name": entry.get("name", ""),
        "description": entry.get("desc", ""),
    }


def _deadline_info(reason: str, basis_date: Optional[date]) -> dict:
    """사유·기준일로부터 신청기한 정보를 만든다 (기준일이 없으면 규정만 안내).

    eligible 하나만 돌려주던 시절에는 "언제까지 신청하나요"라는 질문에 LLM이 마감일을
    직접 계산하지 못해 "3개월 이내"라는 원문 표현만 되뇌었다. 계산은 규칙엔진이 하고
    결과를 근거로 넘긴다.
    """
    rule = WITHDRAWAL_DEADLINE_RULES.get(reason)
    if rule is None:
        return {}

    period = f"{rule.years}년" if rule.years else f"{rule.months}개월"
    info = {
        "deadline_basis_event": rule.basis_event,
        "deadline_period": period,
        "deadline_rule": f"{rule.basis_event}로부터 달력 기준 {period} 이내",
        "source_doc": rule.source_doc,
    }
    if rule.note:
        info["deadline_note"] = rule.note
    if basis_date is not None:
        info["basis_date"] = basis_date.isoformat()
        info["deadline"] = calculate_deadline(reason, basis_date).isoformat()
    return info


@tool
@_safe_tool
def check_early_withdrawal(
    reason: EarlyWithdrawalReason,
    plan_type: Literal["DB", "DC", "IRP"],
    request_date: Optional[str] = None,
    # 요양 (reason="요양")
    medical_expense_last_year: Optional[int] = None,
    prior_year_annual_wage: Optional[int] = None,
    treatment_end_date: Optional[str] = None,
    # 개인회생·파산 (reason="개인회생파산")
    event_type: Optional[Literal["개인회생", "파산선고"]] = None,
    decision_date: Optional[str] = None,
    rehabilitation_still_effective: bool = True,
    is_workout_or_credit_recovery: bool = False,
    # 무주택 전월세보증금 (reason="무주택전월세")
    is_homeless: Optional[bool] = None,
    has_deposit: Optional[bool] = None,
    is_lease_extension: Optional[bool] = None,
    has_deposit_increase: Optional[bool] = None,
    dc_already_used: Optional[bool] = None,
    balance_payment_date: Optional[str] = None,
    # 무주택 주택구입 (reason="무주택주택구입")
    ownership_type: Optional[Literal["본인단독", "부부공동", "증여", "상속"]] = None,
    ownership_registration_date: Optional[str] = None,
    # 재난피해 (reason="재난피해")
    damage_date: Optional[str] = None,
    damage_resolved: bool = True,
) -> dict:
    """5가지 중도인출 사유(요양/개인회생파산/무주택전월세/무주택주택구입/재난피해)에 대해
    **신청기한을 계산**하고, 신청일이 주어지면 **요건 충족 여부까지 판정**한다.
    DB는 어떤 사유든 중도인출이 원천 불가하다.

    두 가지 질문 유형을 모두 처리한다:
      - "언제까지 신청해야 하나요?" -> request_date 없이 기준일만 넘긴다. deadline이 나온다.
      - "이 날짜에 신청하면 되나요?" -> request_date까지 넘긴다. eligible 판정이 함께 나온다.

    ⚠️ request_date를 임의로 채우지 마세요. 사용자가 신청일을 말하지 않았는데 기준일을
    신청일로 넣으면, 묻지도 않은 "그날 신청하면 가능한가" 판정을 하게 됩니다. 기한만
    물었으면 request_date는 비워 두세요.

    사유별 기준일 파라미터:
      요양 -> treatment_end_date / 개인회생파산 -> decision_date /
      무주택전월세 -> balance_payment_date / 무주택주택구입 -> ownership_registration_date /
      재난피해 -> damage_date
    날짜는 모두 "YYYY-MM-DD" 형식 문자열로 전달한다.
    """
    plan = PlanType(plan_type)
    req_date = _parse_date(request_date)

    # 신청기한은 신청일과 무관하게 기준일만으로 확정된다 — 기한 질문에 요건 판정을 강요하지
    # 않도록 항상 먼저 계산해 근거로 넘긴다.
    basis_dates = {
        "요양": treatment_end_date,
        "개인회생파산": decision_date,
        "무주택전월세": balance_payment_date,
        "무주택주택구입": ownership_registration_date,
        "재난피해": damage_date,
    }
    deadline_info = _deadline_info(reason, _parse_date(basis_dates.get(reason)))

    if req_date is None:
        # 신청일이 없으면 요건 판정을 하지 않는다. 기한 정보만 돌려주고, 판정이 필요하면
        # LLM이 사용자에게 신청일을 되묻도록 한다.
        return {
            **deadline_info,
            "eligibility_checked": False,
            "note": "신청일(request_date)이 없어 요건 충족 여부는 판정하지 않았습니다.",
        }

    if reason == "요양":
        result = check_medical_treatment_eligibility(
            plan_type=plan,
            medical_expense_last_year=medical_expense_last_year,
            prior_year_annual_wage=prior_year_annual_wage,
            treatment_end_date=_parse_date(treatment_end_date),
            request_date=req_date,
        )
    elif reason == "개인회생파산":
        result = check_rehabilitation_bankruptcy_eligibility(
            plan_type=plan,
            event_type=event_type,
            decision_date=_parse_date(decision_date),
            request_date=req_date,
            rehabilitation_still_effective=rehabilitation_still_effective,
            is_workout_or_credit_recovery=is_workout_or_credit_recovery,
        )
    elif reason == "무주택전월세":
        result = check_rental_deposit_eligibility(
            plan_type=plan,
            is_homeless=is_homeless,
            has_deposit=has_deposit,
            is_lease_extension=is_lease_extension,
            has_deposit_increase=has_deposit_increase,
            dc_already_used=dc_already_used,
            balance_payment_date=_parse_date(balance_payment_date),
            request_date=req_date,
        )
    elif reason == "무주택주택구입":
        result = check_home_purchase_eligibility(
            plan_type=plan,
            is_homeless=is_homeless,
            ownership_type=ownership_type,
            ownership_registration_date=_parse_date(ownership_registration_date),
            request_date=req_date,
        )
    elif reason == "재난피해":
        result = check_disaster_eligibility(
            plan_type=plan,
            damage_date=_parse_date(damage_date),
            request_date=req_date,
            damage_resolved=damage_resolved,
        )
    else:
        raise ValueError(f"알 수 없는 중도인출 사유입니다: {reason}")

    # 판정 결과와 기한 정보를 함께 낸다 — "기한이 지나서 불가"라는 판정을 받았을 때
    # 사용자에게 "그럼 언제까지였나"를 답하려면 마감일이 근거에 있어야 한다.
    return {**deadline_info, **_to_jsonable(result), "eligibility_checked": True}


# ── 4. check_default_option ──────────────────────────────────────────────

@tool
@_safe_tool
def check_default_option(
    mode: Literal["옵트인가능여부", "자동매수일정"],
    current_holdings_count: Optional[int] = None,
    target_is_same_as_only_holding: bool = False,
    base_date: Optional[str] = None,
    is_new_participant: Optional[bool] = None,
    is_first_occurrence: bool = True,
) -> dict:
    """디폴트옵션(사전지정운용제도) 관련 두 가지를 판정한다.

    mode="옵트인가능여부": current_holdings_count(현재 실제 보유 중인 디폴트옵션 개수, 포트폴리오형은
      1개로 계산), target_is_same_as_only_holding(1개 보유 시 동일 상품 추가매수인지)을 받아
      가입자가 직접 특정 디폴트옵션 상품을 매수(옵트인)할 수 있는지 판정.
    mode="자동매수일정": base_date(기존가입자는 상품 만기일, 신규가입자는 최초 부담금 납입일,
      "YYYY-MM-DD"), is_new_participant, is_first_occurrence(동일 상품 첫 만기/최초납입 여부)를
      받아 자동매수 통지일/적용일을 계산.
    """
    if mode == "옵트인가능여부":
        result = check_optin_eligibility(
            current_holdings_count=current_holdings_count,
            target_is_same_as_only_holding=target_is_same_as_only_holding,
        )
    elif mode == "자동매수일정":
        result = get_auto_purchase_schedule(
            base_date=_parse_date(base_date),
            is_new_participant=is_new_participant,
            is_first_occurrence=is_first_occurrence,
        )
    else:
        raise ValueError(f"알 수 없는 mode입니다: {mode}")

    return _to_jsonable(result)


# ── 5. check_in_kind_transfer ────────────────────────────────────────────

# TRANSFER_BLOCK_CODES의 'flag' 값들을 그대로 파라미터로 노출한다.
_TRANSFER_FLAG_NAMES = sorted(
    v["flag"] for v in TRANSFER_BLOCK_CODES.values() if not v.get("directional")
)


@tool
@_safe_tool
def check_in_kind_transfer(
    small_fund_forced_liquidation: bool = False,
    unbundled_contract: bool = False,
    private_fund: bool = False,
    is_mmf: bool = False,
    has_redemption_fee: bool = False,
    pending_trade_instruction: bool = False,
    seized_or_pledged: bool = False,
    maturity_matching_fund: bool = False,
    equity_or_reit: bool = False,
    is_rp: bool = False,
    is_promissory_note: bool = False,
    rate_linked_insurance: bool = False,
    variable_insurance: bool = False,
    principal_at_risk_derivative_bond: bool = False,
    no_dc_agreement_at_receiver: bool = False,
    exceeds_savings_bank_protection_limit: bool = False,
    own_company_product_at_receiver: bool = False,
    already_matured: bool = False,
    redemption_restricted: bool = False,
    is_default_option_product: bool = False,
    listed_investment_company: bool = False,
    no_product_agreement_at_receiver: bool = False,
) -> dict:
    """보유 상품의 실물이전(퇴직연금 금융기관 변경 시 매도 없이 그대로 이전) 가능 여부를 판정한다.

    각 파라미터는 해당 상품/상황이 그 사유에 해당하면 True로 설정한다 (기본은 전부 False).
    예: 보유 펀드가 사모펀드이면 private_fund=True. 상품제공수수료·상품라인업·재원구분 관련
    3개 사유는 상대 금융기관 확인이 필요해 항상 needs_manual_review_codes에 안내된다.
    """
    flags = {
        "small_fund_forced_liquidation": small_fund_forced_liquidation,
        "unbundled_contract": unbundled_contract,
        "private_fund": private_fund,
        "is_mmf": is_mmf,
        "has_redemption_fee": has_redemption_fee,
        "pending_trade_instruction": pending_trade_instruction,
        "seized_or_pledged": seized_or_pledged,
        "maturity_matching_fund": maturity_matching_fund,
        "equity_or_reit": equity_or_reit,
        "is_rp": is_rp,
        "is_promissory_note": is_promissory_note,
        "rate_linked_insurance": rate_linked_insurance,
        "variable_insurance": variable_insurance,
        "principal_at_risk_derivative_bond": principal_at_risk_derivative_bond,
        "no_dc_agreement_at_receiver": no_dc_agreement_at_receiver,
        "exceeds_savings_bank_protection_limit": exceeds_savings_bank_protection_limit,
        "own_company_product_at_receiver": own_company_product_at_receiver,
        "already_matured": already_matured,
        "redemption_restricted": redemption_restricted,
        "is_default_option_product": is_default_option_product,
        "listed_investment_company": listed_investment_company,
        "no_product_agreement_at_receiver": no_product_agreement_at_receiver,
    }
    result = check_transfer_eligibility(**flags)
    payload = _to_jsonable(result)

    # 코드 번호만 넘기면 LLM이 그 코드가 무슨 뜻인지 몰라 답변에서 지어낸다 — 실측:
    # blocking_codes=["04"]만 근거로 받은 모델이 "계약 종료", "원금과 수익금 회수",
    # "신규 가입 절차" 같은 근거 없는 설명을 덧붙였다. 코드표(doc34)를 함께 넘긴다.
    payload["blocking_reasons"] = [_transfer_code_detail(c) for c in result.blocking_codes]
    payload["needs_manual_review_reasons"] = [
        _transfer_code_detail(c) for c in result.needs_manual_review_codes
    ]
    return payload


# ── 6. check_product_pension_eligibility (③상품 Agent 전용) ─────────────

@tool
@_safe_tool
def check_product_pension_eligibility(
    product_type: str,
    plan_type: Literal["DB", "DC", "IRP"],
    current_risky_asset_ratio_after_purchase: Optional[float] = None,
    is_tdf_qualified: bool = False,
) -> dict:
    """특정 상품유형을 특정 제도(DB/DC/IRP) 계좌에서 매수할 수 있는지 판정한다.

    상품 추천 전 반드시 이 툴로 적합성을 먼저 확인해야 한다 — 예: 국내 상장주식·사모펀드·
    증권예탁증권(DR)은 DB만 가능하고 DC/IRP는 투자금지다.
    current_risky_asset_ratio_after_purchase(0~1)를 넘기면, 이 상품을 매수한 이후 전체
    포트폴리오의 위험자산 비중이 제도별 한도(기본 70%, TDF 조건충족 시 DC/IRP는 100%)를
    넘지 않는지까지 함께 확인한다.
    """
    plan = PlanType(plan_type)
    result = check_product_eligibility(
        product_type=product_type,
        plan_type=plan,
        current_risky_asset_ratio_after_purchase=current_risky_asset_ratio_after_purchase,
        is_tdf_qualified=is_tdf_qualified,
    )
    return _to_jsonable(result)


# ── 7. search_pension_docs (②정보 Agent 전용) ────────────────────────────

@tool
@_safe_tool
def search_pension_docs(query: str, k: int = 5) -> list[dict]:
    """연금 제도·세제·업무 매뉴얼 문서(docs.zip)에서 질의와 관련된 내용을 검색한다.

    계산·판정 툴로 답할 수 없는 개념 설명, 절차 안내, 일반 지식 질문에 사용한다.
    query는 사용자 질문을 그대로 또는 핵심 키워드로 정리해서 넘기면 된다.
    반환된 각 항목의 content가 실제 원문 발췌이며, file_title/section/source_location은
    답변에 근거(출처)로 인용할 때 사용한다.
    관련성이 낮은 결과는 자동으로 제외된다 — 빈 리스트([])가 반환되면 보유 문서에 관련
    내용이 없다는 뜻이므로, 일반 지식으로 메우지 말고 한계를 고지해야 한다.
    """
    results = _search_pension_docs(query=query, k=k)
    return _to_jsonable(results)


# ── 8. search_funds (③상품 Agent 전용) ───────────────────────────────────

@tool
@_safe_tool
def search_funds(
    keyword: Optional[str] = None,
    risk_grade_min: Optional[int] = None,
    risk_grade_max: Optional[int] = None,
    max_expense_ratio: Optional[float] = None,
    min_return_1y: Optional[float] = None,
    min_aum_krw_million: Optional[float] = None,
    sales_channel: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """조건에 맞는 펀드(투자설명서 100개, 판매클래스 단위)를 검색해 후보 목록을 반환한다.

    keyword: 펀드명·상품분류·운용사명에 포함될 키워드(예: "배당", "채권", "솔로몬 국공채").
      띄어쓰기로 여러 토큰을 주면 전부 포함된 상품만 매치된다 (펀드명 붙여쓰기 차이는 무시).
    risk_grade_min/max: 위험등급 숫자 범위 — 1등급이 가장 위험, 6등급이 가장 안전
      (숫자가 클수록 안전). "안전한 상품"을 원하면 risk_grade_min을 크게(예: 4), "공격적 투자"를
      원하면 risk_grade_max를 작게(예: 2) 설정한다.
    max_expense_ratio: 총보수·비용(%) 상한.
    min_return_1y: 최근 1년 수익률(%) 하한.
    min_aum_krw_million: 시장잔고(AUM, 백만원 단위) 하한 — "규모가 큰 펀드"를 원하면 설정
      (예: 100억원 이상이면 10000). 반환되는 aum_krw_million도 백만원 단위이며
      aum_base_date가 그 값의 결산 기준일이다.
    sales_channel: "온라인"/"오프라인" 등 판매방식 필터.
    반환 결과는 최근 1년 수익률 내림차순이며, 상세 정보가 필요하면 get_fund_detail을 이어서 호출한다.
    """
    results = _search_funds(
        keyword=keyword,
        risk_grade_min=risk_grade_min,
        risk_grade_max=risk_grade_max,
        max_expense_ratio=max_expense_ratio,
        min_return_1y=min_return_1y,
        min_aum_krw_million=min_aum_krw_million,
        sales_channel=sales_channel,
        limit=limit,
    )
    return _to_jsonable(results)


# ── 9. get_fund_detail (③상품 Agent 전용) ────────────────────────────────

@tool
@_safe_tool
def get_fund_detail(product_code: str) -> dict:
    """상품코드로 펀드 하나의 전체 상세 정보(펀드 공통정보 + 전체 판매클래스별 보수·수익률 등)를 조회한다.

    search_funds로 후보를 찾은 뒤, 사용자에게 구체적으로 설명하거나 클래스 간 비교가 필요할 때
    이 툴로 전체 상세를 가져온다. 존재하지 않는 상품코드면 빈 결과를 반환한다.
    """
    result = _get_fund_detail(product_code=product_code)
    if result is None:
        return {"found": False, "product_code": product_code}
    payload = _to_jsonable(result)
    payload["found"] = True
    return payload


# ── 10. search_prospectus_text (③상품 Agent 전용) ────────────────────────

@tool
@_safe_tool
def search_prospectus_text(query: str, product_code: Optional[str] = None, k: int = 4) -> list[dict]:
    """투자설명서의 서술형 내용(투자목적/투자전략/투자위험)을 의미 기반으로 검색한다.

    "이 펀드의 투자전략이 뭐예요", "어떤 위험이 있나요" 같은 서술 설명 질의에 사용한다 —
    수치(보수/수익률/위험등급/AUM)는 search_funds·get_fund_detail이 담당하고, 이 툴은
    문장으로 된 설명 원문을 근거로 가져온다.
    특정 펀드에 대한 질문이면 반드시 product_code로 한정하라 (search_funds로 코드를 먼저
    찾은 뒤 호출) — 한정하지 않으면 다른 펀드의 서술이 섞일 수 있다.
    """
    results = _search_prospectus_text(query=query, product_code=product_code, k=k)
    return _to_jsonable(results)


INFO_AGENT_TOOLS = [
    calculate_tax_credit,
    calculate_pension_withdrawal,
    check_early_withdrawal,
    check_default_option,
    check_in_kind_transfer,
    search_pension_docs,
]

PRODUCT_AGENT_TOOLS = [
    check_product_pension_eligibility,
    search_funds,
    get_fund_detail,
    search_prospectus_text,
]
