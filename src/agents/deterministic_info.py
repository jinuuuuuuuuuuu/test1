"""Deterministic shortcuts for high-risk pension information questions.

These handlers cover rule-heavy questions where relying on the LLM to decide
whether to call a tool has repeatedly produced ungrounded answers.
"""

from __future__ import annotations

import re

from src.agents.state import RetrievedItem
from src.rules.comprehensive_tax import (
    ANNUAL_THRESHOLD,
    SEPARATE_TAXATION_RATE_OVER_THRESHOLD,
)
from src.rules.default_option import NOTICE_DELAY_DAYS_EXISTING, WAIT_DAYS_AFTER_NOTICE
from src.rules.in_kind_transfer import TRANSFER_BLOCK_CODES
from src.rules.retirement_tax_reduction import get_deferred_retirement_tax_rate
from src.rules.tax_credit import (
    COMBINED_CREDIT_LIMIT,
    CREDIT_RATE_HIGH,
    CREDIT_RATE_LOW,
    INCOME_THRESHOLD_COMPREHENSIVE,
    INCOME_THRESHOLD_SALARY,
    PENSION_SAVINGS_ONLY_LIMIT,
    TOTAL_CONTRIBUTION_LIMIT,
)
from src.rules.withdrawal_limit import (
    SIX_YEAR_EXCEPTION_CUTOFF,
    SIX_YEAR_EXCEPTION_START,
    UNLIMITED_FROM_YEAR,
)


def deterministic_info_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    text = _compact(question)

    if _is_tax_credit_calculation_missing_question(question):
        return _tax_credit_calculation_missing_response()
    if _is_tax_benefit_overview_question(text):
        return _tax_benefit_overview_response()
    if _is_tax_credit_limit_question(text):
        return _tax_credit_limit_response()
    if _is_early_withdrawal_general_question(text):
        return _early_withdrawal_general_response()
    if _is_default_option_auto_purchase_question(text):
        return _default_option_auto_purchase_response()
    if _is_in_kind_transfer_block_question(text):
        return _in_kind_transfer_block_response()
    if _is_withdrawal_limit_question(text):
        return _withdrawal_limit_response()
    if _is_retirement_tax_reduction_question(text):
        return _retirement_tax_reduction_response()
    if _is_pension_income_tax_question(text):
        return _pension_income_tax_response()
    return None


def should_force_info_agent(question: str) -> bool:
    """Return True for questions that must use information/rule handling."""
    text = _compact(question)
    return any(
        (
            _is_tax_credit_calculation_missing_question(question),
            _is_tax_credit_limit_question(text),
            _is_tax_benefit_overview_question(text),
            _is_early_withdrawal_general_question(text),
            _is_default_option_auto_purchase_question(text),
            _is_in_kind_transfer_block_question(text),
            _is_withdrawal_limit_question(text),
            _is_retirement_tax_reduction_question(text),
            _is_pension_income_tax_question(text),
        )
    )


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _won(amount: int) -> str:
    if amount % 10_000 == 0:
        return f"{amount // 10_000:,}만원"
    return f"{amount:,}원"


def _pct(rate: float) -> str:
    return f"{rate * 100:g}%"


def _context(source: str, content: str) -> list[RetrievedItem]:
    return [{"source": source, "content": content, "node": "info_agent"}]


def _is_tax_credit_calculation_missing_question(question: str) -> bool:
    text = _compact(question)
    if "세액공제" not in text:
        return False
    if not any(word in text for word in ("계산", "받을수", "받을수있는", "금액", "얼마받")):
        return False
    has_contribution = any(word in text for word in ("연금저축", "IRP")) and bool(re.search(r"\d", text))
    has_income = any(word in text for word in ("연봉", "총급여", "종합소득", "소득금액")) and bool(re.search(r"\d", text))
    return not (has_contribution and has_income)


def _tax_credit_calculation_missing_response() -> tuple[str, list[RetrievedItem]]:
    source = "doc41 세액공제 계산 입력값 규칙"
    content = (
        f"세액공제액 계산에는 연금저축 납입액, IRP 납입액, 총급여 또는 종합소득금액이 필요합니다. "
        f"세액공제 대상 납입한도는 연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
        f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
        f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다."
    )
    draft = (
        "세액공제 금액은 납입액과 소득구간이 함께 있어야 계산할 수 있습니다.\n\n"
        "현재 질문에는 실제 계산에 필요한 입력값이 부족하므로, 세액공제액을 임의로 산출하지 않겠습니다.\n\n"
        "현재 답변 가능한 범위는 다음과 같습니다.\n"
        "- 세액공제 대상 한도는 연금저축 단독 연 600만원, 연금저축+IRP 합산 연 900만원입니다.\n"
        "- 세액공제율은 총급여 5,500만원 이하 또는 종합소득금액 4,500만원 이하이면 16.5%, "
        "그 초과 구간이면 13.2%입니다.\n\n"
        "정확한 계산을 위해 다음 정보를 한 번에 알려주세요.\n"
        "1. 올해 연금저축에 납입한 금액은 얼마인가요?\n"
        "2. 올해 IRP에 납입한 금액은 얼마인가요?\n"
        "3. 직장인이라면 총급여, 개인사업자라면 종합소득금액은 얼마인가요?\n"
        "4. 이미 회사 DC/IRP 추가납입 등 다른 연금계좌 납입액이 있다면 함께 알려주세요."
    )
    return draft, _context(source, content)


def _is_tax_benefit_overview_question(text: str) -> bool:
    return any(word in text for word in ("세금혜택", "세제혜택", "절세혜택", "세금상혜택")) and any(
        word in text for word in ("연금계좌", "연금저축", "IRP", "연금")
    )


def _tax_benefit_overview_response() -> tuple[str, list[RetrievedItem]]:
    source = "doc38~doc41 연금계좌 세금혜택 규칙"
    content = (
        f"연금계좌의 세금혜택은 납입 시 세액공제, 운용 중 과세이연, 연금수령 시 저율 과세, "
        f"퇴직금을 연금으로 받을 때 이연퇴직소득세 감면으로 정리됩니다. "
        f"세액공제 대상 납입한도는 연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
        f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
        f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다. "
        f"사적연금소득은 연 {_won(ANNUAL_THRESHOLD)} 초과 여부가 종합과세 판단 기준이며, "
        f"초과 시 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세를 선택할 수 있습니다. "
        f"연금소득세율은 만 55세 이상 70세 미만 5.5%, 70세 이상 80세 미만 4.4%, 80세 이상 3.3%입니다. "
        f"종신연금은 연령과 무관하게 3.3%입니다. "
        f"퇴직금을 연금으로 받을 때 이연퇴직소득세는 연금실제수령연차 1~10년차 30%, "
        f"11~20년차 40%, 21년차 이상 50% 감면됩니다."
    )
    draft = (
        "연금계좌의 세금혜택은 크게 네 가지로 볼 수 있습니다.\n\n"
        "1. 납입할 때 세액공제\n"
        "- 연금저축 단독 세액공제 대상 한도는 연 600만원입니다.\n"
        "- 연금저축과 IRP를 합치면 세액공제 대상 한도는 연 900만원입니다.\n"
        "- 세액공제율은 총급여 5,500만원 이하 또는 종합소득금액 4,500만원 이하이면 16.5%, "
        "그 초과 구간이면 13.2%입니다.\n\n"
        "2. 운용 중 과세이연\n"
        "- 계좌 안에서 발생한 운용수익에 대해 매년 바로 과세하지 않고, 나중에 인출할 때 과세하는 구조입니다.\n\n"
        "3. 연금으로 받을 때 낮은 세율 적용\n"
        "- 세액공제 받은 납입금과 운용수익을 연금으로 받으면 연령에 따라 5.5%, 4.4%, 3.3% 세율이 적용될 수 있습니다.\n"
        "- 종신연금은 연령과 무관하게 3.3%입니다.\n"
        "- 사적연금소득이 연 1,500만원을 초과하면 종합과세 또는 16.5% 분리과세 선택 문제가 생길 수 있습니다.\n\n"
        "4. 퇴직금을 연금으로 받을 때 퇴직소득세 감면\n"
        "- 퇴직금을 연금으로 받으면 이연퇴직소득세가 연금실제수령연차에 따라 30%, 40%, 50% 감면될 수 있습니다.\n\n"
        "정리하면, 연금계좌는 납입 시점에는 세액공제, 운용 중에는 과세이연, 수령 시점에는 저율 과세 또는 "
        "퇴직소득세 감면을 기대할 수 있는 구조입니다."
    )
    return draft, _context(source, content)


def _is_tax_credit_limit_question(text: str) -> bool:
    return "세액공제" in text and any(word in text for word in ("한도", "얼마", "최대", "합쳐", "공제율"))


def _tax_credit_limit_response() -> tuple[str, list[RetrievedItem]]:
    source = "doc41 세액공제 규칙"
    content = (
        f"연금저축+IRP 합산 납입한도는 연 {_won(TOTAL_CONTRIBUTION_LIMIT)}입니다. "
        f"세액공제 대상 납입한도는 연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        "연금저축 600만원과 IRP 900만원을 따로 더한 1,500만원이 세액공제 대상 한도가 아니라, "
        "연금저축 납입액을 포함한 IRP 합산 900만원이 세액공제 대상 한도입니다. "
        f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
        f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
        f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다. "
        f"최대 세액공제액은 {_won(COMBINED_CREDIT_LIMIT)} x {_pct(CREDIT_RATE_LOW)} = 148만 5천원, "
        f"{_won(COMBINED_CREDIT_LIMIT)} x {_pct(CREDIT_RATE_HIGH)} = 118만 8천원입니다."
    )
    draft = (
        "연금저축과 IRP를 합쳐서 볼 때 핵심은 **세액공제 대상 한도는 합산 900만원**이라는 점입니다.\n\n"
        "- 연금저축만 납입하는 경우: 세액공제 대상은 연 600만원까지\n"
        "- 연금저축 + IRP를 함께 납입하는 경우: 두 계좌 합산 세액공제 대상은 연 900만원까지\n"
        "- 단순 납입 가능 한도: 연금저축과 IRP 합산 연 1,800만원까지\n\n"
        "따라서 연금저축 600만원과 IRP 900만원을 각각 따로 더해 1,500만원까지 세액공제되는 구조가 "
        "아니라, **IRP 한도 900만원은 연금저축 납입액을 포함한 합산 한도**입니다.\n\n"
        "세액공제율은 소득 기준에 따라 달라집니다.\n"
        "- 총급여 5,500만원 이하 또는 종합소득금액 4,500만원 이하: 16.5%\n"
        "- 총급여 5,500만원 초과 또는 종합소득금액 4,500만원 초과: 13.2%\n\n"
        "합산 한도 900만원을 모두 채웠다면 최대 세액공제액은 16.5% 적용 시 148만 5천원, "
        "13.2% 적용 시 118만 8천원입니다."
    )
    return draft, _context(source, content)


def _is_early_withdrawal_general_question(text: str) -> bool:
    return "중도인출" in text and any(word in text for word in ("가능", "경우", "사유", "요건", "언제"))


def _early_withdrawal_general_response() -> tuple[str, list[RetrievedItem]]:
    source = "doc46~doc50 중도인출 규칙"
    content = (
        "DB는 중도인출이 허용되지 않습니다. DC와 IRP는 법정 사유가 있으면 중도인출 대상 제도입니다. "
        "가능 사유는 6개월 이상 요양, 개인회생 또는 파산선고, 무주택자 전월세보증금, "
        "무주택자 주택구입, 재난피해입니다. 요양은 DC에서 직전 1년 의료비가 직전년도 연간임금총액의 "
        "12.5%를 초과해야 하며, IRP에는 이 비율 기준이 적용되지 않습니다. 개인회생·파산은 결정일 또는 "
        "선고일로부터 5년 이내 요건이 있습니다. 전월세보증금과 주택구입은 잔금지급일 또는 소유권 이전 "
        "등기일로부터 1개월 이내 신청 요건이 있습니다. 재난피해는 피해발생일로부터 3개월 이내가 원칙입니다."
    )
    draft = (
        "IRP는 중도인출이 가능한 제도이지만, 아무 때나 인출할 수 있는 것은 아니고 법정 사유가 필요합니다.\n\n"
        "대표적인 중도인출 사유는 다음과 같습니다.\n"
        "- 6개월 이상 요양\n"
        "- 개인회생 또는 파산선고\n"
        "- 무주택자 전월세보증금\n"
        "- 무주택자 주택구입\n"
        "- 재난피해\n\n"
        "참고로 DB형은 중도인출이 허용되지 않고, DC와 IRP가 중도인출 대상 제도입니다. "
        "각 사유별로 신청기한이나 추가 요건이 다릅니다. 예를 들어 개인회생·파산은 5년 이내 요건, "
        "전월세보증금·주택구입은 1개월 이내 신청 요건, 재난피해는 3개월 이내 신청 요건이 문제될 수 있습니다."
    )
    return draft, _context(source, content)


def _is_default_option_auto_purchase_question(text: str) -> bool:
    return "디폴트옵션" in text and any(word in text for word in ("자동매수", "사전지정운용", "언제", "시점", "일정"))


def _default_option_auto_purchase_response() -> tuple[str, list[RetrievedItem]]:
    source = "doc29 디폴트옵션 자동매수 규칙"
    content = (
        f"기존가입자는 상품 만기일로부터 4주({NOTICE_DELAY_DAYS_EXISTING}일) 후 통지하고, "
        f"통지 후 2주({WAIT_DAYS_AFTER_NOTICE}일) 대기 뒤 자동매수합니다. "
        f"신규가입자는 최초 부담금 납입 다음 영업일 통지하고, 통지 후 2주({WAIT_DAYS_AFTER_NOTICE}일) "
        "대기 뒤 자동매수합니다. 동일 상품 반복 만기 등 연속성이 유지되는 경우에는 통지·대기 없이 즉시 적용됩니다."
    )
    draft = (
        "디폴트옵션 자동매수 시점은 기존가입자인지, 신규가입자인지에 따라 다릅니다.\n\n"
        "- 기존가입자: 상품 만기일로부터 4주(28일) 후 통지, 그 뒤 2주(14일) 대기 후 자동매수\n"
        "- 신규가입자: 최초 부담금 납입 다음 영업일 통지, 그 뒤 2주(14일) 대기 후 자동매수\n"
        "- 동일 상품 반복 만기처럼 연속성이 유지되는 경우: 통지·대기 없이 즉시 적용\n\n"
        "다만 대기 중 전액을 다른 상품으로 이동해 연속성이 끊기면 다음 만기분부터 다시 통지와 대기 절차를 거칠 수 있습니다."
    )
    return draft, _context(source, content)


def _is_in_kind_transfer_block_question(text: str) -> bool:
    return "실물이전" in text and any(word in text for word in ("안되는", "안되는", "불가", "못", "제한", "상품", "사유"))


def _in_kind_transfer_block_response() -> tuple[str, list[RetrievedItem]]:
    source = "doc34 실물이전 불가사유 코드"
    definite_codes = [code for code, info in TRANSFER_BLOCK_CODES.items() if not info.get("directional")]
    manual_codes = [code for code, info in TRANSFER_BLOCK_CODES.items() if info.get("directional")]
    content = "확정 불가사유: " + "; ".join(
        f"{code}. {TRANSFER_BLOCK_CODES[code]['name']} - {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in definite_codes
    )
    content += " 추가 확인 필요 사유: " + "; ".join(
        f"{code}. {TRANSFER_BLOCK_CODES[code]['name']} - {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in manual_codes
    )
    highlighted = ["03", "04", "09", "10", "11", "12", "14", "15", "22", "23", "24"]
    lines = "\n".join(
        f"- {code}. {TRANSFER_BLOCK_CODES[code]['name']}: {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in highlighted
    )
    draft = (
        "퇴직연금 실물이전이 제한될 수 있는 상품·상황은 여러 가지가 있습니다. 대표적으로는 다음과 같습니다.\n\n"
        f"{lines}\n\n"
        "그 밖에도 소규모 펀드 임의해지, 언번들계약, 환매수수료 존재, 운용지시 진행 중, 압류·질권, "
        "저축은행 예금자보호한도 초과, 자사상품 편입, 만기 도래, 상품협약 미체결 등이 불가 또는 추가 확인 사유가 될 수 있습니다.\n\n"
        "상품제공수수료, 상품라인업, 사용자/가입자부담금 미분리처럼 상대 금융기관 확인이 필요한 항목도 있습니다."
    )
    return draft, _context(source, content)


def _is_withdrawal_limit_question(text: str) -> bool:
    return "연금수령한도" in text or ("연금" in text and "한도" in text and any(word in text for word in ("수령", "인출")))


def _withdrawal_limit_response() -> tuple[str, list[RetrievedItem]]:
    source = "doc39 연금수령한도 규칙"
    content = (
        "연금수령한도 = 연금계좌 평가액 ÷ (11 - 연금수령연차) × 120%입니다. "
        f"연금수령연차 {UNLIMITED_FROM_YEAR}년차 이상부터는 한도 자체가 사라집니다. "
        f"2013.3.1 이전 가입한 연금계좌는 {SIX_YEAR_EXCEPTION_START}년차부터 기산하는 특례가 있습니다. "
        "연금수령 요건은 가입기간 5년 이상, 만 55세 이후, 한도 이내 인출입니다."
    )
    draft = (
        "연금수령한도는 다음 공식으로 계산합니다.\n\n"
        "연금수령한도 = 연금계좌 평가액 ÷ (11 - 연금수령연차) × 120%\n\n"
        "다만 연금수령연차 11년차 이상부터는 한도가 없어져 전액 인출해도 연금수령으로 인정될 수 있습니다. "
        "또 2013.3.1 이전 가입한 연금계좌는 1년차가 아니라 6년차부터 기산하는 특례가 있습니다.\n\n"
        "구체적인 금액 계산을 하려면 연금계좌 평가액과 현재 연금수령연차가 필요합니다."
    )
    return draft, _context(source, content)


def _is_retirement_tax_reduction_question(text: str) -> bool:
    return any(word in text for word in ("퇴직소득세", "이연퇴직소득세")) and any(
        word in text for word in ("감면", "비율", "세율", "납부")
    )


def _retirement_tax_reduction_response() -> tuple[str, list[RetrievedItem]]:
    r1 = get_deferred_retirement_tax_rate(1)
    r11 = get_deferred_retirement_tax_rate(11)
    r21 = get_deferred_retirement_tax_rate(21)
    source = "doc39~doc40 이연퇴직소득세 감면 규칙"
    content = (
        f"연금실제수령연차 1~10년차는 이연퇴직소득세의 {_pct(r1.payment_ratio)}를 납부하고 "
        f"{_pct(r1.reduction_ratio)}를 감면합니다. 11~20년차는 {_pct(r11.payment_ratio)}를 납부하고 "
        f"{_pct(r11.reduction_ratio)}를 감면합니다. 21년차 이상은 {_pct(r21.payment_ratio)}를 납부하고 "
        f"{_pct(r21.reduction_ratio)}를 감면합니다. 연금외수령은 감면 없이 전액 납부합니다."
    )
    draft = (
        "퇴직금을 연금으로 수령하면 이연퇴직소득세가 연차에 따라 감면됩니다.\n\n"
        "- 연금실제수령연차 1~10년차: 이연퇴직소득세의 70% 납부, 30% 감면\n"
        "- 연금실제수령연차 11~20년차: 60% 납부, 40% 감면\n"
        "- 연금실제수령연차 21년차 이상: 50% 납부, 50% 감면\n\n"
        "주의할 점은 여기서 쓰는 기준이 '연금수령연차'가 아니라 실제로 인출한 해만 세는 "
        "'연금실제수령연차'라는 점입니다. 연금외수령이면 감면 없이 이연퇴직소득세 전액을 납부합니다."
    )
    return draft, _context(source, content)


def _is_pension_income_tax_question(text: str) -> bool:
    return any(word in text for word in ("연금소득세", "종합과세", "분리과세")) and any(
        word in text for word in ("1500", "1,500", "세율", "기준", "얼마", "초과")
    )


def _pension_income_tax_response() -> tuple[str, list[RetrievedItem]]:
    source = "doc38 연금소득 종합과세 규칙"
    content = (
        f"사적연금소득은 연 {_won(ANNUAL_THRESHOLD)} 초과 여부가 종합과세 판단 기준입니다. "
        f"초과 시 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세를 선택할 수 있습니다. "
        "1,500만원 판정에는 세액공제 받은 납입금과 운용수익 재원만 포함하고, 세액공제 안 받은 원금과 퇴직금은 제외합니다. "
        "1,500만원 이내 연금소득세율은 만 55세 이상 70세 미만 5.5%, 70세 이상 80세 미만 4.4%, 80세 이상 3.3%입니다."
    )
    draft = (
        "사적연금소득의 종합과세 여부는 연 1,500만원 초과 여부가 기준입니다.\n\n"
        "- 연 1,500만원 이내: 연령별 연금소득세율로 분리과세\n"
        "- 연 1,500만원 초과: 종합과세 또는 16.5% 분리과세 선택 가능\n\n"
        "다만 이 1,500만원 기준에는 세액공제 받은 납입금과 운용수익 재원만 포함됩니다. "
        "세액공제 받지 않은 원금과 퇴직금 재원은 이 판정에서 제외됩니다.\n\n"
        "1,500만원 이내일 때 연금소득세율은 만 55세 이상 70세 미만 5.5%, "
        "70세 이상 80세 미만 4.4%, 80세 이상 3.3%입니다."
    )
    return draft, _context(source, content)
