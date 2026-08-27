"""Deterministic answer generation for high-risk pension information questions.

분류(어느 카테고리인지)와 답변 생성(그 카테고리에서 실제 숫자·조건을 인용하는 것)의 역할을
분리한다 — 근거: "세액공제"+"얼마"만 있으면 무조건 한도질문으로 오분류하던 사고, 그리고
같은 클래스(주제어+느슨한 동반어)의 트리거가 "언제"/"시점"/"상품"/"세율" 등에서도 반복
재발할 위험이 있다는 판단(2026-08-25 재점검).

- candidate_category(): 1단계, 순수 키워드 규칙. 주제어(예: "세액공제")만 보고 "이 카테고리일
  수도 있다"는 느슨한 후보만 낸다. 결정적(같은 입력엔 항상 같은 결과)이고 빠르다.
- DETERMINISTIC_CATEGORIES: 후보로 나올 수 있는 전체 카테고리 목록 — router.py가
  RouterDecision.deterministic_category의 허용값을 정의할 때 이 목록을 그대로 쓴다.
- deterministic_response_for(category, question): 2단계, router의 LLM이 후보를 검증/확정한
  카테고리를 받아 실제 정형 답변(숫자·조건)을 만든다. 여기서부터는 결정론 그대로 유지한다 —
  숫자를 LLM에게 맡기면 학습 지식으로 틀린 값을 지어내는 사고가 실측된 바 있다.

왜 분류까지 전부 규칙으로 하지 않는가: 키워드 매칭은 "얼마"/"언제"/"시점"/"상품"/"세율"처럼
그 자체로 의미가 넓은 동반어를 요구하면 반드시 오탐(다른 의도의 질문을 잘못 걸러냄)을
만든다 — 이게 원래 사고의 근본 원인이었다. 반대로 분류를 통째로 LLM에게 맡기면(동반어 없이
전체 판단) 오탐은 줄지만 판단이 확률적이라 같은 입력에도 흔들릴 수 있고 실패 원인을 코드로
특정하기 어렵다. candidate_category()가 "주제어 존재 여부"라는 가장 안정적인 신호만
규칙으로 담당하고, 그 후보가 맞는지 최종 확정은 router의 LLM(이미 있는 호출에 필드만
추가돼 비용 증가 없음)에게 맡기는 하이브리드로 양쪽의 실패 모드를 서로 보완한다.

## 카테고리는 (도메인 × 작업종류)로 설계한다 — 2026-08-27 재설계

카테고리를 도메인(중도인출·실물이전·세제)으로만 나누면, 같은 도메인 안의 서로 다른
작업이 갈 곳을 잃는다. 실측된 증상:
  - 중도인출 날짜계산 4건(요양/전월세/재난/주택구입)이 전부 후보 ['중도인출_일반']을
    받았다. 목록답변용 카테고리라 라우터가 기각했고, 답변은 사유 목록만 나열했다.
  - "MMF인데 실물이전 되나요"(개별판정)가 ['실물이전_불가사유'](목록답변)만 후보로
    받아 기각됐고, LLM이 자유롭게 툴을 고르다 투자 가능 여부 툴을 불러
    "네, 실물이전 문제없습니다"라는 정반대 답을 냈다.

작업종류는 최소 다음 다섯이며, 새 카테고리를 만들 때 "이 도메인의 이 작업이 이미
있는가"를 먼저 확인한다:
  일반설명   — 제도 자체를 설명 (예: 세액공제_한도)
  목록답변   — 해당 사유·항목을 나열 (예: 중도인출_일반, 실물이전_불가사유)
  기한규칙   — 기준일·기간 규칙을 안내 (exact date는 DB-grounded 답변에서 단정하지 않음)
  요건판정   — 주어진 조건이 요건을 충족하는지 (예: 실물이전_개별판정)
  개인계산   — 사용자 수치를 대입해 산출 (예: 세액공제_계산_입력부족)

⚠️ 사유가 늘 때마다 "요양_신청기한", "주택구입_신청기한"처럼 (도메인×사유×작업)으로
카테고리를 늘리지 말 것. 같은 작업은 하나의 공통 경로가 처리하고, 사유별 차이는
데이터(WITHDRAWAL_DEADLINE_RULES, TRANSFER_BLOCK_CODES)로 표현한다.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal, Optional

from src.agents.state import RetrievedItem
from src.rules.comprehensive_tax import (
    ANNUAL_THRESHOLD,
    SEPARATE_TAXATION_RATE_OVER_THRESHOLD,
    get_pension_income_tax_rate,
)
from src.rules.default_option import NOTICE_DELAY_DAYS_EXISTING, WAIT_DAYS_AFTER_NOTICE
from src.rules.early_withdrawal import (
    WITHDRAWAL_DEADLINE_RULES,
)
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
    calculate_tax_credit,
)
from src.rules.withdrawal_limit import (
    SIX_YEAR_EXCEPTION_CUTOFF,
    SIX_YEAR_EXCEPTION_START,
    UNLIMITED_FROM_YEAR,
)
from src.agents.tax_context import personal_tax_response

DeterministicCategory = Literal[
    "세액공제_계산_입력부족",
    "세액공제_한도",
    "세금혜택_개요",
    "개인세금_입력충분성",
    "중도인출_기한판정",
    "중도인출_요건판정",
    "중도인출_일반",
    "디폴트옵션_자동매수",
    "실물이전_불가사유",
    "실물이전_개별판정",
    "연금수령한도",
    "퇴직소득세감면",
    "연금소득세_종합과세",
    "연금소득세율_연령별",
    "해당없음",
]

# router.py가 프롬프트/RouterDecision의 Literal 정의에 그대로 재사용한다.
DETERMINISTIC_CATEGORIES: tuple[str, ...] = (
    "세액공제_계산_입력부족",
    "세액공제_한도",
    "세금혜택_개요",
    "개인세금_입력충분성",
    "중도인출_기한판정",
    "중도인출_요건판정",
    "중도인출_일반",
    "디폴트옵션_자동매수",
    "실물이전_불가사유",
    "실물이전_개별판정",
    "연금수령한도",
    "퇴직소득세감면",
    "연금소득세_종합과세",
    "연금소득세율_연령별",
    "해당없음",
)


# 라우터가 "해당없음"으로 기각했을 때 코드가 되살려도 되는 카테고리.
#
# 배경: "개인세금_입력충분성"은 후보로 정확히 주어져도 라우터가 3/3 기각했다. 프롬프트
# 문구를 여러 형태로 조정해도 재현됐고(상위 판정 원칙 블록을 완전히 제거한 최소
# 프롬프트에서만 성공), 프롬프트 순종에 기댈 수 없는 실패라 코드로 되살린다.
#
# ## 여기 넣어도 되는 기준: "라우터가 판단할 게 남아있지 않은가"
#
# 후보 목록(candidate_categories)은 "이 카테고리를 검토할 신호가 질문에 있다"까지만
# 말한다. 카테고리에 따라 그 다음 판단이 더 필요할 수 있고, 그 판단은 라우터의 몫이다.
#   - 세액공제_계산_입력부족: "세액공제" 신호만으로는 부족하고, **입력값이 충분한지**를
#     더 봐야 한다. 충분하면 계산 툴로 넘겨야 하는데 이 핸들러는 입력이 충분해도 항상
#     "입력값이 부족합니다"를 반환한다 — 라우터가 정확히 기각한 케이스("연금저축
#     600만원, 총급여 5000만원인데 세액공제 얼마?")를 코드가 되살리면 오답이 나간다.
#   - 여기 담긴 셋은 후보에 오른 시점에 판정이 사실상 끝난다. 개인세금_입력충분성은
#     충분/부족을 핸들러가 스스로 분기하고, 중도인출_기한판정·실물이전_개별판정은
#     사유·상품이 인식되지 않으면 핸들러가 None을 낸다.
#
# 참고: "무관한 질문에도 답하는가"는 더 이상 기준이 아니다. 그 문제는
# deterministic_response_for가 후보 목록을 재확인하도록 고쳐 13개 전부 해결됐다.
CODE_OVERRIDABLE_CATEGORIES: frozenset[str] = frozenset({
    "개인세금_입력충분성",
    "중도인출_기한판정",
    "중도인출_요건판정",
    "실물이전_개별판정",
})


# 나이 표현 — "74세", "만 74세", "제 나이가 74" 등. 연령별 세율 판정의 신호다.
_AGE_MENTION_RE = re.compile(r"(?:만)?\d{1,3}세|나이가?\d{1,3}")

# 세금 관련 표현 — "세율/세금/얼마 떼나" 류를 폭넓게 잡는다. 정확한 제도명("연금소득세")을
# 쓰는 사용자는 소수라, 어휘 목록을 좁게 잡으면 같은 의도의 질문을 놓친다.
_TAX_AMOUNT_WORDS = (
    "세율", "세금", "몇%", "몇퍼센트", "몇프로", "얼마", "떼나", "떼가", "부과", "과세",
)

# "연금을 받는" 문맥 — 납입·가입 얘기("연금저축에 납입")와 구분한다. 연령별 세율은
# 수령 시점의 나이로 결정되므로 수령 문맥이 있어야 이 카테고리가 맞다.
_PENSION_RECEIPT_WORDS = (
    "연금받", "연금수령", "연금으로받", "수령할때", "수령시", "연금개시", "연금탈때",
    "연금을받",
)


def candidate_categories(question: str) -> list[str]:
    """1단계: 주제어만 보고 **느슨한** 후보 목록을 낸다 (동반어 요구 없음).

    router의 LLM이 이 후보들 중 실제로 맞는 카테고리를 확정한다(또는 전부 기각하고
    "해당없음"). 여러 개가 동시에 후보로 나올 수 있다(예: "세액공제" 관련 두 카테고리).

    ⚠️ 이 함수는 **넓게 잡아야 한다**. 정확한 판정은 라우터의 역할이고, 여기서 후보를
    놓치면 라우터는 그 카테고리를 아예 고려조차 못 한다. 실측(2026-08-27): "연금 + 세율"
    AND 조건과 정확한 어휘 목록을 요구하던 시절, 같은 의도의 7개 표현 중 3개가 후보를
    만들지 못했다 —
        "나 74세인데 세금 어떻게 내?"      -> [] (연금이라는 단어가 없음)
        "74세인데 얼마나 떼나요?"           -> [] (세율/세금 어휘가 목록에 없음)
        "제 나이가 74인데 세율 알려주세요"    -> [] (연금이라는 단어가 없음)
    이는 "세액공제+얼마=한도질문"으로 단정하던 사고(7cddb1f)와 같은 클래스다: 표면
    어휘 조합으로 판정하려다 표현이 조금만 달라지면 무너진다. 후보는 넓히고, 대신
    라우터가 후보 밖 카테고리를 고르지 못하도록 코드로 막는다(router._enforce_candidate_scope).
    """
    text = _compact(question)
    candidates: list[str] = []

    if personal_tax_response(question) is not None:
        candidates.append("개인세금_입력충분성")
    if "세액공제" in text:
        candidates.append("세액공제_계산_입력부족")
        candidates.append("세액공제_한도")
    if any(word in text for word in ("세금혜택", "세제혜택", "절세혜택", "세금상혜택")) or (
        "절세" in text and any(word in text for word in ("연금", "irp", "IRP", "개인사업자", "자영업"))
    ):
        candidates.append("세금혜택_개요")
    if "중도인출" in text:
        # 같은 도메인의 두 작업을 모두 후보로 낸다: 사유 목록 나열(중도인출_일반)과
        # 기한 계산·판정(중도인출_기한판정). 사유별로 후보 조건을 따로 쓰면
        # ("요양이고 요양종료일이 있으면...") 사유가 늘 때마다 조건이 늘고, 실제로
        # 요양·주택구입만 잡히고 전월세·재난·개인회생은 빠지는 비대칭이 생겼다.
        candidates.append("중도인출_일반")
        candidates.append("중도인출_기한판정")
        candidates.append("중도인출_요건판정")
    # 제도명을 그대로 쓰지 않는 표현도 잡는다 — "디폴트옵션"이라는 단어 없이
    # "기존가입자인데 언제 자동매수되나요?"처럼 묻는 경우가 많다.
    if any(word in text for word in ("디폴트옵션", "사전지정운용", "자동매수")):
        candidates.append("디폴트옵션_자동매수")
    # 마찬가지로 "실물이전"을 "옮기다/이관/이체"로 말하는 경우를 포함한다.
    if any(word in text for word in ("실물이전", "이전되", "이전가능", "옮길", "옮기", "이관", "이체")):
        # 같은 도메인이라도 "목록 나열"과 "내 상품 판정"은 다른 작업이다. 둘 다 후보로
        # 넣고 라우터가 고르게 한다 — 개별판정 경로가 없던 동안 라우터가 이런 질문을
        # 기각했고, LLM이 자유롭게 툴을 고르다 엉뚱한 툴(투자 가능 여부)을 불러
        # "네, 실물이전 문제없습니다"라는 정반대 답을 낸 실측이 있다.
        candidates.append("실물이전_불가사유")
        candidates.append("실물이전_개별판정")
    if "연금수령한도" in text or ("연금" in text and "한도" in text):
        candidates.append("연금수령한도")
    if any(word in text for word in ("퇴직소득세", "이연퇴직소득세")):
        candidates.append("퇴직소득세감면")
    if any(word in text for word in ("연금소득세", "종합과세", "분리과세")):
        candidates.append("연금소득세_종합과세")

    # 연령별 연금소득세율 — 세금 얘기 + (나이 언급 OR 연금 수령 문맥)일 때 후보에 넣는다.
    # 사용자는 "나 74세인데 세금 어떻게 내?"처럼 제도명을 생략하므로 "연금소득세"라는
    # 정확한 단어를 요구하면 같은 의도의 질문을 놓친다.
    #
    # ⚠️ "연금"이라는 단어만으로는 부족하다. 그러면 "연금저축 600만원 납입하고 총급여
    # 5000만원인데 세액공제 얼마?"가 "연금"(연금저축) + "얼마"만으로 후보가 되어,
    # 나이가 전혀 없는데도 연령별 세율표를 답하는 경로가 열린다(실측). 이 카테고리는
    # 이름 그대로 **연령별**이므로, 나이가 없다면 "연금을 받는" 문맥이라도 있어야 한다.
    has_tax_context = any(word in text for word in _TAX_AMOUNT_WORDS)
    has_receipt_context = any(word in text for word in _PENSION_RECEIPT_WORDS)
    if has_tax_context and (_AGE_MENTION_RE.search(text) or has_receipt_context):
        candidates.append("연금소득세율_연령별")

    return candidates


def deterministic_response_for(
    category: str, question: str
) -> Optional[tuple[str, list[RetrievedItem]]]:
    """2단계: router가 확정한 카테고리로 실제 정형 답변을 만든다.

    category가 "해당없음"이거나, 핸들러가 없거나, **그 카테고리가 이 질문의 후보가 아니면**
    None — 호출측(info_agent)은 LLM+툴 경로로 진행해야 한다.

    ## 왜 여기서 후보를 다시 확인하는가

    원래 설계는 "후보 생성(넓게) → 라우터 확정(정확히) → 핸들러 실행(신뢰)"이었고,
    라우터가 유일한 관문이므로 핸들러는 질문 적합성을 확인하지 않는다. 실제로 13개
    핸들러 중 10개는 반환 타입에 None이 없어 **물러날 방법 자체가 없다**("오늘 점심
    뭐 먹지"에도 세액공제 한도표를 반환한다).

    그 전제는 라우터를 우회하는 경로가 없을 때만 성립한다. router._restore_rejected_category
    (라우터가 잘못 기각한 카테고리를 코드가 되살리는 장치)가 관문을 우회하면서, 검증되지
    않은 카테고리로 정형 답변이 나갈 수 있는 구멍이 생겼다. 실측: 되살릴 대상을 넓게
    잡았더니 "연금저축 600만원 납입하고 총급여 5000만원인데 세액공제 얼마?"가
    연금소득세율_연령별로 되살아나 묻지도 않은 세율표를 답했다.

    핸들러 13개에 각각 가드를 붙이는 방법도 있으나, 새 핸들러가 추가될 때 빠뜨리면
    조용히 다시 뚫린다. candidate_categories가 이미 카테고리별 적합성을 판정하고 있으므로
    (무관한 질문에는 빈 목록을 낸다) 그 판정을 dispatch 한 곳에서 재사용한다 —
    핸들러가 몇 개든, 앞으로 몇 개가 더 생기든 자동으로 같은 보호를 받는다.
    """
    handler = _CATEGORY_HANDLERS.get(category)
    if handler is None:
        return None
    if category not in candidate_categories(question):
        return None
    return handler(question)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _won(amount: int) -> str:
    if amount % 10_000 == 0:
        return f"{amount // 10_000:,}만원"
    return f"{amount:,}원"


def _won_readable(amount: int) -> str:
    if amount % 10_000 == 0:
        return _won(amount)
    if amount >= 10_000:
        man, rest = divmod(amount, 10_000)
        if rest and rest % 1_000 == 0:
            return f"{man:,}만 {rest // 1_000}천원"
    return f"{amount:,}원"


def _pct(rate: float) -> str:
    return f"{rate * 100:g}%"


def _context(source: str, content: str) -> list[RetrievedItem]:
    return [{"source": source, "content": content, "node": "info_agent"}]


_AMOUNT_AFTER_LABEL_RE_TEMPLATE = r"{label}[^\d]{{0,12}}([0-9,]+)\s*(만원|만\s*원|원)"
_AMOUNT_BEFORE_LABEL_RE_TEMPLATE = r"([0-9,]+)\s*(만원|만\s*원|원)[^\n,.;]{{0,12}}{label}"


def _amount_to_won(raw: str, unit: str) -> int:
    value = int(raw.replace(",", ""))
    return value * 10_000 if "만" in unit else value


def _extract_labeled_amount(question: str, labels: tuple[str, ...]) -> int | None:
    compact = _compact(question)
    for label in labels:
        escaped = re.escape(label)
        patterns = (
            _AMOUNT_AFTER_LABEL_RE_TEMPLATE.format(label=escaped),
            _AMOUNT_BEFORE_LABEL_RE_TEMPLATE.format(label=escaped),
        )
        for pattern in patterns:
            match = re.search(pattern, compact, flags=re.IGNORECASE)
            if match:
                return _amount_to_won(match.group(1), match.group(2))
    return None


def _extract_tax_credit_inputs(question: str) -> dict[str, int | None]:
    return {
        "pension_savings_paid": _extract_labeled_amount(question, ("연금저축", "연저")),
        "irp_paid": _extract_labeled_amount(question, ("IRP", "irp", "개인형IRP", "개인형퇴직연금")),
        "total_salary": _extract_labeled_amount(question, ("총급여", "급여", "연봉")),
        "comprehensive_income": _extract_labeled_amount(question, ("종합소득금액", "종합소득", "사업소득")),
    }


def _has_sufficient_tax_credit_inputs(values: dict[str, int | None]) -> bool:
    has_contribution = values["pension_savings_paid"] is not None or values["irp_paid"] is not None
    has_income = values["total_salary"] is not None or values["comprehensive_income"] is not None
    return has_contribution and has_income


def _tax_credit_calculation_missing_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc41 세액공제 계산 입력값 규칙"
    values = _extract_tax_credit_inputs(question)
    if _has_sufficient_tax_credit_inputs(values):
        pension_savings_paid = values["pension_savings_paid"] or 0
        irp_paid = values["irp_paid"] or 0
        result = calculate_tax_credit(
            pension_savings_paid=pension_savings_paid,
            irp_paid=irp_paid,
            total_salary=values["total_salary"],
            comprehensive_income=values["comprehensive_income"],
        )
        income_label = (
            f"총급여 {_won(values['total_salary'])}"
            if values["total_salary"] is not None
            else f"종합소득금액 {_won(values['comprehensive_income'])}"
        )
        income_basis = (
            f"총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하"
            if values["total_salary"] is not None and values["total_salary"] <= INCOME_THRESHOLD_SALARY
            else f"총급여 {_won(INCOME_THRESHOLD_SALARY)} 초과"
            if values["total_salary"] is not None
            else f"종합소득금액 {_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하"
            if values["comprehensive_income"] <= INCOME_THRESHOLD_COMPREHENSIVE
            else f"종합소득금액 {_won(INCOME_THRESHOLD_COMPREHENSIVE)} 초과"
        )
        content = (
            f"세액공제액 계산에는 연금저축 납입액, IRP 납입액, 총급여 또는 종합소득금액이 필요합니다. "
            f"연금저축 단독 세액공제 대상 한도는 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
            f"연금저축+IRP 합산 세액공제 대상 한도는 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
            f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
            f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
            f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다. 입력 조건에서는 세액공제 대상 납입액 "
            f"{_won(result.credited_total)} x {_pct(result.credit_rate)} = {_won_readable(result.tax_credit_amount)}입니다."
        )
        lines = [
            "입력해주신 조건으로 세액공제액을 계산하면 다음과 같습니다.",
            "",
            f"- 연금저축 납입액: {_won(pension_savings_paid)}",
            f"- IRP 납입액: {_won(irp_paid)}",
            f"- 소득 기준: {income_label}",
            f"- 적용 공제율: {_pct(result.credit_rate)} ({income_basis})",
            f"- 세액공제 대상 납입액: {_won(result.credited_total)}",
            "",
            f"예상 세액공제액은 {_won(result.credited_total)} x {_pct(result.credit_rate)} = {_won_readable(result.tax_credit_amount)}입니다.",
        ]
        if result.excess_beyond_credit_limit:
            lines.append(
                f"\n세액공제 대상 한도를 초과한 납입액 {_won(result.excess_beyond_credit_limit)}은 "
                "세액공제액 계산에는 포함되지 않습니다."
            )
        if result.over_contribution_limit:
            lines.append(f"\n연금저축+IRP 합산 납입한도 {_won(TOTAL_CONTRIBUTION_LIMIT)}도 초과합니다.")
        return "\n".join(lines), _context(source, content)

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


def _tax_benefit_overview_response(question: str) -> tuple[str, list[RetrievedItem]]:
    compact = _compact(question)
    if "개인사업자" in compact or "자영업" in compact:
        source = "doc41 세액공제 규칙 및 개인사업자 IRP 가입대상"
        content = (
            "개인사업 대표는 퇴직연금에는 가입할 수 없지만 일반 IRP를 통해 자영업자로 가입할 수 있습니다. "
            f"연금저축+IRP 합산 납입한도는 연 {_won(TOTAL_CONTRIBUTION_LIMIT)}이고, 세액공제 대상 한도는 "
            f"연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, 연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
            f"개인사업자 등 종합소득자는 종합소득금액 {_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 "
            f"{_pct(CREDIT_RATE_LOW)}, 초과이면 {_pct(CREDIT_RATE_HIGH)} 세액공제율을 적용합니다. "
            "연금계좌 운용수익은 인출 전 과세이연되고, 세액공제 받은 금액과 운용수익을 연금으로 수령할 때는 "
            "연령별 연금소득세율과 사적연금소득 1,500만원 기준을 함께 확인합니다."
        )
        draft = (
            "개인사업자라면 연금 상담 범위에서는 주로 **연금저축·IRP를 활용한 절세**를 확인할 수 있습니다.\n\n"
            "- 개인사업 대표는 퇴직연금에는 가입할 수 없지만, 일반 IRP를 통해 자영업자로 가입할 수 있습니다.\n"
            "- 세액공제 대상 한도는 연금저축 단독 연 600만원, 연금저축+IRP 합산 연 900만원입니다.\n"
            "- 개인사업자처럼 종합소득 기준으로 보는 경우, 종합소득금액 4,500만원 이하이면 16.5%, 초과이면 13.2% 세액공제율이 적용됩니다.\n"
            "- 연금계좌 안의 운용수익은 인출 전까지 과세이연되며, 나중에 연금으로 받을 때는 재원·수령방식·연간 수령액에 따라 과세가 달라집니다.\n\n"
            "따라서 구체적인 세액공제액을 계산하려면 올해 연금저축 납입액, IRP 납입액, 종합소득금액을 알려주시면 됩니다."
        )
        return draft, _context(source, content)

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


def _tax_credit_limit_response(question: str) -> tuple[str, list[RetrievedItem]]:
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
    values = _extract_tax_credit_inputs(question)
    pension_savings_paid = values["pension_savings_paid"]
    asks_all_credited = any(word in _compact(question) for word in ("전부", "모두", "다세액공제", "전체"))
    if pension_savings_paid and values["irp_paid"] is None and asks_all_credited:
        credited = min(pension_savings_paid, PENSION_SAVINGS_ONLY_LIMIT)
        excess = max(0, pension_savings_paid - PENSION_SAVINGS_ONLY_LIMIT)
        if excess:
            draft = (
                f"아니요. 연금저축에 {_won(pension_savings_paid)}을 납입했더라도, **연금저축만으로는 "
                f"{_won(PENSION_SAVINGS_ONLY_LIMIT)}까지만 세액공제 대상**입니다.\n\n"
                f"- 세액공제 대상 연금저축 납입액: {_won(credited)}\n"
                f"- 연금저축 단독 한도를 넘는 금액: {_won(excess)}\n\n"
                f"연금저축과 IRP를 함께 활용하면 두 계좌 합산으로 {_won(COMBINED_CREDIT_LIMIT)}까지 "
                "세액공제 대상이 될 수 있지만, 연금저축 단독 한도 자체가 900만원으로 늘어나는 구조는 아닙니다.\n\n"
                "세액공제율은 소득 기준에 따라 16.5% 또는 13.2%가 적용됩니다."
            )
            return draft, _context(source, content)

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


def _parse_korean_dates(text: str) -> list[date]:
    dates = []
    current_year = None
    pattern = re.compile(r"(?:(\d{4})년)?(\d{1,2})월(\d{1,2})일")
    for year, month, day in pattern.findall(text):
        if year:
            current_year = int(year)
        if current_year is None:
            continue
        try:
            dates.append(date(current_year, int(month), int(day)))
        except ValueError:
            continue
    return dates


def _parse_first_korean_date_mention(text: str) -> tuple[int | None, int, int] | None:
    match = re.search(r"(?:(\d{4})년)?(\d{1,2})월(\d{1,2})일", text)
    if not match:
        return None
    year, month, day = match.groups()
    parsed_year = int(year) if year else None
    parsed_month = int(month)
    parsed_day = int(day)
    try:
        date(parsed_year or 2001, parsed_month, parsed_day)
    except ValueError:
        return None
    return parsed_year, parsed_month, parsed_day


def _format_date(value: date) -> str:
    return f"{value.year}년 {value.month}월 {value.day}일"


def _format_month_day(value: date) -> str:
    return f"{value.month}월 {value.day}일"


# 사용자 표현 -> 중도인출 사유. WITHDRAWAL_DEADLINE_RULES의 키와 일치시킨다.
_WITHDRAWAL_REASON_ALIASES: dict[str, tuple[str, ...]] = {
    "요양": ("요양", "치료", "병원", "의료비"),
    "개인회생파산": ("개인회생", "파산", "회생절차"),
    "무주택전월세": ("전월세", "전세", "월세", "임차보증금", "임대차"),
    "무주택주택구입": ("주택구입", "주택매입", "집구입", "소유권이전", "등기"),
    "재난피해": ("재난", "피해", "수해", "화재"),
}
_WITHDRAWAL_REASON_LABELS = {
    "요양": "요양",
    "개인회생파산": "개인회생·파산",
    "무주택전월세": "무주택 전월세보증금",
    "무주택주택구입": "무주택 주택구입",
    "재난피해": "재난피해",
}


def _has_final_consonant(word: str) -> bool:
    """마지막 글자에 받침이 있는지 — 조사(이/가, 이라는/라는) 선택용."""
    if not word:
        return False
    ch = word[-1]
    if not ("가" <= ch <= "힣"):
        return False
    return (ord(ch) - 0xAC00) % 28 != 0


def _with_particle(word: str, with_final: str, without_final: str) -> str:
    """받침 여부에 맞는 조사를 붙인다. 예: _with_particle("잔금지급일", "이라는", "라는")"""
    return word + (with_final if _has_final_consonant(word) else without_final)


def _detect_withdrawal_reason(text: str) -> str | None:
    """질문에서 중도인출 사유를 인식한다 (사유별 핸들러 대신 공통 경로가 쓴다)."""
    for reason, aliases in _WITHDRAWAL_REASON_ALIASES.items():
        if any(alias in text for alias in aliases):
            return reason
    return None


def _early_withdrawal_deadline_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """중도인출 신청기한 규칙을 안내한다. DB-grounded 경로에서는 exact date를 만들지 않는다.

    사유별로 핸들러를 만들면(요양_신청기한판정, 주택구입_신청기한 ...) 사유가 늘 때마다
    같은 코드를 복제하게 되고, 실제로 "요양·주택구입만 있고 전월세·재난·개인회생은 없는"
    비대칭이 생겼다. 사유 인식(_detect_withdrawal_reason)과 기한 규정
    (WITHDRAWAL_DEADLINE_RULES)을 분리해 5개 사유가 같은 경로를 쓰게 한다.

    제공 DB에는 "1개월/3개월/5년 이내"라는 규칙은 있지만, 1개월을 30일로 볼지,
    달력 기준으로 볼지, 초일·말일·휴일을 어떻게 처리할지까지는 명시되어 있지 않다.
    따라서 일반 Agent 답변에서는 마감일이나 신청일 통과 여부를 계산하지 않고, 확인된
    기준일·기간 규칙과 계산 불가 사유만 안내한다.
    """
    text = _compact(question)
    if "중도인출" not in text:
        return None
    if not any(word in text for word in ("언제까지", "기한", "이내", "언제", "신청")):
        return None

    reason = _detect_withdrawal_reason(text)
    if reason is None:
        return None
    rule = WITHDRAWAL_DEADLINE_RULES[reason]
    reason_label = _WITHDRAWAL_REASON_LABELS.get(reason, reason)
    period = f"{rule.years}년" if rule.years else f"{rule.months}개월"

    source = f"{rule.source_doc} 중도인출 {reason_label} 신청기한 규칙"
    content = (
        f"{reason_label} 사유의 중도인출 신청기한은 {rule.basis_event}로부터 {period} 이내입니다. "
        "제공 DB에는 해당 기간을 정확한 날짜로 환산하는 방식이 명시되어 있지 않습니다. "
        f"{rule.note} "
        "중도인출 원문에는 30일/90일 환산 여부, 초일산입 여부, 말일·휴일 처리, "
        "영업시간·신청 도달시점 처리 기준이 명시되어 있지 않습니다. "
        "calculation_basis=not_defined_in_source."
    )
    caveat = (
        "다만 제공 자료에는 이 기간을 정확한 날짜로 계산하는 방식, 초일산입 여부, 말일·휴일 "
        "처리, 신청서 작성일/접수일 기준이 명확히 적혀 있지 않습니다. 따라서 DB 근거만으로는 "
        "특정 날짜가 정확한 마감일인지 또는 신청기한 안인지 단정하지 않겠습니다."
    )

    dates = _parse_korean_dates(text)
    if not dates:
        # 날짜가 없으면 규정만 안내한다(기준일을 지어내지 않는다).
        draft = (
            f"{reason_label} 사유로 중도인출을 신청하는 경우, 신청기한은 "
            f"**{rule.basis_event}로부터 {period} 이내**입니다.\n\n"
            f"{rule.note}\n\n{caveat}"
        )
        return draft, _context(source, content)

    basis_date = dates[0]
    request_dates = dates[1:]
    # 연도 없이 "3월 1일"로만 말한 경우 연도를 붙여 답하면 지어낸 정보가 된다.
    mention = _parse_first_korean_date_mention(text)
    has_year = bool(mention and mention[0] is not None)
    fmt = _format_date if has_year else _format_month_day

    if not request_dates:
        draft = (
            f"**{fmt(basis_date)}이 {_with_particle(rule.basis_event, '이라는', '라는')} 조건**이라면, "
            f"제공 자료에서 확인되는 신청기한 규칙은 **{rule.basis_event}로부터 {period} 이내**입니다.\n\n"
            f"{caveat}"
        )
        return draft, _context(source, content)

    request_date_text = ", ".join(fmt(d) for d in request_dates)
    draft = (
        f"{rule.basis_event}이 {fmt(basis_date)}이면 제공 자료에서 확인되는 신청기한 규칙은 "
        f"**{rule.basis_event}로부터 {period} 이내**입니다.\n\n"
        f"다만 {request_date_text} 신청이 각각 기한 안인지 여부는 DB 근거만으로 정확히 판정하지 않겠습니다.\n\n"
        f"{caveat}"
    )
    return draft, _context(source, content)

def _early_withdrawal_general_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc46~doc50 중도인출 규칙"
    content = (
        "DB는 중도인출이 허용되지 않습니다. DC와 IRP는 법정 사유가 있으면 중도인출 대상 제도입니다. "
        "가능 사유는 6개월 이상 요양, 개인회생 또는 파산선고, 무주택자 전월세보증금, "
        "무주택자 주택구입, 재난피해입니다. 요양은 DC에서 직전 1년 의료비가 직전년도 연간임금총액의 "
        "12.5%를 초과해야 하며, IRP에는 이 비율 기준이 적용되지 않습니다. 개인회생·파산은 결정일 또는 "
        "선고일로부터 5년 이내 요건이 있습니다. 전월세보증금과 주택구입은 잔금지급일 또는 소유권 이전 "
        "등기접수일로부터 달력 기준 1개월 이내 신청 요건이 있습니다. 재난피해는 피해발생일로부터 "
        "달력 기준 3개월 이내가 원칙입니다."
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
        "전월세보증금·주택구입은 달력 기준 1개월 이내 신청 요건, 재난피해는 달력 기준 3개월 이내 "
        "신청 요건이 문제될 수 있습니다."
    )
    return draft, _context(source, content)


def _early_withdrawal_eligibility_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    text = _compact(question)
    if "중도인출" not in text:
        return None
    if not any(word in text for word in ("가능", "되나요", "할수", "되냐", "되는지", "가능한가")):
        return None

    source = "doc46~doc50 중도인출 요건판정 규칙"
    content = (
        "DB형은 중도인출이 허용되지 않습니다. DC와 IRP는 법정 사유가 있으면 중도인출 대상 제도입니다. "
        "가능 사유는 6개월 이상 요양, 개인회생 또는 파산선고, 무주택자 전월세보증금, "
        "무주택자 주택구입, 재난피해입니다. 개인워크아웃제도와 신용회복은 중도인출 사유에 해당하지 않습니다."
    )

    if "DB형" in question or re.search(r"\bDB\b", question, flags=re.IGNORECASE):
        draft = (
            "아니요. DB형 퇴직연금은 중도인출이 허용되지 않습니다.\n\n"
            "전월세보증금 같은 법정 사유가 있더라도 중도인출 대상 제도는 DC와 IRP입니다. "
            "따라서 질문 조건이 DB형이라면 전월세보증금 사유로도 중도인출할 수 없다고 보는 것이 맞습니다."
        )
        return draft, _context(source, content)

    if any(word in text for word in ("개인워크아웃", "워크아웃", "신용회복")):
        draft = (
            "아니요. 제공 자료 기준으로 개인워크아웃이나 신용회복은 퇴직연금 중도인출 사유에 해당하지 않습니다.\n\n"
            "개인회생절차개시 결정 또는 파산선고는 중도인출 사유가 될 수 있지만, 개인워크아웃·신용회복은 그 사유와 구분됩니다."
        )
        return draft, _context(source, content)

    return None


_EXISTING_MEMBER_WORDS = ("기존가입자", "기존 가입자", "기존가입", "만기가 된", "만기된", "만기 도래")
_NEW_MEMBER_WORDS = ("신규가입자", "신규 가입자", "신규가입", "최초 가입", "처음 가입", "신규 납입")


def _default_option_auto_purchase_response(question: str) -> tuple[str, list[RetrievedItem]]:
    """기존/신규가입자가 질문에 이미 특정돼 있으면 그 케이스만, 아니면 역질문+두 케이스를 함께 낸다.

    질문에 이미 없는 정보(정보 손실)는 아니지만, 특정된 사용자에게 무관한 케이스까지 섞어
    답하면 실제로 필요한 정보를 찾기 어려워진다 — 실물이전 사고와 같은 클래스(개인화 미반영)
    의 더 가벼운 변종.
    """
    source = "doc29 디폴트옵션 자동매수 규칙"
    # 일수는 반드시 규칙엔진 상수를 참조한다 — 문구에 숫자를 손으로 적으면 상수만 고쳤을 때
    # 답변이 옛 값을 계속 노출한다(실제로 28/14 오류가 이 경로로 사용자에게 나갔다).
    notice = f"4주 뒤(만기일 + {NOTICE_DELAY_DAYS_EXISTING}일)"
    wait = f"2주 뒤(통지일 + {WAIT_DAYS_AFTER_NOTICE}일)"
    business_day_note = "통지일 또는 매수예정일이 비영업일이면 익영업일로 적용됩니다."
    continuity_note = (
        "다만 대기 중 전액을 다른 상품으로 이동해 연속성이 끊기면 다음 만기분부터 "
        "다시 통지와 대기 절차를 거칠 수 있습니다."
    )
    content = (
        f"기존가입자는 상품 만기일로부터 {notice}에 통지하고, {wait}에 자동매수합니다. "
        f"신규가입자는 최초 부담금 납입 다음 영업일 통지하고, {wait}에 자동매수합니다. "
        "동일 상품 반복 만기 등 연속성이 유지되는 경우에는 통지·대기 없이 즉시 적용됩니다. "
        f"{business_day_note}"
    )

    is_existing = any(word in question for word in _EXISTING_MEMBER_WORDS)
    is_new = any(word in question for word in _NEW_MEMBER_WORDS)

    if is_existing and not is_new:
        draft = (
            f"기존가입자 기준으로 안내드리면, 디폴트옵션 자동매수는 상품 만기일로부터 {notice}에 "
            f"통지하고, {wait}에 이뤄집니다.\n\n"
            f"{business_day_note} {continuity_note}"
        )
    elif is_new and not is_existing:
        draft = (
            "신규가입자 기준으로 안내드리면, 디폴트옵션 자동매수는 최초 부담금 납입 다음 영업일에 "
            f"통지하고, {wait}에 이뤄집니다.\n\n"
            f"{business_day_note}"
        )
    else:
        draft = (
            "기존가입자인지 신규가입자인지에 따라 자동매수 시점이 달라 정확히 안내드리려면 "
            "어느 쪽에 해당하시는지 알려주시면 좋습니다. 다만 일반적으로는 다음과 같습니다.\n\n"
            f"- 기존가입자: 상품 만기일로부터 {notice} 통지, {wait} 자동매수\n"
            f"- 신규가입자: 최초 부담금 납입 다음 영업일 통지, {wait} 자동매수\n"
            "- 동일 상품 반복 만기처럼 연속성이 유지되는 경우: 통지·대기 없이 즉시 적용\n\n"
            f"{business_day_note} {continuity_note}"
        )
    return draft, _context(source, content)


def _in_kind_transfer_block_response(question: str) -> tuple[str, list[RetrievedItem]]:
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
    # ⚠️ 답변 목록을 손으로 고른 하드코딩 리스트로 만들면 안 된다 — 실제로 그렇게 돼 있던
    # 동안 "21. 만기(상환)"이 목록에서 빠져, 만기 때문에 이전이 막힌 사용자에게 근거(content)
    # 에는 있는 사유가 답변 본문에서는 안 보이는 사고가 났다. 근거와 답변이 같은 원천
    # (TRANSFER_BLOCK_CODES)에서 나오도록 해서 둘이 어긋날 수 없게 한다.
    definite_lines = "\n".join(
        f"- {code}. {TRANSFER_BLOCK_CODES[code]['name']}: {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in definite_codes
    )
    manual_lines = "\n".join(
        f"- {code}. {TRANSFER_BLOCK_CODES[code]['name']}: {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in manual_codes
    )
    draft = (
        "퇴직연금 실물이전이 제한될 수 있는 상품·상황은 다음과 같습니다.\n\n"
        f"[실물이전 불가사유]\n{definite_lines}\n\n"
        f"[상대 금융기관 확인이 필요한 사유]\n{manual_lines}\n\n"
        # doc34는 25개 코드 외에 "99.기타"를 두어 목록에 없는 사유도 포괄한다. 이 문장을 빼면
        # 25개가 전부인 것처럼 읽혀 근거를 과대 표현하게 된다(만기지정식 예금, 제도불일치 등이 여기 속한다).
        "이 밖에도 만기지정식 예금, 제도 불일치 등 개별 사유(99. 기타)로 이전이 제한될 수 있습니다.\n\n"
        "보유하신 상품이 어디에 해당하는지 알려주시면(예: MMF, 사모펀드, 만기 도래 여부, "
        "디폴트옵션 상품 여부) 실제 이전 가능 여부를 더 정확히 확인해 드릴 수 있습니다."
    )
    return draft, _context(source, content)


# 사용자가 쓰는 표현 -> doc34 불가사유 코드. 코드표(TRANSFER_BLOCK_CODES)의 name을 그대로
# 쓰되, 실제 질문에서 나오는 축약·구어 표현만 추가로 매핑한다.
_TRANSFER_CODE_ALIASES: dict[str, tuple[str, ...]] = {
    "01": ("소규모펀드", "소규모"),
    "03": ("사모펀드", "사모"),
    "04": ("mmf", "머니마켓", "단기금융"),
    "05": ("환매수수료",),
    "08": ("압류", "질권"),
    "09": ("만기매칭형",),
    "10": ("지분증권", "리츠"),
    "11": ("rp", "환매조건부"),
    "12": ("발행어음",),
    "13": ("금리연동형보험", "금리연동"),
    "14": ("실적배당형보험", "변액보험"),
    "16": ("규약미체결", "규약"),
    "18": ("한도초과", "예금자보호한도"),
    "19": ("자사상품",),
    "21": ("만기", "만기도래", "만기상환", "상환"),
    "22": ("환매불가",),
    "23": ("디폴트옵션",),
    "24": ("상장투자회사", "맥쿼리인프라"),
}


def _detect_transfer_codes(question: str) -> list[str]:
    """질문에서 실물이전 불가사유 코드를 인식한다 (코드표 name + 구어 표현)."""
    text = _compact(question).lower()
    detected: list[str] = []
    for code, info in TRANSFER_BLOCK_CODES.items():
        name_key = _compact(info["name"]).lower()
        aliases = _TRANSFER_CODE_ALIASES.get(code, ())
        if name_key in text or any(alias in text for alias in aliases):
            detected.append(code)
    return detected


def _in_kind_transfer_judgement_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """보유 상품이 특정된 실물이전 질문에 가능/불가를 판정한다 (목록 나열이 아님).

실물이전이 안 되는 경우는?"(목록답변)과 "MMF인데 옮길 수 있나요?"(개별판정)는
    같은 도메인이지만 다른 작업이다. 개별판정용 정형 경로가 없던 동안 라우터가 이런
    질문을 기각했고, LLM이 자유롭게 툴을 고르다 실측에서 실물이전 질문에
    check_product_pension_eligibility(연금계좌 투자 가능 여부)를 호출해 "네, 실물이전
    문제없이 진행 가능합니다"라는 정반대 답을 냈다.

    상품 유형이 인식되지 않으면 None을 돌려 목록답변이나 일반 경로로 넘긴다.
    """
    codes = _detect_transfer_codes(question)
    if not codes:
        return None

    source = "doc34 실물이전 불가사유 코드"
    blocking = [c for c in codes if not TRANSFER_BLOCK_CODES[c].get("directional")]
    manual = [c for c in codes if TRANSFER_BLOCK_CODES[c].get("directional")]

    def _describe(code: str) -> str:
        info = TRANSFER_BLOCK_CODES[code]
        return f"{code}. {info['name']}: {info['desc']}"

    content = "; ".join(_describe(c) for c in codes)

    if blocking:
        reasons = "\n".join(f"- {_describe(c)}" for c in blocking)
        draft = (
            "아니요, 말씀하신 상품은 실물이전이 제한됩니다.\n\n"
            f"해당하는 불가사유는 다음과 같습니다.\n{reasons}\n\n"
            "실물이전이 불가한 상품은 매도(환매) 후 현금으로 이전하는 방법을 검토하실 수 "
            "있습니다. 다만 매도 시점의 손익과 환매 조건은 상품마다 다르므로 가입 금융기관에 "
            "확인하시기 바랍니다."
        )
    else:
        reasons = "\n".join(f"- {_describe(c)}" for c in manual)
        draft = (
            "조건에 따라 다릅니다. 말씀하신 상품은 상대 금융기관 확인이 필요한 사유에 "
            f"해당합니다.\n\n{reasons}\n\n"
            "이관은 가능하더라도 수관 기관의 협약·상품라인업 여부에 따라 최종 가능 여부가 "
            "달라지므로, 옮기려는 금융기관에 확인이 필요합니다."
        )
    return draft, _context(source, content)


def _withdrawal_limit_response(question: str) -> tuple[str, list[RetrievedItem]]:
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


def _retirement_tax_reduction_response(question: str) -> tuple[str, list[RetrievedItem]]:
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


def _pension_income_tax_response(question: str) -> tuple[str, list[RetrievedItem]]:
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


_AGE_RE = re.compile(r"(?:만\s*)?(\d{1,3})\s*세")
_LIFETIME_ANNUITY_WORDS = ("종신연금", "종신형", "종신 연금")


def _extract_age(question: str) -> Optional[int]:
    """질문에서 나이를 뽑는다. 연금 수령 가능 범위(55~120) 밖이면 무시한다.

70세 이상 80세 미만" 같은 제도 설명 인용이 섞이면 오탐이 나므로, 나이가 여러 개
    등장하면 특정하지 않는다(None) — 하나로 단정하는 것보다 조건 부족으로 처리하는 편이 안전하다.
    """
    ages = [int(m.group(1)) for m in _AGE_RE.finditer(question or "")]
    valid = [a for a in ages if 55 <= a <= 120]
    if len(valid) != 1:
        return None
    return valid[0]


def _age_bracket_label(age: int) -> str:
    if age < 70:
        return "만 55세 이상 70세 미만"
    if age < 80:
        return "만 70세 이상 80세 미만"
    return "만 80세 이상"


def _pension_income_tax_rate_response(question: str) -> tuple[str, list[RetrievedItem]]:
    """연령별 연금소득세율 — 조건이 충분하면 확정, 부족하면 구간 확정 + 분기 + 역질문.

    ⚠️ 나이만으로 세율을 단정하면 안 된다(실측 사고: "만 74세 → 3.3%"라고 답했으나 정답은
    4.4%). 같은 나이라도 ①종신연금 여부 ②연 1,500만원 초과 여부 ③인출 재원(퇴직금 재원은
    이연퇴직소득세 체계)에 따라 적용 세율이 달라지기 때문이다. 그래서 확정 가능한 부분만
    규칙엔진으로 확정하고, 나머지는 분기를 그대로 제시한 뒤 부족한 조건을 되묻는다 —
    답변을 포기하지 않으면서 틀린 단정도 하지 않는 방식.
    """
    source = "doc38 연금소득세율 규칙"
    content = (
        f"1,500만원 이내 구간의 연금소득세율은 만 55세 이상 70세 미만 "
        f"{_pct(get_pension_income_tax_rate(55))}, 70세 이상 80세 미만 "
        f"{_pct(get_pension_income_tax_rate(70))}, 80세 이상 "
        f"{_pct(get_pension_income_tax_rate(80))}입니다. 종신연금을 수령하면 연령과 무관하게 "
        f"{_pct(get_pension_income_tax_rate(55, is_lifetime_annuity=True))}가 적용됩니다. "
        f"사적연금소득이 연 {_won(ANNUAL_THRESHOLD)}을 초과하면 종합과세 또는 "
        f"{_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세를 선택합니다. "
        "이 세율은 세액공제 받은 납입금과 운용수익 재원에 적용되며, 퇴직금(이연퇴직소득) "
        "재원은 이연퇴직소득세 감면 체계가 별도로 적용됩니다."
    )

    age = _extract_age(question)
    is_lifetime = any(word in question for word in _LIFETIME_ANNUITY_WORDS)

    if age is None:
        draft = (
            "연금소득세율은 연금을 받는 시점의 나이에 따라 달라집니다.\n\n"
            f"- {_age_bracket_label(55)}: {_pct(get_pension_income_tax_rate(55))}\n"
            f"- {_age_bracket_label(70)}: {_pct(get_pension_income_tax_rate(70))}\n"
            f"- {_age_bracket_label(80)}: {_pct(get_pension_income_tax_rate(80))}\n\n"
            f"다만 종신연금으로 받으시면 나이와 관계없이 "
            f"{_pct(get_pension_income_tax_rate(55, is_lifetime_annuity=True))}가 적용됩니다.\n\n"
            f"또한 위 세율은 사적연금소득이 연 {_won(ANNUAL_THRESHOLD)} 이내일 때 적용되며, "
            f"초과하면 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세 중 "
            "선택하게 됩니다.\n\n"
            "정확한 적용 세율을 확인하시려면 연금 수령 시점의 나이, 종신연금 여부, 연간 "
            "예상 수령액을 알려주세요."
        )
        return draft, _context(source, content)

    bracket = _age_bracket_label(age)
    base_rate = get_pension_income_tax_rate(age)
    lifetime_rate = get_pension_income_tax_rate(age, is_lifetime_annuity=True)

    if is_lifetime:
        draft = (
            f"만 {age}세에 종신연금으로 수령하시는 경우 연금소득세율은 "
            f"**{_pct(lifetime_rate)}**입니다. 종신연금은 연령과 무관하게 동일한 세율이 적용됩니다.\n\n"
            f"다만 이 세율은 사적연금소득이 연 {_won(ANNUAL_THRESHOLD)} 이내일 때 적용되며, "
            f"초과하면 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세 중 "
            "선택하게 됩니다."
        )
    else:
        draft = (
            f"만 {age}세는 '{bracket}' 구간에 해당하며, 이 구간의 연금소득세율은 "
            f"**{_pct(base_rate)}**입니다.\n\n"
            "다만 아래 조건에 해당하면 적용 세율이 달라집니다.\n"
            f"- 종신연금으로 수령하는 경우: 나이와 무관하게 {_pct(lifetime_rate)}\n"
            f"- 사적연금소득이 연 {_won(ANNUAL_THRESHOLD)}을 초과하는 경우: 종합과세 또는 "
            f"{_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세 중 선택\n\n"
            "또한 이 세율은 세액공제를 받은 납입금과 운용수익 재원에 적용됩니다. 퇴직금을 "
            "연금으로 받는 경우에는 이연퇴직소득세 감면 체계가 별도로 적용됩니다.\n\n"
            "종신연금 여부와 연간 예상 수령액을 알려주시면 더 정확히 안내드릴 수 있습니다."
        )
    return draft, _context(source, content)


_CATEGORY_HANDLERS = {
    "세액공제_계산_입력부족": _tax_credit_calculation_missing_response,
    "세액공제_한도": _tax_credit_limit_response,
    "세금혜택_개요": _tax_benefit_overview_response,
    "개인세금_입력충분성": personal_tax_response,
    "중도인출_기한판정": _early_withdrawal_deadline_response,
    "중도인출_요건판정": _early_withdrawal_eligibility_response,
    "중도인출_일반": _early_withdrawal_general_response,
    "디폴트옵션_자동매수": _default_option_auto_purchase_response,
    "실물이전_불가사유": _in_kind_transfer_block_response,
    "실물이전_개별판정": _in_kind_transfer_judgement_response,
    "연금수령한도": _withdrawal_limit_response,
    "퇴직소득세감면": _retirement_tax_reduction_response,
    "연금소득세_종합과세": _pension_income_tax_response,
    "연금소득세율_연령별": _pension_income_tax_rate_response,
}
