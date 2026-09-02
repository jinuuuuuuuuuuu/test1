"""Input sufficiency gate for personal pension tax questions.

This layer intentionally stays separate from rule calculators. Rule modules
calculate only when explicit values are present; this module decides whether
the question provided enough values to call them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.agents.state import RetrievedItem
from src.rules.comprehensive_tax import (
    ANNUAL_THRESHOLD,
    determine_comprehensive_tax,
)
from src.rules.retirement_tax_reduction import get_deferred_retirement_tax_rate

# 사용자가 "세금"이라는 주제를 직접 꺼냈는지 판정할 때 쓰는 어휘.
#
# ⚠️ 이 판정과 _is_personal_tax_question()을 혼동하면 안 된다. 후자는 "계산 게이트로
# 보낼까"를 판정하는 함수라, 세금을 물었더라도 순수 세율 비교 질문이면 False를 낸다
# (실측 no.115 "퇴직금 일시금 vs 연금 중 세금이 더 적은 쪽은?" -> asks_rate_comparison에
# 걸려 False). Guardian의 EXPLICIT_USER_TOPIC 판정에 그 함수를 쓰면 "세금을 이미 물은
# 질문"에 파수꾼이 또 세금 이야기를 덧붙이는 중복이 생긴다.
TAX_TOPIC_WORDS: tuple[str, ...] = (
    "세금",
    "세율",
    "세액공제",
    "공제한도",
    "연금소득세",
    "종합과세",
    "분리과세",
    "퇴직소득세",
    "이연퇴직소득세",
    "감면",
    "절세",
    "과세",
)

ReceiptType = Literal["pension", "non_pension"]
SourceType = Literal[
    "tax_deducted_contribution_and_return",
    "retirement_benefit",
    "non_tax_deducted_principal",
]
TaxBranch = Literal[
    "tax_deducted_pension",
    "tax_deducted_non_pension",
    "retirement_benefit_pension",
    "retirement_benefit_non_pension",
    "non_tax_deducted_principal",
]


@dataclass
class TaxContext:
    age: int | None = None
    receipt_type: ReceiptType | None = None
    source_type: SourceType | None = None
    lifetime_annuity: bool | None = None
    annual_private_pension_income: int | None = None
    actual_pension_year: int | None = None
    asks_to_assume: bool = False


def personal_tax_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    compact = _compact(question)
    if not _is_personal_tax_question(compact):
        return None

    context = extract_tax_context(question)
    branch = determine_tax_branch(context)

    if branch is None:
        return _branch_clarification_response(context)

    missing = required_missing_fields(branch, context)
    if missing:
        return _branch_missing_fields_response(branch, context, missing)

    return _complete_tax_response(branch, context)


def extract_tax_context(question: str) -> TaxContext:
    compact = _compact(question)
    return TaxContext(
        age=_extract_age(compact),
        receipt_type=_extract_receipt_type(compact),
        source_type=_extract_source_type(compact),
        lifetime_annuity=_extract_lifetime_annuity(compact),
        annual_private_pension_income=_extract_annual_amount(compact),
        actual_pension_year=_extract_actual_pension_year(compact),
        asks_to_assume=any(word in compact for word in ("알아서", "대충", "가정", "적당히")),
    )


def determine_tax_branch(context: TaxContext) -> TaxBranch | None:
    if context.source_type == "non_tax_deducted_principal":
        return "non_tax_deducted_principal"
    if context.source_type == "retirement_benefit":
        if context.receipt_type == "pension":
            return "retirement_benefit_pension"
        if context.receipt_type == "non_pension":
            return "retirement_benefit_non_pension"
    if context.source_type == "tax_deducted_contribution_and_return":
        if context.receipt_type == "pension":
            return "tax_deducted_pension"
        if context.receipt_type == "non_pension":
            return "tax_deducted_non_pension"
    return None


def required_missing_fields(branch: TaxBranch, context: TaxContext) -> list[str]:
    missing = []
    if branch == "tax_deducted_pension":
        if context.annual_private_pension_income is None:
            missing.append("annual_private_pension_income")
        if context.age is None:
            missing.append("age")
        if (
            context.annual_private_pension_income is None
            or context.annual_private_pension_income <= ANNUAL_THRESHOLD
        ) and context.lifetime_annuity is None:
            missing.append("lifetime_annuity")
    elif branch == "retirement_benefit_pension":
        if context.actual_pension_year is None:
            missing.append("actual_pension_year")
    return missing


def _is_personal_tax_question(compact: str) -> bool:
    """개인세금_입력충분성 게이트로 보낼지 판정한다.

    이 게이트는 "세율(%) 자체가 아니라 실제 원화 세액·한도를 계산해 달라"는 질문
    전용이다. 세율 자체를 묻는 질문(나이만 있으면 답이 완결됨)은
    연금소득세율_연령별 핸들러가 이미 구간표 확정 + 종신연금/1,500만원 초과 같은
    남은 조건은 분기로 안내하도록 정교하게 처리하므로, 여기서 가로채면 그 핸들러가
    아예 호출되지 못한다.

    ⚠️ 실측(500문항 평가): 나이 언급 하나만으로 has_personal_signal이 True가 되던
    이전 로직은 "제가 60세인데 연금소득세율이 몇 %인가요?"조차 계산 질문으로 오판해
    45건 중 43건(96%)을 역질문으로 되돌렸다 — asks_general_rate_rule 어휘("세율표",
    "연령별" 등 4개)가 "세율이 몇 %"·"세율이 얼마" 같은 실제 표현을 못 잡았기 때문이다.
    핵심 판별선은 어휘 목록을 늘리는 게 아니라 "세율(%)만 물었나 vs 세액(원)이나
    복합 조건을 물었나"이므로, 그 기준 자체를 판정에 반영한다.
    """
    # ⚠️ TAX_TOPIC_WORDS를 그대로 쓰지 않는다 — 그 목록은 Guardian의 "세금 주제를
    # 물었나" 판정용이라 "감면/절세/과세"까지 포함하는데, 이 계산 게이트에 그대로
    # 적용하면 "절세 방법 알려줘" 같은 일반 질문까지 계산 경로로 끌어온다.
    has_tax_word = any(
        word in compact
        for word in (
            "세금",
            "세율",
            "연금소득세",
            "종합과세",
            "분리과세",
            "퇴직소득세",
            "이연퇴직소득세",
        )
    )
    if not has_tax_word:
        return False

    # 세율(%) 자체를 묻는 표현 — 나이 구간만 알면 완결되는 질문. 이 신호가 있고
    # 아래 계산 신호가 없으면 연금소득세율_연령별에 맡긴다.
    #
    # "세율"이라는 단어 자체를 기본 신호로 삼는다 — "세율이 몇%", "세율 기준",
    # "세율표" 등 표현이 다양해서 특정 조사·어미 조합만 나열하면 계속 빠진다
    # (실측: "세율 기준이 어떻게 돼?"가 "세율이/세율은/세율표" 어디에도 안 걸려
    # 계산 게이트로 잘못 넘어갔다). "세금 얼마나 떼요/떼나요"도 "세율"이라는 단어
    # 없이 같은 유형(나이 구간만으로 완결)이라 별도로 포함한다.
    # ⚠️ "세율"이라는 단어에만 의존하면 안 된다 — 실측 no.77("연금소득세는 몇 %인가요?"),
    # no.336("세금이 적은 쪽은 뭔가요?")처럼 세율을 묻는 가장 흔한 표현에 정작 "세율"이라는
    # 단어가 없다. 이 게이트가 가로채는 바람에 나이를 알면서도 세율을 말하지 않고
    # "수령 방식과 재원"을 되물었다. 판별선은 어휘가 아니라 "%(비율)를 물었나"이므로,
    # 퍼센트 기호·"몇 프로" 같은 비율 표현과 세율 비교 질문을 함께 본다.
    asks_percent = bool(re.search(r"몇\s*%|몇\s*퍼센트|몇\s*프로|%인가요|%예요|%인지", compact))
    # 세율 비교("55세와 70세 중 세금이 적은 쪽") — 금액이 아니라 구간별 세율 비교가 답이다.
    asks_rate_comparison = any(
        word in compact for word in ("적은쪽", "낮은쪽", "유리한쪽", "차이가얼마", "차이나")
    )
    asks_rate_only = (
        "세율" in compact
        or asks_percent
        or asks_rate_comparison
        or any(word in compact for word in ("얼마나떼", "얼마떼"))
    )

    # 세율 하나로 안 끝나는 신호 — 실제 원화 세액, 구체적 재원, 한도 계산 등
    # 세율표만으로는 답할 수 없는 조건이 함께 제시된 경우.
    #
    # ⚠️ "얼마나 떼나요/얼마 떼요"는 여기 넣지 않는다 — "세금을 떼다"는 세율만
    # 묻는 질문에도 자연스럽게 쓰이는 표현이라("여든 살인데 세금 얼마나 떼요?") 계산
    # 신호로 잡으면 오히려 순수 세율 질문을 계산 게이트로 도로 밀어넣는다(실측).
    # ⚠️ "세금얼마"는 부분문자열이라 "세금 얼마나 떼요"(순수 세율 질문)에도 걸린다
    # ("세금얼마" + "나떼요"). "세금 얼마 내나요/얼마인가요"처럼 뒤에 "나"가 붙지 않는
    # 형태만 계산 신호로 잡는다(실측 오탐 확인 후 좁힘).
    has_actual_calculation_signal = any(
        word in compact
        for word in (
            "제경우",
            "내경우",
            "실제",
            "계산",
            "얼마내",
            "얼마를내",
            "어떻게내",
        )
    ) or bool(re.search(r"세금얼마(?!나)", compact)) or bool(
        re.search(r"\d[\d,]*\s?(?:만원|억원|원)", compact)
    )  # 구체적 금액 언급
    if asks_rate_only and not has_actual_calculation_signal:
        return False

    has_personal_signal = bool(_extract_age(compact)) or any(
        word in compact
        for word in (
            "나",
            "내",
            "제가",
            "받아",
            "받고",
            "수령",
            "인출",
            "계산",
            "얼마나",
            "얼마",
        )
    )
    return has_personal_signal


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _extract_age(compact: str) -> int | None:
    match = re.search(r"(?:만)?(\d{2,3})세", compact)
    if not match:
        return None
    age = int(match.group(1))
    if age < 0 or age > 120:
        return None
    return age


# "연금이 아니라 한꺼번에"처럼 연금수령을 **부정**하는 표현. 이게 있으면 문장에
# "연금으로"가 함께 있어도 연금외수령으로 본다 — 부정을 놓치면 정반대로 판정한다.
_NON_PENSION_NEGATION_PATTERNS = (
    "연금으로안", "연금이아니", "연금말고", "연금안받", "연금대신",
)

# 그 자체로 "연금외수령"을 뜻하는 용어 — 문맥 없이도 판정할 수 있다.
_LUMP_SUM_WORDS = ("연금외", "일시금", "일시인출", "한도초과", "전액인출")

# 수령/납입 어느 쪽에도 쓰이는 모호한 구어 표현. 이것만으로 판정하면 안 된다.
# ⚠️ 실측 no.325 "연말에 한꺼번에 900만원 넣어도 세액공제 되나요?"는 **납입**을 묻는
# 질문인데 "한꺼번에"만 보고 연금외수령으로 오판했다(당시엔 재원이 없어 branch=None으로
# 남아 답변에는 영향이 없었지만, 재원이 함께 언급되면 정반대 세제 판정이 나간다).
#
# "중도"도 같은 이유로 확정 목록(_LUMP_SUM_WORDS)이 아니라 여기에 둔다 — "중도인출
# 가능한가요"처럼 재원 언급 없는 자격 질문이 47건 중 대다수라, 문맥 없이 확정하면
# 그 질문들까지 영향권에 들어온다. "인출"이 이미 _RECEIPT_CONTEXT_WORDS에 있어
# "중도인출"은 자동으로 조합 조건을 만족한다(실측 no.106 "세액공제 받은 원금과
# 운용수익을 중도에 찾으면"은 source_type은 잡히는데 receipt_type만 None이라
# 역질문으로 빠졌다 — 명백한 연금외수령인데도).
_AMBIGUOUS_LUMP_SUM_WORDS = ("한번에", "한꺼번에", "목돈으로", "통째로", "중도")

# 위 모호한 표현이 "수령" 문맥일 때만 연금외수령으로 본다.
_RECEIPT_CONTEXT_WORDS = ("받", "수령", "찾", "인출", "빼", "해지", "지급")


def _extract_receipt_type(compact: str) -> ReceiptType | None:
    if any(pattern in compact for pattern in _NON_PENSION_NEGATION_PATTERNS):
        return "non_pension"
    if any(word in compact for word in _LUMP_SUM_WORDS):
        return "non_pension"
    if any(word in compact for word in _AMBIGUOUS_LUMP_SUM_WORDS) and any(
        word in compact for word in _RECEIPT_CONTEXT_WORDS
    ):
        return "non_pension"
    if any(word in compact for word in ("연금수령", "연금으로", "연금받", "종신연금")):
        return "pension"
    return None


def _extract_source_type(compact: str) -> SourceType | None:
    if any(
        word in compact
        for word in (
            "세액공제안받은",
            "세액공제받지않은",
            "세액공제받지않는",
            "비과세원금",
        )
    ):
        return "non_tax_deducted_principal"
    if any(word in compact for word in ("퇴직금", "퇴직소득", "이연퇴직")):
        return "retirement_benefit"
    if any(word in compact for word in ("세액공제받은", "세액공제받고", "세액공제된", "운용수익")):
        return "tax_deducted_contribution_and_return"
    return None


def _extract_lifetime_annuity(compact: str) -> bool | None:
    if any(word in compact for word in ("비종신", "종신아니", "종신연금아니", "종신연금아님")):
        return False
    if "종신연금" in compact:
        return True
    return None


def _extract_annual_amount(compact: str) -> int | None:
    patterns = (
        r"연(?:간)?([0-9,]+)만원",
        r"연(?:간)?([0-9,]+)원",
    )
    for pattern in patterns:
        match = re.search(pattern, compact)
        if not match:
            continue
        raw = int(match.group(1).replace(",", ""))
        return raw * 10_000 if pattern.endswith("만원") else raw
    return None


def _extract_actual_pension_year(compact: str) -> int | None:
    match = re.search(r"(?:연금실제수령연차|실제수령연차|수령연차)(\d{1,2})년차", compact)
    if not match:
        match = re.search(r"(\d{1,2})년차", compact)
    if not match:
        return None
    year = int(match.group(1))
    return year if year >= 1 else None


def _branch_clarification_response(context: TaxContext) -> tuple[str, list[RetrievedItem]]:
    source = "doc38~doc40 개인 연금세금 입력 충분성 규칙"
    content = (
        "연금계좌 세금은 수령 방식과 재원 종류에 따라 계산 방식이 달라집니다. 세액공제 받은 "
        "납입금과 운용수익은 연 1,500만원 초과 여부와 연령별 세율을 확인합니다. 퇴직금 재원은 "
        "연금실제수령연차별 이연퇴직소득세 감면율을 확인합니다. 세액공제 받지 않은 원금과 퇴직금은 "
        "사적연금소득 1,500만원 판정에서 제외합니다."
    )
    known = _known_conditions(context)
    prefix = f"현재 확인된 정보는 {known}입니다.\n\n" if known else ""
    assumption_warning = (
        "사용자가 임의 가정을 요청하더라도 세금 계산에 필요한 전제는 대신 채우지 않겠습니다.\n\n"
        if context.asks_to_assume
        else ""
    )
    questions = []
    if context.receipt_type is None:
        questions.append("연금으로 받고 있나요, 연금 외 방식으로 인출하나요?")
    if context.source_type is None:
        questions.append("받는 돈은 퇴직금인가요, 세액공제 받은 납입금·운용수익인가요, 세액공제 받지 않은 원금인가요?")
    question_lines = "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions, start=1))

    draft = (
        f"{prefix}{assumption_warning}"
        "현재 질문만으로는 적용할 세금 계산 방식을 확정할 수 없습니다. "
        "연금계좌 세금은 먼저 **수령 방식**과 **돈의 출처**가 확인되어야 합니다.\n\n"
        "추가로 필요한 정보는 다음과 같습니다.\n"
        f"{question_lines}\n\n"
        "이 정보가 확인되지 않은 상태에서는 나이만으로 세율이나 세금을 확정하지 않겠습니다."
    )
    return draft, _context(source, content)


def _branch_missing_fields_response(
    branch: TaxBranch,
    context: TaxContext,
    missing: list[str],
) -> tuple[str, list[RetrievedItem]]:
    source = _source_for_branch(branch)
    content = _content_for_branch(branch, context)
    known = _known_conditions(context)
    missing_lines = "\n".join(f"- {_question_for_field(field)}" for field in missing)
    draft = (
        f"현재 확인된 정보는 {known}입니다.\n\n"
        f"{_branch_summary(branch)} "
        "다만 아직 계산에 필요한 값이 부족해서 세율이나 세금을 확정하지 않겠습니다.\n\n"
        "추가로 필요한 정보는 다음과 같습니다.\n"
        f"{missing_lines}"
    )
    return draft, _context(source, content)


def _complete_tax_response(branch: TaxBranch, context: TaxContext) -> tuple[str, list[RetrievedItem]]:
    source = _source_for_branch(branch)
    content = _content_for_branch(branch, context)

    if branch == "tax_deducted_pension":
        result = determine_comprehensive_tax(
            context.annual_private_pension_income,
            context.age,
            context.lifetime_annuity,
        )
        if result.exceeds_threshold:
            flat_tax_amount = round(
                context.annual_private_pension_income * result.optional_flat_rate_if_exceeded
            )
            draft = (
                "세액공제 받은 납입금·운용수익 재원을 연금으로 수령하는 경우입니다.\n\n"
                f"연간 과세대상 사적연금소득이 {_won(context.annual_private_pension_income)}이므로 "
                f"{_won(ANNUAL_THRESHOLD)}을 초과합니다. 이 경우 종합과세 또는 16.5% 분리과세를 선택할 수 있습니다. "
                f"16.5% 분리과세를 선택한다면 단순 계산 세액은 {_won(context.annual_private_pension_income)} x 16.5% = "
                f"{_won(flat_tax_amount)}입니다.\n\n"
                "1,500만원 초과 여부 판단에는 세액공제 받은 납입금과 운용수익 재원만 포함하고, "
                "세액공제 받지 않은 원금과 퇴직금 재원은 제외합니다. 종합과세를 선택하는 경우의 실제 세액은 "
                "다른 종합소득과 공제 항목에 따라 달라질 수 있습니다."
            )
        else:
            tax_amount = round(context.annual_private_pension_income * result.separate_tax_rate)
            draft = (
                "세액공제 받은 납입금·운용수익 재원을 연금으로 수령하는 경우입니다.\n\n"
                f"연간 과세대상 사적연금소득이 {_won(context.annual_private_pension_income)}으로 "
                f"{_won(ANNUAL_THRESHOLD)} 이내이므로 연령별 연금소득세율로 분리과세됩니다. "
                f"현재 입력 조건의 적용 세율은 {_pct(result.separate_tax_rate)}입니다.\n\n"
                "예상 연금소득세는 다음과 같습니다.\n"
                f"{_won(context.annual_private_pension_income)} x {_pct(result.separate_tax_rate)} = {_won(tax_amount)}\n\n"
                f"따라서 입력 조건 기준 예상 세금은 {_won(tax_amount)}입니다."
            )
        return draft, _context(source, content)

    if branch == "retirement_benefit_pension":
        result = get_deferred_retirement_tax_rate(context.actual_pension_year, is_pension_receipt=True)
        draft = (
            "퇴직금을 연금으로 수령하는 경우에는 연금실제수령연차에 따라 이연퇴직소득세 감면율이 달라집니다.\n\n"
            f"연금실제수령연차가 {context.actual_pension_year}년차이면 이연퇴직소득세의 "
            f"{_pct(result.payment_ratio)}를 납부하고 {_pct(result.reduction_ratio)}를 감면합니다. "
            "여기서 연금실제수령연차는 실제로 인출한 해만 누적되는 값입니다."
        )
        return draft, _context(source, content)

    if branch == "retirement_benefit_non_pension":
        draft = (
            "퇴직금 재원을 연금외수령하는 경우에는 이연퇴직소득세 감면이 적용되지 않고 전액 납부 대상입니다. "
            "연금실제수령연차별 감면은 퇴직금을 연금으로 수령하는 경우에 적용됩니다."
        )
        return draft, _context(source, content)

    if branch == "non_tax_deducted_principal":
        draft = (
            "세액공제 받지 않은 원금은 사적연금소득 1,500만원 초과 여부를 판단할 때 포함하지 않는 재원입니다. "
            "따라서 이 재원만 묻는 질문이라면 사적연금소득 기준이나 퇴직금 감면 계산으로 넘어가지 않습니다. "
            "다만 실제 인출 처리와 과세 여부는 해당 금액이 정말 세액공제 받지 않은 원금인지 증빙되는지가 중요합니다."
        )
        return draft, _context(source, content)

    if branch == "tax_deducted_non_pension":
        draft = (
            "세액공제 받은 납입금과 운용수익 재원을 연금외수령(연금이 아닌 방식으로 인출)하면, "
            "전액 16.5% 기타소득세가 부과됩니다. 연금으로 받을 때 적용되는 연령별 연금소득세율"
            "(3.3~5.5%)이나 1,500만원 초과 시의 종합과세·16.5% 분리과세 선택 규정은 연금수령 "
            "전용이라 여기에는 적용되지 않습니다."
        )
        return draft, _context(
            "doc38 연금수령한도 초과 기타소득세·인출순서 표",
            "기타소득세: 세액공제 납입금과 운용수익에서 연금외수령(또는 연금수령한도 초과 수령)하는 "
            "금액은 전액 16.5% 기타소득세가 부과됩니다. 연금의 형태로 받더라도 한도 초과분은 "
            "연금수령이 아닙니다.",
        )

    draft = (
        "세액공제 받은 납입금·운용수익을 연금외 방식으로 인출하는 경우는 연금수령과 과세 방식이 달라질 수 있습니다. "
        "현재 규칙에서는 이 경우의 구체 세액 계산을 단정하지 않고, 연금외수령 여부와 적용 사유를 추가 확인해야 합니다."
    )
    return draft, _context(source, content)


def _context(source: str, content: str) -> list[RetrievedItem]:
    return [{"source": source, "content": content, "node": "info_agent"}]


def _known_conditions(context: TaxContext) -> str:
    parts = []
    if context.age is not None:
        parts.append(f"나이 {context.age}세")
    if context.receipt_type is not None:
        parts.append("연금수령" if context.receipt_type == "pension" else "연금외수령")
    if context.source_type is not None:
        parts.append(_source_label(context.source_type))
    if context.lifetime_annuity is not None:
        parts.append("종신연금" if context.lifetime_annuity else "비종신연금")
    if context.annual_private_pension_income is not None:
        parts.append(f"연간 과세대상 사적연금소득 {_won(context.annual_private_pension_income)}")
    if context.actual_pension_year is not None:
        parts.append(f"연금실제수령연차 {context.actual_pension_year}년차")
    return ", ".join(parts)


def _source_label(source_type: SourceType) -> str:
    if source_type == "tax_deducted_contribution_and_return":
        return "세액공제 받은 납입금·운용수익"
    if source_type == "retirement_benefit":
        return "퇴직금"
    return "세액공제 받지 않은 원금"


def _branch_summary(branch: TaxBranch) -> str:
    if branch == "tax_deducted_pension":
        return "세액공제 받은 납입금·운용수익을 연금으로 받는 경우입니다."
    if branch == "retirement_benefit_pension":
        return "퇴직금을 연금으로 받는 경우입니다."
    if branch == "retirement_benefit_non_pension":
        return "퇴직금을 연금외 방식으로 인출하는 경우입니다."
    if branch == "non_tax_deducted_principal":
        return "세액공제 받지 않은 원금을 인출하는 경우입니다."
    return "세액공제 받은 납입금·운용수익을 연금외 방식으로 인출하는 경우입니다."


def _question_for_field(field: str) -> str:
    return {
        "annual_private_pension_income": "세액공제 받은 납입금·운용수익 재원에서 올해 연금으로 받을 금액은 얼마인가요?",
        "age": "연금수령일 현재 만 나이는 몇 세인가요?",
        "lifetime_annuity": "종신연금으로 받나요, 비종신 연금으로 받나요?",
        "actual_pension_year": "퇴직금을 실제로 연금으로 받은 해가 올해 몇 년차인가요? 즉, 연금실제수령연차가 몇 년차인가요?",
    }[field]


def _source_for_branch(branch: TaxBranch) -> str:
    if branch.startswith("tax_deducted"):
        return "doc38 연금소득 종합과세 및 연령별 세율 규칙"
    if branch.startswith("retirement_benefit"):
        return "doc39~doc40 이연퇴직소득세 감면 규칙"
    return "doc38 재원별 사적연금소득 판정 제외 규칙"


def _content_for_branch(branch: TaxBranch, context: TaxContext | None = None) -> str:
    if branch.startswith("tax_deducted"):
        content = (
            f"세액공제 받은 납입금과 운용수익 재원은 사적연금소득 {_won(ANNUAL_THRESHOLD)} 초과 여부를 "
            "판단합니다. 1,500만원 이내이면 연령별 연금소득세율로 분리과세하고, 70세 이상 80세 미만은 "
            "4.4%, 종신연금은 연령과 무관하게 3.3%입니다. 1,500만원 초과 시 종합과세 또는 16.5% "
            "분리과세를 선택할 수 있습니다."
        )
        if (
            context is not None
            and context.source_type == "tax_deducted_contribution_and_return"
            and context.receipt_type == "pension"
            and context.annual_private_pension_income is not None
            and context.age is not None
            and context.lifetime_annuity is not None
        ):
            result = determine_comprehensive_tax(
                context.annual_private_pension_income,
                context.age,
                context.lifetime_annuity,
            )
            if result.exceeds_threshold:
                flat_tax_amount = round(
                    context.annual_private_pension_income * result.optional_flat_rate_if_exceeded
                )
                content += (
                    f" 입력 조건에서는 {_won(context.annual_private_pension_income)} x "
                    f"{_pct(result.optional_flat_rate_if_exceeded)} = {_won(flat_tax_amount)}입니다."
                )
            else:
                tax_amount = round(context.annual_private_pension_income * result.separate_tax_rate)
                content += (
                    f" 입력 조건에서는 {_won(context.annual_private_pension_income)} x "
                    f"{_pct(result.separate_tax_rate)} = {_won(tax_amount)}입니다."
                )
        return content
    if branch.startswith("retirement_benefit"):
        return (
            "퇴직금을 연금으로 수령하면 연금실제수령연차 1~10년차는 이연퇴직소득세의 70%를 납부하고 "
            "30%를 감면합니다. 11~20년차는 60%를 납부하고 40%를 감면합니다. "
            "21년차 이상은 50%를 납부하고 50%를 감면합니다. 연금외수령은 감면 없이 전액 납부합니다."
        )
    return (
        "세액공제 받지 않은 원금과 퇴직금 재원은 사적연금소득 1,500만원 초과 여부 판단에서 제외합니다."
    )


def _won(amount: int) -> str:
    if amount % 10_000 == 0:
        return f"{amount // 10_000:,}만원"
    return f"{amount:,}원"


def _pct(rate: float) -> str:
    return f"{rate * 100:g}%"
