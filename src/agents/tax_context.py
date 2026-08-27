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

    has_actual_calculation_signal = any(
        word in compact
        for word in (
            "제경우",
            "내경우",
            "실제",
            "계산",
            "얼마내",
            "얼마를내",
            "세금얼마",
            "어떻게내",
            "얼마나떼",
            "얼마떼",
        )
    )
    asks_general_rate_rule = any(word in compact for word in ("세율표", "연령별", "기준", "설명"))
    if asks_general_rate_rule and not has_actual_calculation_signal:
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


def _extract_receipt_type(compact: str) -> ReceiptType | None:
    if any(word in compact for word in ("연금외", "일시금", "일시인출", "한도초과")):
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
