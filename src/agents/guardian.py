"""후단 파수꾼 체크.

Guardian은 검증된 Core Answer를 수정하지 않는다. 사용자가 직접 묻지 않은 중요 포인트 중,
사전 정의된 Rule과 독립 근거가 존재하고 Core와 중복되지 않는 경우에만 최종 조립 단계에서
최대 1건을 추가한다.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from src.agents.context import merge_drafts
from src.agents.deterministic_info import extract_tax_credit_inputs
from src.agents.in_kind_transfer_intent import (
    asks_in_kind_transfer_procedure_only,
    normalize_for_guard_match,
)
from src.agents.state import PensionAgentState, RetrievedItem
from src.agents.tax_context import (
    TAX_TOPIC_WORDS,
    determine_tax_branch,
    extract_tax_context,
)
from src.rules.tax_credit import COMBINED_CREDIT_LIMIT, PENSION_SAVINGS_ONLY_LIMIT

GUARD_HEADING = "🛡️ 파수꾼 체크"

GuardDisabledReason = Literal[
    "CORE_NOT_GROUNDED",
    "REQUIREMENTS_NOT_MET",
    "NEEDS_CLARIFICATION",
    "NON_COMPLETE_RESPONSE",
    "OUT_OF_SCOPE",
    "GUARD_FACT_ALREADY_ASKED",
    "NO_CANDIDATE",
    "NO_GUARD_EVIDENCE",
    "CORE_ALREADY_COVERS_TOPIC",
    # 사용자가 그 주제를 직접 물어본 경우 — Core Answer가 답할 몫이라 파수꾼은 침묵한다.
    "EXPLICIT_USER_TOPIC",
]


class GuardianRule(TypedDict):
    candidate_id: str
    guard_type: str
    trigger: str
    topic: str
    priority: int
    question_markers: tuple[str, ...]
    guard_fact: str
    required_evidence: tuple[RetrievedItem, ...]


_WITHDRAWAL_HOUSING_TAX_EVIDENCE: RetrievedItem = {
    "source": "doc38~doc40 주택 관련 중도인출 재원별 과세 규칙",
    "content": (
        "무주택자인 가입자의 본인 명의 주택 구입과 주거 목적 전세보증금 부담은 근퇴법상 "
        "중도인출 사유이나 세법상 부득이한 사유는 아니며, 인출 재원에 따라 과세가 달라집니다."
    ),
    "node": "guardian",
}


_IN_KIND_TRANSFER_RESTRICTION_EVIDENCE: RetrievedItem = {
    "source": "doc34 실물이전 불가사유 코드",
    "content": (
        "모든 보유 상품을 실물이전할 수 있는 것은 아닙니다. 사모펀드·MMF·소규모 펀드"
        "(잔고 50억 미만)·환매수수료 부과 상품·지분증권·리츠 등은 실물이전이 제한되거나 "
        "상대 금융기관 확인이 필요합니다."
    ),
    "node": "guardian",
}


_RETIREMENT_NON_PENSION_EVIDENCE: RetrievedItem = {
    "source": "doc39~doc40 이연퇴직소득세 감면 규칙",
    "content": (
        "퇴직금 재원을 연금으로 수령하면 연금실제수령연차에 따라 이연퇴직소득세의 "
        "70%(1~10년차)·60%(11~20년차)·50%(21년차 이상)만 납부합니다. 반면 연금외수령"
        "(일시금 등)은 감면이 적용되지 않아 이연퇴직소득세를 전액 납부합니다."
    ),
    "node": "guardian",
}


GUARDIAN_RULES: tuple[GuardianRule, ...] = (
    {
        "candidate_id": "housing_deposit_documents",
        "guard_type": "ACTION",
        "trigger": "housing_deposit_documents_only",
        "topic": "withdrawal_tax",
        "priority": 90,
        "question_markers": ("전월세", "전세", "월세", "보증금", "임차보증금", "임대차"),
        "guard_fact": (
            "세금도 함께 확인하세요. 전월세보증금 중도인출은 세법상 '부득이한 사유'가 "
            "아니며, 인출 재원에 따라 과세가 달라질 수 있습니다."
        ),
        "required_evidence": (_WITHDRAWAL_HOUSING_TAX_EVIDENCE,),
    },
    {
        "candidate_id": "home_purchase_documents",
        "guard_type": "ACTION",
        "trigger": "home_purchase_documents_only",
        "topic": "withdrawal_tax",
        "priority": 90,
        "question_markers": ("주택구입", "주택매입", "집구입", "집을사", "집사", "주택을사"),
        "guard_fact": (
            "세금도 함께 확인하세요. 무주택 주택구입 중도인출은 세법상 '부득이한 사유'가 "
            "아니며, 인출 재원에 따라 과세가 달라질 수 있습니다."
        ),
        "required_evidence": (_WITHDRAWAL_HOUSING_TAX_EVIDENCE,),
    },
    {
        # A2 — 퇴직금 재원을 연금외수령(일시금 등)하려는 행동. 사용자가 절차만 묻고
        # 세금을 묻지 않으면 Core는 침묵하는데(tax_context의 계산 게이트가 "세금"류
        # 어휘를 요구한다), 그 행동으로 이연퇴직소득세 감면(연차별 30~50%)을 통째로
        # 잃는다는 사실은 아무도 말해주지 않는다. 파수꾼이 그 지점을 짚는다.
        "candidate_id": "retirement_non_pension_tax_loss",
        "guard_type": "ACTION",
        "trigger": "retirement_lump_sum_action",
        "topic": "retirement_tax_reduction",
        "priority": 100,
        # 이 규칙은 어휘 매칭이 아니라 tax_context의 구조적 판정(재원+수령방식)을 쓴다.
        "question_markers": (),
        "guard_fact": (
            "퇴직금 재원을 연금외수령하면 연금으로 받을 때 적용되는 이연퇴직소득세 감면"
            "(연금실제수령연차에 따라 30~50%)을 적용받지 못합니다."
        ),
        "required_evidence": (_RETIREMENT_NON_PENSION_EVIDENCE,),
    },
    {
        # A1 — 실물이전 절차·방법을 묻는 질문(상품이 특정되지 않음)에는 라우터가 이미
        # "해당없음"으로 정확히 분류해 LLM이 절차를 답한다(실측 확인: candidate_categories
        # 단계에서는 실물이전_불가사유가 후보에 오르지만, 라우터 프롬프트에 "절차 자체를
        # 묻는 질문은 해당없음"이라는 지시가 있고 실제로 그렇게 분류된다). 다만 그 답변은
        # "이 상품이 이전 가능한지"는 확인 안 된 채 절차만 말하므로, 옮기려는 상품이
        # 실물이전 제한 대상일 수 있다는 점을 짚어준다.
        "candidate_id": "in_kind_transfer_procedure",
        "guard_type": "ACTION",
        "trigger": "in_kind_transfer_procedure_without_product",
        "topic": "in_kind_transfer_eligibility",
        "priority": 95,
        # A1은 단순 어휘 목록이 아니라 _asks_in_kind_transfer_procedure_only()의
        # 구조적 판정만 사용한다. 일반 이전 표현이 다른 공통 게이트에 섞이지 않게 비운다.
        "question_markers": (),
        "guard_fact": (
            "모든 상품을 그대로 실물이전할 수 있는 것은 아닙니다. 이전하려는 상품이 실물이전 "
            "가능 대상인지도 함께 확인해야 합니다."
        ),
        "required_evidence": (_IN_KIND_TRANSFER_RESTRICTION_EVIDENCE,),
    },
)

_RULES_BY_ID: dict[str, GuardianRule] = {rule["candidate_id"]: rule for rule in GUARDIAN_RULES}


def _disabled(reason: GuardDisabledReason) -> dict:
    return {
        "enabled": False,
        "candidate_id": None,
        "guard_type": None,
        "trigger": None,
        "topic": None,
        "message": None,
        "guard_fact": None,
        "evidence_ids": [],
        "priority": None,
        "disabled_reason": reason,
    }


def _enabled(rule: GuardianRule) -> dict:
    evidence = list(rule["required_evidence"])
    return {
        "enabled": True,
        "candidate_id": rule["candidate_id"],
        "guard_type": rule["guard_type"],
        "trigger": rule["trigger"],
        "topic": rule["topic"],
        "message": f"{GUARD_HEADING}\n{rule['guard_fact']}",
        "guard_fact": rule["guard_fact"],
        "evidence_ids": [item["source"] for item in evidence],
        "priority": rule["priority"],
        "disabled_reason": None,
    }


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _core_gate(state: PensionAgentState) -> GuardDisabledReason | None:
    verification = state.get("verification") or {}
    if state.get("scope") != "범위내":
        return "OUT_OF_SCOPE"
    if state.get("needs_clarification"):
        return "NEEDS_CLARIFICATION"
    if state.get("response_mode") != "complete":
        return "NON_COMPLETE_RESPONSE"
    if verification.get("grounded") is not True:
        return "CORE_NOT_GROUNDED"
    if verification.get("requirements_met") is not True:
        return "REQUIREMENTS_NOT_MET"
    return None


def _explicit_topics(question: str) -> set[str]:
    text = normalize_for_guard_match(question)
    topics: set[str] = set()
    if _contains_any(text, ("세금", "세율", "과세", "기타소득세", "퇴직소득세")):
        topics.add("withdrawal_tax")
    if _contains_any(text, ("언제까지", "기한", "신청기한", "시기")):
        topics.add("withdrawal_deadline")
    if _contains_any(text, ("서류", "필요서류", "구비서류", "징구서류", "제출서류", "준비서류", "준비할", "챙겨")):
        topics.add("withdrawal_documents")
    return topics


def _is_documents_only_withdrawal_question(question: str) -> bool:
    topics = _explicit_topics(question)
    return "withdrawal_documents" in topics and not (topics - {"withdrawal_documents"})


def _asks_tax_topic(question: str) -> bool:
    """사용자가 세금·감면 주제를 직접 꺼냈는지 판정한다.

    ⚠️ tax_context._is_personal_tax_question()을 쓰면 안 된다 — 그 함수는 "계산 게이트로
    보낼까"를 판정하므로, 세금을 물었어도 순수 세율 비교면 False를 낸다(실측 no.115
    "퇴직금 일시금 vs 연금 중 세금이 더 적은 쪽은?"). 그걸 EXPLICIT_USER_TOPIC 판정에
    쓰면 이미 세금을 물은 질문에 파수꾼이 또 세금 이야기를 덧붙이게 된다.
    """
    return _contains_any(normalize_for_guard_match(question), TAX_TOPIC_WORDS)


def _select_retirement_non_pension_rule(question: str) -> GuardianRule | None:
    """A2 — 퇴직금 재원을 연금외수령하려는 행동을 감지한다.

    재원(퇴직금)과 행동(연금외수령)이 **둘 다** 확인될 때만 후보가 된다. "IRP에서 돈
    빼고 싶어요"처럼 재원이 불명확하면 감면 상실을 단정할 수 없으므로 침묵한다.
    판정은 tax_context의 기존 추출기를 그대로 재사용한다 — 어휘 목록을 여기서 새로
    만들면 두 곳이 반드시 어긋난다(실측: _MEDICAL_CONTEXT_WORDS가 같은 이유로 갈렸다).
    """
    context = extract_tax_context(question)
    if determine_tax_branch(context) != "retirement_benefit_non_pension":
        return None
    return _RULES_BY_ID.get("retirement_non_pension_tax_loss")


def _asks_in_kind_transfer_procedure_only(question: str) -> bool:
    return asks_in_kind_transfer_procedure_only(question)


def _select_in_kind_transfer_rule(question: str) -> GuardianRule | None:
    if not _asks_in_kind_transfer_procedure_only(question):
        return None
    return _RULES_BY_ID.get("in_kind_transfer_procedure")


def _select_unused_tax_credit_capacity_rule(question: str) -> GuardianRule | None:
    """O1 — 제공된 납입정보로 확정 가능한 세액공제 미사용 한도를 탐지한다.

    ⚠️ '추가 납입 추천'이 아니라 '미사용 혜택 탐지'다. "○○만원 남아 있습니다"까지만
    말하고 "더 넣으세요"는 말하지 않는다. 세액공제율·예상 절세액은 소득 정보가 없으면
    계산하지 않는다 — 잔여 한도(연금저축+IRP 합산 900만원 기준)는 소득과 무관하게
    확정할 수 있어 분리한다.

    필수 입력은 Rule 계산에 실제 필요한 값만 요구한다 — 납입액(연금저축 또는 IRP
    중 하나라도)만 있으면 잔여 한도를 계산할 수 있다. 소득이 없어도 계산 자체는
    막지 않는다(세율만 못 붙일 뿐).
    """
    values = extract_tax_credit_inputs(question)
    pension_savings_paid = values["pension_savings_paid"] or 0
    irp_paid = values["irp_paid"] or 0
    total_paid = pension_savings_paid + irp_paid
    if total_paid <= 0:
        return None

    credited_pension_savings = min(pension_savings_paid, PENSION_SAVINGS_ONLY_LIMIT)
    credited_total = min(credited_pension_savings + irp_paid, COMBINED_CREDIT_LIMIT)
    remaining = COMBINED_CREDIT_LIMIT - credited_total
    if remaining <= 0:
        # 이미 한도를 채웠거나 넘겼다 — 미사용 혜택이 없으므로 탐지할 것이 없다.
        return None

    remaining_label = f"{remaining // 10_000:,}만원"
    pension_savings_remaining = max(0, PENSION_SAVINGS_ONLY_LIMIT - pension_savings_paid)
    pension_savings_remaining_label = f"{pension_savings_remaining // 10_000:,}만원"
    if pension_savings_paid and not irp_paid:
        if pension_savings_remaining:
            guard_fact = (
                f"제공한 올해 납입정보 기준으로 연금저축 단독 한도는 "
                f"{pension_savings_remaining_label}, IRP를 포함한 합산 한도 기준으로는 "
                f"{remaining_label} 남아 있습니다."
            )
        else:
            guard_fact = (
                "제공한 올해 납입정보 기준으로 연금저축 단독 한도는 이미 채워졌고, "
                f"IRP를 포함한 합산 한도 기준으로는 {remaining_label} 남아 있습니다."
            )
    else:
        guard_fact = (
            f"제공한 올해 납입정보 기준으로 IRP를 포함한 합산 한도 기준 세액공제 대상 "
            f"납입한도가 {remaining_label} 남아 있습니다."
        )
    evidence: RetrievedItem = {
        "source": "doc41 세액공제 규칙",
        "content": (
            f"연금저축+IRP 합산 세액공제 대상 납입한도는 연 {COMBINED_CREDIT_LIMIT // 10_000:,}만원"
            f"(연금저축 단독은 {PENSION_SAVINGS_ONLY_LIMIT // 10_000:,}만원)입니다. "
            f"현재까지 확인된 납입액은 {total_paid // 10_000:,}만원이고, "
            f"연금저축 납입액 중 세액공제 대상 반영액은 {credited_pension_savings // 10_000:,}만원입니다. "
            f"현재 세액공제 대상 반영액은 {credited_total // 10_000:,}만원이므로, "
            f"합산 한도 기준으로 {remaining_label}이 남아 있습니다."
        ),
        "node": "guardian",
    }
    return {
        "candidate_id": "unused_tax_credit_capacity",
        "guard_type": "OPPORTUNITY",
        "trigger": "unused_tax_credit_capacity",
        "topic": "tax_credit_capacity",
        "priority": 80,
        "question_markers": (),
        "guard_fact": guard_fact,
        "required_evidence": (evidence,),
    }


def _select_rule(question: str) -> GuardianRule | None:
    """규칙별 트리거를 우선순위 순으로 확인해 하나만 고른다 (LLM에 맡기지 않는다).

    순서는 명세의 priority 정책을 그대로 따른다: 확정적인 세금상 손실(A2, 100) >
    행동/이전 제약(A1, 95) > 중도인출 재원별 과세(90) > 확정적인 미사용 혜택(O1, 80).
    """
    retirement = _select_retirement_non_pension_rule(question)
    if retirement is not None:
        return retirement

    transfer = _select_in_kind_transfer_rule(question)
    if transfer is not None:
        return transfer

    text = normalize_for_guard_match(question)
    if "중도인출" in text and _is_documents_only_withdrawal_question(question):
        for rule in GUARDIAN_RULES:
            if rule["guard_type"] != "ACTION" or not rule["question_markers"]:
                continue
            if rule["candidate_id"] == "in_kind_transfer_procedure":
                continue
            if _contains_any(text, rule["question_markers"]):
                return rule

    return _select_unused_tax_credit_capacity_rule(question)


def _has_guard_domain_marker(question: str) -> bool:
    text = normalize_for_guard_match(question)
    return "중도인출" in text and any(
        _contains_any(text, rule["question_markers"]) for rule in GUARDIAN_RULES
    )


def _core_text(state: PensionAgentState) -> str:
    return merge_drafts(state.get("info_draft"), state.get("product_draft")) or ""


def _core_covers_topic(core: str, topic: str) -> bool:
    text = normalize_for_guard_match(core)
    if topic == "withdrawal_tax":
        return _contains_any(text, ("세법상부득이한사유", "기타소득세", "퇴직소득세", "재원별과세"))
    if topic == "retirement_tax_reduction":
        # Core가 이미 감면 상실을 설명했으면 파수꾼은 침묵한다 — tax_context의
        # retirement_benefit_non_pension 브랜치가 "이연퇴직소득세 감면이 적용되지 않고
        # 전액 납부 대상"이라고 사실상 같은 내용을 답한다.
        return _contains_any(
            text,
            ("이연퇴직소득세", "감면이적용되지", "감면이없", "전액납부", "연금실제수령연차"),
        )
    if topic == "in_kind_transfer_eligibility":
        # Core가 이미 "이 상품은/모든 상품이" 실물이전 가능 여부를 언급했으면 중복이다.
        return _contains_any(
            text, ("실물이전이제한", "실물이전불가", "실물이전대상", "이전가능한상품", "이전이제외")
        )
    if topic == "tax_credit_capacity":
        # Core가 이미 세액공제 한도(남은 금액·합산 900만원 기준)를 언급했으면 중복이다.
        return _contains_any(text, ("남아있습니다", "한도가남", "합산900만원", "합산9백만원"))
    return False


def evaluate_guardian(state: PensionAgentState) -> tuple[dict, list[RetrievedItem]]:
    """State를 평가해 GuardianResult와 guardian_evidence를 반환한다."""
    gate_reason = _core_gate(state)
    if gate_reason:
        return _disabled(gate_reason), []

    question = state.get("question") or ""
    if "withdrawal_tax" in _explicit_topics(question) and _has_guard_domain_marker(question):
        return _disabled("GUARD_FACT_ALREADY_ASKED"), []

    rule = _select_rule(question)
    if rule is None:
        return _disabled("NO_CANDIDATE"), []

    # 사용자가 그 주제를 직접 물었으면 Core Answer가 답할 몫이다. 특히 A2는 Core의
    # retirement_benefit_non_pension 브랜치가 이미 "감면이 적용되지 않고 전액 납부
    # 대상"이라고 거의 같은 문장으로 답하므로, 여기서 막지 않으면 중복이 된다. O1도
    # "세액공제 한도 얼마 남았어?"처럼 직접 물으면 Core(세액공제_한도/계산 카테고리)가
    # 이미 답하므로 같은 원칙을 적용한다.
    if rule["topic"] in ("retirement_tax_reduction", "tax_credit_capacity") and _asks_tax_topic(question):
        return _disabled("EXPLICIT_USER_TOPIC"), []

    evidence = list(rule["required_evidence"])
    if not evidence:
        return _disabled("NO_GUARD_EVIDENCE"), []

    if _core_covers_topic(_core_text(state), rule["topic"]):
        return _disabled("CORE_ALREADY_COVERS_TOPIC"), []

    return _enabled(rule), evidence


def build_guardian_node():
    def guardian_node(state: PensionAgentState) -> dict:
        result, evidence = evaluate_guardian(state)
        return {
            "guardian_result": result,
            "guardian_evidence": evidence,
        }

    return guardian_node
