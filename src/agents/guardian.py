"""후단 파수꾼 체크.

Guardian은 검증된 Core Answer를 수정하지 않는다. 사용자가 직접 묻지 않은 중요 포인트 중,
사전 정의된 Rule과 독립 근거가 존재하고 Core와 중복되지 않는 경우에만 최종 조립 단계에서
최대 1건을 추가한다.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from src.agents.context import merge_drafts
from src.agents.state import PensionAgentState, RetrievedItem
from src.agents.tax_context import (
    TAX_TOPIC_WORDS,
    determine_tax_branch,
    extract_tax_context,
)

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
            "전월세보증금 중도인출은 근퇴법상 사유에는 해당할 수 있지만, 세법상 '부득이한 "
            "사유'에는 해당하지 않습니다. 인출 재원에 따라 세금이 달라질 수 있어 재원 구분도 "
            "함께 확인해야 합니다."
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
            "무주택 주택구입 중도인출은 근퇴법상 사유에는 해당할 수 있지만, 세법상 '부득이한 "
            "사유'에는 해당하지 않습니다. 인출 재원에 따라 세금이 달라질 수 있어 재원 구분도 "
            "함께 확인해야 합니다."
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


def _compact(text: str | None) -> str:
    return (text or "").replace(" ", "").lower()


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
    text = _compact(question)
    topics: set[str] = set()
    if _contains_any(text, ("세금", "세율", "과세", "기타소득세", "퇴직소득세")):
        topics.add("withdrawal_tax")
    if _contains_any(text, ("언제까지", "기한", "신청기한", "시기")):
        topics.add("withdrawal_deadline")
    if _contains_any(text, ("서류", "필요서류", "구비서류", "징구서류")):
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
    return _contains_any(_compact(question), TAX_TOPIC_WORDS)


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


def _select_rule(question: str) -> GuardianRule | None:
    """규칙별 트리거를 우선순위 순으로 확인해 하나만 고른다 (LLM에 맡기지 않는다)."""
    retirement = _select_retirement_non_pension_rule(question)
    if retirement is not None:
        return retirement

    text = _compact(question)
    if "중도인출" not in text or not _is_documents_only_withdrawal_question(question):
        return None
    for rule in GUARDIAN_RULES:
        if rule["guard_type"] != "ACTION" or not rule["question_markers"]:
            continue
        if _contains_any(text, rule["question_markers"]):
            return rule
    return None


def _has_guard_domain_marker(question: str) -> bool:
    text = _compact(question)
    return "중도인출" in text and any(
        _contains_any(text, rule["question_markers"]) for rule in GUARDIAN_RULES
    )


def _core_text(state: PensionAgentState) -> str:
    return merge_drafts(state.get("info_draft"), state.get("product_draft")) or ""


def _core_covers_topic(core: str, topic: str) -> bool:
    text = _compact(core)
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
    # 대상"이라고 거의 같은 문장으로 답하므로, 여기서 막지 않으면 중복이 된다.
    if rule["topic"] == "retirement_tax_reduction" and _asks_tax_topic(question):
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
