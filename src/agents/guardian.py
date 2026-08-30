"""후단 파수꾼 체크.

Guardian은 검증된 Core Answer를 수정하지 않는다. 사용자가 직접 묻지 않은 중요 포인트 중,
사전 정의된 Rule과 독립 근거가 존재하고 Core와 중복되지 않는 경우에만 최종 조립 단계에서
최대 1건을 추가한다.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from src.agents.context import merge_drafts
from src.agents.state import PensionAgentState, RetrievedItem

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
)


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


def _select_rule(question: str) -> GuardianRule | None:
    text = _compact(question)
    if "중도인출" not in text or not _is_documents_only_withdrawal_question(question):
        return None
    for rule in GUARDIAN_RULES:
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
