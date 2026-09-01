"""중도인출 질문에서 사용자가 이미 제공한 사실을 보존하는 상태 객체.

이 모듈은 답을 추론하는 Planner가 아니다. 질문에 명시됐거나 코드로 확정할 수 있는
사유·재원·수령방식과 사용자가 직접 요구한 주제를 후속 노드에 전달한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

WithdrawalReason = Literal[
    "MEDICAL",
    "PERSONAL_REHABILITATION",
    "BANKRUPTCY",
    "PERSONAL_WORKOUT",
    "HOUSING_DEPOSIT",
    "HOME_PURCHASE",
    "DISASTER",
]
WithdrawalSourceType = Literal["RETIREMENT_PAY"]
WithdrawalReceiptMode = Literal["PENSION", "NON_PENSION", "SPLIT"]
WithdrawalTopic = Literal["ELIGIBILITY", "DOCUMENTS", "DEADLINE", "TAX"]
ContextSource = Literal["explicit_user", "deterministic_rule"]


@dataclass(frozen=True)
class WithdrawalContext:
    reason: WithdrawalReason | None = None
    reason_source: ContextSource | None = None
    source_type: WithdrawalSourceType | None = None
    source_type_source: ContextSource | None = None
    receipt_mode: WithdrawalReceiptMode | None = None
    receipt_mode_source: ContextSource | None = None
    explicit_topics: frozenset[WithdrawalTopic] = field(default_factory=frozenset)
    task_required_topics: frozenset[str] = field(default_factory=frozenset)
    locked_fields: frozenset[str] = field(default_factory=frozenset)

    def to_state_dict(self) -> dict:
        return {
            "reason": self.reason,
            "reason_source": self.reason_source,
            "source_type": self.source_type,
            "source_type_source": self.source_type_source,
            "receipt_mode": self.receipt_mode,
            "receipt_mode_source": self.receipt_mode_source,
            "explicit_topics": sorted(self.explicit_topics),
            "task_required_topics": sorted(self.task_required_topics),
            "locked_fields": sorted(self.locked_fields),
        }


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _detect_reason(text: str) -> WithdrawalReason | None:
    # 충돌 가능성이 있는 표현부터 먼저 판정한다.
    has_workout = any(word in text for word in ("개인워크아웃", "워크아웃", "신용회복"))
    has_rehabilitation = any(word in text for word in ("개인회생", "회생절차"))
    if has_workout and has_rehabilitation:
        # 비교 질문에서는 어느 한쪽을 사용자 확정 사유로 잠그지 않는다.
        return None
    if has_workout:
        return "PERSONAL_WORKOUT"
    if has_rehabilitation:
        return "PERSONAL_REHABILITATION"
    if any(word in text for word in ("파산선고", "파산")):
        return "BANKRUPTCY"
    if any(word in text for word in ("재난피해", "재난", "수해", "화재피해")):
        return "DISASTER"
    if any(word in text for word in ("전월세", "전세", "월세", "임차보증금", "임대차")):
        return "HOUSING_DEPOSIT"
    if any(word in text for word in ("주택구입", "주택매입", "집구입", "집을사", "집사", "소유권이전")):
        return "HOME_PURCHASE"
    if any(word in text for word in ("요양", "의료비", "치료비", "질병", "부상")):
        return "MEDICAL"
    return None


def _detect_topics(text: str) -> frozenset[WithdrawalTopic]:
    topics: set[WithdrawalTopic] = set()
    if any(word in text for word in ("서류", "구비", "제출", "준비할", "챙겨")):
        topics.add("DOCUMENTS")
    if any(word in text for word in ("언제까지", "신청기한", "기한", "시기")):
        topics.add("DEADLINE")
    if any(word in text for word in ("세금", "세율", "과세", "퇴직소득세", "기타소득세", "얼마나떼")):
        topics.add("TAX")
    if any(word in text for word in ("가능한가", "가능한지", "가능해", "가능하", "되나요", "되는지", "할수있")):
        topics.add("ELIGIBILITY")
    return frozenset(topics)


def extract_withdrawal_context(question: str) -> WithdrawalContext | None:
    text = _compact(question)
    if "중도인출" not in text and not (
        "퇴직금" in text and any(word in text for word in ("일부", "나머지", "남은"))
    ):
        return None

    reason = _detect_reason(text)
    topics = _detect_topics(text)
    source_type: WithdrawalSourceType | None = "RETIREMENT_PAY" if "퇴직금" in text else None

    has_partial = "일부" in text or "중도인출" in text
    has_remaining = any(word in text for word in ("나머지", "남은", "잔여"))
    has_pension = any(word in text for word in ("연금으로", "연금수령", "연금으로받"))
    if source_type and has_partial and has_remaining and has_pension:
        receipt_mode: WithdrawalReceiptMode | None = "SPLIT"
    elif "중도인출" in text:
        receipt_mode = "NON_PENSION"
    elif source_type and has_pension:
        receipt_mode = "PENSION"
    else:
        receipt_mode = None

    locked_fields = set()
    if reason:
        locked_fields.add("reason")
    if source_type:
        locked_fields.add("source_type")
    if receipt_mode:
        locked_fields.add("receipt_mode")

    required_topics: set[str] = set()
    if receipt_mode == "SPLIT" and "TAX" in topics:
        required_topics.add("ELIGIBILITY_PRECONDITION")

    return WithdrawalContext(
        reason=reason,
        reason_source="explicit_user" if reason else None,
        source_type=source_type,
        source_type_source="explicit_user" if source_type else None,
        receipt_mode=receipt_mode,
        receipt_mode_source="deterministic_rule" if receipt_mode else None,
        explicit_topics=topics,
        task_required_topics=frozenset(required_topics),
        locked_fields=frozenset(locked_fields),
    )
