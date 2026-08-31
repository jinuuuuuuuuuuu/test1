"""실물이전 의도 판정의 단일 진실 공급원.

Router는 이 판정으로 정보형 Core 경로를 고르고, Guardian A1은 여기에 절차·직접 주제
조건을 더해 추가 안내 여부만 결정한다. 두 계층이 서로 다른 키워드 목록을 가지면 자연어
표현에서 Router와 Guardian의 판정이 어긋나므로 공용 모듈로 둔다.
"""

from __future__ import annotations

import re


_ASSET_PRESERVING_MARKERS = (
    "상품그대로",
    "보유상품그대로",
    "매도없이",
    "매도하지않고",
    "매도하고싶지않",
    "현금화없이",
    "현금화하지않고",
    "상품유지한채",
)

# 목적지 표현("다른 금융사" 등)은 이전 행동이 아니므로 동사만 Gate로 사용한다.
_TRANSFER_ACTION_MARKERS = ("옮기", "이전", "이관")

_PROCEDURE_MARKERS = (
    "절차",
    "방법",
    "신청",
    "어떻게",
    "옮기려면",
    "이전하려면",
    "이관하려면",
)

# 가능 여부·제한사항·불가사유를 직접 물으면 해당 내용은 Core가 답할 몫이다.
# "할수있"처럼 넓은 조각은 "이전할 수 있는 방법"까지 막으므로 쓰지 않는다.
_EXPLICIT_GUARD_TOPIC_MARKERS = (
    "가능해",
    "가능한가",
    "가능한지",
    "가능하나요",
    "가능여부",
    "되는지",
    "되나요",
    "할수있어",
    "할수있나요",
    "안되는",
    "안되나요",
    "불가능",
    "제한되는",
    "제한되는지",
    "제한사항",
    "제한사유",
    "불가사유",
    "대상인지",
    "대상여부",
)


def normalize_for_guard_match(text: str | None) -> str:
    """트리거 매칭 전용 정규화. 사용자 원문과 답변 생성용 문자열은 바꾸지 않는다."""
    return re.sub(r"\s+", "", (text or "").lower())


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def has_in_kind_transfer_intent(question: str) -> bool:
    """보유 자산을 처분하지 않고 이전하려는 의도가 명확한지 판정한다.

    "실물이전" 명시는 그 자체로 충분하다. 전문용어가 없으면 자산보존 의도와 이전 행동이
    모두 있어야 한다. 일반 계좌이전이나 단순 보유 방법은 여기서 추정하지 않는다.
    """
    text = normalize_for_guard_match(question)
    explicit_in_kind = "실물이전" in text
    asset_preserving_intent = _contains_any(text, _ASSET_PRESERVING_MARKERS)
    transfer_action = _contains_any(text, _TRANSFER_ACTION_MARKERS)
    return explicit_in_kind or (asset_preserving_intent and transfer_action)


def asks_in_kind_transfer_procedure_only(question: str) -> bool:
    """A1용: 실물이전 절차를 묻되 제한 주제를 직접 묻지 않았는지 판정한다."""
    text = normalize_for_guard_match(question)
    return (
        has_in_kind_transfer_intent(question)
        and _contains_any(text, _PROCEDURE_MARKERS)
        and not _contains_any(text, _EXPLICIT_GUARD_TOPIC_MARKERS)
    )
