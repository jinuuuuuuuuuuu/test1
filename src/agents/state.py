"""LangGraph 파이프라인(①라우터~⑤생성기)이 공유하는 State 스키마.

평가 API 응답 스키마(question_id/question/retrieved_context/think_trace/answer)와
최대한 맞닿도록 설계했다 — ⑤생성기가 이 State를 그대로 응답으로 변환한다.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

Intent = Literal["정보형", "상품형"]


class RetrievedItem(TypedDict, total=False):
    source: str
    content: str
    node: str

    # 문서 RAG 근거 정보
    chunk_id: str
    document_id: str
    file_title: str
    section: str
    source_location: str

class PensionAgentState(TypedDict, total=False):
    question_id: str
    question: str

    # 멀티턴 문맥 — chat.py 등 호출자가 이전 턴(question/answer)들을 담아 매 invoke() 호출마다
    # 입력으로 넘긴다. 그래프 내부에서 갱신되는 값이 아니라 호출자가 통째로 넘기는 값이라
    # reducer 없이 일반 필드로 둔다. (대화 자체는 여전히 싱글턴 invoke — 세션 스레드/checkpointer는
    # 쓰지 않고, chat.py가 리스트를 들고 있다가 다음 호출에 다시 넣어주는 방식.)
    conversation_history: list[dict[str, str]]

    # ① 라우터/가드레일 출력
    is_safe: bool
    safety_reason: str | None
    intent: list[Intent]  # 복합형이면 ["정보형", "상품형"] 둘 다 포함

    # ②③ 출력 — retrieved_context는 여러 노드가 이어서 채우므로 누적(operator.add) 리듀서 사용
    retrieved_context: Annotated[list[RetrievedItem], operator.add]
    info_draft: str | None
    product_draft: str | None

# 상품 추천 전 추가 정보가 필요한지
    needs_clarification: bool

    # ④ 검증/Grounding 출력
    verification: dict | None

    # ⑤ 생성기 출력 (최종 응답)
    answer: str | None
    think_trace: str | None
