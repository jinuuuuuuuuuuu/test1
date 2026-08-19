"""LangGraph 파이프라인(①라우터~⑤생성기)이 공유하는 State 스키마.

평가 API 응답 스키마(question_id/question/retrieved_context/think_trace/answer)와
최대한 맞닿도록 설계했다 — ⑤생성기가 이 State를 그대로 응답으로 변환한다.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

Intent = Literal["정보형", "상품형"]
Scope = Literal["범위내", "부분관련", "범위외"]
ResponseMode = Literal["complete", "conditional", "clarification_included"]


class RetrievedItem(TypedDict):
    source: str   # 툴 이름(예: calculate_tax_credit) 또는 "RAG"
    content: str  # 근거 텍스트 / 툴 반환값 (JSON 문자열 또는 요약)
    node: str     # 어느 노드에서 생성됐는지: "info_agent" | "product_agent"
    chunk_id: NotRequired[str]
    document_id: NotRequired[str]
    file_title: NotRequired[str]
    section: NotRequired[str]
    source_location: NotRequired[str]


class ToolCallRecord(TypedDict):
    """②③이 실제로 호출한 툴 1건의 기록 — think_trace 서사화용.

    retrieved_context는 근거 '내용'만 남기고 "어떤 인자로 무엇을 호출했는지"는 버리기
    때문에 별도로 모은다 (대회 평가지표 "추론 논리성" 대응).
    """
    node: str     # "info_agent" | "product_agent"
    tool: str     # 툴 이름
    args: str     # 호출 인자 요약 (예: 'query="세액공제 한도", k=5')
    result: str   # 결과 요약 (예: "3건 검색: 연금저축계좌·IRP 세액공제 안내 …")


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
    # 서비스 범위 판정 — 안전성(is_safe)과는 별개 축이다: "개인사업 절세법"은 안전하지만
    # 부분관련이고, 범위외면 ②③④를 건너뛰고 정형 한계 고지로 응답한다.
    scope: Scope
    scope_note: str | None  # 부분관련: 연금 관점 재조준 방향 / 범위외: 사유

    # ②③ 출력 — retrieved_context는 여러 노드가 이어서 채우므로 누적(operator.add) 리듀서 사용.
    # repair 재실행 시 같은 근거가 중복 누적되므로 읽는 쪽은 context.dedupe_context를 거친다.
    retrieved_context: Annotated[list[RetrievedItem], operator.add]
    # 툴 호출 기록도 실행 순서대로 누적한다 — ⑤가 이걸 시간순 서사(think_trace)로 재구성한다.
    # repair 재실행 시 같은 노드 기록이 뒤에 다시 붙어, 서사에서 "재실행" 구간으로 드러난다.
    tool_trace: Annotated[list[ToolCallRecord], operator.add]
    info_draft: str | None
    product_draft: str | None
    # ②③이 조건 불충분으로 역질문 초안([추가 확인 필요] 마커)을 낸 경우 True —
    # 복합형에서 ③ 스킵, ④의 요구사항 검증 면제, ⑤의 답변 보충 금지가 걸린다.
    needs_clarification: bool
    # ④ 탈락으로 ②③을 재실행한 적이 있으면 True — repair 루프를 1회로 제한하는 가드.
    repair_attempted: bool
    # 상품 추천 멀티턴에서 수집한 투자자 조건. chat.py/API 호출자가 다음 invoke에 다시 넘긴다.
    recommendation_profile: dict
    # clarification/type_recommendation/specific_recommendation — 상품 추천 2단계 흐름 식별용.
    recommendation_stage: str | None
    # 싱글턴 평가 응답 내부 메타데이터 — 외부 EvaluationResponse에는 노출하지 않는다.
    missing_information: list[str]
    clarification_questions: list[str]
    response_mode: ResponseMode
    # 세액공제·중도인출 등 고위험 정보질의를 결정론 답변 경로로 처리했는지 여부.
    deterministic_info: bool

    # ④ 검증/Grounding 출력
    verification: dict | None

    # ⑤ 생성기 출력 (최종 응답)
    answer: str | None
    think_trace: str | None
