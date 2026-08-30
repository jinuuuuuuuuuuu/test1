"""①~⑤ 전체 파이프라인을 LangGraph StateGraph로 조립한다.

build_graph()는 노드 등록·엣지 연결·compile()까지 전부 수행하지만, 각 노드가 실제로
ChatClovaX를 호출하는 건 노드 함수가 "실행"될 때(.invoke() 시점)뿐이다. 따라서
build_graph() 자체는 CLOVASTUDIO_API_KEY 환경변수가 (더미 값이라도) 설정만 되어 있으면
네트워크 호출 없이 완료된다 — 그래프 "배선"과 "실행"이 분리되어 있다.

라우팅 규칙 (설계 결정, 2026-08-11 / scope 축 추가 2026-08-16):
- is_safe=False -> 바로 generator (정형 거절 응답)
- scope="범위외" -> 바로 generator (정형 한계 고지 응답, "정보한계 대응" 평가지표 대응)
- 분류 실패(빈 intent) -> info_agent 폴백 (generator 직행은 사실상 무응답이 됨)
- intent에 "정보형" 포함 -> info_agent 먼저 (복합형이면 이후 product_agent로 순차 이동)
- intent가 "상품형"만 -> product_agent 곧장 실행
- 복합형은 항상 ②→③ 순차 실행이며 병렬 처리하지 않는다 (③이 ②의 retrieved_context를
  State에서 읽어 쓰기 때문).
- ②가 역질문(needs_clarification)이면 ③ 스킵 (조건 없이 추천 시도 방지)
- ④ 탈락(grounded=False 또는 requirements_met=False) 시 담당 에이전트로 1회만 재실행
  (bounded repair loop — repair_attempted 가드, 역질문은 대상 아님)
- ④ 통과 + complete 응답이면 ⑤ 직전 Guardian이 근거 있는 추가 점검 1건만 붙일 수 있다
  (Core Answer는 수정하지 않는 후단 안전 계층).
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.generator import build_generator_node
from src.agents.grounding import build_grounding_node
from src.agents.guardian import build_guardian_node
from src.agents.info_agent import build_info_agent_node
from src.agents.product_agent import build_product_agent_node
from src.agents.router import build_router_node
from src.agents.state import PensionAgentState

NODE_NAMES = ("router", "info_agent", "product_agent", "grounding", "guardian", "generator")


def _route_after_router(state: PensionAgentState) -> str:
    if state.get("is_safe") is False:
        return "generator"
    if state.get("scope") == "범위외":
        # 연금과 접점이 없는 질문 — ②③④를 건너뛰고 ⑤가 정형 한계 고지 응답을 만든다.
        return "generator"
    if state.get("deterministic_category", "해당없음") != "해당없음":
        # 라우터가 확정한 정형 답변 카테고리 — ②가 이 카테고리로 결정론 답변을 만든다.
        return "info_agent"
    intent = state.get("intent") or []
    if "정보형" in intent:
        return "info_agent"
    if "상품형" in intent:
        return "product_agent"
    # 분류 실패(빈 intent) 시 검색 툴을 가진 정보 Agent로 폴백해 최소한의 답변을 시도한다
    # (생성기 직행은 초안·근거가 전부 없어 사실상 무응답이 된다).
    return "info_agent"


def _route_after_info(state: PensionAgentState) -> str:
    if state.get("needs_clarification"):
        # ②가 조건 불충분으로 역질문한 상태 — ③이 조건 없이 추천을 시도하면 안 되므로 스킵.
        return "grounding"
    if "상품형" in (state.get("intent") or []):
        return "product_agent"
    return "grounding"


def _route_after_product(state: PensionAgentState) -> str:
    if state.get("needs_clarification"):
        return "generator"
    if state.get("recommendation_stage") == "type_recommendation":
        return "generator"
    return "grounding"


def _route_after_grounding(state: PensionAgentState) -> str:
    """④ 탈락 시 담당 에이전트로 1회 되돌리는 bounded repair loop.

    역질문 초안은 repair 대상이 아니고(의도된 유보), repair_attempted가 이미 True면
    두 번째 재실행 없이 ⑤로 넘긴다 — ⑤가 검증결과를 반영해 방어적으로 답을 조립한다.
    """
    if state.get("needs_clarification"):
        return "generator"
    if state.get("recommendation_stage") == "type_recommendation":
        return "generator"
    verification = state.get("verification") or {}
    failed = verification.get("grounded") is False or verification.get("requirements_met") is False
    if failed:
        if state.get("repair_attempted"):
            return "generator"
        intent = state.get("intent") or []
        if "정보형" in intent:
            return "info_agent"  # 복합형이면 기존 엣지를 따라 다시 ②→③ 순차 실행된다
        if "상품형" in intent:
            return "product_agent"
    if _guardian_route_possible(state):
        return "guardian"
    return "generator"


def _guardian_route_possible(state: PensionAgentState) -> bool:
    verification = state.get("verification") or {}
    return (
        state.get("scope") == "범위내"
        and state.get("needs_clarification") is not True
        and state.get("response_mode") == "complete"
        and verification.get("grounded") is True
        and verification.get("requirements_met") is True
    )


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(PensionAgentState)

    graph.add_node("router", build_router_node())
    graph.add_node("info_agent", build_info_agent_node())
    graph.add_node("product_agent", build_product_agent_node())
    graph.add_node("grounding", build_grounding_node())
    graph.add_node("guardian", build_guardian_node())
    graph.add_node("generator", build_generator_node())

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"info_agent": "info_agent", "product_agent": "product_agent", "generator": "generator"},
    )
    graph.add_conditional_edges(
        "info_agent",
        _route_after_info,
        {"product_agent": "product_agent", "grounding": "grounding"},
    )
    graph.add_conditional_edges(
        "product_agent",
        _route_after_product,
        {"grounding": "grounding", "generator": "generator"},
    )
    graph.add_conditional_edges(
        "grounding",
        _route_after_grounding,
        {
            "info_agent": "info_agent",
            "product_agent": "product_agent",
            "guardian": "guardian",
            "generator": "generator",
        },
    )
    graph.add_edge("guardian", "generator")
    graph.add_edge("generator", END)

    return graph.compile()
