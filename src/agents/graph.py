"""①~⑤ 전체 파이프라인을 LangGraph StateGraph로 조립한다.

build_graph()는 노드 등록·엣지 연결·compile()까지 전부 수행하지만, 각 노드가 실제로
ChatClovaX를 호출하는 건 노드 함수가 "실행"될 때(.invoke() 시점)뿐이다. 따라서
build_graph() 자체는 CLOVASTUDIO_API_KEY 환경변수가 (더미 값이라도) 설정만 되어 있으면
네트워크 호출 없이 완료된다 — 그래프 "배선"과 "실행"이 분리되어 있다.

라우팅 규칙 (설계 결정, 2026-08-11):
- is_safe=False -> 바로 generator (정형 거절 응답)
- intent에 "정보형" 포함 -> info_agent 먼저 (복합형이면 이후 product_agent로 순차 이동)
- intent가 "상품형"만 -> product_agent 곧장 실행
- 복합형은 항상 ②→③ 순차 실행이며 병렬 처리하지 않는다 (③이 ②의 retrieved_context를
  State에서 읽어 쓰기 때문).
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.generator import build_generator_node
from src.agents.grounding import build_grounding_node
from src.agents.info_agent import build_info_agent_node
from src.agents.product_agent import build_product_agent_node
from src.agents.router import build_router_node
from src.agents.state import PensionAgentState

NODE_NAMES = ("router", "info_agent", "product_agent", "grounding", "generator")


def _route_after_router(state: PensionAgentState) -> str:
    if state.get("is_safe") is False:
        return "generator"
    intent = state.get("intent") or []
    if "정보형" in intent:
        return "info_agent"
    if "상품형" in intent:
        return "product_agent"
    # 분류 실패(빈 intent) 시 안전하게 생성기로 보내 정형 응답을 내도록 한다.
    return "generator"


def _route_after_info(state: PensionAgentState) -> str:
    if "상품형" in (state.get("intent") or []):
        return "product_agent"
    return "grounding"


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(PensionAgentState)

    graph.add_node("router", build_router_node())
    graph.add_node("info_agent", build_info_agent_node())
    graph.add_node("product_agent", build_product_agent_node())
    graph.add_node("grounding", build_grounding_node())
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
    graph.add_edge("product_agent", "grounding")
    graph.add_edge("grounding", "generator")
    graph.add_edge("generator", END)

    return graph.compile()
