"""그래프 '배선'이 올바른지만 검증한다 (실제 LLM 응답 품질/행동은 검증 대상이 아님 — 그건
CLOVASTUDIO_API_KEY가 있어야 가능하다).

더미 API 키로 ChatClovaX 인스턴스 생성 자체는 되므로(네트워크 호출 없음), build_graph()가
에러 없이 끝까지 조립·컴파일되는지, 노드/엣지 구조가 설계한 대로인지까지는 키 없이도
확인할 수 있다.
"""

import os

import pytest

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

from src.agents.graph import (
    NODE_NAMES,
    _route_after_grounding,
    _route_after_info,
    _route_after_router,
    build_graph,
)


def test_graph_builds_and_compiles():
    app = build_graph()
    node_names = set(app.get_graph().nodes.keys())
    for name in NODE_NAMES:
        assert name in node_names


def test_route_after_router_unsafe_goes_to_generator():
    assert _route_after_router({"is_safe": False, "intent": ["정보형"]}) == "generator"


def test_route_after_router_info_intent():
    assert _route_after_router({"is_safe": True, "intent": ["정보형"]}) == "info_agent"


def test_route_after_router_product_only_intent():
    assert _route_after_router({"is_safe": True, "intent": ["상품형"]}) == "product_agent"


def test_route_after_router_composite_intent_goes_to_info_first():
    # 복합형(정보+상품)은 항상 정보 Agent가 먼저 실행된다 (순차 실행 설계)
    assert _route_after_router({"is_safe": True, "intent": ["정보형", "상품형"]}) == "info_agent"


def test_route_after_router_empty_intent_falls_back_to_info_agent():
    # 분류 실패 시 생성기 직행(무응답)이 아니라 검색 툴이 있는 정보 Agent로 폴백한다.
    assert _route_after_router({"is_safe": True, "intent": []}) == "info_agent"


def test_route_after_router_out_of_scope_goes_to_generator():
    # 범위외 질문은 ②③④를 건너뛰고 ⑤가 정형 한계 고지 응답을 만든다.
    assert (
        _route_after_router({"is_safe": True, "scope": "범위외", "intent": ["정보형"]})
        == "generator"
    )


def test_route_after_router_partial_scope_proceeds_normally():
    # 부분관련은 차단하지 않고 해당 에이전트가 연금 관점으로 재조준해 답한다.
    assert (
        _route_after_router({"is_safe": True, "scope": "부분관련", "intent": ["정보형"]})
        == "info_agent"
    )


def test_route_after_router_unsafe_wins_over_scope():
    assert (
        _route_after_router({"is_safe": False, "scope": "범위외", "intent": ["정보형"]})
        == "generator"
    )


def test_route_after_info_composite_goes_to_product():
    assert _route_after_info({"intent": ["정보형", "상품형"]}) == "product_agent"


def test_route_after_info_info_only_goes_to_grounding():
    assert _route_after_info({"intent": ["정보형"]}) == "grounding"


def test_route_after_info_clarification_skips_product_agent():
    # ②가 역질문한 복합형 — 조건 없이 ③이 추천을 시도하면 안 되므로 곧장 검증으로 간다.
    assert (
        _route_after_info({"intent": ["정보형", "상품형"], "needs_clarification": True})
        == "grounding"
    )


# ── bounded repair loop (④ 탈락 시 1회 재실행) ───────────────────────────


def test_route_after_grounding_pass_goes_to_generator():
    state = {"verification": {"grounded": True, "requirements_met": True}, "intent": ["정보형"]}
    assert _route_after_grounding(state) == "generator"


def test_route_after_grounding_failure_repairs_info_agent():
    state = {"verification": {"grounded": False, "requirements_met": True}, "intent": ["정보형"]}
    assert _route_after_grounding(state) == "info_agent"


def test_route_after_grounding_failure_repairs_product_agent():
    state = {"verification": {"grounded": True, "requirements_met": False}, "intent": ["상품형"]}
    assert _route_after_grounding(state) == "product_agent"


def test_route_after_grounding_repair_is_bounded_to_one_attempt():
    # 두 번째 탈락은 재실행 없이 ⑤로 — ⑤가 검증결과를 반영해 방어적으로 조립한다.
    state = {
        "verification": {"grounded": False, "requirements_met": False},
        "intent": ["정보형"],
        "repair_attempted": True,
    }
    assert _route_after_grounding(state) == "generator"


def test_route_after_grounding_clarification_is_not_repaired():
    # 역질문은 의도된 유보라 repair 대상이 아니다.
    state = {
        "verification": {"grounded": True, "requirements_met": False},
        "intent": ["상품형"],
        "needs_clarification": True,
    }
    assert _route_after_grounding(state) == "generator"


@pytest.mark.skipif(
    os.environ.get("CLOVASTUDIO_API_KEY", "").startswith("dummy-"),
    reason="실제 CLOVASTUDIO_API_KEY가 있어야 엔드투엔드 실행이 가능합니다",
)
def test_graph_end_to_end_smoke():
    """실제 API 키가 있을 때만 도는 엔드투엔드 스모크 테스트. 지금은 항상 skip된다."""
    app = build_graph()
    result = app.invoke({"question_id": "smoke-1", "question": "연금저축 세액공제 한도가 얼마인가요?"})
    assert result.get("answer")
