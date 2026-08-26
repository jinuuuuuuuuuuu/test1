"""평가용 API 서버 — 요강 스키마 준수와 장애 대응 검증.

그래프를 가짜로 갈아끼워 모델 호출 없이 돌린다. API 계층이 담당하는 것은
"그래프 결과를 주최측 스키마로 옮기는 일"뿐이므로, 답변 품질이 아니라
스키마·에러 처리·필드 매핑만 확인하면 된다.
"""

import os

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main

# 요강 p8이 정한 응답 필드. 이 목록이 바뀌면 제출 스키마 위반이다.
REQUIRED_FIELDS = {"question_id", "question", "retrieved_context", "think_trace", "answer"}


class _FakeGraph:
    """invoke() 한 번에 고정 결과를 돌려주는 가짜 그래프."""

    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client(monkeypatch):
    """lifespan이 실제 그래프를 만들지 않도록 build_graph를 막고 TestClient를 준다."""

    def _make(graph):
        monkeypatch.setattr(api_main, "build_graph", lambda: graph)
        api_main._state.clear()
        return TestClient(api_main.app)

    return _make


def _ok_graph():
    return _FakeGraph({
        "answer": "연금저축 단독 600만원, IRP 합산 900만원입니다.",
        "think_trace": "[① 질문 분류] ...",
        "retrieved_context": [
            {"source": "doc41 세액공제 규칙", "content": "연금저축은 연 600만원", "node": "info_agent"},
        ],
    })


def test_response_matches_contest_schema(client):
    """응답은 요강이 정한 5개 필드로만 구성된다 — 누락도 추가도 없어야 한다."""
    with client(_ok_graph()) as c:
        body = c.get("/answer", params={"question_id": "Q-001", "question": "세액공제 한도는?"}).json()

    assert set(body) == REQUIRED_FIELDS
    assert body["question_id"] == "Q-001"
    assert body["question"] == "세액공제 한도는?"
    assert "600만원" in body["answer"]


def test_retrieved_context_is_flattened_to_string(client):
    """State의 근거 리스트를 문자열 필드로 옮기되 출처를 남긴다 (근거 완전성 확인용)."""
    with client(_ok_graph()) as c:
        body = c.get("/answer", params={"question": "세액공제 한도는?"}).json()

    assert isinstance(body["retrieved_context"], str)
    assert "doc41 세액공제 규칙" in body["retrieved_context"]
    assert "연금저축은 연 600만원" in body["retrieved_context"]


def test_question_id_is_generated_when_omitted(client):
    """평가측이 id를 안 보내도 응답에는 항상 id가 있어야 한다."""
    with client(_ok_graph()) as c:
        body = c.get("/answer", params={"question": "세액공제 한도는?"}).json()

    assert body["question_id"]


def test_single_turn_invocation(client):
    """평가는 문항당 단발 호출이다 — 대화 이력을 섞어 넣지 않는다."""
    graph = _ok_graph()
    with client(graph) as c:
        c.get("/answer", params={"question": "세액공제 한도는?"})

    assert graph.calls[0]["conversation_history"] == []


def test_pipeline_failure_returns_200_with_disclosure(client):
    """파이프라인이 죽어도 500을 던지지 않는다 — 무응답은 그 문항이 0점이 된다."""
    with client(_FakeGraph(error=RuntimeError("boom"))) as c:
        response = c.get("/answer", params={"question_id": "Q-9", "question": "세액공제 한도는?"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == REQUIRED_FIELDS
    assert body["question_id"] == "Q-9"
    assert "오류" in body["answer"] or "답변을 드리지 못했" in body["answer"]
    # 원인은 삼켜지지 않고 think_trace에 남아야 디버깅이 된다
    assert "RuntimeError" in body["think_trace"]


def test_missing_answer_falls_back_to_disclosure(client):
    """그래프가 answer 없이 돌아와도 빈 문자열을 내보내지 않는다."""
    with client(_FakeGraph({"think_trace": "t"})) as c:
        body = c.get("/answer", params={"question": "세액공제 한도는?"}).json()

    assert body["answer"]


def test_health_reports_graph_readiness(client):
    with client(_ok_graph()) as c:
        body = c.get("/health").json()

    assert body["status"] == "ok"
    assert body["graph_ready"] is True


def test_question_parameter_is_required(client):
    """question 없이 호출하면 스키마 검증에서 걸린다."""
    with client(_ok_graph()) as c:
        assert c.get("/answer", params={"question_id": "Q-1"}).status_code == 422
