import src.agents.grounding as grounding_module
import src.agents.router as router_module
from src.agents.router import _should_answer_with_financial_correction


class _StructuredOutputRecorder:
    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, kwargs))
        return object()


def test_financial_misconception_is_not_treated_as_unsafe():
    assert _should_answer_with_financial_correction(
        "확실하게 원금 보장되면서 수익률도 높은 펀드 추천해주세요."
    ) is True
    assert _should_answer_with_financial_correction(
        "내년에 제일 많이 오를 IRP 펀드를 알려주세요."
    ) is True


def test_genuinely_unsafe_request_is_not_overridden():
    assert _should_answer_with_financial_correction(
        "소득을 숨겨 탈세하면서 원금보장 상품을 사는 법을 알려줘."
    ) is False
    assert _should_answer_with_financial_correction(
        "주민등록번호로 연금계좌를 조회하고 원금보장 상품도 추천해줘."
    ) is False


def test_router_requests_clova_json_schema_mode(monkeypatch):
    recorder = _StructuredOutputRecorder()
    monkeypatch.setattr(router_module, "get_llm", lambda *args, **kwargs: recorder)

    router_module.build_router_node()

    assert recorder.calls
    assert recorder.calls[0][1] == {"method": "json_schema"}


def test_grounding_requests_clova_json_schema_mode(monkeypatch):
    recorder = _StructuredOutputRecorder()
    monkeypatch.setattr(grounding_module, "get_llm", lambda *args, **kwargs: recorder)

    grounding_module.build_grounding_node()

    assert recorder.calls
    assert recorder.calls[0][1] == {"method": "json_schema"}
