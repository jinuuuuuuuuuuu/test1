"""정보 Agent의 결정론 실물이전 판정 선택 규칙."""

import os

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

from src.agents.info_agent import _select_deterministic_response


def test_generic_in_kind_transfer_does_not_create_product_judgement():
    result = _select_deterministic_response(
        "해당없음", "IRP 상품 그대로 이전할 수 있는 방법 알려줘"
    )

    assert result is None


def test_explicit_in_kind_transfer_product_uses_deterministic_judgement():
    result = _select_deterministic_response("해당없음", "RP 상품은 실물이전 가능한가요?")

    assert result is not None
    draft, context = result
    assert "환매조건부채권" in draft
    assert context[0]["source"] == "doc34 실물이전 불가사유 코드"


def test_forced_doc_search_is_not_skipped_for_clarification_answers():
    """역질문 답변이어도 근거가 0건이면 강제 검색을 건너뛰면 안 된다.

    예전 조건은 `not needs_clarification and not retrieved_context`였다. 그런데
    시스템 프롬프트는 조건이 부족한 질문에도 "현재 답변 가능한 일반 기준"을 함께
    쓰라고 지시한다 — 역질문 답변에도 사실 서술이 들어간다. 안전망만 "역질문이면
    근거가 필요 없다"고 가정한 탓에, 되묻기 직전에 쓴 일반 기준이 근거 0건으로
    나갔다(실측 501문항 6건: no.86/99/104/140/321/483이 전부 이 경로였다).

    구현을 직접 호출하면 실제 LLM·벡터DB가 필요하므로, 조건식이 되돌아가지
    않았다는 것만 소스에서 확인한다.
    """
    import inspect

    from src.agents import info_agent

    source = inspect.getsource(info_agent.build_info_agent_node)
    assert "if not retrieved_context:" in source, (
        "근거 0건일 때의 강제 검색 조건에 needs_clarification이 다시 끼어들면, "
        "역질문 답변이 근거 없이 수치를 지어내는 경로가 되살아난다."
    )
