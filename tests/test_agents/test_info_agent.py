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
