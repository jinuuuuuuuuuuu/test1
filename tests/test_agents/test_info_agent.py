import os

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

from src.agents.info_agent import build_info_agent_node


def test_deterministic_generic_withdrawal_documents_sets_clarification_mode():
    node = build_info_agent_node()

    result = node({
        "question": "IRP 중도인출 필요서류 알려줘",
        "deterministic_category": "복합정보_태스크플랜",
    })

    assert result["needs_clarification"] is True
    assert result["response_mode"] == "clarification_included"
    assert "추가로 필요한 정보는 다음과 같습니다" in result["info_draft"]
    assert result["deterministic_info"] is True
