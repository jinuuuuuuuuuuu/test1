from src.agents.generator import _grounding_failure_answer, build_generator_node


def test_grounding_failure_answer_does_not_use_ungrounded_draft_without_context():
    answer = _grounding_failure_answer(
        {
            "info_draft": "근거 없는 일반 지식 답변입니다.",
            "verification": {"grounded": False, "issues": ["근거 없음"]},
            "retrieved_context": [],
        }
    )

    assert "관련 근거 검색이 되지 않았습니다" in answer
    assert "근거 없이 숫자나 요건을 단정" in answer
    assert "근거 없는 일반 지식 답변입니다." not in answer


def test_grounding_failure_answer_uses_only_retrieved_context_when_present():
    answer = _grounding_failure_answer(
        {
            "info_draft": "1,500만원까지 세액공제됩니다.",
            "verification": {"grounded": False, "issues": ["근거와 불일치"]},
            "retrieved_context": [
                {
                    "source": "doc41",
                    "content": "연금저축+IRP 합산 세액공제 대상 한도는 900만원입니다.",
                    "node": "info_agent",
                }
            ],
        }
    )

    assert "900만원" in answer
    assert "1,500만원까지 세액공제됩니다." not in answer


def test_generator_returns_type_recommendation_draft_directly():
    node = build_generator_node()
    result = node(
        {
            "question": "상품 관련해서 추천해줘",
            "product_draft": "TDF와 채권혼합형 펀드를 먼저 고려하세요.\n`상품추천`을 입력해 주세요.",
            "recommendation_stage": "type_recommendation",
            "verification": {"grounded": False},
            "retrieved_context": [],
        }
    )

    assert "TDF와 채권혼합형" in result["answer"]
    assert "`상품추천`" in result["answer"]


def test_generator_returns_deterministic_info_draft_directly():
    node = build_generator_node()
    result = node(
        {
            "question": "세액공제 얼마까지 되나요?",
            "info_draft": "세액공제 대상 한도는 합산 900만원입니다.",
            "deterministic_info": True,
            "verification": {"grounded": True},
            "retrieved_context": [],
        }
    )

    assert result["answer"] == "세액공제 대상 한도는 합산 900만원입니다."
