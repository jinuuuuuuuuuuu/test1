"""④검증(grounding) 노드 테스트 — 특히 결정론 답변 검증 면제 경로."""

import src.agents.grounding as grounding


class _FakeLLM:
    """with_structured_output 체이닝만 흉내낸다 — 실제로는 호출되면 안 되는 자리에 둔다."""

    def with_structured_output(self, *_args, **_kwargs):
        return self


def _fail_if_called(_llm, _messages):
    raise AssertionError("결정론 답변 경로에서는 grounding LLM이 호출되면 안 된다")


def test_deterministic_answer_skips_llm_verification(monkeypatch):
    """deterministic_info=True면 LLM을 호출하지 않고 grounded=True로 확정해야 한다.

    회귀 방지: 501문항 실측에서 결정론 답변(LLM이 전혀 개입하지 않고
    deterministic_response_for()가 그대로 낸 draft) 184건 중 13건이 grounded=False로
    탈락했고, 그 지적은 대부분 부정확했다(④ 스스로 "문제가 되지 않음"이라 써놓고도
    False를 낸 사례, ④ 쪽이 사실관계를 틀린 사례 포함). 47건(26%)이 불필요한 repair를
    돌았는데, repair는 같은 결정론 함수를 다시 불러 100% 동일한 draft를 재생산하므로
    LLM 호출만 낭비되고 결과는 바뀌지 않는다.
    """
    monkeypatch.setattr(grounding, "get_llm", lambda *a, **k: _FakeLLM())
    monkeypatch.setattr(grounding, "invoke_with_retry", _fail_if_called)

    node = grounding.build_grounding_node()
    result = node({
        "question": "세액공제 한도가 얼마인가요?",
        "info_draft": "연금저축 단독 600만원, 합산 900만원입니다.",
        "product_draft": None,
        "retrieved_context": [
            {"source": "doc41", "content": "연금저축 단독 600만원, 합산 900만원", "node": "info_agent"}
        ],
        "deterministic_info": True,
    })

    verification = result["verification"]
    assert verification["grounded"] is True
    assert verification["requirements_met"] is True
    assert verification["issues"] == []
    assert verification["missing_requirements"] == []


def test_deterministic_composite_with_product_draft_still_verifies(monkeypatch):
    """product_draft가 함께 있는 복합형은 결정론 면제 대상에서 제외해야 한다.

    ③상품 Agent는 deterministic_info 플래그를 쓰지 않으므로, product_draft가 있다는
    것은 그 부분이 LLM 생성이라는 뜻이다 — 면제하면 그 부분의 환각을 놓친다.
    """
    called = {"count": 0}

    def fake_invoke(_llm, _messages):
        called["count"] += 1

        class R:
            grounded = True
            issues: list = []
            unsupported_numbers_confirmed: list = []
            premise_issues: list = []
            requirements_met = True
            missing_requirements: list = []

        return R()

    monkeypatch.setattr(grounding, "get_llm", lambda *a, **k: _FakeLLM())
    monkeypatch.setattr(grounding, "invoke_with_retry", fake_invoke)

    node = grounding.build_grounding_node()
    node({
        "question": "세액공제도 알려주고 상품도 추천해주세요.",
        "info_draft": "연금저축 단독 600만원, 합산 900만원입니다.",
        "product_draft": "추천 상품: OO펀드",
        "retrieved_context": [],
        "deterministic_info": True,
    })

    assert called["count"] == 1, "복합형(product_draft 존재)은 LLM 검증을 거쳐야 한다"


def test_non_deterministic_answer_still_verifies(monkeypatch):
    """deterministic_info=False(일반 LLM 답변)는 기존대로 LLM 검증을 거쳐야 한다."""
    called = {"count": 0}

    def fake_invoke(_llm, _messages):
        called["count"] += 1

        class R:
            grounded = True
            issues: list = []
            unsupported_numbers_confirmed: list = []
            premise_issues: list = []
            requirements_met = True
            missing_requirements: list = []

        return R()

    monkeypatch.setattr(grounding, "get_llm", lambda *a, **k: _FakeLLM())
    monkeypatch.setattr(grounding, "invoke_with_retry", fake_invoke)

    node = grounding.build_grounding_node()
    node({
        "question": "IRP는 무엇인가요?",
        "info_draft": "IRP는 개인형 퇴직연금제도입니다.",
        "product_draft": None,
        "retrieved_context": [{"source": "doc1", "content": "IRP 설명", "node": "info_agent"}],
        "deterministic_info": False,
    })

    assert called["count"] == 1
