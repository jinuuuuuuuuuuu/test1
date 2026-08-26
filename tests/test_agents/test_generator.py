"""⑤생성기의 결정론적 분기(LLM 호출 없이 처리되는 경로) 검증 — 더미 API 키로 실행 가능."""

import os

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

from src.agents.generator import build_generator_node


def test_unsafe_returns_canned_refusal_without_llm():
    node = build_generator_node()
    result = node({"question": "탈세 방법 알려줘", "is_safe": False, "safety_reason": "불법 행위 문의"})
    assert "답변드릴 수 없습니다" in result["answer"]
    assert "불법 행위 문의" in result["think_trace"]


def test_out_of_scope_returns_canned_limitation_without_llm():
    # 범위외 질문은 LLM 호출 없이(더미 키에서 네트워크 호출이 나면 이 테스트가 에러로 실패)
    # 정형 한계 고지로 답한다 — "정보한계 대응" 평가지표 경로.
    node = build_generator_node()
    result = node({
        "question": "삼성전자 주식 지금 사도 돼?",
        "is_safe": True,
        "scope": "범위외",
        "scope_note": "개별 주식 투자 판단은 연금 상담과 무관",
    })
    assert "상담 범위" in result["answer"]
    assert "연금" in result["answer"]
    assert "범위외" in result["think_trace"]
    assert "개별 주식 투자 판단은 연금 상담과 무관" in result["think_trace"]


# ── ④ 검증 결과의 ⑤ 강제 반영 (F-3) ──────────────────────────────────

def _ctx(*sources):
    return [{"source": s, "content": ""} for s in sources]


def test_enforce_adds_limit_disclosure_when_requirement_missing():
    """④가 '답 못한 항목'을 짚었는데 답변이 무시하면 코드가 한계를 명시한다 (실측 S1/M2)."""
    from src.agents.generator import _enforce_verification

    out = _enforce_verification(
        "연금계좌 세금혜택은 네 가지입니다.",
        {"missing_requirements": ["2027년 개편안 확정 내용"], "premise_issues": []},
        _ctx("doc38~41 세금혜택 규칙"),
    )

    assert "확인이 어려워" in out
    assert "2027년 개편안 확정 내용" in out


def test_enforce_prefers_limit_disclosure_over_premise_when_duplicated():
    """④가 같은 항목을 두 필드에 넣으면 한계 고지로 처리한다 (실측 S1).

    자료에 없어 못 답한 것을 '사실과 다른 전제'라고 말하면 부정확하다.
    """
    from src.agents.generator import _enforce_verification

    item = "2027년 개편안 확정 내용"
    out = _enforce_verification(
        "현행 제도는 이렇습니다.",
        {"missing_requirements": [item], "premise_issues": [item]},
        _ctx("doc38"),
    )

    assert "확인이 어려워" in out
    assert not out.startswith("먼저 질문에 담긴 전제")
    assert out.count(item) == 1  # 같은 문구가 두 번 나오면 안 된다


def test_enforce_skips_limit_disclosure_in_clarification_mode():
    """역질문 초안은 의도적 유보이므로 '확인이 어렵다'를 덧붙이지 않는다."""
    from src.agents.generator import _enforce_verification

    answer = "추천을 위해 계좌유형과 투자성향을 알려주세요."
    out = _enforce_verification(
        answer,
        {"missing_requirements": ["투자성향"], "premise_issues": [], "clarification_mode": True},
        [],
    )

    assert "확인이 어려워" not in out


def test_enforce_replaces_placeholders_and_appends_reference_line():
    """출처 표기는 코드가 확정한다 — LLM에 맡기면 '[근거 1]'이 그대로 나간다 (실측 5/7)."""
    from src.agents.generator import _enforce_verification

    out = _enforce_verification(
        "한도는 900만원입니다 (출처: [근거 1]).",
        {"missing_requirements": [], "premise_issues": []},
        _ctx("doc41 세액공제 규칙"),
    )

    assert "[근거" not in out
    assert "doc41 세액공제 규칙" in out
    assert "참고 근거:" in out


def test_enforce_leaves_clean_answer_untouched_except_reference():
    """위반이 없으면 출처 줄 외에는 답변을 건드리지 않는다 (과잉 개입 방지)."""
    from src.agents.generator import _enforce_verification

    answer = "연금저축과 IRP 합산 세액공제 한도는 900만원입니다.\n\n참고 근거: doc41"
    out = _enforce_verification(answer, {"missing_requirements": [], "premise_issues": []}, _ctx("doc41"))

    assert out == answer
