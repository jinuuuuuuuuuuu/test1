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


# ── ④가 확정한 근거 없는 수치가 답변에 남으면 경고 (500문항 실측) ─────────
#
# grounded=False/issues는 ⑤ 시스템 프롬프트에 "쓰지 마세요"라는 부탁만 있고 코드
# 강제가 없었다. 실측(no.67): "연간 납입액 700만원까지 가능합니다"(옛 한도, 현행
# 900만원)가 ④에서 근거 없음으로 확정됐는데도 최종 답변에 그대로 남았다.


def test_enforce_flags_confirmed_unsupported_number_left_in_answer():
    from src.agents.generator import _enforce_verification

    answer = "일반적인 경우 연간 납입액 700만원까지 가능합니다."
    verification = {
        "grounded": False,
        "unsupported_numbers_confirmed": ["700만원"],
        "missing_requirements": [],
        "premise_issues": [],
    }
    out = _enforce_verification(answer, verification, [])

    assert "700만원" in out  # 삭제하지 않는다 — 문장이 깨진다
    assert "확인되지 않아 참고용입니다" in out
    assert "700만원" in out.split("참고용입니다")[1]  # 경고 문구에도 수치가 나열됨


def test_enforce_does_not_flag_number_used_in_negation_context():
    """수치를 '틀렸다'고 바로잡는 데 쓰였으면 경고를 붙이지 않는다.

    실측(no.5): "평균 임금의 60%가 아니라 30일분에 계속근로기간을 곱하여"처럼 틀린
    수치를 부정하며 교정한 답변까지 위반으로 잡으면, 오히려 올바른 답변에 불필요한
    경고가 붙는다.
    """
    from src.agents.generator import _enforce_verification

    answer = (
        "퇴직급여는 평균 임금의 60%가 아니라 30일분에 계속근로기간을 곱하여 계산됩니다."
    )
    verification = {
        "grounded": False,
        "unsupported_numbers_confirmed": ["60%"],
        "missing_requirements": [],
        "premise_issues": [],
    }
    out = _enforce_verification(answer, verification, [])

    assert out == answer  # 손대지 않는다
    assert "참고용입니다" not in out


# ── ④가 확정한 "근거에 없는 수치" 목록을 ⑤ 프롬프트로 전달 ──────────────


def _capture_generator_prompt(monkeypatch, verification: dict) -> str:
    """⑤가 LLM에 실제로 보내는 프롬프트를 가로채 반환한다 (네트워크 호출 없음)."""
    import src.agents.generator as gen

    captured = {}

    class FakeLLM:
        pass

    def fake_invoke(_llm, messages):
        captured["prompt"] = messages[-1]["content"]

        class R:
            content = "최종 답변입니다."

        return R()

    monkeypatch.setattr(gen, "get_llm", lambda *a, **k: FakeLLM())
    monkeypatch.setattr(gen, "invoke_with_retry", fake_invoke)

    node = gen.build_generator_node()
    node({
        "question": "연금저축 세액공제율이 얼마인가요?",
        "is_safe": True,
        "scope": "범위내",
        "info_draft": "세액공제율은 16.5%입니다.",
        "retrieved_context": [{"source": "doc41", "content": "세액공제 관련 규정", "node": "info_agent"}],
        "verification": verification,
    })
    return captured["prompt"]


def test_confirmed_unsupported_numbers_are_passed_to_generator_prompt(monkeypatch):
    """④가 '근거에 없다'고 확정한 수치 목록이 ⑤ 프롬프트에 실제로 전달돼야 한다.

    이 목록 없이 "숫자를 근거와 대조하라"고만 시키면 ⑤가 그 대조를 처음부터 다시 해야 한다 —
    ④는 L0(기계적 토큰 대조) + L1(LLM 확인)을 거쳐 답을 이미 갖고 있는데도 넘기지 않던
    누락을 막는다.
    """
    prompt = _capture_generator_prompt(
        monkeypatch,
        {
            "grounded": False,
            "issues": ["근거 원문에 없는 수치입니다: 16.5%"],
            "unsupported_numbers_confirmed": ["16.5%", "900만원"],
            "premise_issues": [],
            "requirements_met": True,
            "missing_requirements": [],
        },
    )

    assert "근거에 없는 것으로 확정된 수치" in prompt
    assert "16.5%" in prompt
    assert "900만원" in prompt


def test_no_unsupported_numbers_line_when_list_is_empty(monkeypatch):
    """확정 목록이 비면 그 줄을 넣지 않는다 — 빈 목록을 보여주면 프롬프트만 길어진다."""
    prompt = _capture_generator_prompt(
        monkeypatch,
        {
            "grounded": True,
            "issues": [],
            "unsupported_numbers_confirmed": [],
            "premise_issues": [],
            "requirements_met": True,
            "missing_requirements": [],
        },
    )

    assert "근거에 없는 것으로 확정된 수치" not in prompt


# ── Guardian 최종 조립: Core 불변 + 근거 격리 ─────────────────────────────


def test_enforce_can_skip_reference_append_for_finalizer():
    from src.agents.generator import _enforce_verification

    answer = "연금저축과 IRP 합산 세액공제 한도는 900만원입니다."
    out = _enforce_verification(
        answer,
        {"missing_requirements": [], "premise_issues": []},
        _ctx("doc41"),
        append_reference=False,
    )

    assert out == answer


def test_finalize_answer_keeps_core_unchanged_when_guardian_off():
    from src.agents.generator import _finalize_answer

    core = "추천을 위해 투자성향을 알려주세요."
    out = _finalize_answer(
        core,
        {"guardian_result": {"enabled": False}, "guardian_evidence": []},
        [],
    )

    assert out == core


def test_finalize_answer_appends_single_guardian_block_and_reference():
    from src.agents.generator import _finalize_answer

    out = _finalize_answer(
        "필요서류는 다음과 같습니다.",
        {
            "guardian_result": {
                "enabled": True,
                "message": "🛡️ 파수꾼 체크\n재원 구분도 함께 확인해야 합니다.",
            },
            "guardian_evidence": [
                {"source": "guard-doc", "content": "세금 주의", "node": "guardian"}
            ],
        },
        [{"source": "core-doc", "content": "서류", "node": "info_agent"}],
    )

    assert "필요서류는 다음과 같습니다." in out
    assert out.count("🛡️ 파수꾼 체크") == 1
    assert "참고 근거: core-doc; guard-doc" in out


def test_generator_prompt_excludes_guardian_evidence_but_final_references_include_it(monkeypatch):
    import src.agents.generator as gen

    captured = {}

    class FakeLLM:
        pass

    def fake_invoke(_llm, messages):
        captured["prompt"] = messages[-1]["content"]

        class R:
            content = "핵심 답변입니다."

        return R()

    monkeypatch.setattr(gen, "get_llm", lambda *a, **k: FakeLLM())
    monkeypatch.setattr(gen, "invoke_with_retry", fake_invoke)

    node = gen.build_generator_node()
    result = node({
        "question": "전세보증금 중도인출 필요서류 알려줘",
        "is_safe": True,
        "scope": "범위내",
        "response_mode": "complete",
        "info_draft": "필요서류는 다음과 같습니다.",
        "retrieved_context": [{"source": "core-doc", "content": "서류", "node": "info_agent"}],
        "guardian_result": {
            "enabled": True,
            "message": "🛡️ 파수꾼 체크\n재원 구분도 함께 확인해야 합니다.",
        },
        "guardian_evidence": [{"source": "guard-doc", "content": "세금 주의", "node": "guardian"}],
        "verification": {
            "grounded": True,
            "requirements_met": True,
            "missing_requirements": [],
            "premise_issues": [],
        },
    })

    assert "guard-doc" not in captured["prompt"]
    assert "세금 주의" not in captured["prompt"]
    assert "참고 근거: core-doc; guard-doc" in result["answer"]
