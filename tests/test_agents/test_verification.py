"""L0 결정론적 검증(src/agents/verification.py) — 실제 실패 사례("근거 0건인데 노란우산공제
연 500만원 같은 학습 지식 숫자로 답변, ④가 grounded=True로 통과")의 재발을 코드 수준에서
막는지 검증한다.
"""

from src.agents.verification import (
    apply_clarification_override,
    apply_l0_overrides,
    extract_number_tokens,
    find_unsupported_numbers,
)

# ── extract_number_tokens ────────────────────────────────────────────────


def test_extracts_numbers_with_units():
    text = "연금저축 600만원, 합산 900만원까지 세액공제되며 공제율은 16.5%입니다. 55세부터 10년간 수령 시."
    tokens = extract_number_tokens(text)
    assert "600만원" in tokens
    assert "900만원" in tokens
    assert "16.5%" in tokens
    assert "55세" in tokens
    assert "10년" in tokens


def test_ignores_bare_list_markers():
    # 단위 없는 맨 숫자(목록 번호 등)는 오탐을 만들므로 L0에서 잡지 않는다.
    text = "확인이 필요합니다: 1. 계좌유형 2. 투자기간 3. 위험선호"
    assert extract_number_tokens(text) == []


def test_ignores_tax_compound_words():
    # '세액공제', '세율' 같은 복합어의 '세'는 나이 단위가 아니다.
    assert extract_number_tokens("세액공제와 세율 안내") == []


def test_extracts_risk_grade_and_bare_man_units():
    tokens = extract_number_tokens("위험등급 2등급, 노란우산공제는 연 500만 한도입니다.")
    assert "2등급" in tokens
    assert "500만" in tokens


def test_deduplicates_tokens():
    assert extract_number_tokens("한도는 900만원이며, 900만원을 넘으면 안 됩니다.") == ["900만원"]


# ── find_unsupported_numbers ─────────────────────────────────────────────


def test_supported_numbers_are_not_flagged():
    draft = "연금저축과 IRP를 합산해 900만원까지 세액공제됩니다."
    evidence = ["연금저축·IRP 합산 세액공제 한도는 900만원이다."]
    assert find_unsupported_numbers(draft, evidence) == []


def test_comma_notation_difference_is_normalized():
    draft = "총 1,200만원까지 납입할 수 있습니다."
    evidence = ["연간 납입 한도는 1200만원이다."]
    assert find_unsupported_numbers(draft, evidence) == []


def test_hallucinated_number_is_flagged():
    # 실제 실패 사례: 근거에 전혀 없는 "노란우산공제 연 500만원"
    draft = "노란우산공제로 연 500만원까지 공제받을 수 있습니다."
    evidence = ["연금저축 세액공제 한도는 600만원이다."]
    assert find_unsupported_numbers(draft, evidence) == ["500만원"]


def test_user_stated_numbers_are_not_flagged():
    # 역질문 흐름 턴3: 사용자가 말한 조건(월 30만원, 10년)을 초안이 되받아 정리하는 것은
    # 할루시네이션이 아니다 — 이전 턴 사용자 발화가 지원 근거로 인정돼야 한다.
    draft = "정리하면 월 30만 원, 투자 기간 10년 이상, 공격적인 성향이시군요."
    user_texts = ["좋은 연금 상품 하나 추천해주세요", "월 투자 가능한 금액은 30만원 정도야", "예상 투자 기간은 10년 이상 될것 같아"]
    assert find_unsupported_numbers(draft, [], user_texts=user_texts) == []


def test_prior_answer_numbers_do_not_legitimize_reuse():
    # 사용자 발화가 아닌 곳(과거 '답변' 등)의 수치는 지원 근거가 아니다 — user_texts에
    # 답변을 넣지 않는 호출 규약을 전제로, 근거 없는 수치는 여전히 잡혀야 한다.
    draft = "앞서 말씀드린 대로 세액공제 한도는 900만원입니다."
    assert find_unsupported_numbers(draft, [], user_texts=["그 한도 다시 알려줘"]) == ["900만원"]


def test_empty_evidence_flags_all_numbers():
    draft = "비용처리와 함께 노란우산공제 연 500만원, 공제율 16.5%가 대표적입니다."
    suspects = find_unsupported_numbers(draft, [])
    assert "500만원" in suspects
    assert "16.5%" in suspects


# ── apply_l0_overrides ───────────────────────────────────────────────────


def _llm_pass() -> dict:
    """LLM이 (잘못) 전부 통과시킨 판정 — 실패 사례에서 실제로 관찰된 출력."""
    return {
        "grounded": True,
        "issues": [],
        "unsupported_numbers_confirmed": [],
        "premise_issues": [],
        "requirements_met": True,
        "missing_requirements": [],
    }


def test_no_evidence_with_numbers_forces_grounded_false():
    # 핵심 재발 방지: 근거 0건 + 수치 존재면 LLM이 통과시켜도 무조건 불합격.
    result = apply_l0_overrides(_llm_pass(), suspects=["500만원"], has_evidence=False)
    assert result["grounded"] is False
    assert any("500만원" in issue for issue in result["issues"])


def test_llm_confirmed_unsupported_number_forces_grounded_false():
    verification = _llm_pass()
    verification["unsupported_numbers_confirmed"] = ["500만원"]
    result = apply_l0_overrides(verification, suspects=["500만원"], has_evidence=True)
    assert result["grounded"] is False


def test_llm_cannot_invent_confirmed_numbers_outside_suspects():
    # 확인 목록은 의심 목록의 부분집합으로 강제 — LLM이 지어낸 수치는 무시된다.
    verification = _llm_pass()
    verification["unsupported_numbers_confirmed"] = ["999만원"]
    result = apply_l0_overrides(verification, suspects=["500만원"], has_evidence=True)
    assert result["grounded"] is True
    assert result["unsupported_numbers_confirmed"] == []


def test_llm_cleared_suspects_keep_grounded_true():
    # 근거가 있고 LLM이 의심 수치를 표기 차이로 판정해 전부 걸렀으면 통과 유지.
    result = apply_l0_overrides(_llm_pass(), suspects=["900만원"], has_evidence=True)
    assert result["grounded"] is True
    assert result["l0_suspect_numbers"] == ["900만원"]


def test_llm_grounded_false_is_preserved():
    verification = _llm_pass()
    verification["grounded"] = False
    result = apply_l0_overrides(verification, suspects=[], has_evidence=True)
    assert result["grounded"] is False


def test_numberless_draft_with_no_evidence_passes():
    # 순수 개념 설명(수치 없음)은 근거 0건이어도 L0가 강제로 떨어뜨리지 않는다.
    result = apply_l0_overrides(_llm_pass(), suspects=[], has_evidence=False)
    assert result["grounded"] is True


# ── apply_clarification_override ─────────────────────────────────────────


def test_clarification_override_exempts_requirements_check():
    # ③의 의도적 역질문을 ④가 "추천 누락"으로 판정 → ⑤가 추천을 되살리는 경로 차단.
    verification = _llm_pass()
    verification["requirements_met"] = False
    verification["missing_requirements"] = ["상품 추천"]
    result = apply_clarification_override(verification)
    assert result["requirements_met"] is True
    assert result["missing_requirements"] == []
    assert result["clarification_mode"] is True


def test_clarification_override_keeps_grounded_verdict():
    # 역질문 문장에 근거 없는 수치가 섞인 경우의 grounded=False는 면제되지 않는다.
    verification = _llm_pass()
    verification["grounded"] = False
    result = apply_clarification_override(verification)
    assert result["grounded"] is False
