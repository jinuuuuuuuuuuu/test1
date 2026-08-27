"""L0 결정론적 검증(src/agents/verification.py) — 실제 실패 사례("근거 0건인데 노란우산공제
연 500만원 같은 학습 지식 숫자로 답변, ④가 grounded=True로 통과")의 재발을 코드 수준에서
막는지 검증한다.
"""

import pytest

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


def test_issues_present_forces_grounded_false():
    """④ LLM이 issues에 위반을 적고도 grounded=True로 통과시키는 모순을 코드가 바로잡는다.

    실측: 개별 펀드 2건을 근거로 "연금저축 펀드는 일반적으로 환매 제한이 없다"고 단정한
    초안에 대해, LLM이 issues에는 "일부 펀드 정보를 일반 규칙으로 단정했다"고 정확히
    지적하면서도 grounded=True를 반환했다. 판정과 근거가 어긋나면 위반 쪽으로 확정한다.
    """
    from src.agents.verification import apply_l0_overrides

    result = apply_l0_overrides(
        {
            "grounded": True,
            "issues": ["일부 펀드에만 해당하는 정보를 전체 제도의 규칙으로 단정했습니다."],
            "unsupported_numbers_confirmed": [],
        },
        suspects=[],
        has_evidence=True,
    )

    assert result["grounded"] is False
    assert result["issues"]


def test_no_issues_keeps_grounded_true():
    """위반 지적이 없으면 grounded=True는 그대로 유지되어야 한다(과잉 차단 방지)."""
    from src.agents.verification import apply_l0_overrides

    result = apply_l0_overrides(
        {"grounded": True, "issues": [], "unsupported_numbers_confirmed": []},
        suspects=[],
        has_evidence=True,
    )

    assert result["grounded"] is True


# ── ⑤ 생성기 출력 강제 (F-3) ──────────────────────────────────────────
#
# ④의 판정을 ⑤가 무시하던 경로를 코드로 막는다. 실측 실패 4건(S1/S2/M2/C3)이
# 전부 "④는 정확히 지적했는데 최종 답변에 반영이 안 됨"이었다.


def test_missing_requirements_appends_limit_disclosure():
    """답하지 못한 요구 항목이 있으면 한계를 명시한다 — 요강 '정보한계 대응'."""
    from src.agents.verification import enforce_missing_requirements

    out = enforce_missing_requirements("연금계좌 세금혜택은 네 가지입니다.", ["2027년 개편안 확정 내용"])

    assert "확인이 어려워" in out
    assert "2027년 개편안 확정 내용" in out


def test_missing_requirements_skips_when_already_disclosed():
    """이미 한계를 고지한 답변에 또 붙이면 중복이 된다."""
    from src.agents.verification import enforce_missing_requirements

    answer = "그 부분은 제공된 자료로는 확인이 어렵습니다."
    assert enforce_missing_requirements(answer, ["무언가"]) == answer


def test_missing_requirements_noop_when_empty():
    from src.agents.verification import enforce_missing_requirements

    answer = "정상 답변입니다."
    assert enforce_missing_requirements(answer, []) == answer


def test_premise_issues_prepends_correction():
    """잘못된 전제를 초안이 안 짚었으면 앞머리에 교정문을 붙인다 — 요강 '정확성'."""
    from src.agents.verification import enforce_premise_issues

    out = enforce_premise_issues("IRP는 사유가 있어야 인출됩니다.", ["IRP는 중도인출이 자유롭다"])

    assert out.startswith("먼저 질문에 담긴 전제")
    assert "IRP는 중도인출이 자유롭다" in out


def test_premise_issues_skips_when_already_corrected():
    """초안이 이미 전제를 바로잡았으면 덧붙이지 않는다 (실측 S2 패턴)."""
    from src.agents.verification import enforce_premise_issues

    answer = "말씀하신 것처럼 IRP는 중도인출이 완전히 자유로운 것은 아닙니다."
    assert enforce_premise_issues(answer, ["IRP는 중도인출이 자유롭다"]) == answer


def test_evidence_placeholders_replaced_with_source_names():
    """'[근거 N]' 내부 표기를 실제 출처명으로 치환한다 — 요강 '근거 문서 표시'."""
    from src.agents.verification import replace_evidence_placeholders

    context = [{"source": "doc41 세액공제 규칙"}, {"source": "doc38 연금소득세율"}]
    out = replace_evidence_placeholders("한도는 900만원입니다 (출처: [근거 1]). [근거 2]도 참고.", context)

    assert "doc41 세액공제 규칙" in out
    assert "doc38 연금소득세율" in out
    assert "[근거" not in out
    # 이미 "출처:" 라벨이 있는 자리에 접두사를 또 붙이면 안 된다
    assert "출처: 출처:" not in out


def test_evidence_placeholder_out_of_range_is_dropped():
    """근거 개수를 넘는 번호는 LLM이 지어낸 것이므로 표기를 지운다."""
    from src.agents.verification import replace_evidence_placeholders

    out = replace_evidence_placeholders("근거입니다 [근거 7].", [{"source": "doc41"}])

    assert "[근거" not in out
    assert "7" not in out


# ── 인라인 서식이 L0를 뚫던 구멍 ──────────────────────────────────────
#
# LLM이 수치를 강조할 때 숫자만 감싸고 단위를 밖에 두면(**16.5**%) "숫자+단위" 패턴이
# 서식 문자로 갈라져 L0가 토큰을 아예 추출하지 못했다. 표기 미관 문제가 아니라
# 할루시네이션 방어선의 구멍이다 — 지어낸 수치도 강조로 감싸면 통과했다.


@pytest.mark.parametrize("text,expected", [
    ("16.5%", "16.5%"),
    ("**16.5**%", "16.5%"),      # 굵게: 단위가 밖으로 (실측된 형태)
    ("**16.5%**", "16.5%"),      # 굵게: 전체를 감쌈
    ("*16.5*%", "16.5%"),        # 기울임
    ("__16.5__%", "16.5%"),      # 밑줄
    ("`16.5`%", "16.5%"),        # 코드
    ("1,**800**만원", "1,800만원"),  # 숫자 내부에 서식
    ("**74**세", "74세"),
    ("**10**년", "10년"),
])
def test_number_extraction_survives_inline_markup(text, expected):
    """어떤 인라인 서식이 섞여도 수치를 추출해야 한다."""
    from src.agents.verification import extract_number_tokens

    assert extract_number_tokens(text) == [expected]


def test_fabricated_number_is_caught_even_when_emphasized():
    """근거에 없는 수치는 강조로 감싸도 잡혀야 한다 — 이게 원래 뚫려 있던 구멍이다."""
    from src.agents.verification import find_unsupported_numbers

    evidence = ["세액공제율은 16.5% 또는 13.2%"]

    assert find_unsupported_numbers("세액공제율은 **99.9**%입니다", evidence) == ["99.9%"]
    assert find_unsupported_numbers("세액공제율은 99.9%입니다", evidence) == ["99.9%"]


def test_supported_number_still_passes_when_emphasized():
    """근거에 있는 수치는 강조 여부와 무관하게 통과한다 (과잉 차단 방지)."""
    from src.agents.verification import find_unsupported_numbers

    evidence = ["세액공제율은 16.5% 또는 13.2%"]

    assert find_unsupported_numbers("세액공제율은 **16.5**%입니다", evidence) == []


def test_markup_in_evidence_does_not_break_matching():
    """근거 쪽에 서식이 있어도 대조가 어긋나면 안 된다."""
    from src.agents.verification import find_unsupported_numbers

    assert find_unsupported_numbers("한도는 1,800만원입니다", ["한도는 1,**800**만원"]) == []


# ── premise_issues 오분류 재분류 (2026-08-27) ──────────────────────────
#
# ④가 premise_issues에 "답변의 결함"을 적어 넣으면 최종 답변이 "먼저 질문에 담긴
# 전제를 짚고 넘어가겠습니다: 초안이 날짜에 직접 답하지 않음"처럼, 사용자가 하지도
# 않은 말을 전제라고 지적하는 문장으로 시작한다.


@pytest.mark.parametrize("text", [
    "초안이 날짜에 직접 답하지 않음",
    "초안이 구체적인 마감일을 제시하지 않음",
    "전월세 중도인출 기한이 누락됨",
    # 인용 어미("라고 하여")가 결함 서술에 섞인 실측 사례 — 인용 표현만으로 진짜
    # 전제라고 단정하면 이 문장을 놓쳐 답변이 이상한 도입부로 시작한다.
    "질문은 '나 74세인데 세금 어떻게 내?'라고 하여 구체적인 상황 정보를 제공하지 않고 있음",
])
def test_answer_defect_statements_are_detected(text):
    from src.agents.verification import is_answer_defect_statement

    assert is_answer_defect_statement(text)


@pytest.mark.parametrize("text", [
    "IRP는 중도인출이 자유롭다던데",
    "명퇴수당을 연금계좌에 넣으면 세금 감면이 '어마어마하다던데'",
    "연금저축은 아무 때나 인출할 수 있다고 들었는데",
])
def test_real_user_premises_are_kept(text):
    """사용자 발화를 인용한 진짜 전제는 재분류하지 않는다."""
    from src.agents.verification import is_answer_defect_statement

    assert not is_answer_defect_statement(text)


def test_split_premise_issues_separates_by_nature():
    from src.agents.verification import split_premise_issues

    real, misfiled = split_premise_issues([
        "IRP는 중도인출이 자유롭다던데",
        "초안이 날짜에 직접 답하지 않음",
    ])

    assert real == ["IRP는 중도인출이 자유롭다던데"]
    assert misfiled == ["초안이 날짜에 직접 답하지 않음"]
