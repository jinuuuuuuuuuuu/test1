from src.agents.tax_context import _is_personal_tax_question, _compact, personal_tax_response


def test_age_only_personal_tax_question_asks_branch_fields():
    draft, context = personal_tax_response("나 74세인데 세금 어떻게 내?")

    assert "나이 74세" in draft
    assert "수령 방식" in draft
    assert "돈의 출처" in draft
    assert "나이만으로 세율이나 세금을 확정하지 않겠습니다" in draft
    assert "4.4%" not in draft
    assert context


def test_pension_receipt_without_source_type_does_not_calculate():
    draft, _ = personal_tax_response("74세고 연금으로 연 1,000만원 받아. 세금은?")

    assert "연금수령" in draft
    assert "돈의 출처" in draft
    assert "연금으로 받고 있나요" not in draft
    assert "4.4%" not in draft


def test_tax_deducted_pension_without_lifetime_annuity_does_not_calculate():
    draft, _ = personal_tax_response("74세고 세액공제 받은 돈을 연금으로 연 1,000만원 받아. 세금은?")

    assert "세액공제 받은 납입금·운용수익을 연금으로 받는 경우" in draft
    assert "종신연금으로 받나요, 비종신 연금으로 받나요?" in draft
    assert "4.4%" not in draft


def test_tax_deducted_non_lifetime_pension_with_all_fields_calculates():
    draft, context = personal_tax_response("74세고 세액공제 받은 돈을 비종신 연금으로 연 1,000만원 받아. 세금은?")

    assert "세액공제 받은 납입금·운용수익 재원" in draft
    assert "1,500만원 이내" in draft
    assert "4.4%" in draft
    assert "1,000만원 x 4.4% = 44만원" in draft
    assert "예상 세금은 44만원" in draft
    assert "1,000만원 x 4.4% = 44만원" in context[0]["content"]


def test_tax_deducted_over_threshold_shows_optional_flat_tax_amount():
    draft, context = personal_tax_response("74세고 세액공제 받은 돈을 비종신 연금으로 연 2,000만원 받아. 세금은?")

    assert "1,500만원을 초과" in draft
    assert "16.5% 분리과세" in draft
    assert "2,000만원 x 16.5% = 330만원" in draft
    assert "종합과세를 선택하는 경우의 실제 세액" in draft
    assert "2,000만원 x 16.5% = 330만원" in context[0]["content"]


def test_non_tax_deducted_principal_does_not_ask_irrelevant_fields():
    draft, _ = personal_tax_response("74세이고 세액공제 안 받은 원금 인출해. 세금은?")

    assert "1,500만원 초과 여부를 판단할 때 포함하지 않는 재원" in draft
    assert "퇴직금 감면 계산으로 넘어가지 않습니다" in draft
    assert "추가로 필요한 정보" not in draft
    assert "연금실제수령연차" not in draft


def test_retirement_benefit_pension_requires_actual_pension_year():
    draft, _ = personal_tax_response("74세이고 퇴직금을 연금으로 받고 있어요. 세금은?")

    assert "퇴직금을 연금으로 받는 경우" in draft
    assert "연금실제수령연차" in draft
    assert "4.4%" not in draft


def test_retirement_benefit_pension_with_actual_year_calculates_reduction():
    draft, _ = personal_tax_response("퇴직금을 연금으로 받고 있고 연금실제수령연차 11년차야. 세금은?")

    assert "연금실제수령연차가 11년차" in draft
    assert "60%" in draft
    assert "40%" in draft


def test_lifetime_annuity_without_source_type_still_does_not_calculate():
    draft, _ = personal_tax_response("74세, 종신연금 아니야. 세금 어떻게 내?")

    assert "돈의 출처" in draft
    assert "4.4%" not in draft


def test_assumption_request_does_not_fill_missing_fields():
    draft, _ = personal_tax_response("74세인데 세금 알아서 적당히 가정해서 계산해줘")

    assert "임의 가정을 요청하더라도" in draft
    assert "대신 채우지 않겠습니다" in draft
    assert "4.4%" not in draft


# ── 개인세금 게이트 vs 연령별 세율 게이트 판별 (500문항 실측) ─────────────
#
# 실측 사고: 나이 언급 하나만으로 이 게이트가 발동하던 이전 로직은 "제가 60세인데
# 연금소득세율이 몇 %인가요?"조차 계산 질문으로 오판해, 501문항 중
# 개인세금_입력충분성으로 확정된 45건의 96%(43건)를 역질문으로 되돌렸다. 나이 구간
# 만으로 답이 완결되는 순수 세율 질문(연금소득세율_연령별이 정확히 답할 수 있는
# 영역)까지 전부 삼킨 것 — 판별선은 "세율(%)만 물었나 vs 세액(원)·복합조건을
# 물었나"여야 한다.


def test_rate_only_question_is_not_personal_tax_gate():
    """나이 구간만으로 완결되는 세율 질문은 개인세금 게이트가 아니다."""
    questions = [
        "제가 만 60세인데 연금소득세율이 몇 %인가요?",
        "70세 이상 세율 기준이 어떻게 돼?",
        "여든 살인데 연금 받으면 세금 얼마나 떼요?",
        "연금 받으면 세율이 얼마인가요?",
        # ⚠️ "세율"이라는 단어가 없는 세율 질문 — 실측에서 게이트가 가로채,
        # 나이를 알면서도 세율을 말하지 않고 수령방식·재원을 되물었다.
        "55세에 연금을 개시해서 10년째 받고 있는데 연금소득세는 몇 %인가요?",  # no.77
        "같은 금액이라도 55세에 받는 것과 70세에 받는 것 중 세금이 적은 쪽은 뭔가요?",  # no.336
        "65세인데 종신연금으로 받을지 고민 중이에요. 세율 차이가 얼마나 나나요?",  # no.282
    ]
    for q in questions:
        assert not _is_personal_tax_question(_compact(q)), q


def test_actual_calculation_question_stays_personal_tax_gate():
    """금액·재원 등 세율표만으로 안 끝나는 조건이 있으면 계산 게이트를 유지한다."""
    questions = [
        "나 74세인데 세금 어떻게 내?",
        "제가 만 68세인데 연금세금이 얼마나 나오나요?",
        "제가 60세이고 세액공제받은 재원에서 연 2000만원을 인출하는데 세율이 어떻게 되나요?",
        "74세 세율이 궁금한데 제 경우 실제 얼마 내나요?",
    ]
    for q in questions:
        assert _is_personal_tax_question(_compact(q)), q


# ── _extract_receipt_type: 구어 표현·부정 표현 ─────────────────────────


def test_receipt_type_detects_colloquial_lump_sum():
    """"일시금"이라는 용어 없이 "한 번에/한꺼번에"로 말해도 연금외수령으로 본다."""
    from src.agents.tax_context import _compact, _extract_receipt_type

    for question in (
        "퇴직금을 한 번에 받으려면 어떻게 해야 하나요?",
        "퇴직금 한꺼번에 인출하고 싶어요.",
        "IRP 목돈으로 받고 싶은데요.",
    ):
        assert _extract_receipt_type(_compact(question)) == "non_pension", question


def test_receipt_type_negation_beats_pension_keyword():
    """"연금으로 안 받고"처럼 연금수령을 부정하면 연금외수령으로 판정해야 한다.

    회귀 방지: "연금으로"라는 부분문자열이 있어 부정을 놓치면 정반대(pension)로
    판정된다 — 이연퇴직소득세 감면 여부가 뒤집히는 치명적 오판이다.
    """
    from src.agents.tax_context import _compact, _extract_receipt_type

    for question in (
        "퇴직금을 연금으로 안 받고 한꺼번에 인출하고 싶어요.",
        "연금이 아니라 일시금으로 받으려고요.",
        "연금 말고 목돈으로 받을 수 있나요?",
    ):
        assert _extract_receipt_type(_compact(question)) == "non_pension", question


def test_receipt_type_still_detects_pension_receipt():
    """정상적인 연금수령 표현은 그대로 pension으로 판정한다(과잉 확장 방지)."""
    from src.agents.tax_context import _compact, _extract_receipt_type

    for question in (
        "퇴직금을 연금으로 받으려면 어떻게 하나요?",
        "종신연금으로 수령하면 세율이 얼마인가요?",
    ):
        assert _extract_receipt_type(_compact(question)) == "pension", question


def test_receipt_type_ignores_contribution_context_lump_sum():
    """"한꺼번에 넣는다"(납입)를 연금외수령으로 오판하면 안 된다.

    회귀 방지: 실측 no.325 "연말에 한꺼번에 900만원 넣어도 세액공제 되나요?"는
    납입 방식을 묻는 질문인데, 구어 표현("한꺼번에")만 보고 연금외수령으로 판정했다.
    당시엔 재원이 없어 branch=None으로 남아 답변에 영향이 없었지만, 재원이 함께
    언급되면 정반대 세제 판정(이연퇴직소득세 감면 여부)이 나간다.
    """
    from src.agents.tax_context import _compact, _extract_receipt_type

    for question in (
        "연말에 한꺼번에 900만원 넣어도 세액공제 되나요?",
        "매달 나눠 넣지 않고 한번에 납입해도 되나요?",
    ):
        assert _extract_receipt_type(_compact(question)) is None, question
