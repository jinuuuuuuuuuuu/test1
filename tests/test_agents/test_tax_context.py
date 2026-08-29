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
