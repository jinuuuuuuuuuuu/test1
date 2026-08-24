from src.agents.deterministic_info import deterministic_info_response, should_force_info_agent


def test_tax_credit_limit_question_gets_grounded_deterministic_answer():
    draft, context = deterministic_info_response("연금저축이랑 IRP 다 합쳐서 세액공제 얼마까지 되나요?")

    assert "합산 900만원" in draft
    assert "1,500만원까지 세액공제되는 구조가 아니라" in draft
    assert context
    assert "연금저축+IRP 합산 900만원" in context[0]["content"]


def test_tax_benefit_overview_question_gets_grounded_deterministic_answer():
    draft, context = deterministic_info_response("연금계좌의 세금혜택에 대해 알려주세요")

    assert "납입할 때 세액공제" in draft
    assert "운용 중 과세이연" in draft
    assert "연금으로 받을 때 낮은 세율" in draft
    assert "퇴직소득세 감면" in draft
    assert context
    assert "세액공제 대상 납입한도" in context[0]["content"]
    assert "1,500만원" in context[0]["content"]


def test_tax_credit_calculation_missing_inputs_asks_all_required_values():
    question = "제가 받을 수 있는 세액공제 금액을 계산해 주세요."
    draft, context = deterministic_info_response(question)

    assert should_force_info_agent(question) is True
    assert "임의로 산출하지 않겠습니다" in draft
    assert "연금저축에 납입한 금액" in draft
    assert "IRP에 납입한 금액" in draft
    assert "총급여" in draft
    assert "종합소득금액" in draft
    assert context


def test_early_withdrawal_general_question_gets_reasons():
    draft, context = deterministic_info_response("IRP에서 중도인출은 어떤 경우에 가능한가요?")

    assert "6개월 이상 요양" in draft
    assert "개인회생" in draft
    assert "무주택자 주택구입" in draft
    assert context


def test_early_withdrawal_specific_reason_answers_that_reason_not_generic_list():
    """사유가 특정된 질문은 5개 사유 나열이 아니라 그 사유의 요건을 답해야 한다.

    실측 실패 사례: 사유가 서로 다른 10개 질문이 전부 동일한 5개 사유 나열 답변을 받았다.
    """
    rehab, _ = deterministic_info_response("개인회생 중인데 IRP 중도인출이 가능한가요?")
    disaster, _ = deterministic_info_response("태풍 피해를 입었는데 DC형에서 중도인출 가능한가요?")

    assert "5년 이내" in rehab
    assert "회생절차의 효력이 진행 중" in rehab
    assert "3개월 이내" in disaster
    # 서로 다른 사유는 서로 다른 답이어야 한다 (정형 답변 반복 방지).
    assert rehab != disaster


def test_early_withdrawal_db_plan_is_rejected_outright():
    question = "DB형인데 무주택자 전세자금이 급해서 중도인출하고 싶어요. 가능한가요?"
    draft, context = deterministic_info_response(question)

    assert should_force_info_agent(question) is True
    assert "DB(확정급여형)는 중도인출 자체가 허용되지 않는" in draft
    assert context


def test_early_withdrawal_tax_question_states_limitation_instead_of_inventing_rate():
    """규칙엔진에 중도인출 세율 데이터가 없으므로 숫자를 지어내지 말고 한계를 고지해야 한다."""
    draft, _ = deterministic_info_response(
        "개인회생 중인데 IRP 중도인출하면 세금은 얼마나 나오나요? 서류는 뭐가 필요한가요?"
    )

    assert "세율은 보유 자료로 확인이 어렵습니다" in draft
    assert "서류의 구체적인 목록은 보유 자료로 확인이 어렵습니다" in draft


def test_tax_credit_question_with_full_inputs_is_left_to_calculation_tool():
    """납입액+소득이 이미 있으면 한도 일반론으로 가로채지 말고 계산 경로로 보내야 한다."""
    question = "연봉 5천만원인데 연금저축에 400만원, IRP에 300만원 넣으면 세액공제 얼마 받나요?"

    assert deterministic_info_response(question) is None
    assert should_force_info_agent(question) is False


def test_default_option_auto_purchase_question_gets_schedule_rules():
    draft, context = deterministic_info_response("디폴트옵션은 언제 자동으로 매수되나요?")

    assert "4주(28일)" in draft
    assert "2주(14일)" in draft
    assert context


def test_in_kind_transfer_block_question_forces_info_agent():
    question = "퇴직연금 실물이전이 안 되는 상품은 뭐가 있나요?"
    draft, context = deterministic_info_response(question)

    assert should_force_info_agent(question) is True
    assert "사모펀드" in draft
    assert "디폴트옵션" in draft
    assert context


def test_pension_income_tax_question_gets_threshold_rules():
    draft, context = deterministic_info_response("연금소득세 종합과세 기준 1500만원이 뭐야?")

    assert "1,500만원" in draft
    assert "16.5%" in draft
    assert context
