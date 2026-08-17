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


def test_early_withdrawal_general_question_gets_reasons():
    draft, context = deterministic_info_response("IRP에서 중도인출은 어떤 경우에 가능한가요?")

    assert "6개월 이상 요양" in draft
    assert "개인회생" in draft
    assert "무주택자 주택구입" in draft
    assert context


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
