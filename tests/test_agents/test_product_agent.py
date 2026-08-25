from src.agents.product_agent import (
    _fallback_product_recommendation,
    _has_principal_guarantee_premise,
    _is_forecast_question,
    _recommendation_flow_response,
    _requires_account_eligibility_check,
    build_product_agent_node,
)


def test_account_eligibility_question_is_not_intercepted_by_generic_recommendation():
    """계좌유형+제한상품유형 조합은 정형 추천으로 가로채지 말고 LLM+툴로 넘겨야 한다.

    실측 실패: "IRP로 사모펀드 투자 가능한가요?"가 사모펀드 언급 없이 TDF/채권혼합형을
    추천했다 — 퇴직연금계좌는 사모펀드 등 투자 제한 상품유형이 있는데, 정형 응답은 이
    제도적 제약을 모른다.
    """
    assert _requires_account_eligibility_check("IRP로 사모펀드 투자 가능한가요? 추천해주세요.") is True
    result = _recommendation_flow_response({"question": "IRP로 사모펀드 투자 가능한가요? 추천해주세요."})
    assert result is None


def test_recommendation_flow_starts_with_one_question():
    draft, context, profile, needs_clarification = _recommendation_flow_response({"question": "상품 추천해줘"})

    for word in ("계좌유형", "위험성향", "투자기간", "투자금액", "투자목적"):
        assert word in draft
    assert "특정 펀드명이나 상품코드를 임의로 추천하지 않겠습니다" in draft
    assert context == []
    assert profile == {}
    assert needs_clarification is True


def test_recommendation_flow_keeps_context_and_asks_next_missing_field():
    state = {
        "question": "월 30만원 정도 가능해",
        "conversation_history": [
            {"question": "상품 추천해줘", "answer": "한 달에 어느 정도 금액을 투자할 예정인가요?"},
        ],
    }

    draft, context, profile, needs_clarification = _recommendation_flow_response(state)

    assert profile["monthly_investment"] == "월 30만원 정도"
    assert "투자 가능 금액 또는 월 납입금액" not in draft
    assert "계좌에서 투자할 예정" in draft
    assert "안정형, 중립형, 공격형" in draft
    assert "예상 투자기간" in draft
    assert context == []
    assert needs_clarification is True


def test_recommendation_flow_accepts_plain_amount_follow_up():
    state = {
        "question": "20만원",
        "conversation_history": [
            {"question": "상품 추천해줘", "answer": "한 달에 어느 정도 금액을 투자할 예정인가요?"},
        ],
    }

    draft, context, profile, needs_clarification = _recommendation_flow_response(state)

    assert profile["monthly_investment"] == "월 20만원"
    assert "투자 가능 금액 또는 월 납입금액" not in draft
    assert "계좌에서 투자할 예정" in draft
    assert "안정형, 중립형, 공격형" in draft
    assert context == []
    assert needs_clarification is True


def test_recommendation_flow_accepts_amount_with_suffix_follow_up():
    state = {
        "question": "30만원 이상",
        "recommendation_profile": {
            "account_type": "IRP",
            "risk_profile": "안정형",
            "investment_goal": "노후/은퇴 준비",
        },
        "conversation_history": [
            {
                "question": "적당한 IRP 상품 추천해주세요. 안정적이였음 좋겠어요",
                "answer": "한 달에 어느 정도 금액을 투자할 예정인가요?",
            },
        ],
    }

    draft, context, profile, needs_clarification = _recommendation_flow_response(state)

    assert profile["monthly_investment"] == "월 30만원 이상"
    # 계좌유형·위험성향·투자목적·투자금액이 다 있으니(투자기간만 없음) 완전 역질문 대신
    # 상품 유형 안내 + 남은 정보 요청으로 degrade한다.
    assert "투자기간" in draft
    assert context == []
    assert needs_clarification is False


def test_recommendation_flow_infers_retirement_goal_from_irp_request():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "적당한 IRP 상품 추천해주세요. 안정적이였음 좋겠어요"
    })

    assert profile["account_type"] == "IRP"
    assert profile["risk_profile"] == "안정형"
    assert profile["investment_goal"] == "노후/은퇴 준비"
    # 위험성향(안정형)을 알고 있으니 상품 유형 수준 답은 가능 — 남은 항목(투자금액·투자기간)만
    # 안내에 덧붙여 물어본다. 완전 역질문은 아니다.
    assert "투자금액" in draft
    assert "투자기간" in draft
    assert context == []
    assert needs_clarification is False


def test_recommendation_flow_with_missing_account_asks_for_all_missing_info():
    state = {
        "question": "노후 준비 목적이야",
        "recommendation_profile": {
            "monthly_investment": "월 30만원",
            "risk_profile": "중립형",
            "investment_horizon": "20년 이상",
        },
        "conversation_history": [
            {"question": "상품 추천해줘", "answer": "월 투자 가능 금액을 알려주세요."},
            {"question": "월 30만원", "answer": "투자성향을 알려주세요."},
            {"question": "중립형이고 20년 이상", "answer": "투자 목적을 알려주세요."},
        ],
    }

    draft, context, profile, needs_clarification = _recommendation_flow_response(state)

    # 위험성향을 이미 알고 있어 계좌유형만 빠져도 상품 유형 수준 답변으로 degrade한다.
    assert "계좌유형" in draft
    assert context == []
    assert needs_clarification is False


def test_product_node_marks_clarification_stage_for_incomplete_recommendation():
    node = build_product_agent_node()
    result = node(
        {
            "question": "요즘 TDF가 핫하다던데 상품 관련해서 추천해줘",
            "recommendation_profile": {
                "monthly_investment": "월 30만원",
                "risk_profile": "중립형",
                "investment_horizon": "20년 이상",
                "investment_goal": "노후/은퇴 준비",
            },
        }
    )

    # 계좌유형만 빠졌고 위험성향은 있으니 완전 역질문이 아니라 상품 유형 안내로 degrade —
    # response_mode는 "conditional"(부분 정보 기반 답변).
    assert result["recommendation_stage"] == "type_recommendation"
    assert result["response_mode"] == "conditional"
    assert "계좌유형" in result["product_draft"]
    assert result["retrieved_context"] == []


def test_recommendation_flow_sufficient_profile_uses_fund_db():
    state = {
        "question": "IRP 계좌에서 월 30만원씩 20년간 투자할 중립형 상품을 추천해 주세요.",
        "recommendation_profile": {
            "account_type": "IRP",
            "monthly_investment": "월 30만원",
            "risk_profile": "중립형",
            "investment_horizon": "20년 이상",
            "investment_goal": "노후/은퇴 준비",
            "recommended_product_types": ["TDF", "채권혼합형 펀드"],
        },
    }

    draft, context, profile, needs_clarification = _recommendation_flow_response(state)

    assert "상품코드" not in draft
    assert "위험등급" in draft
    assert "총보수" in draft
    assert context
    assert all("상품코드=" in item["content"] for item in context)
    assert needs_clarification is False


def test_recommendation_flow_does_not_capture_specific_product_question():
    assert _recommendation_flow_response({"question": "미래에셋솔로몬단기국공채 펀드는 어때?"}) is None


def test_specific_recommendation_with_missing_info_asks_only_needed_question():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "위험등급 낮고 수수료 적은 상품을 구체적으로 추천해줘"
    })

    for word in ("계좌유형", "투자기간", "투자금액", "투자목적"):
        assert word in draft
    assert context == []
    assert profile["risk_profile"] == "안정형"
    # 위험성향(안정형)만으로도 상품 유형 수준 답은 가능해 degrade된다 — 완전 역질문은 아니다.
    assert needs_clarification is False


def test_forecast_question_states_prediction_is_impossible():
    """미래 수익률 질문은 '예측 불가'를 먼저 짚어야 한다.

    실측 실패: "이 펀드가 내년에 몇 % 수익 날지"가 지시어('이 펀드') 참조로 분류돼
    "어떤 펀드인지 알려달라"고만 답했다 — 상품코드만 주면 내년 수익률을 알려줄 수
    있다는 인상을 주는 답이라 안전성·신뢰성에서 위험하다.
    """
    draft, context, _, needs_clarification = _recommendation_flow_response({
        "question": "이 펀드가 내년에 정확히 몇 % 수익 날지 투자설명서를 보고 알려주세요."
    })

    assert "예측할 수 없" in draft
    assert "과거 수익률" in draft
    assert "어떤 상품을 말하는지" not in draft  # 지시어 참조 분기로 새면 안 된다
    assert context == []
    assert needs_clarification is True


def test_principal_guarantee_premise_is_corrected_and_concentration_discouraged():
    draft, _, _, _ = _recommendation_flow_response({
        "question": "IRP는 원금이 무조건 보장되는 계좌잖아요. 그러니까 가장 수익률 높은 상품에 전액 투자해도 안전하죠?"
    })

    assert "계좌 자체가 원금을 보장하는 상품이 아닙니다" in draft
    assert "실적배당형" in draft
    assert "전액 투자" in draft  # 집중투자 요구도 함께 짚어야 한다


def test_safety_guards_do_not_block_legitimate_requests():
    """원리금보장'형 상품'을 찾는 정당한 요청까지 전제 교정으로 막으면 안 된다."""
    for question in (
        "원금보장형 상품 추천해줘",
        "원리금보장 상품이 IRP에서 가능한가요?",
        "최근 1년 수익률이 좋은 채권형 펀드 알려줘",
    ):
        assert _has_principal_guarantee_premise(question) is False
        assert _is_forecast_question(question) is False


def test_context_reference_without_history_asks_for_product_identity():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "그중 두 번째 상품은 어때?"
    })

    assert "이전 대화 내용" in draft
    assert "상품명 또는 상품코드" in draft
    assert context == []
    assert profile == {}
    assert needs_clarification is True


def test_fallback_product_recommendation_uses_fund_metrics():
    state = {
        "question": "나는 IRP 계좌 이용자이고, 투자기간은 5년 이상이야. 약간의 변동성을 감수 가능해.",
        "conversation_history": [
            {"question": "58세인데 크게 잃지 않을 상품 하나 추천해줘", "answer": "계좌유형과 투자기간을 알려주세요."},
        ],
    }

    draft, context = _fallback_product_recommendation(state)

    assert "위험등급" in draft
    assert "총보수" in draft
    assert "1년" in draft
    assert "최우선 후보" in draft
    assert context
    assert all("위험등급=" in item["content"] for item in context)


def test_fallback_product_recommendation_skips_without_account_type():
    state = {
        "question": "약간의 변동성을 감수 가능해.",
        "conversation_history": [
            {"question": "상품 추천해줘", "answer": "계좌유형을 알려주세요."},
        ],
    }

    draft, context = _fallback_product_recommendation(state)

    assert draft == ""
    assert context == []
