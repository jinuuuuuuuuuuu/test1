from src.agents.product_agent import (
    _fallback_product_recommendation,
    _recommendation_flow_response,
    build_product_agent_node,
)


def test_recommendation_flow_starts_with_one_question():
    draft, context, profile, needs_clarification = _recommendation_flow_response({"question": "상품 추천해줘"})

    assert "한 달에 어느 정도" in draft
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
    assert "안정형 / 중립형 / 공격형" in draft
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
    assert "한 달에 어느 정도" not in draft
    assert "안정형 / 중립형 / 공격형" in draft
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
    assert "한 달에 어느 정도" not in draft
    assert "예상 투자 기간" in draft
    assert context == []
    assert needs_clarification is True


def test_recommendation_flow_infers_retirement_goal_from_irp_request():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "적당한 IRP 상품 추천해주세요. 안정적이였음 좋겠어요"
    })

    assert profile["account_type"] == "IRP"
    assert profile["risk_profile"] == "안정형"
    assert profile["investment_goal"] == "노후/은퇴 준비"
    assert "한 달에 어느 정도" in draft
    assert context == []
    assert needs_clarification is True


def test_recommendation_flow_recommends_product_types_before_specific_funds():
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

    assert "TDF" in draft
    assert "채권혼합형" in draft
    assert "`상품추천`" in draft
    assert "상품코드" not in draft
    assert context == []
    assert profile["recommended_product_types"] == ["TDF", "채권혼합형 펀드"]
    assert needs_clarification is False


def test_product_node_marks_type_recommendation_stage():
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

    assert result["recommendation_stage"] == "type_recommendation"
    assert "`상품추천`" in result["product_draft"]
    assert result["retrieved_context"] == []


def test_recommendation_flow_specific_keyword_uses_fund_db():
    state = {
        "question": "상품추천",
        "recommendation_profile": {
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
    assert _recommendation_flow_response({"question": "이 펀드 위험이 뭐야?"}) is None


def test_specific_recommendation_with_missing_info_asks_only_needed_question():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "위험등급 낮고 수수료 적은 상품을 구체적으로 추천해줘"
    })

    assert "한 달에 어느 정도" in draft
    assert context == []
    assert profile["risk_profile"] == "안정형"
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
