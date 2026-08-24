from src.agents.product_agent import (
    _extract_recommendation_profile,
    _fallback_product_recommendation,
    _financial_limit_note,
    _recommendation_flow_response,
    build_product_agent_node,
)


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
    assert "투자 가능 금액 또는 월 납입금액" not in draft
    assert "예상 투자기간" in draft
    assert context == []
    assert needs_clarification is True


def test_recommendation_flow_infers_retirement_goal_from_irp_request():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "적당한 IRP 상품 추천해주세요. 안정적이였음 좋겠어요"
    })

    assert profile["account_type"] == "IRP"
    assert profile["risk_profile"] == "안정형"
    assert profile["investment_goal"] == "노후/은퇴 준비"
    assert "투자 가능 금액 또는 월 납입금액" in draft
    assert "예상 투자기간" in draft
    assert context == []
    assert needs_clarification is True


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

    assert "계좌유형" in draft
    assert "IRP, DC, DB, 연금저축" in draft
    assert context == []
    assert needs_clarification is True


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

    assert result["recommendation_stage"] == "clarification"
    assert result["response_mode"] == "clarification_included"
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


def test_recommendation_uses_core_profile_without_requiring_amount():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": (
            "저는 60세이고 IRP 계좌에서 3년 정도 투자하려고 합니다. "
            "손실을 최대한 줄이고 싶은데 상품 추천해주세요."
        )
    })

    assert profile["account_type"] == "IRP"
    assert profile["investment_horizon"] == "3년 정도"
    assert profile["risk_profile"] == "안정형"
    assert "monthly_investment" not in profile
    assert context
    assert "위험등급" in draft
    assert needs_clarification is False


def test_recommendation_parses_age_long_horizon_and_profit_seeking():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "30대이고 은퇴까지 오래 남아서 수익성을 추구하는 연금 상품을 추천해주세요."
    })

    assert profile["age_or_retirement_horizon"] == "30대"
    assert profile["investment_horizon"] == "장기"
    assert profile["risk_profile"] == "공격형"
    assert "인덱스 펀드" in draft
    assert "계좌유형" in draft
    assert context == []
    assert needs_clarification is True


def test_recommendation_parses_lump_sum_and_high_equity_preference():
    profile = _extract_recommendation_profile({
        "question": (
            "30대이고 IRP 계좌에서 5천만원을 굴리려고 합니다. "
            "주식형 비중을 최대한 높이고 싶은데 상품 추천해주세요."
        )
    })

    assert profile["account_type"] == "IRP"
    assert profile["monthly_investment"] == "5천만원"
    assert profile["risk_profile"] == "공격형"
    assert profile["preferred_product_type"] == "주식형"
    assert profile["age_or_retirement_horizon"] == "30대"


def test_contradictory_return_request_gets_premise_correction_note():
    note = _financial_limit_note("확실하게 원금 보장되면서 수익률도 높은 펀드 추천해주세요.")

    assert "동시에 확정하는 펀드는 없습니다" in note
    assert "원리금보장형" in note
    assert "원금 손실 가능성" in note


def test_account_specific_eligibility_request_reaches_tool_agent():
    assert _recommendation_flow_response({
        "question": "DC 계좌에서 투자 가능한 국내 상장주식형 상품 추천해주세요."
    }) is None


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
    assert needs_clarification is True


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
