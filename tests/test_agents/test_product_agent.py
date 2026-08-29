from src.agents.product_agent import (
    _fallback_product_recommendation,
    _recommendation_flow_response,
    build_product_agent_node,
)


def test_recommendation_flow_starts_with_one_question():
    draft, context, profile, needs_clarification = _recommendation_flow_response({"question": "상품 추천해줘"})

    for word in ("계좌유형", "위험성향", "투자기간"):
        assert word in draft
    for word in ("투자금액", "투자목적", "시장 상황이 바뀐다면", "TDF", "채권형"):
        assert word not in draft
    assert "상품별 근거를 바탕으로 비교하겠습니다" in draft
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
    assert "예상 투자기간" in draft
    assert context == []
    assert needs_clarification is True
    assert "투자 가능 금액 또는 월 납입금액" not in draft
    assert "채권형" not in draft
    assert "원리금보장형" not in draft
    assert "상품별 근거를 바탕으로 비교하겠습니다" in draft


def test_incomplete_recommendation_is_clarification_only_without_guidance():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "IRP에 넣을 안전한 상품 추천해주세요."
    })

    assert profile["account_type"] == "IRP"
    assert profile["risk_profile"] == "안정형"
    assert "예상 투자기간" in draft
    assert "채권형" not in draft
    assert "원리금보장형" not in draft
    assert "IRP/DC에는 위험자산 투자 제한" not in draft
    assert "시장 상황이 바뀐다면" not in draft
    assert "상품코드=" not in draft
    assert context == []
    assert needs_clarification is True


def test_overseas_equity_clarification_does_not_include_what_if_scenarios():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "IRP에서 미국 주식에 투자하는 상품 추천해줘"
    })

    assert profile["account_type"] == "IRP"
    assert profile["preferred_product_type"] == "해외주식형 펀드"
    assert "미국·해외 증시" not in draft
    assert "원/달러 환율" not in draft
    assert "시장 상황이 바뀐다면" not in draft
    assert "투자성향" in draft
    assert "투자기간" in draft
    assert "투자금액" not in draft
    assert [item["source"] for item in context] == ["doc56~doc58 적립금 운용 및 투자한도 규칙"]
    assert needs_clarification is True


def test_overseas_equity_follow_up_uses_fund_db_and_rule_based_scenario():
    state = {
        "question": "중립형이고 20년 이상이야",
        "conversation_history": [
            {
                "question": "IRP에서 미국 주식에 투자하는 상품 추천해줘",
                "answer": "투자성향과 투자기간을 알려주세요.",
            }
        ],
    }

    draft, context, profile, needs_clarification = _recommendation_flow_response(state)

    assert profile["account_type"] == "IRP"
    assert profile["preferred_product_type"] == "해외주식형 펀드"
    assert profile["risk_profile"] == "중립형"
    assert profile["investment_horizon"] == "20년 이상"
    assert "위험등급" in draft
    assert "총보수" in draft
    assert "상품 속성 기반 시나리오 점검" in draft
    assert "시장 상황이 바뀐다면" not in draft
    assert "미국·해외 증시" not in draft
    assert "원/달러 환율" not in draft
    assert context
    assert any("상품코드=" in item["content"] for item in context)
    assert any(item["source"] == "상품 시나리오 규칙 — 후보 속성 기반 점검" for item in context)
    assert needs_clarification is False


def test_sp500_recommendation_does_not_reuse_previous_safe_profile():
    state = {
        "question": "IRP에서 S&P500 ETF 같은 상품 추천해줘.",
        "recommendation_profile": {
            "account_type": "IRP",
            "risk_profile": "안정형",
            "investment_goal": "노후/은퇴 준비",
        },
        "conversation_history": [
            {"question": "IRP에 넣을 안전한 상품 추천해주세요.", "answer": "투자기간과 금액을 알려주세요."},
        ],
    }

    draft, context, profile, needs_clarification = _recommendation_flow_response(state)

    assert profile["account_type"] == "IRP"
    assert profile["preferred_product_type"] == "해외주식형 펀드"
    assert "risk_profile" not in profile
    assert "- 투자성향: 안정형" not in draft
    assert "미국·해외 증시" not in draft
    assert "원/달러 환율" not in draft
    assert "시장 상황이 바뀐다면" not in draft
    assert [item["source"] for item in context] == ["doc56~doc58 적립금 운용 및 투자한도 규칙"]
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
    assert "상품 속성 기반 시나리오 점검" in draft
    assert context
    assert any("상품코드=" in item["content"] for item in context)
    assert any(item["source"] == "상품 시나리오 규칙 — 후보 속성 기반 점검" for item in context)
    assert needs_clarification is False


def test_recommendation_flow_does_not_capture_specific_product_question():
    assert _recommendation_flow_response({"question": "미래에셋솔로몬단기국공채 펀드는 어때?"}) is None


def test_specific_recommendation_with_missing_info_asks_only_needed_question():
    draft, context, profile, needs_clarification = _recommendation_flow_response({
        "question": "위험등급 낮고 수수료 적은 상품을 구체적으로 추천해줘"
    })

    for word in ("계좌유형", "투자기간"):
        assert word in draft
    for word in ("투자금액", "투자목적", "시장 상황이 바뀐다면"):
        assert word not in draft
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


def test_fallback_does_not_trigger_on_institutional_question():
    """추천 의도가 없는 제도 질문에는 상품 후보 폴백이 발동하면 안 된다.

    실측 사고: "연금저축 펀드 환매하는데 제한기간이 있나요?"가 _is_product_recommendation
    ("펀드" 포함)과 _has_account_type("연금저축" 포함)을 둘 다 통과해, 추천 요청이 아닌데도
    임의의 펀드 3개를 근거로 끌어와 "연금저축 펀드는 환매 제한이 없다"는 일반화를 만들었다.
    더 나쁜 것은 이 폴백이 LLM이 툴을 호출하지 않았을 때 발동한다는 점이라, LLM이 "제도
    질문이라 상품 데이터로 답할 수 없다"는 지시를 올바르게 따를수록 폴백에 덮여버렸다.
    """
    from src.agents.product_agent import _fallback_product_recommendation

    draft, context = _fallback_product_recommendation(
        {"question": "연금저축 펀드 환매하는데 제한기간이 있나요?", "conversation_history": []}
    )

    assert draft == ""
    assert context == []


def test_fallback_still_triggers_on_real_recommendation_request():
    """실제 추천 요청에는 폴백이 그대로 동작해야 한다(과잉 차단 방지)."""
    from src.agents.product_agent import _is_recommendation_intent, _has_account_type

    text = "IRP에 넣을 펀드 추천해줘"
    assert _is_recommendation_intent(text) is True
    assert _has_account_type(text) is True


# ── 대화 이력 오염 차단 (2026-08-27) ───────────────────────────────────
#
# conversation_history 전체를 무차별로 합치면 직전 제도·세제 질문의 수치가 상품 추천
# 프로필로 새어 들어간다 (실측: "만 74세 연금 세율" 다음에 "IRP에서 S&P500 ETF 살 수
# 있나요"를 물었더니 추천 조건에 "투자기간 2026년, 나이 74세"가 섞였다).
# 나이·날짜만 빼는 방식은 그 다음에 금액이 새는 식으로 반복되므로 "같은 흐름의 턴인가"로 거른다.


def test_history_from_other_topics_is_excluded():
    from src.agents.product_agent import _combined_user_text

    combined = _combined_user_text({
        "question": "IRP에서 S&P500 ETF 살 수 있나요?",
        "conversation_history": [
            {"question": "만 74세인데 연금 세율이 몇 %인가요?"},
            {"question": "2026년에 인출하면 어떻게 되나요?"},
        ],
    })

    assert "74세" not in combined
    assert "2026년" not in combined
    assert "S&P500" in combined


def test_product_flow_history_is_kept():
    """상품 추천 흐름의 이전 턴은 이어받아야 슬롯 채우기가 동작한다."""
    from src.agents.product_agent import _combined_user_text

    combined = _combined_user_text({
        "question": "안정형이야",
        "conversation_history": [{"question": "IRP에 넣을 펀드 추천해줘"}],
    })

    assert "IRP에 넣을 펀드 추천해줘" in combined
    assert "안정형이야" in combined


def test_product_flow_profile_is_kept_for_short_follow_up():
    state = {
        "question": "30만원 이상",
        "recommendation_profile": {
            "account_type": "IRP",
            "risk_profile": "안정형",
            "investment_goal": "노후/은퇴 준비",
        },
        "conversation_history": [{"question": "IRP에 넣을 안전한 상품 추천해주세요."}],
    }

    _, _, profile, _ = _recommendation_flow_response(state)

    assert profile["account_type"] == "IRP"
    assert profile["risk_profile"] == "안정형"
    assert profile["monthly_investment"] == "월 30만원 이상"


def test_long_question_with_condition_words_is_not_flow():
    """조건 어휘가 섞였다고 긴 질문까지 흐름으로 오인하면 안 된다."""
    from src.agents.product_agent import _is_product_flow_turn

    assert not _is_product_flow_turn("2026년에 인출하면 어떻게 되나요?")
    assert _is_product_flow_turn("월 30만원")


# ── 추천 의도 어휘 확장 + LLM 실패 시 무응답 방지 (500문항 실측) ──────────
#
# 501문항 평가에서 파이프라인 실패 5건 중 4건이 product_agent에서 났다. 원인은
# CLOVA의 간헐적 400 오류(재시도 예산 소진) 자체가 아니라, 그 뒤의 폴백이 "골라
# 주세요"/"투자하고 싶어요" 같은 자연스러운 추천 표현을 인식하지 못해 조건 불충분
# 으로 재실패하고, 그 경우 raise로 예외를 다시 던져 그래프 전체가 죽었다는 점이다.
# "지연보다 무응답이 압도적으로 비싸다"(llm.py) 원칙에 따라 raise를 제거했다.


def test_recommendation_intent_recognizes_natural_phrasings():
    """"골라주세요"/"투자하고 싶어요"처럼 "추천"이라는 단어를 안 쓰는 자연스러운
    표현도 추천 의도로 인식해야 한다.

    실측: "DC형 추가납입금으로 위험등급 3등급 이내, 총보수 낮은 상품 골라주세요"와
    "IRP 계좌에 목돈 5천만원을 위험등급 3등급 정도로 투자하고 싶어요"가 둘 다
    추천 의도로 인식되지 않아 폴백이 빈 결과를 반환했다.
    """
    from src.agents.product_agent import _is_recommendation_intent

    assert _is_recommendation_intent("위험등급 3등급 이내, 총보수 낮은 상품 골라주세요.")
    assert _is_recommendation_intent("목돈 5천만원을 위험등급 3등급 정도로 투자하고 싶어요.")


def test_recommendation_intent_does_not_overtrigger_on_institutional_questions():
    """어휘를 넓혔다고 제도 질문까지 추천 의도로 오판하면 안 된다(과잉 확장 방지)."""
    from src.agents.product_agent import _is_recommendation_intent

    assert not _is_recommendation_intent("DC형 계좌에 위험자산을 얼마나 투자할 수 있나요?")
    assert not _is_recommendation_intent("퇴직연금 계좌에서 국내주식에 투자할 수 있나요?")


def test_product_agent_never_raises_when_llm_and_fallback_both_fail(monkeypatch):
    """LLM 호출도 실패하고 폴백도 조건 불충분으로 실패하면, 예외 대신 계좌유형을
    되묻는 역질문으로 응답해야 한다 (무응답으로 그 문항이 0점 처리되는 것을 막는다).
    """
    import src.agents.product_agent as product_agent_module

    def always_fail(*_args, **_kwargs):
        raise RuntimeError("simulated CLOVA 400 Unsupported function")

    monkeypatch.setattr(product_agent_module, "invoke_with_retry", always_fail)

    node = product_agent_module.build_product_agent_node()
    # 계좌유형이 없어 _fallback_product_recommendation도 빈 결과를 반환하는 질문
    state = {
        "question": "채권형이면서 총보수 낮고 최근 3년 수익률도 좋은 상품 하나만 콕 집어서 추천해주세요.",
        "conversation_history": [],
        "recommendation_profile": {},
        "intent": ["상품형"],
    }

    result = node(state)  # 예외를 던지면 이 호출 자체가 실패한다

    assert result["needs_clarification"] is True
    assert result["product_fallback_used"] is True
    assert result["clarification_questions"]
    assert "계좌" in result["product_draft"]
