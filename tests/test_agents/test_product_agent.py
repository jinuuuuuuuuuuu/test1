import os

import pytest

from src.agents.product_agent import (
    _fallback_product_recommendation,
    _recommendation_flow_response,
    _risk_asset_limit_note,
    _specific_product_recommendation,
    build_product_agent_node,
)
from src.storage.queries import DEFAULT_DB_PATH

_HAS_PROSPECTUS_DB = os.path.exists(DEFAULT_DB_PATH)


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


def test_overseas_equity_request_does_not_present_domestic_fund_as_match():
    """해외주식형을 요청했는데 실제 후보가 국내 상품이면, 그 사실을 확정 어조로 숨기면 안 된다.

    회귀 방지: search_funds의 keyword 매칭은 "주식"처럼 넓은 문자열만 보고 국내/해외를
    구분하지 못한다. 실측: DB 100개 펀드 중 해외 투자대상이 이름에 드러나는 펀드는
    2개뿐이라(1개는 채권형), "미국 주식 투자하는 상품 추천해줘"에 keyword="주식"으로
    검색하면 삼성퇴직연금KOSPI200(국내 지수 추종) 같은 국내 상품이 섞여 나온다.
    "이 조건에서는 이 상품을 우선 후보로 보겠습니다"처럼 확정 어조로 제시하면, 사용자는
    국내 자산을 해외 자산으로 오인해 원치 않는 국가·통화 노출을 갖게 된다.
    """
    from src.agents.product_agent import _specific_product_recommendation

    profile = {
        "account_type": "IRP",
        "risk_profile": "공격형",
        "preferred_product_type": "해외주식형 펀드",
        "investment_horizon": "장기",
    }
    draft, _ = _specific_product_recommendation(profile, {"question": "미국 주식 투자하는 상품 추천해줘"})

    assert "해외 투자대상이 확인되는 후보를 찾지 못했습니다" in draft
    # 확정 어조로 특정 상품을 "이 조건에서는 이 상품을" 이라고 못 박지 않는다.
    assert "이 조건에서는" not in draft
    assert "가입 금융기관에 별도로 문의" in draft


def test_domestic_equity_request_is_unaffected_by_overseas_check():
    """국내주식형 요청은 해외 불일치 검사의 영향을 받지 않아야 한다(과잉 차단 방지)."""
    from src.agents.product_agent import _specific_product_recommendation

    profile = {
        "account_type": "IRP",
        "risk_profile": "공격형",
        "preferred_product_type": "국내주식형 펀드",
        "investment_horizon": "장기",
    }
    draft, _ = _specific_product_recommendation(profile, {"question": "국내 주식 투자하는 상품 추천해줘"})

    assert "우선 후보로 보겠습니다" in draft
    assert "해외 투자대상이 확인되는 후보를 찾지 못했습니다" not in draft


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
    assert "우선 후보로 보겠습니다" in draft
    # 지시문(LLM에게 주는 명령형 문장)이 최종 답변에 그대로 남으면 안 된다.
    # 실측 사고: react_agent 예외 경로에서 이 draft가 재호출 없이 그대로 사용자에게
    # 노출되는데, "~하세요/~설명하세요" 같은 지시문 스타일이 그대로 나갈 수 있었다.
    for instruction in ("반영하세요", "제시하세요", "설명하세요", "쓰세요", "덧붙이세요"):
        assert instruction not in draft
    assert context
    fund_context = [item for item in context if item["source"] != "doc58 퇴직연금 적립금 운용 및 투자한도 안내"]
    assert fund_context
    assert all("위험등급=" in item["content"] for item in fund_context)


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


# ── _risk_asset_limit_note (위험자산 70%/TDF 100% 한도 검증) ────────────


def test_risk_asset_limit_note_warns_when_all_candidates_are_risky():
    """추천 후보 전부가 위험자산이면 계좌유형별 한도 초과 가능성을 답변에 명시해야 한다.

    회귀 방지: check_product_pension_eligibility 툴은 존재했지만 결정론 추천 경로에서는
    호출되지 않았다(실측: 상품형 62건 중 실제 호출 2건, 3%). _search_args_from_profile도
    account_type을 전혀 쓰지 않아 DB/DC/IRP가 완전히 동일한 검색 인자를 냈다 — 위험자산
    70%(또는 TDF 100%) 한도를 검증하는 코드가 결정론 경로 어디에도 없었다.
    """
    candidates = [
        {"fund_name": "삼성퇴직연금KOSPI200증권자투자신탁 제1호[주식]", "fund_category": "증권(주식형)"},
        {"fund_name": "KB 그로스 포커스 증권 자투자신탁(주식)", "fund_category": "증권(주식형)"},
    ]
    note, context = _risk_asset_limit_note(candidates, {"account_type": "IRP"})

    assert "위험자산으로 분류됩니다" in note
    assert "70%" in note
    assert context


def test_risk_asset_limit_note_silent_when_safe_asset_included():
    candidates = [
        {"fund_name": "한국투자 퇴직연금 증권 자투자신탁 1호(국공채)", "fund_category": "증권(채권형)"},
        {"fund_name": "삼성퇴직연금KOSPI200증권자투자신탁 제1호[주식]", "fund_category": "증권(주식형)"},
    ]
    note, context = _risk_asset_limit_note(candidates, {"account_type": "IRP"})

    assert note == ""
    assert context == []


def test_risk_asset_limit_note_silent_without_account_type():
    """계좌유형을 모르면 한도 자체를 특정할 수 없으므로 과잉 경고하지 않는다."""
    candidates = [{"fund_name": "주식형 펀드", "fund_category": "증권(주식형)"}]
    note, _ = _risk_asset_limit_note(candidates, {})

    assert note == ""


def test_risk_asset_limit_note_relaxed_for_tdf_preference():
    """TDF를 선호 상품유형으로 명시하면 DC/IRP는 100% 한도가 적용되어 경고하지 않는다."""
    candidates = [{"fund_name": "TDF2050 증권투자신탁", "fund_category": "증권(주식혼합-재간접형)"}]
    note, _ = _risk_asset_limit_note(
        candidates, {"account_type": "IRP", "preferred_product_type": "TDF"}
    )

    assert note == ""


def test_risk_asset_limit_note_still_applies_to_db_even_with_tdf_preference():
    """TDF 100% 특례는 DC/IRP 전용이다 — DB는 TDF를 선호해도 여전히 70% 한도가 적용된다."""
    candidates = [{"fund_name": "TDF2050 증권투자신탁", "fund_category": "증권(주식혼합-재간접형)"}]
    note, _ = _risk_asset_limit_note(
        candidates, {"account_type": "DB", "preferred_product_type": "TDF"}
    )

    assert "위험자산으로 분류됩니다" in note
    assert "70%" in note


@pytest.mark.skipif(not _HAS_PROSPECTUS_DB, reason="data/processed/prospectus.db가 아직 없습니다")
def test_account_type_reaches_search_and_limit_note_end_to_end():
    """account_type이 프로필에 있으면 실제 추천 답변에 한도 경고가 붙는지 종단 확인한다."""
    profile = {
        "account_type": "IRP",
        "risk_profile": "공격형",
        "preferred_product_type": "주식형",
        "investment_horizon": "장기",
    }
    draft, _ = _specific_product_recommendation(profile, {"question": "주식형 펀드 추천해줘"})

    assert "위험자산으로 분류됩니다" in draft


# ── _apply_clarification_policy (근거 없는 시장 전망 차단) ──────────────


def test_clarification_policy_blocks_market_outlook_paraphrases():
    """정확한 문구가 아니라 같은 뜻의 다른 표현으로 쓴 시장 전망도 차단해야 한다.

    회귀 방지: _CLARIFICATION_BLOCKED_MARKERS는 죽은 코드였던 _what_if_scenario_block이
    만들던 정확한 문구("시장 상황이 바뀐다면?" 등)에만 맞춰져 있어, 같은 뜻의 다른
    표현은 전혀 못 걸렀다. 직접 재현: "요즘 증시가 좋아서 주식형이 유리합니다" 등
    5개 paraphrase가 전부 필터를 통과했다. clarification 상태(계좌유형·위험성향도
    아직 모름)에서 이런 조건부 시장 전망은 근거 없는 주장이다.
    """
    from src.agents.product_agent import _apply_clarification_policy

    paraphrases = [
        "요즘 증시가 좋아서 주식형 상품이 유리할 수 있습니다.",
        "최근 금리 인하 기조라 채권형이 매력적입니다.",
        "달러 강세가 이어지면 환노출 상품에 유리합니다.",
        "경기 침체 우려가 있어 안전자산 비중을 늘리는 게 좋습니다.",
        "주가지수가 상승세라 지금이 매수 적기일 수 있습니다.",
        "금리가 오르면 채권형이 불리해질 수 있습니다.",
    ]
    for outlook in paraphrases:
        answer = f"조건을 확인 후 답변드리겠습니다.\n\n{outlook}\n\n[추가 확인 필요]"
        out = _apply_clarification_policy(answer)
        assert outlook not in out, outlook


def test_clarification_policy_does_not_over_block_normal_text():
    """정상적인 조건 안내·제도 설명 문장은 차단되면 안 된다(과잉 차단 방지)."""
    from src.agents.product_agent import _apply_clarification_policy

    safe_lines = [
        "계좌유형은 IRP, DC, DB 중에서 알려주세요.",
        "위험성향은 안정형, 중립형, 공격형 중 어디에 가까우신가요?",
        "IRP/DC에는 위험자산 투자 제한이 있으므로 실제 상품 선정 전 계좌 내 투자 가능 범위를 확인해야 합니다.",
        "투자기간이 길고 공격적인 성향이라면 주식 등 성장자산 비중이 상대적으로 높은 상품이나 TDF를 검토할 수 있습니다.",
        "원리금보장형 상품은 안정적인 수익을 제공합니다.",
    ]
    answer = "\n".join(safe_lines)
    out = _apply_clarification_policy(answer)
    for line in safe_lines:
        assert line in out, line


# ── 모호한 정도 표현 해석 회귀 ────────────────────────────────────────────────
# "안정적/약간 공격적" 같은 표현은 코드가 위험등급으로 확정하는데, 이 판정의 오판은
# 곧 "사용자가 원하지 않은 상품군만 보여주는" 결과가 된다.


def test_risk_profile_does_not_match_fee_words():
    """'보수'(수수료)를 보수적 성향으로 오판하지 않는다 (실측 V23)."""
    from src.agents.product_agent import _extract_risk_profile

    for text in ("합리적인 수준의 총보수를 가진 펀드", "총보수가 낮은 펀드", "운용보수 얼마인가요"):
        assert _extract_risk_profile(text) is None, text
    # 진짜 보수적 성향은 계속 잡아야 한다
    assert _extract_risk_profile("보수적으로 운용하고 싶어요") == "안정형"


def test_hedged_aggressive_is_softened_to_neutral():
    """'약간 공격적'을 공격형(1~3등급)으로 읽으면 완충 표현을 무시하게 된다 (V11)."""
    from src.agents.product_agent import _extract_risk_profile

    assert _extract_risk_profile("약간 공격적으로 투자하고 싶은데") == "중립형"
    assert _extract_risk_profile("공격적으로 굴리고 싶어요") == "공격형"


def test_hedged_risk_mention_is_read_as_neutral():
    """'위험은 조금만 감수하고'를 놓치면 이미 말한 성향을 다시 되묻게 된다 (V10)."""
    from src.agents.product_agent import _extract_risk_profile

    assert _extract_risk_profile("위험은 조금만 감수하고 수익 좀 보고 싶어요") == "중립형"


def test_abstract_quality_words_stay_unresolved():
    """'괜찮은/좋은'은 기준이 없으므로 확정하지 않고 역질문으로 보낸다."""
    from src.agents.product_agent import _extract_risk_profile

    for text in ("괜찮은 연금저축 펀드 하나 골라주세요", "좋은 퇴직연금 상품이 뭔가요"):
        assert _extract_risk_profile(text) is None, text


def test_risk_disclosure_matches_actual_search_filter():
    """사용자에게 고지하는 해석과 실제 검색 필터가 어긋나면 안 된다."""
    from src.agents.product_agent import (
        _RISK_GRADE_DISCLOSURE,
        _format_profile_summary,
        _search_args_from_profile,
    )

    expected = {
        "안정형": {"risk_grade_min": 5},
        "중립형": {"risk_grade_min": 4},
        "공격형": {"risk_grade_max": 3},
    }
    for profile_value, filters in expected.items():
        args = _search_args_from_profile({"risk_profile": profile_value})
        for key, value in filters.items():
            assert args[key] == value, profile_value
        # 고지 문구에 실제 등급 숫자가 들어 있어야 한다
        assert profile_value in _RISK_GRADE_DISCLOSURE
        summary = _format_profile_summary({"risk_profile": profile_value})
        assert _RISK_GRADE_DISCLOSURE[profile_value] in summary


def test_profile_summary_discloses_interpretation():
    from src.agents.product_agent import _format_profile_summary

    summary = _format_profile_summary({"account_type": "IRP", "risk_profile": "안정형"})

    assert "위험등급 5~6등급" in summary
    assert "IRP" in summary
