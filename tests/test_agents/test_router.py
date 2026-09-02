"""①라우터의 결정론적 보정 로직 검증 — LLM 호출 없이 순수 함수만 테스트한다."""

import os

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

from src.agents.deterministic_info import candidate_categories
import src.agents.router as router_module
from src.agents.router import (
    _apply_asset_scope_override,
    _apply_condition_search_intent_override,
    _apply_in_kind_transfer_intent_override,
    _prioritize_collision_category,
)


# ── scope 오버라이드 (F-2) ────────────────────────────────────────────
#
# 라우터 LLM은 우리가 무엇을 보유했는지 모른 채 "답할 수 있는가"를 판정한다.
# 실측: DB에 실재하는 펀드를 두고 "개별 상품 정보는 제공 불가능"이라며 범위외 판정(3/3).
# 코드가 조회한 사실이 판정과 어긋나면 사실을 따른다 (④ L0 오버라이드와 같은 사상).


def test_override_flips_out_of_scope_when_fund_exists():
    """보유 상품이 조회됐는데 범위외로 판정했으면 뒤집는다."""
    scope, note, intent = _apply_asset_scope_override(
        "범위외", "구체적인 연금 관련 내용이 없어 범위 외임", ["상품형"],
        ["미래에셋솔로몬단기국공채증권자투자신탁1호(채권)"],
    )

    assert scope == "범위내"
    assert "보유 DB에 있어" in note
    assert intent == ["상품형"]


def test_override_fills_intent_when_router_gave_none():
    """범위외 판정과 함께 intent가 비어 나온 경우 상품형으로 보정한다."""
    _, _, intent = _apply_asset_scope_override("범위외", None, [], ["어떤펀드"])

    assert intent == ["상품형"]


def test_override_does_not_touch_in_scope_decisions():
    """이미 범위내면 개입하지 않는다 — scope_note도 그대로 둔다."""
    scope, note, intent = _apply_asset_scope_override(
        "범위내", None, ["상품형"], ["미래에셋솔로몬단기국공채증권자투자신탁1호(채권)"],
    )

    assert (scope, note, intent) == ("범위내", None, ["상품형"])


def test_override_adds_product_intent_when_fund_matched_but_intent_is_info_only():
    """DB에 있는 상품을 물었는데 intent가 '정보형'만이면 '상품형'을 보탠다.

    실측 사고(no.202/210/216/221/227): "한국투자 퇴직연금 증권 자투자신탁
    1호(국공채)의 위험등급이 몇 등급인가요?"처럼 DB에 이름이 정확히 있는 상품을
    물었는데 intent=['정보형']으로 분류돼 ③상품 Agent가 아예 실행되지 않았다.
    ②정보 Agent에는 search_funds가 없어 문서 검색만 하다 "제공된 자료에서 확인할
    수 없습니다"로 답을 포기했다 — 5건 전부 같은 경로로 실패했다.

    예전 구현은 scope="범위외"일 때만 개입해서 이 경우(scope는 이미 범위내)를
    그냥 통과시켰다.
    """
    scope, _, intent = _apply_asset_scope_override(
        "범위내", None, ["정보형"], ["한국투자 퇴직연금 증권 자투자신탁 1호(국공채)"],
    )

    assert scope == "범위내"
    assert "상품형" in intent
    # 정보형은 유지한다 — 제도 설명이 함께 필요한 복합 질문이면 ②→③ 순차 실행이 맞다
    assert "정보형" in intent


def test_override_does_not_duplicate_existing_product_intent():
    _, _, intent = _apply_asset_scope_override("범위내", None, ["상품형"], ["어떤펀드"])

    assert intent == ["상품형"]


def test_override_keeps_out_of_scope_when_no_fund_matched():
    """조회가 비었으면 범위외 판정을 유지한다 — 게이트를 무력화하면 안 된다.

    이 가드가 없으면 "삼성전자 주가", "부동산 양도세" 같은 진짜 범위 밖 질문까지
    통과해, scope 게이트를 1차 방어선으로 신뢰하는 다른 계층(RAG 거리 임계값 등)이
    함께 무너진다.
    """
    scope, note, intent = _apply_asset_scope_override(
        "범위외", "주식 시세는 연금 상담 범위 밖", ["정보형"], [],
    )

    assert scope == "범위외"
    assert note == "주식 시세는 연금 상담 범위 밖"


def test_override_preserves_partial_scope():
    """'부분관련'은 연금 관점으로 답하라는 별도 지시라 건드리지 않는다."""
    scope, note, _ = _apply_asset_scope_override(
        "부분관련", "개인사업자 연금계좌 세액공제 관점으로 답변", ["정보형"], ["어떤펀드"],
    )

    assert scope == "부분관련"
    assert note == "개인사업자 연금계좌 세액공제 관점으로 답변"


# ── 자연어 실물이전 의도 라우팅 ───────────────────────────────────────


def test_in_kind_transfer_intent_routes_asset_preserving_procedure_to_info_only():
    """'상품 그대로 이전할 수 있는 방법'은 상품 추천이 아니라 이전 제도 질문이다."""
    question = "IRP 상품 그대로 이전할 수 있는 방법 알려줘"

    assert candidate_categories(question) == []
    assert _apply_in_kind_transfer_intent_override(["상품형"], question) == ["정보형"]


def test_in_kind_transfer_intent_routes_direct_eligibility_to_info_for_core():
    """가능 여부 질문은 Guardian이 아닌 Core가 답하므로 역시 상품 추천 경로로 보내지 않는다."""
    question = "이 상품 그대로 이전할 수 있나요?"

    assert _apply_in_kind_transfer_intent_override(["상품형"], question) == ["정보형"]


def test_in_kind_transfer_intent_does_not_override_generic_transfer_or_non_transfer_questions():
    for question in (
        "IRP 이전신청 방법 알려줘",
        "매도 없이 펀드를 계속 보유하는 방법 알려줘",
        "매도 없이 다른 금융사 상품을 사는 방법 알려줘",
    ):
        assert _apply_in_kind_transfer_intent_override(["상품형"], question) == ["상품형"]


# ── deterministic category collision ───────────────────────────────────


def test_personal_tax_beats_age_rate_when_question_asks_actual_case():
    question = "74세 세율이 궁금한데 제 경우 실제 얼마 내나요?"
    candidates = candidate_categories(question)

    assert "개인세금_입력충분성" in candidates
    assert "연금소득세율_연령별" in candidates
    assert (
        _prioritize_collision_category("연금소득세율_연령별", candidates, question)
        == "개인세금_입력충분성"
    )


def test_composite_category_beats_single_task_categories():
    question = "전세보증금 때문에 IRP 중도인출하려고 하는데, 언제까지 신청해야 하고 필요한 서류랑 세금은 어떻게 되나요?"
    candidates = candidate_categories(question)

    assert "복합정보_태스크플랜" in candidates
    assert "개인세금_입력충분성" in candidates
    assert (
        _prioritize_collision_category("개인세금_입력충분성", candidates, question)
        == "복합정보_태스크플랜"
    )


def test_withdrawal_eligibility_beats_general_when_condition_is_given():
    question = "DB형인데 집 사려고 중도인출할래"
    candidates = candidate_categories(question)

    assert "중도인출_일반" in candidates
    assert "중도인출_요건판정" in candidates
    assert (
        _prioritize_collision_category("중도인출_일반", candidates, question)
        == "중도인출_요건판정"
    )


def test_general_age_rate_question_is_not_forced_to_personal_tax():
    question = "연령별 연금소득세율 표 알려줘"
    candidates = candidate_categories(question)

    assert "개인세금_입력충분성" not in candidates
    assert (
        _prioritize_collision_category("연금소득세율_연령별", candidates, question)
        == "연금소득세율_연령별"
    )


def test_router_node_prioritizes_personal_tax_collision(monkeypatch):
    """실제 라우터 노드 후처리에서도 개인 실제 세금 질문은 부족정보 Gate가 이긴다."""

    class FakeLLM:
        def with_structured_output(self, *args, **kwargs):
            return self

    def fake_invoke(_llm, _messages):
        return router_module.RouterDecision(
            intent=["정보형"],
            scope="범위내",
            is_safe=True,
            deterministic_category="연금소득세율_연령별",
        )

    monkeypatch.setattr(router_module, "get_llm", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(router_module, "invoke_with_retry", fake_invoke)
    monkeypatch.setattr(router_module, "find_asset_overlap", lambda _question: [])

    node = router_module.build_router_node()
    result = node({"question": "74세 세율이 궁금한데 제 경우 실제 얼마 내나요?"})

    assert result["deterministic_category"] == "개인세금_입력충분성"


def test_router_restore_rejected_withdrawal_eligibility():
    question = "개인워크아웃 중인데 퇴직연금 중도인출 가능한가요?"
    candidates = candidate_categories(question)

    assert "중도인출_요건판정" in candidates
    assert router_module._restore_rejected_category("해당없음", candidates, question) == "중도인출_요건판정"


def test_router_restores_rejected_composite_task_plan():
    question = "무주택자인데 집을 사려고 퇴직연금 중도인출하려고 해요. 신청기한, 필요한 서류, DB형에서도 가능한지 알려주세요."
    candidates = candidate_categories(question)

    assert candidates[0] == "복합정보_태스크플랜"
    assert router_module._restore_rejected_category("해당없음", candidates, question) == "복합정보_태스크플랜"


def test_condition_search_gains_product_intent():
    """조건 수치로 상품을 찾는 질문은 상품형이 강제돼야 한다.

    실측 no.207/227: "위험등급 1등급 펀드 중에 총보수가 1% 미만인 상품이 있나요"가
    intent=['정보형']으로만 잡혀 ③상품 Agent가 실행되지 않았다. find_asset_overlap은
    특정 상품명이 지목된 질문만 잡으므로("○○펀드"), 이름 없이 조건만으로 후보를
    찾아달라는 질문은 놓친다.
    """
    for question in (
        "위험등급 1등급 펀드 중에 총보수가 1% 미만인 상품이 있나요?",
        "위험등급 1등급 펀드랑 5등급 펀드랑 최근 1년 수익률 차이가 얼마나 나나요?",
    ):
        result = _apply_condition_search_intent_override(["정보형"], question)
        assert "상품형" in result, question


def test_condition_search_override_does_not_catch_concept_questions():
    """등급 자체의 의미를 묻는 개념 질문은 상품형으로 새면 안 된다(과잉 확장 방지).

    "맞나요/아닌가요"는 조회 요청이 아니라 개념 확인이라, search_funds로 상품을
    찾아봐야 답이 나오는 게 아니다 — search_pension_docs(제도 문서)로 등급 체계
    설명을 찾는 게 정답 경로다.
    """
    for question in (
        "위험등급 6등급 펀드가 1등급보다 안전한 게 맞나요?",
        "디폴트옵션 상품도 위험등급이 다양한가요?",
        "같은 펀드에서 판매클래스가 여러 개면 수익률도 다른가요?",
    ):
        result = _apply_condition_search_intent_override(["정보형"], question)
        assert "상품형" not in result, question
