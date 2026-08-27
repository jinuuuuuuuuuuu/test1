"""①라우터의 결정론적 보정 로직 검증 — LLM 호출 없이 순수 함수만 테스트한다."""

import os

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

from src.agents.deterministic_info import candidate_categories
import src.agents.router as router_module
from src.agents.router import _apply_asset_scope_override, _prioritize_collision_category


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
