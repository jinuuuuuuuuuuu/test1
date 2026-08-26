"""①라우터의 결정론적 보정 로직 검증 — LLM 호출 없이 순수 함수만 테스트한다."""

import os

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

from src.agents.router import _apply_asset_scope_override


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
