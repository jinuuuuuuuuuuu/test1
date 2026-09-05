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


# ── 정도부사 25문항 회귀: is_safe 과잉 차단 (V02/V14/V18) ────────────────────
# 최상급·과장 부사를 "확정 수익률 요구"로 오인해 정상 질문 3건이 차단됐다.
# 차단은 사용자가 답을 아예 못 받는 출력이라, 프롬프트만으로 두지 않고 코드로 막는다.


def test_override_releases_superlative_only_questions():
    """강조 부사만 있고 보장 요구가 없으면 차단을 해제한다."""
    from src.agents.router import _override_overblocked_safety

    reason = "확정 수익률/원금 보장을 요구하거나 암시하는 질문이므로 안전하지 않음"
    for question in (
        "IRP 수익률이 엄청 좋다던데 얼마나 벌 수 있어요?",
        "제일 안전한 연금 상품이 뭔가요?",
        "세금 완전히 안 내는 방법 있나요?",
    ):
        assert _override_overblocked_safety(False, reason, question) == (True, None)


def test_override_releases_tax_question_with_absolute_adverb():
    """세금·제도 문맥의 절대 수식어는 투자 보장 요구가 아니다 (V04/V18)."""
    from src.agents.router import _override_overblocked_safety

    reason = "확정 수익률/원금 보장을 요구하거나 암시하는 질문이므로 안전하지 않음"
    for question in (
        "퇴직소득세 감면을 미친듯이 많이 받으려면 어떻게 해야 하나요?",
        "세금 완전히 안 내는 방법 있나요?",
    ):
        assert _override_overblocked_safety(False, reason, question) == (True, None)


def test_override_keeps_real_guarantee_demand_blocked():
    """차단이 정당한 두 유형은 그대로 유지한다.

    (A) 절대 수식어 + 무손실 요구  — 수익 요구가 없어도 성립 불가라 차단 (V19/V20)
    (B) 무손실 + 고수익 동시 요구  — 양립 불가한 모순이라 차단 (V17)
    """
    from src.agents.router import _override_overblocked_safety

    reason = "확정 수익률/원금 보장을 요구하거나 암시하는 질문이므로 안전하지 않음"
    for question in (
        "무조건 원금 보장되면서 수익도 나는 상품 있나요?",
        "절대 손실 안 나는 안전한 상품으로 추천해주세요.",
        "IRP 100% 안전하게 굴리고 싶어요. 어떻게 해야 하나요?",
        "원금 보장되면서 연 10% 수익 나는 상품 추천해줘",
        "리스크 없이 예금보다 훨씬 많이 버는 상품 알려줘",
    ):
        is_safe, _ = _override_overblocked_safety(False, reason, question)
        assert is_safe is False, question


def test_override_distinguishes_safety_query_from_guarantee_demand():
    """'안전한 상품이 뭐냐'(사실 질의)와 '절대 손실 없게 해달라'(보장 요구)의 경계."""
    from src.agents.router import _override_overblocked_safety

    reason = "확정 수익률/원금 보장을 요구하거나 암시하는 질문이므로 안전하지 않음"
    # 절대 수식어가 없으면 사실 질의 — 통과
    assert _override_overblocked_safety(False, reason, "제일 안전한 연금 상품이 뭔가요?")[0] is True
    # 절대 수식어가 붙으면 보장 요구 — 차단
    assert _override_overblocked_safety(False, reason, "절대 안전한 연금 상품 주세요")[0] is False


def test_override_does_not_touch_other_block_reasons():
    """탈세·개인정보 차단은 재검토 대상이 아니다 — 풀면 실제 위험이 통과한다."""
    from src.agents.router import _override_overblocked_safety

    for reason, question in (
        ("탈세 방법을 묻는 질문", "소득 숨겨서 세금 안 내는 방법"),
        ("개인정보 조회 요청", "제 주민번호는 900101-1234567인데 조회해주세요"),
    ):
        is_safe, _ = _override_overblocked_safety(False, reason, question)
        assert is_safe is False


def test_override_leaves_safe_questions_untouched():
    from src.agents.router import _override_overblocked_safety

    assert _override_overblocked_safety(True, None, "연금저축 세액공제 한도는?") == (True, None)


# ── 세제 폴백: 후보 0건이어도 정형 경로를 되살린다 ──────────────────────────────
#
# candidate_categories는 키워드로 "검토할 근거"를 판정하는데, 세제 영역에서 이 방식이
# 반복적으로 무너졌다 — 사용자는 제도 용어를 모른 채 일상어로 묻기 때문이다
# ("퇴직소득세" 대신 "퇴직금", "1,500만원 기준" 대신 본인 금액 "1,600만원").
# 표현은 무한하고 키워드 목록은 유한해 커버리지 확장으로는 구조적으로 진다.
# 그래서 세제에 한해 순서를 뒤집었다: 키워드로 차단하지 않고, 라우터가 "해당없음"을
# 냈을 때 세제 핸들러에게 직접 물어본다(핸들러는 자기 소관이 아니면 None을 낸다).


def test_tax_fallback_restores_category_when_candidates_empty():
    """세제 질문은 후보가 비어도 정형 경로로 되살아난다."""
    from src.agents.deterministic_info import candidate_categories
    from src.agents.router import _restore_rejected_category

    for question, expected in (
        ("연금소득이 1600만원이야", "연금소득세_종합과세"),
        ("일흔 넘었는데 연금소득세율이 어떻게 되나요?", "연금소득세율_연령별"),
    ):
        restored = _restore_rejected_category(
            "해당없음", candidate_categories(question), question
        )
        assert restored == expected, question


def test_tax_fallback_does_not_hijack_non_tax_questions():
    """세제와 무관한 질문은 "해당없음"이 유지돼야 한다(오탈취 방지).

    이 폴백의 안전은 "핸들러가 자기 소관이 아니면 스스로 None을 낸다"에 달려 있다.
    잘 작동하는 다른 영역(중도인출·실물이전·디폴트옵션·상품추천)을 세제 카테고리가
    가로채면 안 된다.
    """
    from src.agents.deterministic_info import candidate_categories
    from src.agents.router import _restore_rejected_category

    for question in (
        "안정적인 연금 상품 추천해줘",
        "IRP 중도인출 신청은 어디서 하나요?",
        "디폴트옵션 상품이 뭔가요?",
        "솔로몬 국공채 단기랑 장기 뭐가 달라요?",
    ):
        restored = _restore_rejected_category(
            "해당없음", candidate_categories(question), question
        )
        assert restored == "해당없음", question

    assert _restore_rejected_category(
        "해당없음",
        candidate_categories("DB형과 DC형 운용주체가 어떻게 다른가요?"),
        "DB형과 DC형 운용주체가 어떻게 다른가요?",
    ) == "퇴직연금_유형비교"


def test_calculation_shortage_stays_out_of_overridable_set():
    """세액공제_계산_입력부족은 CODE_OVERRIDABLE에 넣으면 안 된다(후보 순서 회귀).

    이 집합은 후보가 **있을 때의** 첫 루프에도 쓰이는데, 후보 순서상 이 카테고리가
    세액공제_한도보다 앞이라 넣는 순간 "얼마나 넣어야 하나요?"류 질문이 한도 안내
    대신 "입력값이 부족하다"는 계산 보류 답변으로 바뀐다(실측 회귀).
    """
    from src.agents.deterministic_info import (
        CODE_OVERRIDABLE_CATEGORIES,
        TAX_FALLBACK_CATEGORIES,
    )

    assert "세액공제_계산_입력부족" not in CODE_OVERRIDABLE_CATEGORIES
    # 후보 0건 폴백에서는 여전히 쓸 수 있어야 한다
    assert "세액공제_계산_입력부족" in TAX_FALLBACK_CATEGORIES
