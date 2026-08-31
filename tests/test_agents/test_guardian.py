from src.agents.guardian import GUARD_HEADING, evaluate_guardian


def _verified_state(question: str, *, draft: str = "필요서류는 다음과 같습니다.") -> dict:
    return {
        "question": question,
        "scope": "범위내",
        "response_mode": "complete",
        "needs_clarification": False,
        "info_draft": draft,
        "verification": {"grounded": True, "requirements_met": True},
    }


def test_guardian_turns_on_for_housing_deposit_documents_only():
    result, evidence = evaluate_guardian(_verified_state("전세보증금 중도인출 필요서류 알려줘"))

    assert result["enabled"] is True
    assert result["candidate_id"] == "housing_deposit_documents"
    assert result["message"].startswith(GUARD_HEADING)
    assert "세금도 함께 확인하세요" in result["message"]
    assert "전월세보증금 중도인출" in result["message"]
    assert "무주택 주택구입 중도인출" not in result["message"]
    assert evidence
    assert evidence[0]["node"] == "guardian"


def test_guardian_turns_on_for_home_purchase_documents_only():
    result, _ = evaluate_guardian(_verified_state("무주택 주택구입 중도인출 구비서류 알려줘"))

    assert result["enabled"] is True
    assert result["candidate_id"] == "home_purchase_documents"
    assert "세금도 함께 확인하세요" in result["message"]
    assert "무주택 주택구입 중도인출" in result["message"]
    assert "전월세보증금 중도인출" not in result["message"]


def test_guardian_turns_on_for_natural_documents_paraphrases():
    for question, expected_id in (
        ("전세계약 때문에 IRP에서 중도인출하려는데 뭐 챙겨야 해?", "housing_deposit_documents"),
        ("집 사려고 퇴직연금 중도인출할 때 준비할 것 알려줘", "home_purchase_documents"),
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is True, question
        assert result["candidate_id"] == expected_id, question
        assert evidence, question


def test_guardian_stays_off_when_question_needs_clarification():
    state = _verified_state("IRP 중도인출 필요서류 알려줘")
    state["needs_clarification"] = True
    state["response_mode"] = "clarification_included"

    result, evidence = evaluate_guardian(state)

    assert result["enabled"] is False
    assert result["disabled_reason"] == "NEEDS_CLARIFICATION"
    assert evidence == []


def test_guardian_stays_off_for_generic_or_unsupported_documents_questions():
    for question in (
        "IRP 중도인출 필요서류 알려줘",
        "재난피해 중도인출 필요서류 알려줘",
        "개인회생 때문에 IRP 중도인출 서류 뭐 필요해?",
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is False
        assert result["disabled_reason"] == "NO_CANDIDATE"
        assert evidence == []


def test_guardian_stays_off_when_tax_is_explicitly_asked():
    result, evidence = evaluate_guardian(_verified_state("전세보증금 중도인출 필요서류랑 세금 알려줘"))

    assert result["enabled"] is False
    assert result["disabled_reason"] == "GUARD_FACT_ALREADY_ASKED"
    assert evidence == []


def test_guardian_stays_off_when_core_already_covers_topic():
    state = _verified_state(
        "전세보증금 중도인출 필요서류 알려줘",
        draft="전월세보증금 중도인출은 세법상 부득이한 사유가 아니며 재원별 과세가 달라집니다.",
    )

    result, evidence = evaluate_guardian(state)

    assert result["enabled"] is False
    assert result["disabled_reason"] == "CORE_ALREADY_COVERS_TOPIC"
    assert evidence == []


def test_guardian_core_gate_disabled_reasons():
    cases = [
        ({"scope": "범위외"}, "OUT_OF_SCOPE"),
        ({"response_mode": "conditional"}, "NON_COMPLETE_RESPONSE"),
        ({"verification": {"grounded": False, "requirements_met": True}}, "CORE_NOT_GROUNDED"),
        ({"verification": {"grounded": True, "requirements_met": False}}, "REQUIREMENTS_NOT_MET"),
    ]

    for update, expected in cases:
        state = _verified_state("전세보증금 중도인출 필요서류 알려줘")
        state.update(update)
        result, evidence = evaluate_guardian(state)

        assert result["enabled"] is False
        assert result["disabled_reason"] == expected
        assert evidence == []


# ── A2: 퇴직금 연금외수령 Action Guard ──────────────────────────────────


def test_guardian_turns_on_for_retirement_lump_sum_action():
    """세금을 묻지 않고 퇴직금을 일시금으로 받으려는 행동에는 감면 상실을 짚어준다.

    Core Answer는 이 사실을 말하지 못한다 — tax_context의 계산 게이트가 "세금/세율"류
    어휘를 요구해서, 절차만 묻는 질문은 personal_tax_response가 None을 반환한다
    (실측 확인). 그래서 사용자는 이연퇴직소득세 감면(연차별 30~50%)을 통째로 잃는다는
    사실을 모른 채 진행하게 된다.
    """
    for question in (
        "퇴직금을 IRP에서 한 번에 받으려면 어떻게 해야 하나요?",
        "퇴직금을 일시금으로 찾으려고 하는데 절차가 어떻게 되나요?",
        "퇴직금을 연금으로 안 받고 한꺼번에 인출하고 싶어요.",
        "IRP에 있는 퇴직금 일시금으로 받는 방법 알려주세요.",
        "퇴직금 연금외수령하려면 뭐부터 해야 하죠?",
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is True, question
        assert result["candidate_id"] == "retirement_non_pension_tax_loss", question
        assert "이연퇴직소득세 감면" in result["message"], question
        assert evidence, question
        assert evidence[0]["node"] == "guardian", question


def test_guardian_stays_off_when_user_already_asked_about_tax():
    """사용자가 세금·감면을 직접 물었으면 Core가 답할 몫이라 파수꾼은 침묵한다.

    ⚠️ 이 판정에 tax_context._is_personal_tax_question()을 쓰면 안 된다 — 그 함수는
    "계산 게이트로 보낼까"를 판정해서, 세금을 물었어도 순수 세율 비교면 False를 낸다
    (실측 no.115). 그걸 쓰면 이미 세금을 물은 질문에 파수꾼이 또 세금 이야기를 덧붙인다.
    """
    for question in (
        "퇴직금 일시금으로 받으면 세금이 어떻게 되나요?",
        "퇴직금을 일시금으로 받는 것과 연금으로 받는 것 중 세금이 더 적은 쪽은?",
        "퇴직금 일시금으로 받으면 이연퇴직소득세 감면을 받을 수 있나요?",
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is False, question
        assert result["disabled_reason"] == "EXPLICIT_USER_TOPIC", question
        assert evidence == [], question


def test_guardian_stays_off_when_retirement_source_is_unclear():
    """재원이 퇴직금인지 불명확하면 감면 상실을 단정할 수 없으므로 침묵한다."""
    for question in (
        "IRP에서 돈 빼고 싶어요.",
        "연금저축 일시금으로 찾으려면 어떻게 하나요?",
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is False, question
        assert evidence == [], question


def test_guardian_stays_off_for_pension_receipt_action():
    """연금으로 받으려는 행동은 감면 대상이라 경고할 내용이 없다."""
    result, evidence = evaluate_guardian(
        _verified_state("퇴직금을 연금으로 받으려면 어떻게 하나요?")
    )

    assert result["enabled"] is False
    assert result["disabled_reason"] == "NO_CANDIDATE"
    assert evidence == []


def test_guardian_stays_off_when_core_already_explains_reduction():
    """Core가 이미 감면 상실을 설명했으면 중복이므로 침묵한다.

    tax_context의 retirement_benefit_non_pension 브랜치가 "이연퇴직소득세 감면이
    적용되지 않고 전액 납부 대상"이라고 사실상 같은 내용을 답하는 경로가 있다.
    """
    state = _verified_state(
        "퇴직금을 일시금으로 찾으려면 절차가 어떻게 되나요?",
        draft="퇴직금 재원을 연금외수령하면 이연퇴직소득세 감면이 적용되지 않고 전액 납부 대상입니다.",
    )
    result, evidence = evaluate_guardian(state)

    assert result["enabled"] is False
    assert result["disabled_reason"] == "CORE_ALREADY_COVERS_TOPIC"
    assert evidence == []


# ── A1: 실물이전 절차 Action Guard ───────────────────────────────────────


def test_guardian_turns_on_for_in_kind_transfer_procedure_question():
    """실물이전 절차·방법만 묻고 상품을 특정하지 않으면 이전 제한 가능성을 짚어준다.

    회귀 방지의 진짜 원인은 라우팅이 아니라 response_mode 누락이었다 — 실측에서
    "IRP 실물이전 절차 알려줘"는 라우터가 정확히 "해당없음"으로 분류하고 grounded=True/
    requirements_met=True까지 통과했는데도 response_mode가 None으로 남아
    _guardian_route_possible이 항상 실패했다(guardian 노드 자체를 못 탐). info_agent/
    product_agent의 LLM 자유 응답 경로가 이 키를 채운 적이 없었기 때문이다.
    """
    result, evidence = evaluate_guardian(
        _verified_state("IRP 실물이전 절차 알려줘", draft="실물이전 신청은 영업점 또는 앱에서 가능합니다.")
    )

    assert result["enabled"] is True
    assert result["candidate_id"] == "in_kind_transfer_procedure"
    assert "실물이전 가능 대상인지" in result["message"]
    assert evidence
    assert evidence[0]["node"] == "guardian"


def test_guardian_detects_asset_preserving_transfer_procedure_paraphrases():
    """전문용어 없이도 자산보존 의도와 이전 행동이 명확하면 A1을 켠다."""
    for question in (
        "IRP 상품 그대로 이전신청하려면 어떻게 해?",
        "보유 펀드를 매도 없이 다른 금융사로 옮기는 방법 알려줘",
        "IRP 상품 그대로 이전할 수 있는 방법 알려줘",
        "연금계좌 옮기면서 펀드는 매도하고 싶지 않아. 방법 알려줘",
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is True, question
        assert result["candidate_id"] == "in_kind_transfer_procedure", question
        assert evidence, question


def test_guardian_does_not_treat_generic_transfer_or_destination_as_in_kind_transfer():
    """일반 이전이나 목적지 표현만으로 실물이전 의도를 추정하지 않는다."""
    for question in (
        "IRP 이전신청 방법 알려줘",
        "연금계좌를 다른 금융사로 옮기는 방법 알려줘",
        "다른 증권사로 계약이전하고 싶어",
        "매도 없이 펀드를 계속 보유하는 방법 알려줘",
        "매도 없이 다른 금융사 상품을 사는 방법 알려줘",
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is False, question
        assert evidence == [], question


def test_guardian_stays_off_for_product_specified_or_list_questions():
    """상품을 특정했거나 불가사유 목록 자체를 물으면 Core(개별판정/목록)가 답할 몫이다."""
    for question in (
        "이 펀드 실물이전 가능해?",
        "실물이전 안 되는 상품 알려줘",
        "매도 없이 옮길 수 있어?",
        "이 상품 그대로 이전 가능한가?",
        "이 상품 그대로 이전할 수 있나요?",
        "실물이전 절차랑 안 되는 상품도 알려줘",
        "실물이전 절차와 제한사항 알려줘",
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is False, question
        assert evidence == [], question


def test_guardian_stays_off_when_core_already_lists_transfer_eligible_products():
    """Core가 이미 실물이전 대상/제외 상품을 설명했으면 중복이므로 침묵한다.

    실측: LLM이 실물이전 절차를 답할 때 근거 문서의 "가능·제외 상품" 섹션을 함께
    검색해 대상/제외 목록까지 자연스럽게 포함시키는 경우가 있었다. 이럴 때 파수꾼이
    또 "이전 가능 대상인지 확인해야 한다"고 덧붙이면 중복이다.
    """
    draft = (
        "실물이전 신청은 영업점 또는 앱에서 가능합니다. 예금, GIC 등은 실물이전 대상이지만 "
        "디폴트옵션 상품, 사모펀드, MMF 등은 이전이 제외됩니다."
    )
    result, evidence = evaluate_guardian(_verified_state("IRP 실물이전 절차 알려줘", draft=draft))

    assert result["enabled"] is False
    assert result["disabled_reason"] == "CORE_ALREADY_COVERS_TOPIC"
    assert evidence == []


# ── O1: 세액공제 미사용 한도 Opportunity Guard ──────────────────────────


def test_guardian_turns_on_for_unused_tax_credit_capacity():
    """납입정보로 확정 가능한 세액공제 미사용 한도를 짚어준다.

    핵심은 '추가 납입 추천'이 아니라 '미사용 혜택 탐지'다 — "○○만원 남아 있습니다"
    까지만 말하고 "더 넣으세요"는 말하지 않는다.
    """
    for question in (
        "올해 IRP에 600만원 넣었는데 납입내역은 어디서 봐?",
        "올해 연금저축이랑 IRP에 400만원 납입한 금액 확인하고 싶어",
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is True, question
        assert result["candidate_id"] == "unused_tax_credit_capacity", question
        assert "남아 있습니다" in result["message"], question
        # 추천 문구를 절대 쓰지 않는다.
        assert "더 넣으세요" not in result["message"], question
        assert "추가로 납입" not in result["message"], question
        assert evidence, question
        assert evidence[0]["node"] == "guardian", question


def test_guardian_computes_correct_remaining_amount():
    """잔여 한도는 합산 900만원 - 확인된 납입액으로 정확히 계산해야 한다."""
    result, _ = evaluate_guardian(
        _verified_state("올해 IRP에 600만원 넣었는데 납입내역은 어디서 봐?")
    )
    assert "300만원" in result["message"]


def test_guardian_respects_pension_savings_only_limit_when_computing_remaining_amount():
    """연금저축 납입액은 단독 600만원까지만 합산 세액공제 한도에 반영된다."""
    result, evidence = evaluate_guardian(
        _verified_state("올해 연금저축 700만원 넣었는데 납입내역은 어디서 봐?")
    )

    assert result["enabled"] is True
    assert "연금저축 단독 한도는 이미 채워졌고" in result["message"]
    assert "IRP를 포함한 합산 한도 기준으로는 300만원 남아 있습니다" in result["message"]
    assert "200만원" not in result["message"]
    assert "세액공제 대상 반영액은 600만원" in evidence[0]["content"]


def test_guardian_explains_pension_savings_and_combined_remaining_amounts():
    result, _ = evaluate_guardian(
        _verified_state("올해 연금저축 500만원 넣었는데 납입내역은 어디서 봐?")
    )

    assert result["enabled"] is True
    assert "연금저축 단독 한도는 100만원" in result["message"]
    assert "IRP를 포함한 합산 한도 기준으로는 400만원 남아 있습니다" in result["message"]


def test_guardian_stays_off_when_limit_already_used_up():
    """합산 세액공제 한도를 이미 채웠거나 초과했으면 미사용 혜택이 없으므로 침묵한다."""
    result, evidence = evaluate_guardian(
        _verified_state("연금저축 600만원, IRP 300만원 넣었는데 확인 좀 해줘")
    )

    assert result["enabled"] is False
    assert result["disabled_reason"] == "NO_CANDIDATE"
    assert evidence == []


def test_guardian_stays_off_when_user_already_asked_tax_credit():
    """세액공제 한도를 직접 물었으면 Core가 답할 몫이라 파수꾼은 침묵한다."""
    for question in (
        "연금저축 600만원 넣었는데 세액공제 한도 얼마 남았어?",
        "IRP 500만원 넣었는데 세액공제 받으려면 얼마 더 넣어야 해?",
    ):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is False, question
        assert result["disabled_reason"] == "EXPLICIT_USER_TOPIC", question
        assert evidence == [], question


def test_guardian_stays_off_without_contribution_amount():
    """납입액이 확인되지 않으면 잔여 한도를 계산할 근거가 없으므로 침묵한다."""
    for question in ("IRP가 뭐야?", "연금저축 세액공제 한도가 궁금해요"):
        result, evidence = evaluate_guardian(_verified_state(question))

        assert result["enabled"] is False, question
        assert evidence == [], question
