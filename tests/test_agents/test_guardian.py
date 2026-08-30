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
    assert "전월세보증금 중도인출" in result["message"]
    assert "무주택 주택구입 중도인출" not in result["message"]
    assert evidence
    assert evidence[0]["node"] == "guardian"


def test_guardian_turns_on_for_home_purchase_documents_only():
    result, _ = evaluate_guardian(_verified_state("무주택 주택구입 중도인출 구비서류 알려줘"))

    assert result["enabled"] is True
    assert result["candidate_id"] == "home_purchase_documents"
    assert "무주택 주택구입 중도인출" in result["message"]
    assert "전월세보증금 중도인출" not in result["message"]


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
