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
