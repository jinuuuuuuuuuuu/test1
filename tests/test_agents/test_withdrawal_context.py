from src.agents.withdrawal_context import extract_withdrawal_context


def test_disaster_deadline_context_locks_explicit_reason_and_topic():
    context = extract_withdrawal_context("재난피해로 중도인출하려는데 언제까지 신청해야 해?")

    assert context is not None
    assert context.reason == "DISASTER"
    assert context.reason_source == "explicit_user"
    assert context.explicit_topics == {"DEADLINE"}
    assert "reason" in context.locked_fields


def test_rehabilitation_and_workout_are_distinct_reasons():
    rehabilitation = extract_withdrawal_context("개인회생 때문에 중도인출 가능한가요?")
    workout = extract_withdrawal_context("개인워크아웃 중인데 중도인출 가능한가요?")

    assert rehabilitation.reason == "PERSONAL_REHABILITATION"
    assert workout.reason == "PERSONAL_WORKOUT"


def test_rehabilitation_workout_comparison_does_not_lock_one_reason():
    context = extract_withdrawal_context("개인회생과 개인워크아웃 중도인출 가능 여부 차이가 뭐야?")

    assert context is not None
    assert context.reason is None
    assert "reason" not in context.locked_fields


def test_retirement_pay_split_is_one_source_with_two_receipt_branches():
    context = extract_withdrawal_context(
        "퇴직금 일부는 중도인출하고 나머지는 연금으로 받으면 세금 어떻게 돼?"
    )

    assert context is not None
    assert context.source_type == "RETIREMENT_PAY"
    assert context.receipt_mode == "SPLIT"
    assert context.explicit_topics == {"TAX"}
    assert context.task_required_topics == {"ELIGIBILITY_PRECONDITION"}
    assert {"source_type", "receipt_mode"} <= context.locked_fields


def test_all_lump_sum_or_all_pension_is_not_misread_as_split():
    lump_sum = extract_withdrawal_context("퇴직금 전부 일시금으로 받으면 세금이 어떻게 돼?")
    pension = extract_withdrawal_context("퇴직금 전부 연금으로 받으면 세금이 어떻게 돼?")

    assert lump_sum is None or lump_sum.receipt_mode != "SPLIT"
    assert pension is None or pension.receipt_mode != "SPLIT"
