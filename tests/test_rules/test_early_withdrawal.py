import pytest
from datetime import date
from src.rules.early_withdrawal import (
    PlanType,
    check_plan_type_eligible,
    check_medical_treatment_eligibility,
    check_rehabilitation_bankruptcy_eligibility,
    check_rental_deposit_eligibility,
    check_home_purchase_eligibility,
    check_disaster_eligibility,
)


def test_db_always_blocked():
    r = check_plan_type_eligible(PlanType.DB)
    assert r.eligible is False


def test_dc_and_irp_allowed():
    assert check_plan_type_eligible(PlanType.DC).eligible is True
    assert check_plan_type_eligible(PlanType.IRP).eligible is True


# ── 요양 (doc46) ─────────────────────────────────────────────────────

def test_medical_dc_under_ratio_blocked():
    r = check_medical_treatment_eligibility(
        plan_type=PlanType.DC,
        medical_expense_last_year=1_000_000,
        prior_year_annual_wage=50_000_000,  # 12.5% = 6,250,000, 의료비가 이보다 작음
        treatment_end_date=date(2026, 1, 1),
        request_date=date(2026, 1, 15),
    )
    assert r.eligible is False


def test_medical_dc_over_ratio_allowed():
    r = check_medical_treatment_eligibility(
        plan_type=PlanType.DC,
        medical_expense_last_year=7_000_000,
        prior_year_annual_wage=50_000_000,
        treatment_end_date=date(2026, 1, 1),
        request_date=date(2026, 1, 15),
    )
    assert r.eligible is True


def test_medical_irp_ratio_not_applied():
    # IRP는 12.5% 기준 자체가 적용되지 않음 -> 의료비가 적어도 가능
    r = check_medical_treatment_eligibility(
        plan_type=PlanType.IRP,
        medical_expense_last_year=100_000,
        prior_year_annual_wage=50_000_000,
        treatment_end_date=date(2026, 1, 1),
        request_date=date(2026, 1, 15),
    )
    assert r.eligible is True


def test_medical_deadline_exceeded():
    r = check_medical_treatment_eligibility(
        plan_type=PlanType.IRP,
        medical_expense_last_year=100_000,
        prior_year_annual_wage=50_000_000,
        treatment_end_date=date(2026, 1, 1),
        request_date=date(2026, 3, 1),  # 1개월 훌쩍 지남
    )
    assert r.eligible is False


# ── 개인회생·파산 (doc47) ────────────────────────────────────────────

def test_rehabilitation_within_5_years_and_effective():
    r = check_rehabilitation_bankruptcy_eligibility(
        plan_type=PlanType.IRP,
        event_type="개인회생",
        decision_date=date(2022, 1, 1),
        request_date=date(2026, 1, 1),
        rehabilitation_still_effective=True,
    )
    assert r.eligible is True


def test_rehabilitation_terminated_blocked():
    r = check_rehabilitation_bankruptcy_eligibility(
        plan_type=PlanType.IRP,
        event_type="개인회생",
        decision_date=date(2022, 1, 1),
        request_date=date(2026, 1, 1),
        rehabilitation_still_effective=False,
    )
    assert r.eligible is False


def test_bankruptcy_discharge_status_irrelevant():
    # 파산선고는 면책/복권 여부 무관
    r = check_rehabilitation_bankruptcy_eligibility(
        plan_type=PlanType.IRP,
        event_type="파산선고",
        decision_date=date(2022, 1, 1),
        request_date=date(2026, 1, 1),
    )
    assert r.eligible is True


def test_workout_not_eligible():
    r = check_rehabilitation_bankruptcy_eligibility(
        plan_type=PlanType.IRP,
        event_type="개인회생",
        decision_date=date(2022, 1, 1),
        request_date=date(2026, 1, 1),
        is_workout_or_credit_recovery=True,
    )
    assert r.eligible is False


def test_over_5_years_blocked():
    r = check_rehabilitation_bankruptcy_eligibility(
        plan_type=PlanType.IRP,
        event_type="파산선고",
        decision_date=date(2018, 1, 1),
        request_date=date(2026, 1, 1),
    )
    assert r.eligible is False


# ── 무주택 전월세보증금 (doc48) ──────────────────────────────────────

def test_rental_deposit_basic_eligible():
    r = check_rental_deposit_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        has_deposit=True,
        is_lease_extension=False,
        has_deposit_increase=False,
        dc_already_used=False,
        balance_payment_date=date(2026, 1, 1),
        request_date=date(2026, 1, 20),
    )
    assert r.eligible is True


def test_rental_deposit_monthly_rent_only_blocked():
    r = check_rental_deposit_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        has_deposit=False,
        is_lease_extension=False,
        has_deposit_increase=False,
        dc_already_used=False,
        balance_payment_date=date(2026, 1, 1),
        request_date=date(2026, 1, 20),
    )
    assert r.eligible is False


def test_rental_deposit_extension_without_increase_blocked():
    r = check_rental_deposit_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        has_deposit=True,
        is_lease_extension=True,
        has_deposit_increase=False,
        dc_already_used=False,
        balance_payment_date=date(2026, 1, 1),
        request_date=date(2026, 1, 20),
    )
    assert r.eligible is False


def test_rental_deposit_dc_second_use_blocked():
    r = check_rental_deposit_eligibility(
        plan_type=PlanType.DC,
        is_homeless=True,
        has_deposit=True,
        is_lease_extension=False,
        has_deposit_increase=False,
        dc_already_used=True,
        balance_payment_date=date(2026, 1, 1),
        request_date=date(2026, 1, 20),
    )
    assert r.eligible is False


def test_rental_deposit_irp_unlimited_uses():
    r = check_rental_deposit_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        has_deposit=True,
        is_lease_extension=False,
        has_deposit_increase=False,
        dc_already_used=True,  # IRP는 이 플래그와 무관
        balance_payment_date=date(2026, 1, 1),
        request_date=date(2026, 1, 20),
    )
    assert r.eligible is True


# ── 무주택 주택구입 (doc49) ──────────────────────────────────────────

def test_home_purchase_eligible():
    r = check_home_purchase_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        ownership_type="본인단독",
        ownership_registration_date=date(2026, 1, 1),
        request_date=date(2026, 1, 20),
    )
    assert r.eligible is True


def test_home_purchase_inheritance_blocked():
    r = check_home_purchase_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        ownership_type="상속",
        ownership_registration_date=date(2026, 1, 1),
        request_date=date(2026, 1, 20),
    )
    assert r.eligible is False


def test_home_purchase_deadline_exceeded():
    r = check_home_purchase_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        ownership_type="본인단독",
        ownership_registration_date=date(2026, 1, 1),
        request_date=date(2026, 3, 1),
    )
    assert r.eligible is False


# ── 재난피해 (doc50) ─────────────────────────────────────────────────

def test_disaster_within_3_months():
    r = check_disaster_eligibility(
        plan_type=PlanType.IRP,
        damage_date=date(2026, 1, 1),
        request_date=date(2026, 2, 1),
    )
    assert r.eligible is True


def test_disaster_over_3_months_but_unresolved_still_eligible():
    r = check_disaster_eligibility(
        plan_type=PlanType.IRP,
        damage_date=date(2026, 1, 1),
        request_date=date(2026, 6, 1),
        damage_resolved=False,
    )
    assert r.eligible is True


def test_disaster_over_3_months_resolved_blocked():
    r = check_disaster_eligibility(
        plan_type=PlanType.IRP,
        damage_date=date(2026, 1, 1),
        request_date=date(2026, 6, 1),
        damage_resolved=True,
    )
    assert r.eligible is False


# ── 신청기한 공통 계산 (task_type 공통화, 2026-08-27) ──────────────────
#
# 사유별로 마감일 계산을 따로 구현하던 시절, "요양·주택구입만 날짜 핸들러가 있고
# 전월세·재난은 없는" 비대칭이 생겨 전월세 날짜 질문이 사유 목록 답변으로 빠졌다.
# 기한 규정을 테이블로 선언하고 계산은 calculate_deadline 한 곳에서만 한다.


def test_all_five_reasons_have_deadline_rules():
    """5개 사유 전부 기한 규정이 선언돼 있어야 한다 — 하나라도 빠지면 그 사유는 날짜를 못 답한다."""
    from src.rules.early_withdrawal import WITHDRAWAL_DEADLINE_RULES

    assert set(WITHDRAWAL_DEADLINE_RULES) == {
        "요양", "개인회생파산", "무주택전월세", "무주택주택구입", "재난피해",
    }
    for reason, rule in WITHDRAWAL_DEADLINE_RULES.items():
        assert rule.basis_event, f"{reason}: 기준일 표현이 없다"
        assert rule.months or rule.years, f"{reason}: 기간이 없다"
        assert rule.source_doc, f"{reason}: 근거 문서가 없다"


@pytest.mark.parametrize("reason,basis,expected", [
    ("요양", date(2026, 1, 15), date(2026, 2, 15)),
    ("무주택전월세", date(2026, 1, 31), date(2026, 2, 28)),      # 말일 보정
    ("무주택주택구입", date(2026, 3, 20), date(2026, 4, 20)),
    ("재난피해", date(2026, 5, 10), date(2026, 8, 10)),          # 3개월
    ("개인회생파산", date(2026, 6, 1), date(2031, 6, 1)),        # 5년
])
def test_calculate_deadline_per_reason(reason, basis, expected):
    from src.rules.early_withdrawal import calculate_deadline

    assert calculate_deadline(reason, basis) == expected


def test_calendar_months_handles_month_end_and_leap_year():
    """원문이 "1개월"이라고만 하므로 30일 환산이 아니라 달력 기준으로 계산해야 한다.

    timedelta(days=30)을 쓰던 시절 1월 31일 기준 마감일이 3월 2일로 나와 이틀 관대했다.
    """
    from src.rules.early_withdrawal import add_calendar_months

    assert add_calendar_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_calendar_months(date(2024, 1, 31), 1) == date(2024, 2, 29)  # 윤년
    assert add_calendar_months(date(2026, 3, 31), 1) == date(2026, 4, 30)
    assert add_calendar_months(date(2026, 5, 15), 1) == date(2026, 6, 15)


def test_unknown_reason_raises():
    from src.rules.early_withdrawal import calculate_deadline

    with pytest.raises(ValueError):
        calculate_deadline("없는사유", date(2026, 1, 1))

# ── 달력 기준 마감일 회귀 (dana 브랜치에서 병합, 2026-08-27) ─────────
#
# 원문이 "1개월"이라고만 하므로 30일 환산은 실제보다 관대하게 판정한다.
# calculate_deadline 공통화 이후에도 각 사유의 판정 함수가 같은 결과를 내는지 지킨다.


def test_medical_deadline_uses_calendar_month_not_30_days():
    r = check_medical_treatment_eligibility(
        plan_type=PlanType.IRP,
        medical_expense_last_year=100_000,
        prior_year_annual_wage=50_000_000,
        treatment_end_date=date(2026, 1, 31),
        request_date=date(2026, 2, 28),
    )
    assert r.eligible is True

    late = check_medical_treatment_eligibility(
        plan_type=PlanType.IRP,
        medical_expense_last_year=100_000,
        prior_year_annual_wage=50_000_000,
        treatment_end_date=date(2026, 1, 31),
        request_date=date(2026, 3, 1),
    )
    assert late.eligible is False
    # 메시지 표현은 "1개월(달력 기준 2026-02-28까지)" 형태다 — 달력 기준임과
    # 실제 마감일이 함께 드러나야 사용자가 언제까지였는지 알 수 있다.
    assert "달력 기준" in late.reason
    assert "2026-02-28" in late.reason


def test_medical_deadline_handles_leap_year_month_end():
    r = check_medical_treatment_eligibility(
        plan_type=PlanType.IRP,
        medical_expense_last_year=100_000,
        prior_year_annual_wage=50_000_000,
        treatment_end_date=date(2024, 1, 31),
        request_date=date(2024, 2, 29),
    )
    assert r.eligible is True


def test_rental_deposit_deadline_uses_calendar_month():
    r = check_rental_deposit_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        has_deposit=True,
        is_lease_extension=False,
        has_deposit_increase=False,
        dc_already_used=False,
        balance_payment_date=date(2026, 5, 10),
        request_date=date(2026, 6, 10),
    )
    assert r.eligible is True

    late = check_rental_deposit_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        has_deposit=True,
        is_lease_extension=False,
        has_deposit_increase=False,
        dc_already_used=False,
        balance_payment_date=date(2026, 5, 10),
        request_date=date(2026, 6, 11),
    )
    assert late.eligible is False


def test_home_purchase_deadline_uses_registration_receipt_calendar_month():
    r = check_home_purchase_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        ownership_type="본인단독",
        ownership_registration_date=date(2026, 8, 31),
        request_date=date(2026, 9, 30),
    )
    assert r.eligible is True

    late = check_home_purchase_eligibility(
        plan_type=PlanType.IRP,
        is_homeless=True,
        ownership_type="본인단독",
        ownership_registration_date=date(2026, 8, 31),
        request_date=date(2026, 10, 1),
    )
    assert late.eligible is False
    assert "등기접수일" in late.reason


def test_disaster_deadline_uses_calendar_months_not_90_days():
    r = check_disaster_eligibility(
        plan_type=PlanType.IRP,
        damage_date=date(2026, 8, 31),
        request_date=date(2026, 11, 30),
        damage_resolved=True,
    )
    assert r.eligible is True

    late = check_disaster_eligibility(
        plan_type=PlanType.IRP,
        damage_date=date(2026, 8, 31),
        request_date=date(2026, 12, 1),
        damage_resolved=True,
    )
    assert late.eligible is False
