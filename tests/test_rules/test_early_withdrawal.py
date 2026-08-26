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
    assert "달력 기준 1개월" in late.reason


def test_medical_deadline_handles_leap_year_month_end():
    r = check_medical_treatment_eligibility(
        plan_type=PlanType.IRP,
        medical_expense_last_year=100_000,
        prior_year_annual_wage=50_000_000,
        treatment_end_date=date(2024, 1, 31),
        request_date=date(2024, 2, 29),
    )
    assert r.eligible is True


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
