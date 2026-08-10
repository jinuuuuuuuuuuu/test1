"""중도인출 요건판정기 — 근거: doc46(요양), doc47(개인회생·파산), doc48(무주택 전월세),
doc49(무주택 주택구입), doc50(재난피해)

공통 규칙 (5개 문서 공통 서두):
- DC(확정기여형), IRP(개인형퇴직연금)만 중도인출 가능. DB(확정급여형)는 중도인출 자체가 불가.
  (근로자퇴직급여보장법 시행령 제14조·제18조 각호)
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class PlanType(Enum):
    DC = "DC"
    IRP = "IRP"
    DB = "DB"


@dataclass
class WithdrawalEligibilityResult:
    eligible: bool
    reason: str


def check_plan_type_eligible(plan_type: PlanType) -> WithdrawalEligibilityResult:
    """DB는 애초에 중도인출 제도 자체가 없다 — 모든 사유 판정 전에 선행 체크."""
    if plan_type == PlanType.DB:
        return WithdrawalEligibilityResult(False, "DB(확정급여형)는 중도인출이 허용되지 않습니다.")
    return WithdrawalEligibilityResult(True, "DC/IRP는 중도인출 대상 제도입니다.")


# ── doc46: 6개월 이상 요양 ──────────────────────────────────────────────
MEDICAL_EXPENSE_RATIO_THRESHOLD = 0.125  # 직전년도 연간임금총액 대비 의료비 비율 기준 (DC만 적용)


def check_medical_treatment_eligibility(
    plan_type: PlanType,
    medical_expense_last_year: int,
    prior_year_annual_wage: int,
    treatment_end_date: date,
    request_date: date,
) -> WithdrawalEligibilityResult:
    """6개월 이상 요양 사유 중도인출 요건을 판정한다 (doc46).

    medical_expense_last_year: 신청일 기준 직전 1년간 근로자 본인이 부담한 의료비 총액.
    prior_year_annual_wage: 직전년도 연간임금총액.
    IRP는 12.5% 비율 기준 자체가 적용되지 않는다 (요양 사유만 있으면 됨) — DC만 이 비율을 따진다.
    신청시기: 요양종료일로부터 1개월 이내.
    """
    plan_check = check_plan_type_eligible(plan_type)
    if not plan_check.eligible:
        return plan_check

    if plan_type == PlanType.DC:
        if medical_expense_last_year <= prior_year_annual_wage * MEDICAL_EXPENSE_RATIO_THRESHOLD:
            return WithdrawalEligibilityResult(
                False,
                "DC는 직전 1년 의료비 총액이 직전년도 연간임금총액의 12.5%를 초과해야 신청 가능합니다.",
            )

    deadline = treatment_end_date + timedelta(days=30)
    if request_date > deadline:
        return WithdrawalEligibilityResult(False, "요양종료일로부터 1개월(약 30일)이 지나 신청 기한이 지났습니다.")

    return WithdrawalEligibilityResult(True, "요양 사유 중도인출 요건을 충족합니다.")


# ── doc47: 개인회생·파산 ────────────────────────────────────────────────

def check_rehabilitation_bankruptcy_eligibility(
    plan_type: PlanType,
    event_type: str,  # "개인회생" | "파산선고"
    decision_date: date,
    request_date: date,
    rehabilitation_still_effective: bool = True,
    is_workout_or_credit_recovery: bool = False,
) -> WithdrawalEligibilityResult:
    """개인회생·파산 사유 중도인출 요건을 판정한다 (doc47).

    개인회생: 개시결정일로부터 5년 이내이면서, 신청 시점에 회생절차 효력이 '진행 중'이어야 함
      (폐지결정·면책결정으로 효력이 종료되면 신청 불가).
    파산선고: 선고일로부터 5년 이내 (면책·복권 여부는 무관).
    개인워크아웃, 신용회복지원은 이 사유의 대상이 아니다.
    """
    plan_check = check_plan_type_eligible(plan_type)
    if not plan_check.eligible:
        return plan_check

    if is_workout_or_credit_recovery:
        return WithdrawalEligibilityResult(False, "개인워크아웃·신용회복지원은 중도인출 대상 사유가 아닙니다.")

    if event_type not in ("개인회생", "파산선고"):
        raise ValueError("event_type은 '개인회생' 또는 '파산선고'여야 합니다")

    deadline = decision_date.replace(year=decision_date.year + 5)
    if request_date > deadline:
        return WithdrawalEligibilityResult(False, f"{event_type} 결정일로부터 5년이 지나 신청 기한이 지났습니다.")

    if event_type == "개인회생" and not rehabilitation_still_effective:
        return WithdrawalEligibilityResult(False, "개인회생절차가 폐지·면책 결정으로 이미 종료되어 신청할 수 없습니다.")

    return WithdrawalEligibilityResult(True, f"{event_type} 사유 중도인출 요건을 충족합니다.")


# ── doc48: 무주택자 전월세보증금 ────────────────────────────────────────

def check_rental_deposit_eligibility(
    plan_type: PlanType,
    is_homeless: bool,
    has_deposit: bool,  # 임차보증금 존재 여부 — 월세만 있고 보증금이 없으면 대상 아님
    is_lease_extension: bool,
    has_deposit_increase: bool,  # 연장계약일 때만 의미 있음: 보증금 인상분이 있어야 신청 가능
    dc_already_used: bool,       # DC는 하나의 사업장에서 재직 중 1회 한정
    balance_payment_date: date,
    request_date: date,
) -> WithdrawalEligibilityResult:
    """무주택자 전월세보증금 사유 중도인출 요건을 판정한다 (doc48).

    신청시기: 주택임대차계약 체결일로부터 잔금지급일 이후 1개월 이내.
    DC는 동일 사업장 재직 중 1회 한정, 개인형IRP는 횟수 제한 없음.
    """
    plan_check = check_plan_type_eligible(plan_type)
    if not plan_check.eligible:
        return plan_check

    if not is_homeless:
        return WithdrawalEligibilityResult(False, "신청일 기준 근로자 본인 명의 주택이 없어야 합니다 (배우자 등 세대원 주택 소유는 무관).")

    if not has_deposit:
        return WithdrawalEligibilityResult(False, "임차보증금 없이 월세금만 있는 계약은 중도인출 대상이 아닙니다.")

    if is_lease_extension and not has_deposit_increase:
        return WithdrawalEligibilityResult(False, "연장계약은 보증금 인상분이 있어야 신청 가능합니다 (월세금만 인상은 불가).")

    if plan_type == PlanType.DC and dc_already_used:
        return WithdrawalEligibilityResult(False, "DC는 하나의 사업장에서 재직 중 1회만 가능하며, 이미 사용한 이력이 있습니다.")

    deadline = balance_payment_date + timedelta(days=30)
    if request_date > deadline:
        return WithdrawalEligibilityResult(False, "잔금지급일로부터 1개월(약 30일)이 지나 신청 기한이 지났습니다.")

    return WithdrawalEligibilityResult(True, "무주택자 전월세보증금 사유 중도인출 요건을 충족합니다.")


# ── doc49: 무주택자 주택구입 ────────────────────────────────────────────

def check_home_purchase_eligibility(
    plan_type: PlanType,
    is_homeless: bool,
    ownership_type: str,  # "본인단독" | "부부공동" | "증여" | "상속"
    ownership_registration_date: date,
    request_date: date,
) -> WithdrawalEligibilityResult:
    """무주택자 주택구입 사유 중도인출 요건을 판정한다 (doc49).

    신청시기: 주택매매계약 체결일로부터 소유권 이전 등기 후 1개월 이내.
    증여, 상속으로 취득하는 경우는 대상이 아니다.
    """
    plan_check = check_plan_type_eligible(plan_type)
    if not plan_check.eligible:
        return plan_check

    if not is_homeless:
        return WithdrawalEligibilityResult(False, "신청일 기준 근로자 본인 명의 주택이 없어야 합니다 (배우자 등 세대원 주택 소유는 무관).")

    if ownership_type in ("증여", "상속"):
        return WithdrawalEligibilityResult(False, "증여·상속으로 취득하는 주택구입은 중도인출 대상이 아닙니다.")

    deadline = ownership_registration_date + timedelta(days=30)
    if request_date > deadline:
        return WithdrawalEligibilityResult(False, "소유권 이전 등기일로부터 1개월(약 30일)이 지나 신청 기한이 지났습니다.")

    return WithdrawalEligibilityResult(True, "무주택자 주택구입 사유 중도인출 요건을 충족합니다.")


# ── doc50: 재난피해 ─────────────────────────────────────────────────────

def check_disaster_eligibility(
    plan_type: PlanType,
    damage_date: date,
    request_date: date,
    damage_resolved: bool = True,
) -> WithdrawalEligibilityResult:
    """재난피해 사유 중도인출 요건을 판정한다 (doc50).

    신청시기: 피해발생일로부터 3개월 이내. 단, 3개월이 지나도 해당 사유가 해소되지 않았음을
    증명하면 그 사유가 해소되기 전까지 신청 가능하다.
    """
    plan_check = check_plan_type_eligible(plan_type)
    if not plan_check.eligible:
        return plan_check

    deadline = damage_date + timedelta(days=90)
    if request_date > deadline and damage_resolved:
        return WithdrawalEligibilityResult(False, "피해발생일로부터 3개월이 지났고 피해 사유도 이미 해소되어 신청 기한이 지났습니다.")

    return WithdrawalEligibilityResult(True, "재난피해 사유 중도인출 요건을 충족합니다.")
