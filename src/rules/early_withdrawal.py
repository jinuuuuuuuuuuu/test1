"""중도인출 요건판정기 — 근거: doc46(요양), doc47(개인회생·파산), doc48(무주택 전월세),
doc49(무주택 주택구입), doc50(재난피해)

공통 규칙 (5개 문서 공통 서두):
- DC(확정기여형), IRP(개인형퇴직연금)만 중도인출 가능. DB(확정급여형)는 중도인출 자체가 불가.
  (근로자퇴직급여보장법 시행령 제14조·제18조 각호)

신청기한 (5개 사유 공통 구조 — WITHDRAWAL_DEADLINE_RULES):
사유마다 문서가 다르지만 규정 형태는 전부 "기준일 + 기간 이내"로 같다. 사유별로 마감일
계산을 따로 구현하면 사유가 늘 때마다 같은 코드를 복제하게 되고, 실제로 "요양·주택구입만
날짜 핸들러가 있고 전월세·재난은 없는" 비대칭이 생겼다(실측: 전월세 날짜 질문이 사유 목록
답변으로 빠짐). 그래서 기한 규정을 테이블로 선언하고 계산은 한 곳(calculate_deadline)에서만
한다 — 사유가 추가돼도 테이블 한 줄이면 된다.

⚠️ 기간 단위: 원문은 "1개월", "3개월", "5년"으로만 표현하고 30일/90일 환산을 정의하지
않는다. 따라서 고정 일수가 아니라 달력 기준으로 계산한다(2026-01-31 + 1개월 = 2026-02-28).
과거에 timedelta(days=30)을 쓰던 시기에는 1월 31일 기준 마감일이 3월 2일로 나와 실제보다
이틀 관대하게 판정됐다.

⚠️ 휴일 처리: 중도인출 원문에는 휴일·영업시간·신청 도달시점(작성일 기준인지 접수일
기준인지) 처리 기준이 명시되어 있지 않다. 따라서 여기서는 마감일을 영업일로 별도
연장하지 않는다 — 원문에 없는 규칙을 만들지 않기 위해서다. 답변에서는 이 부분이
문서로 확인되지 않는다는 점을 함께 안내해야 한다.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum


class PlanType(Enum):
    DC = "DC"
    IRP = "IRP"
    DB = "DB"


@dataclass
class WithdrawalEligibilityResult:
    eligible: bool
    reason: str


@dataclass
class WithdrawalDeadlineRule:
    """한 사유의 신청기한 규정 — "무엇을 기준으로, 얼마 이내"."""

    basis_event: str          # 기준일이 되는 사건 (사용자에게 그대로 보여줄 표현)
    months: int | None        # 기간(개월). years와 둘 중 하나만 채운다.
    years: int | None = None
    source_doc: str = ""      # 근거 문서
    note: str = ""            # 기한 외 유의사항 (예: 재난피해의 미해소 예외)


# 사유 코드 -> 신청기한 규정. reason 값은 tools.check_early_withdrawal의 EarlyWithdrawalReason과 일치.
WITHDRAWAL_DEADLINE_RULES: dict[str, WithdrawalDeadlineRule] = {
    "요양": WithdrawalDeadlineRule(
        basis_event="요양종료일", months=1, source_doc="doc46",
        note="요양사유 확인일로부터 요양종료일 이후 1개월 이내가 원문 표현입니다.",
    ),
    "개인회생파산": WithdrawalDeadlineRule(
        basis_event="개인회생절차개시 결정일 또는 파산선고일", months=None, years=5,
        source_doc="doc47",
        note="개인회생은 신청 시점에 절차 효력이 진행 중이어야 합니다(폐지·면책 결정 시 불가).",
    ),
    "무주택전월세": WithdrawalDeadlineRule(
        basis_event="잔금지급일", months=1, source_doc="doc48",
        note="주택임대차계약 체결일로부터 잔금지급일 이후 1개월 이내가 원문 표현입니다.",
    ),
    "무주택주택구입": WithdrawalDeadlineRule(
        basis_event="소유권 이전 등기접수일", months=1, source_doc="doc49",
        note="주택매매계약 체결일로부터 소유권 이전 등기 후 1개월 이내가 원문 표현입니다.",
    ),
    "재난피해": WithdrawalDeadlineRule(
        basis_event="피해발생일", months=3, source_doc="doc50",
        note="3개월이 지나도 피해 사유가 해소되지 않았음을 증명하면 해소 전까지 신청할 수 있습니다.",
    ),
}


def add_calendar_months(base_date: date, months: int) -> date:
    """달력 기준으로 개월을 더한다 (말일은 그 달의 마지막 날로 맞춘다).

    2026-01-31 + 1개월 = 2026-02-28, 2024-01-31 + 1개월 = 2024-02-29.
    """
    import calendar

    month_index = base_date.month - 1 + months
    year = base_date.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(base_date.day, last_day))


def calculate_deadline(reason: str, basis_date: date) -> date:
    """사유와 기준일을 받아 신청 마감일을 계산한다 — 5개 사유 공통 진입점.

    요건 충족 판정(check_*_eligibility)과 분리해 둔 이유: 사용자가 "언제까지 신청해야
    하나요"만 물었을 때 신청일을 요구하면 안 되기 때문이다. 실측에서 신청일이 필수라
    LLM이 기준일을 신청일로 그대로 넣어, 묻지도 않은 "그날 신청하면 가능한가" 판정을
    해버리는 오작동이 있었다.
    """
    rule = WITHDRAWAL_DEADLINE_RULES.get(reason)
    if rule is None:
        raise ValueError(f"알 수 없는 중도인출 사유입니다: {reason}")
    if rule.years is not None:
        return add_calendar_months(basis_date, rule.years * 12)
    return add_calendar_months(basis_date, rule.months)


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

    deadline = calculate_deadline("요양", treatment_end_date)
    if request_date > deadline:
        return WithdrawalEligibilityResult(
            False, f"요양종료일로부터 1개월(달력 기준 {deadline}까지)이 지나 신청 기한이 지났습니다."
        )

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

    deadline = calculate_deadline("개인회생파산", decision_date)
    if request_date > deadline:
        return WithdrawalEligibilityResult(
            False, f"{event_type} 결정일로부터 5년(달력 기준 {deadline}까지)이 지나 신청 기한이 지났습니다."
        )

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

    deadline = calculate_deadline("무주택전월세", balance_payment_date)
    if request_date > deadline:
        return WithdrawalEligibilityResult(
            False, f"잔금지급일로부터 1개월(달력 기준 {deadline}까지)이 지나 신청 기한이 지났습니다."
        )

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

    deadline = calculate_deadline("무주택주택구입", ownership_registration_date)
    if request_date > deadline:
        return WithdrawalEligibilityResult(
            False,
            f"소유권 이전 등기접수일로부터 1개월(달력 기준 {deadline}까지)이 지나 신청 기한이 지났습니다.",
        )

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

    deadline = calculate_deadline("재난피해", damage_date)
    if request_date > deadline and damage_resolved:
        return WithdrawalEligibilityResult(
            False,
            f"피해발생일로부터 3개월(달력 기준 {deadline}까지)이 지났고 피해 사유도 이미 "
            "해소되어 신청 기한이 지났습니다.",
        )

    return WithdrawalEligibilityResult(True, "재난피해 사유 중도인출 요건을 충족합니다.")
