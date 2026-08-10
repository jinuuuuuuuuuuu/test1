"""세액공제 계산기 — 근거: doc41 (연금저축계좌, IRP 세액공제 안내)

핵심 수치 (doc41 원문 대조 완료):
- 연금저축+IRP 합산 납입한도: 연 1,800만원 (세액공제와 무관하게 납입만 가능한 한도)
- 세액공제 대상 납입한도: 연금저축 단독 600만원 / 연금저축+IRP 합산 900만원
- 세액공제율: 총급여 5,500만원 이하(종합소득금액 4,500만원 이하) → 16.5%, 초과 → 13.2%
- 최대 절세액: 900만원 x 16.5% = 148만 5천원 / 900만원 x 13.2% = 118만 8천원
"""

from dataclasses import dataclass

PENSION_SAVINGS_ONLY_LIMIT = 6_000_000  # 연금저축 단독 세액공제 납입한도
COMBINED_CREDIT_LIMIT = 9_000_000       # 연금저축+IRP 합산 세액공제 납입한도
TOTAL_CONTRIBUTION_LIMIT = 18_000_000   # 연금저축+IRP 합산 납입한도 (세액공제 무관)

INCOME_THRESHOLD_SALARY = 55_000_000        # 총급여 기준
INCOME_THRESHOLD_COMPREHENSIVE = 45_000_000  # 종합소득금액 기준

CREDIT_RATE_LOW = 0.165   # 소득 기준 이하
CREDIT_RATE_HIGH = 0.132  # 소득 기준 초과


@dataclass
class TaxCreditResult:
    credit_rate: float                # 적용 세액공제율
    credited_pension_savings: int     # 세액공제 대상으로 인정된 연금저축 납입액
    credited_total: int               # 세액공제 대상 합계 (연금저축 + IRP, 900만원 한도 적용 후)
    tax_credit_amount: int            # 실제 세액공제액
    over_contribution_limit: bool     # 1,800만원 총 납입한도 초과 여부
    excess_beyond_credit_limit: int   # 세액공제 한도(900만원) 초과 납입액 — 과세이연 등 비과세 혜택은 있으나 세액공제는 없음


def calculate_tax_credit(
    pension_savings_paid: int,
    irp_paid: int,
    total_salary: int | None = None,
    comprehensive_income: int | None = None,
) -> TaxCreditResult:
    """연금저축·IRP 납입액을 받아 세액공제액을 계산한다.

    total_salary(총급여) 또는 comprehensive_income(종합소득금액) 중 하나는 반드시 제공해야 한다.
    둘 다 제공되면 total_salary를 우선 적용한다 (doc41은 직장인=총급여, 종합소득자=종합소득금액 기준으로 구분).
    """
    if total_salary is None and comprehensive_income is None:
        raise ValueError("total_salary 또는 comprehensive_income 중 하나는 필요합니다")

    if total_salary is not None:
        credit_rate = CREDIT_RATE_LOW if total_salary <= INCOME_THRESHOLD_SALARY else CREDIT_RATE_HIGH
    else:
        credit_rate = CREDIT_RATE_LOW if comprehensive_income <= INCOME_THRESHOLD_COMPREHENSIVE else CREDIT_RATE_HIGH

    credited_pension_savings = min(pension_savings_paid, PENSION_SAVINGS_ONLY_LIMIT)
    credited_total = min(credited_pension_savings + irp_paid, COMBINED_CREDIT_LIMIT)
    tax_credit_amount = round(credited_total * credit_rate)

    total_paid = pension_savings_paid + irp_paid
    over_contribution_limit = total_paid > TOTAL_CONTRIBUTION_LIMIT
    excess_beyond_credit_limit = max(0, total_paid - COMBINED_CREDIT_LIMIT)

    return TaxCreditResult(
        credit_rate=credit_rate,
        credited_pension_savings=credited_pension_savings,
        credited_total=credited_total,
        tax_credit_amount=tax_credit_amount,
        over_contribution_limit=over_contribution_limit,
        excess_beyond_credit_limit=excess_beyond_credit_limit,
    )
