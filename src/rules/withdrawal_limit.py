"""연금수령한도 계산기 — 근거: doc39 (연금수령한도 안내)

핵심 수치 (doc39 원문 대조 완료):
- 연금수령한도 = 연금계좌 평가액(매년 1/1 또는 연금개시일 기준) ÷ (11 - 연금수령연차) × 120%
- 연금수령연차 11년차 이상부터는 한도 자체가 사라짐 (전액 일시인출도 연금수령으로 인정)
- 6년차 특례: 2013.3.1 이전 가입한 연금계좌(IRP·연금저축)는 1년차가 아닌 6년차로 기산 시작.
  2013.3.1 이전 가입한 퇴직연금(DC/DB)도, 그 퇴직금 전액을 신규 연금계좌로 이체하는 경우에 한해 특례 적용.
- 연금수령 요건 3가지: ① 연금계좌 가입기간 5년 이상 (계좌에 퇴직금이 있으면 이 요건은 면제) ② 만 55세 이후 ③ 한도 이내 인출
"""

from dataclasses import dataclass

UNLIMITED_FROM_YEAR = 11  # 이 연차부터 한도 소멸
SIX_YEAR_EXCEPTION_START = 6  # 특례 적용 시 시작 연차
SIX_YEAR_EXCEPTION_CUTOFF = "2013-03-01"


@dataclass
class WithdrawalLimitResult:
    pension_payment_year: int          # 적용된 연금수령연차 (특례 반영 후)
    limit_amount: int | None           # 연금수령한도 금액. None이면 한도 없음(11년차 이상)
    is_unlimited: bool                 # 11년차 이상 여부


def resolve_pension_payment_year(
    years_since_eligible: int,
    six_year_exception: bool = False,
) -> int:
    """연금개시 가능 시점부터 경과한 연수(1부터 시작)를 받아 실제 연금수령연차를 계산한다.

    six_year_exception=True면 2013.3.1 이전 가입 특례를 적용해 6년차부터 기산한다
    (즉, 개시 가능 첫 해가 6년차).
    """
    if years_since_eligible < 1:
        raise ValueError("years_since_eligible은 1 이상이어야 합니다 (개시 가능한 해가 1년차)")
    if six_year_exception:
        return years_since_eligible + (SIX_YEAR_EXCEPTION_START - 1)
    return years_since_eligible


def calculate_withdrawal_limit(
    account_value: int,
    pension_payment_year: int,
) -> WithdrawalLimitResult:
    """연금계좌 평가액과 연금수령연차로 올해 연금수령한도를 계산한다."""
    if pension_payment_year < 1:
        raise ValueError("연금수령연차는 1 이상이어야 합니다")

    if pension_payment_year >= UNLIMITED_FROM_YEAR:
        return WithdrawalLimitResult(
            pension_payment_year=pension_payment_year,
            limit_amount=None,
            is_unlimited=True,
        )

    denominator = 11 - pension_payment_year
    limit_amount = round(account_value / denominator * 1.2)
    return WithdrawalLimitResult(
        pension_payment_year=pension_payment_year,
        limit_amount=limit_amount,
        is_unlimited=False,
    )


def is_within_limit(withdrawal_amount: int, limit_result: WithdrawalLimitResult) -> bool:
    """인출 금액이 연금수령한도 이내인지(=연금수령 vs 연금외수령) 판정한다."""
    if limit_result.is_unlimited:
        return True
    return withdrawal_amount <= limit_result.limit_amount
