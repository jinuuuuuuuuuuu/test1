"""연금 인출 종합 계산기 — withdrawal_limit / retirement_tax_reduction / comprehensive_tax를
하나의 질문("연금 인출하면 얼마까지 받을 수 있고 세금은 얼마인가")에 대한 답으로 묶는다.

세 모듈은 서로 다른 축을 계산한다 (doc39/doc40 — "연금수령연차"와 "연금실제수령연차"는 별개 개념,
모듈별 docstring 참고):
  - withdrawal_limit: 올해 연금수령한도가 얼마인지 (연금수령연차 기준)
  - retirement_tax_reduction: 퇴직금(이연퇴직소득) 재원의 세율 감면율 (연금실제수령연차 기준)
  - comprehensive_tax: 세액공제받은 재원+운용수익의 종합과세 여부/세율 (나이 기준)
이 모듈은 세 결과를 계산해 하나의 응답으로 합치기만 하며, 각 계산의 정확성은 개별 모듈의
테스트가 보증한다.
"""

from dataclasses import dataclass

from src.rules.comprehensive_tax import ComprehensiveTaxResult, determine_comprehensive_tax
from src.rules.retirement_tax_reduction import RetirementTaxReductionResult, get_deferred_retirement_tax_rate
from src.rules.withdrawal_limit import WithdrawalLimitResult, calculate_withdrawal_limit


@dataclass
class PensionWithdrawalResult:
    withdrawal_limit: WithdrawalLimitResult
    retirement_tax_reduction: RetirementTaxReductionResult
    comprehensive_tax: ComprehensiveTaxResult


def calculate_pension_withdrawal(
    account_value: int,
    pension_payment_year: int,
    actual_receipt_year: int,
    age: int,
    tax_credited_principal_and_gains_withdrawn: int,
    is_lifetime_annuity: bool = False,
) -> PensionWithdrawalResult:
    """연금계좌 인출 시나리오 하나를 받아 한도/퇴직소득세감면/종합과세 세 가지를 한 번에 계산한다.

    account_value, pension_payment_year: 올해 연금수령한도 계산용 (withdrawal_limit.py).
    actual_receipt_year: 퇴직금(이연퇴직소득) 재원의 감면율 계산용 — 연금수령연차와는 다른 값이니
      혼동하지 말 것 (retirement_tax_reduction.py 모듈 docstring 참고).
    age, tax_credited_principal_and_gains_withdrawn, is_lifetime_annuity: 세액공제받은
      원금+운용수익 재원의 종합과세 여부 계산용 (comprehensive_tax.py).
    """
    limit = calculate_withdrawal_limit(account_value, pension_payment_year)
    retirement_tax = get_deferred_retirement_tax_rate(actual_receipt_year, is_pension_receipt=True)
    comprehensive = determine_comprehensive_tax(
        tax_credited_principal_and_gains_withdrawn, age, is_lifetime_annuity,
    )
    return PensionWithdrawalResult(
        withdrawal_limit=limit,
        retirement_tax_reduction=retirement_tax,
        comprehensive_tax=comprehensive,
    )
