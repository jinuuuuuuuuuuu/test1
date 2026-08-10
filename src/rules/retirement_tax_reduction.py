"""퇴직소득세 감면율(이연퇴직소득세) 계산기 — 근거: doc39, doc40

⚠️ 러프 설계요약(연금Agent_설계요약.md)에는 "1~10년차 70%, 11년차 이후 60%"로 2단계만 기재돼
있었으나, 원본 doc39/doc40 대조 결과 실제로는 3단계다. 반드시 아래 수치를 기준으로 한다.

핵심 수치:
- 연금실제수령연차 1~10년차: 이연퇴직소득세의 70% 납부 (30% 감면)
- 연금실제수령연차 11~20년차: 이연퇴직소득세의 60% 납부 (40% 감면)
- 연금실제수령연차 21년차 이상: 이연퇴직소득세의 50% 납부 (50% 감면)
- 연금외수령(한도초과분 포함) 시에는 감면 없이 이연퇴직소득세 전액 납부

⚠️ 매우 중요한 구분 (doc40):
- "연금수령연차"(withdrawal_limit.py 대상)는 연금수령한도를 결정하며, 실제 인출 여부와
  무관하게 개시 가능 시점부터 매년 자동 누적된다.
- "연금실제수령연차"(이 모듈의 대상)는 퇴직소득세 감면율을 결정하며, 실제로 그 해에
  최소 1만원이라도 인출해야만 누적된다. 인출을 거른 해는 카운트되지 않는다.
  두 연차는 같은 계좌라도 서로 다른 값일 수 있다.
"""

from dataclasses import dataclass


@dataclass
class RetirementTaxReductionResult:
    actual_receipt_year: int   # 적용된 연금실제수령연차
    payment_ratio: float       # 납부해야 하는 비율 (0.7 / 0.6 / 0.5)
    reduction_ratio: float     # 감면율 (1 - payment_ratio)


def get_deferred_retirement_tax_rate(
    actual_receipt_year: int,
    is_pension_receipt: bool = True,
) -> RetirementTaxReductionResult:
    """연금실제수령연차를 받아 이연퇴직소득세 납부비율/감면율을 계산한다.

    is_pension_receipt=False (연금외수령)이면 감면이 전혀 적용되지 않는다 (전액 납부).
    """
    if actual_receipt_year < 1:
        raise ValueError("연금실제수령연차는 1 이상이어야 합니다")

    if not is_pension_receipt:
        return RetirementTaxReductionResult(
            actual_receipt_year=actual_receipt_year,
            payment_ratio=1.0,
            reduction_ratio=0.0,
        )

    if actual_receipt_year <= 10:
        payment_ratio = 0.7
    elif actual_receipt_year <= 20:
        payment_ratio = 0.6
    else:
        payment_ratio = 0.5

    return RetirementTaxReductionResult(
        actual_receipt_year=actual_receipt_year,
        payment_ratio=payment_ratio,
        reduction_ratio=round(1.0 - payment_ratio, 2),
    )
