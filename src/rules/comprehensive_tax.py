"""종합과세 기준판정 — 근거: doc38 (연금소득 종합과세 안내), doc39 (인출순서/재원별 과세)

핵심 규칙 (doc38 원문 대조 완료):
- 사적연금소득이 연 1,500만원(전 금융기관 연금계좌 합산)을 초과하는지가 종합과세 판단 기준.
- 단, 이 1,500만원 판정에는 "세액공제 안 받은 원금"과 "퇴직금(이연퇴직소득)"은 포함되지 않는다.
  오직 "세액공제 받은 납입금 + 운용수익" 재원만 1,500만원 판정 대상이다.
- 1,500만원 이내: 연령별 세율로 분리과세 (종합소득 합산 불필요)
- 1,500만원 초과: 종합과세(다른 소득과 합산, 6.6~49.5%) 또는 16.5% 분리과세 중 선택 가능.
  16.5% 선택 시 초과분이 아니라 "전체 금액"에 16.5%가 적용된다는 점에 유의.

연금소득세율표 (만 나이 기준, 연금수령일 현재):
- 55세 이상 70세 미만: 5.5% (종신연금 수령 시 3.3%)
- 70세 이상 80세 미만: 4.4% (종신연금 수령 시 3.3%)
- 80세 이상: 3.3% (종신연금 수령 시에도 동일 3.3%)

doc38 원문 표(docx raw XML의 vMerge 속성까지 확인 — 세 행 모두 병합 없이 독립 셀)를 재대조한
결과 "종신연금 수령 시" 열은 세 연령 구간 모두 3.3%로 명시되어 있다. 즉 종신연금 수령이면
연령과 무관하게 항상 3.3%가 적용된다.
"""

from dataclasses import dataclass

ANNUAL_THRESHOLD = 15_000_000  # 종합과세 판정 기준 (연금소득세 대상 재원만)
SEPARATE_TAXATION_RATE_OVER_THRESHOLD = 0.165  # 초과 시 선택 가능한 분리과세율 (전체 금액에 적용)

COMPREHENSIVE_TAX_RATE_MIN = 0.066  # 지방소득세 포함 최저
COMPREHENSIVE_TAX_RATE_MAX = 0.495  # 지방소득세 포함 최고


def get_pension_income_tax_rate(age: int, is_lifetime_annuity: bool = False) -> float:
    """1,500만원 이내 구간에 적용되는 연령별 연금소득세율(분리과세)을 반환한다."""
    if age < 55:
        raise ValueError("연금 수령은 만 55세 이후부터 가능합니다 (doc39)")
    if is_lifetime_annuity:
        return 0.033
    if age < 70:
        return 0.055
    if age < 80:
        return 0.044
    return 0.033


@dataclass
class ComprehensiveTaxResult:
    taxable_pension_income: int          # 세액공제받은 원금+운용수익 재원 중 연금수령분 (1,500만원 판정 대상)
    exceeds_threshold: bool              # 1,500만원 초과 여부
    separate_tax_rate: float | None      # 초과 안 했을 때 적용되는 연령별 분리과세율
    optional_flat_rate_if_exceeded: float | None  # 초과했을 때 선택 가능한 16.5% 분리과세율


def determine_comprehensive_tax(
    tax_credited_principal_and_gains_withdrawn: int,
    age: int,
    is_lifetime_annuity: bool = False,
) -> ComprehensiveTaxResult:
    """세액공제받은 원금+운용수익 재원의 연간 인출액을 받아 종합과세 대상 여부를 판정한다.

    tax_credited_principal_and_gains_withdrawn 인자에는 반드시 "세액공제 받은 납입금 +
    운용수익" 재원에서 인출한 금액만 넣는다 — 비과세 원금, 퇴직금(이연퇴직소득) 인출액은
    이 판정과 무관하므로 포함하면 안 된다 (doc38).
    """
    exceeds = tax_credited_principal_and_gains_withdrawn > ANNUAL_THRESHOLD

    if exceeds:
        return ComprehensiveTaxResult(
            taxable_pension_income=tax_credited_principal_and_gains_withdrawn,
            exceeds_threshold=True,
            separate_tax_rate=None,
            optional_flat_rate_if_exceeded=SEPARATE_TAXATION_RATE_OVER_THRESHOLD,
        )

    return ComprehensiveTaxResult(
        taxable_pension_income=tax_credited_principal_and_gains_withdrawn,
        exceeds_threshold=False,
        separate_tax_rate=get_pension_income_tax_rate(age, is_lifetime_annuity),
        optional_flat_rate_if_exceeded=None,
    )
