"""디폴트옵션(사전지정운용제도) 상태판정기 — 근거: doc29 (디폴트옵션 고객/매수 중심 FAQ 100선)

doc29는 100개 FAQ + AI분기포인트로 구성된 방대한 문서라, 전체를 규칙으로 하드코딩하지 않고
가장 정형적이고 반복적으로 쓰이는 두 축만 결정론적 함수로 만든다:
  1) 옵트인(가입자 직접매수) 가능 여부 — FAQ 1~22, 51~55 (보유 개수 기반)
  2) 자동매수(사전지정운용) 통지/적용 일정 — FAQ 26~28, 33~38, 48~50 (기존/신규가입자, 반복만기)
그 외 세부 예외(FAQ 23~25, 40~46, 56~70, 71~100 등 텍스트 성격이 강한 항목)는 RAG 검색
대상으로 남겨둔다 — 결정론적 규칙과 텍스트 검색을 병행하는 것이 설계 원칙이다.

핵심 규칙 (원문 대조 완료):
- 옵트인: 실제 보유 디폴트옵션 0개 → 1개 상품 직접매수 가능
          실제 보유 정확히 1개 → "동일 상품"에 한해 추가매수 가능(예외), 다른 상품은 불가
          실제 보유 2개 이상(복수보유) → 어떤 옵트인도 불가, 먼저 1개로 정리(전량매도) 필요 (일부매도로는 해소 안 됨)
          포트폴리오형은 구성상품이 여럿이어도 "1개 상품"으로 계산 (FAQ 51, 54, 55)
          일반상품(비-디폴트옵션) 보유 여부는 옵트인 판단과 무관 (FAQ 11~13, 20)
          "지정"(향후 자동매수 예정 상품)과 "실제 보유"는 별개 — 옵트인 판단은 실제 보유 기준으로만 (FAQ 1, 17, 19)
          옵트인에는 4주·2주 대기기간이 없음 — 즉시 매수 가능 (FAQ 14, 86)
- 자동매수: 기존가입자 = 만기일 + 4주(28일) 통지 → 통지 + 2주(14일) 대기 → 자동매수
            신규가입자 = 최초 부담금 납입 다음 영업일 통지 → 2주(14일) 대기 → 자동매수 (4주 유예 없음)
            휴일이면 익영업일 적용
            동일 상품 반복 만기: 최초 1회만 통지·대기, 이후 즉시 적용
            단, "대기 중(아직 자동매수 전)"에 전액을 다른 상품으로 이동하면 연속성 단절 → 다음 만기분부터 재통지·재대기 필요 (FAQ 35)
            반면 "이미 자동매수 완료된 후" 매도(부분이든 전량이든)한 경우는 연속성이 유지되어, 다음 동일상품 만기분은 즉시 적용 (FAQ 37, 38)
- 만기 인정 대상: 정기예금·GIC·ELB의 만기/조기상환, 만기·상환조건이 확정된 채권·만기형펀드·ETN → 만기 O
                  운용기간이 정해지지 않은 일반 펀드·ETF의 청산 → 만기 아님, 자동매수 대상 아님 (FAQ 40~43)
"""

from dataclasses import dataclass
from datetime import date, timedelta

NOTICE_DELAY_DAYS_EXISTING = 28   # 기존가입자: 만기일 + 4주
WAIT_DAYS_AFTER_NOTICE = 14       # 통지 후 2주 대기 (기존/신규 공통)

MATURITY_ELIGIBLE_PRODUCT_TYPES = {
    "정기예금", "GIC", "ELB", "만기확정채권", "만기형펀드", "만기형ETN",
}
MATURITY_INELIGIBLE_PRODUCT_TYPES = {
    "일반펀드", "ETF",  # 운용기간/상환조건이 정해지지 않아 청산되어도 '만기'로 보지 않음
}


@dataclass
class OptinEligibilityResult:
    eligible: bool
    reason: str


def check_optin_eligibility(
    current_holdings_count: int,
    target_is_same_as_only_holding: bool = False,
) -> OptinEligibilityResult:
    """디폴트옵션 옵트인(가입자 직접매수) 가능 여부를 판정한다.

    current_holdings_count: 현재 실제로 운용 중인 디폴트옵션 상품 개수
      (포트폴리오형은 구성상품 개수와 무관하게 1개로 카운트).
    target_is_same_as_only_holding: 보유 개수가 1개일 때, 사려는 상품이 그 보유 상품과
      동일한 상품인지 여부. 동일 상품 추가매수는 예외적으로 허용된다.
    """
    if current_holdings_count < 0:
        raise ValueError("current_holdings_count는 0 이상이어야 합니다")

    if current_holdings_count == 0:
        return OptinEligibilityResult(True, "실제 보유 중인 디폴트옵션이 없어 1개 상품을 직접 매수할 수 있습니다.")

    if current_holdings_count == 1:
        if target_is_same_as_only_holding:
            return OptinEligibilityResult(True, "이미 보유 중인 동일 상품에 대한 추가 매수는 예외적으로 허용됩니다.")
        return OptinEligibilityResult(False, "이미 다른 디폴트옵션을 보유 중이므로 다른 유형으로의 옵트인은 제한됩니다. 기존 상품을 전량 매도해야 합니다.")

    return OptinEligibilityResult(False, "디폴트옵션을 복수 보유 중이므로 추가 옵트인이 제한됩니다. 먼저 1개 상품만 남도록 전량 정리해야 합니다 (일부 매도로는 해소되지 않습니다).")


def is_maturity_event(product_type: str) -> bool:
    """상품 유형이 '만기'로 인정되어 사전지정운용제도 대상이 되는지 판정한다."""
    if product_type in MATURITY_ELIGIBLE_PRODUCT_TYPES:
        return True
    if product_type in MATURITY_INELIGIBLE_PRODUCT_TYPES:
        return False
    raise ValueError(f"알 수 없는 상품 유형입니다: {product_type}")


@dataclass
class AutoPurchaseSchedule:
    notice_date: date
    apply_date: date
    waited: bool  # False면 반복만기 등으로 대기 없이 즉시 적용됨


def _next_business_day_if_holiday(d: date, is_holiday_fn) -> date:
    while is_holiday_fn(d):
        d += timedelta(days=1)
    return d


def get_auto_purchase_schedule(
    base_date: date,
    is_new_participant: bool,
    is_first_occurrence: bool = True,
    is_holiday_fn=lambda d: False,
) -> AutoPurchaseSchedule:
    """자동매수(사전지정운용) 통지일/적용일을 계산한다.

    base_date: 기존가입자는 상품 만기일, 신규가입자는 최초 부담금 납입일.
    is_first_occurrence: 동일 상품의 첫 만기(또는 최초 납입)인지 여부.
      False(반복 만기 등)면 통지·대기 없이 즉시 적용된다 (연속성 유지 전제).
    is_holiday_fn: 특정 날짜가 휴일인지 판정하는 함수. 기본은 항상 영업일로 가정.
    """
    if not is_first_occurrence:
        applied = _next_business_day_if_holiday(base_date, is_holiday_fn)
        return AutoPurchaseSchedule(notice_date=applied, apply_date=applied, waited=False)

    if is_new_participant:
        notice_date = base_date + timedelta(days=1)  # 다음 영업일 (간이 구현: 휴일 캘린더 미적용 시 +1일)
    else:
        notice_date = base_date + timedelta(days=NOTICE_DELAY_DAYS_EXISTING)

    notice_date = _next_business_day_if_holiday(notice_date, is_holiday_fn)
    apply_date = notice_date + timedelta(days=WAIT_DAYS_AFTER_NOTICE)
    apply_date = _next_business_day_if_holiday(apply_date, is_holiday_fn)

    return AutoPurchaseSchedule(notice_date=notice_date, apply_date=apply_date, waited=True)


def is_continuity_broken(moved_during_wait_in_full: bool) -> bool:
    """반복 만기 시 '연속성'이 끊겼는지 판정한다.

    moved_during_wait_in_full: 대기 중(아직 자동매수되기 전)에 해당 금액 전액을
      다른 상품으로 이동시켰는지 여부. True면 연속성이 끊겨 다음 만기분부터 다시
      통지·대기를 거쳐야 한다. 이미 자동매수가 완료된 후의 매도(부분/전량 무관)는
      연속성에 영향을 주지 않는다 — 이 함수의 대상이 아니다.
    """
    return moved_during_wait_in_full
