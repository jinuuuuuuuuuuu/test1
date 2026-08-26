from datetime import date
import pytest
from src.rules.default_option import (
    check_optin_eligibility,
    is_maturity_event,
    get_auto_purchase_schedule,
    is_continuity_broken,
)


# ── 옵트인 가능 여부 (FAQ 1~22) ──────────────────────────────────────

def test_no_holdings_can_optin():
    r = check_optin_eligibility(current_holdings_count=0)
    assert r.eligible is True


def test_one_holding_same_product_exception_allowed():
    # FAQ 4, 13, 20: 동일 상품 추가매수는 예외적으로 허용
    r = check_optin_eligibility(current_holdings_count=1, target_is_same_as_only_holding=True)
    assert r.eligible is True


def test_one_holding_different_product_blocked():
    # FAQ 3: 이미 보유 중이면 다른 유형 옵트인 불가
    r = check_optin_eligibility(current_holdings_count=1, target_is_same_as_only_holding=False)
    assert r.eligible is False


def test_multiple_holdings_always_blocked_even_if_same():
    # FAQ 6, 7: 복수보유 중에는 같은 상품이든 다른 상품이든 옵트인 불가
    r = check_optin_eligibility(current_holdings_count=2, target_is_same_as_only_holding=True)
    assert r.eligible is False


def test_negative_holdings_raises():
    with pytest.raises(ValueError):
        check_optin_eligibility(current_holdings_count=-1)


# ── 만기 인정 여부 (FAQ 40~43) ────────────────────────────────────────

def test_maturity_eligible_products():
    assert is_maturity_event("정기예금") is True
    assert is_maturity_event("GIC") is True
    assert is_maturity_event("만기형펀드") is True


def test_maturity_ineligible_products():
    # FAQ 40: 만기 없는 실적배당형(일반펀드/ETF)은 만기로 보지 않음
    assert is_maturity_event("일반펀드") is False
    assert is_maturity_event("ETF") is False


def test_unknown_product_type_raises():
    with pytest.raises(ValueError):
        is_maturity_event("존재하지않는상품")


# ── 자동매수 통지/적용 일정 (FAQ 26~28, 48) ───────────────────────────

def test_existing_participant_schedule_4weeks_2weeks():
    # FAQ 26 수정문: 만기일 + 29일 통지, 통지일 + 15일 적용.
    # 원문 예시("6월 1일 만기 → 6월 29일 또는 30일 무렵 통지")와 일치하는지 함께 확인한다.
    maturity = date(2026, 6, 1)
    sched = get_auto_purchase_schedule(base_date=maturity, is_new_participant=False)
    assert sched.notice_date == date(2026, 6, 30)
    assert sched.apply_date == date(2026, 7, 15)
    assert sched.waited is True


def test_new_participant_schedule_next_day_2weeks():
    # FAQ 48 + FAQ 26 수정문: 신규가입자는 최초 부담금 다음 영업일 통지, 통지일 + 15일 적용
    contribution = date(2026, 6, 1)
    sched = get_auto_purchase_schedule(base_date=contribution, is_new_participant=True)
    assert sched.notice_date == date(2026, 6, 2)
    assert sched.apply_date == date(2026, 6, 17)


def test_repeat_maturity_no_wait():
    # FAQ 33, 36: 동일상품 반복 만기는 최초 1회만 대기, 이후 즉시 적용
    maturity = date(2026, 9, 1)
    sched = get_auto_purchase_schedule(base_date=maturity, is_new_participant=False, is_first_occurrence=False)
    assert sched.waited is False
    assert sched.notice_date == sched.apply_date == maturity


def test_holiday_pushes_to_next_business_day():
    maturity = date(2026, 6, 1)

    def is_holiday(d):
        return d == date(2026, 6, 30)

    sched = get_auto_purchase_schedule(base_date=maturity, is_new_participant=False, is_holiday_fn=is_holiday)
    assert sched.notice_date == date(2026, 7, 1)


# ── 연속성 판단 (FAQ 35 vs 37/38) ─────────────────────────────────────

def test_continuity_broken_when_moved_during_wait():
    assert is_continuity_broken(moved_during_wait_in_full=True) is True


def test_continuity_kept_when_not_moved_during_wait():
    assert is_continuity_broken(moved_during_wait_in_full=False) is False
