import pytest
from src.rules.withdrawal_limit import (
    calculate_withdrawal_limit,
    resolve_pension_payment_year,
    is_within_limit,
)


def test_year_1_denominator_10():
    # doc39: 1년차에는 분모가 10
    r = calculate_withdrawal_limit(account_value=100_000_000, pension_payment_year=1)
    assert r.limit_amount == round(100_000_000 / 10 * 1.2)  # 12,000,000
    assert r.is_unlimited is False


def test_year_10_denominator_1_full_120pct():
    # doc39: 10년차가 되면 분모가 1이 되어 평가액 전체의 120%까지
    r = calculate_withdrawal_limit(account_value=100_000_000, pension_payment_year=10)
    assert r.limit_amount == 120_000_000


def test_year_11_unlimited():
    r = calculate_withdrawal_limit(account_value=100_000_000, pension_payment_year=11)
    assert r.is_unlimited is True
    assert r.limit_amount is None


def test_year_20_still_unlimited():
    r = calculate_withdrawal_limit(account_value=100_000_000, pension_payment_year=20)
    assert r.is_unlimited is True


def test_resolve_year_without_exception():
    assert resolve_pension_payment_year(years_since_eligible=1) == 1
    assert resolve_pension_payment_year(years_since_eligible=5) == 5


def test_resolve_year_with_six_year_exception():
    # 2013.3.1 이전 가입: 개시 가능 첫 해가 6년차
    assert resolve_pension_payment_year(years_since_eligible=1, six_year_exception=True) == 6
    assert resolve_pension_payment_year(years_since_eligible=5, six_year_exception=True) == 10


def test_is_within_limit():
    r = calculate_withdrawal_limit(account_value=100_000_000, pension_payment_year=1)
    assert is_within_limit(10_000_000, r) is True
    assert is_within_limit(13_000_000, r) is False


def test_is_within_limit_unlimited_always_true():
    r = calculate_withdrawal_limit(account_value=100_000_000, pension_payment_year=11)
    assert is_within_limit(999_999_999, r) is True


def test_invalid_year_raises():
    with pytest.raises(ValueError):
        calculate_withdrawal_limit(account_value=100_000_000, pension_payment_year=0)
