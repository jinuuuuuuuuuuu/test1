import pytest
from src.rules.retirement_tax_reduction import get_deferred_retirement_tax_rate


def test_tier1_1_to_10_years_70pct():
    r = get_deferred_retirement_tax_rate(1)
    assert r.payment_ratio == 0.7
    assert r.reduction_ratio == 0.3
    r10 = get_deferred_retirement_tax_rate(10)
    assert r10.payment_ratio == 0.7


def test_tier2_11_to_20_years_60pct():
    r = get_deferred_retirement_tax_rate(11)
    assert r.payment_ratio == 0.6
    assert r.reduction_ratio == 0.4
    r20 = get_deferred_retirement_tax_rate(20)
    assert r20.payment_ratio == 0.6


def test_tier3_21_years_and_above_50pct():
    r = get_deferred_retirement_tax_rate(21)
    assert r.payment_ratio == 0.5
    assert r.reduction_ratio == 0.5
    r100 = get_deferred_retirement_tax_rate(100)
    assert r100.payment_ratio == 0.5


def test_non_pension_receipt_no_reduction():
    # 연금외수령은 감면 없이 전액 납부
    r = get_deferred_retirement_tax_rate(15, is_pension_receipt=False)
    assert r.payment_ratio == 1.0
    assert r.reduction_ratio == 0.0


def test_invalid_year_raises():
    with pytest.raises(ValueError):
        get_deferred_retirement_tax_rate(0)
