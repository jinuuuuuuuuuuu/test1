import pytest
from src.rules.comprehensive_tax import determine_comprehensive_tax, get_pension_income_tax_rate


def test_under_threshold_uses_age_rate():
    r = determine_comprehensive_tax(tax_credited_principal_and_gains_withdrawn=14_000_000, age=60)
    assert r.exceeds_threshold is False
    assert r.separate_tax_rate == 0.055
    assert r.optional_flat_rate_if_exceeded is None


def test_over_threshold_offers_choice():
    r = determine_comprehensive_tax(tax_credited_principal_and_gains_withdrawn=16_000_000, age=60)
    assert r.exceeds_threshold is True
    assert r.separate_tax_rate is None
    assert r.optional_flat_rate_if_exceeded == 0.165


def test_exactly_at_threshold_not_exceeded():
    # doc38: "1,500만원 초과" 시에만 종합과세 -> 정확히 1,500만원이면 초과 아님
    r = determine_comprehensive_tax(tax_credited_principal_and_gains_withdrawn=15_000_000, age=60)
    assert r.exceeds_threshold is False


def test_age_rate_table():
    assert get_pension_income_tax_rate(55) == 0.055
    assert get_pension_income_tax_rate(69) == 0.055
    assert get_pension_income_tax_rate(70) == 0.044
    assert get_pension_income_tax_rate(79) == 0.044
    assert get_pension_income_tax_rate(80) == 0.033


def test_lifetime_annuity_special_rate_55_to_70():
    # 종신연금 수령 시 55~70세 구간은 3.3% (doc38에 명시된 부분만 반영)
    assert get_pension_income_tax_rate(60, is_lifetime_annuity=True) == 0.033


def test_lifetime_annuity_over_70_falls_back_to_normal_rate():
    # 70세 이상은 원문에 종신연금 특례가 명시되지 않아 일반 세율 적용
    assert get_pension_income_tax_rate(75, is_lifetime_annuity=True) == 0.044


def test_under_55_raises():
    with pytest.raises(ValueError):
        get_pension_income_tax_rate(50)
