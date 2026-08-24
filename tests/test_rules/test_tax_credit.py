from src.rules.tax_credit import calculate_tax_credit


def test_max_credit_low_income_doc41_example():
    # doc41: 900만원 납입, 16.5% -> 148만 5천원
    r = calculate_tax_credit(pension_savings_paid=6_000_000, irp_paid=3_000_000, total_salary=50_000_000)
    assert r.credit_rate == 0.165
    assert r.credited_total == 9_000_000
    assert r.tax_credit_amount == 1_485_000


def test_max_credit_high_income_doc41_example():
    # doc41: 900만원 납입, 13.2% -> 118만 8천원
    r = calculate_tax_credit(pension_savings_paid=6_000_000, irp_paid=3_000_000, total_salary=60_000_000)
    assert r.credit_rate == 0.132
    assert r.tax_credit_amount == 1_188_000


def test_income_threshold_boundary_is_low_rate():
    r = calculate_tax_credit(pension_savings_paid=1_000_000, irp_paid=0, total_salary=55_000_000)
    assert r.credit_rate == 0.165


def test_pension_savings_only_capped_at_6m():
    # 연금저축만 700만원 넣어도 세액공제 대상은 600만원까지만
    r = calculate_tax_credit(pension_savings_paid=7_000_000, irp_paid=0, total_salary=50_000_000)
    assert r.credited_pension_savings == 6_000_000
    assert r.credited_total == 6_000_000
    assert r.tax_credit_amount == 990_000  # 600만 x 16.5%
    assert r.excess_beyond_credit_limit == 1_000_000


def test_combined_capped_at_9m_even_if_pension_savings_maxed():
    r = calculate_tax_credit(pension_savings_paid=6_000_000, irp_paid=5_000_000, total_salary=50_000_000)
    assert r.credited_total == 9_000_000
    assert r.excess_beyond_credit_limit == 2_000_000  # 1,100만원 납입 - 900만원


def test_comprehensive_income_threshold_used_when_no_salary():
    r_low = calculate_tax_credit(pension_savings_paid=1_000_000, irp_paid=0, comprehensive_income=40_000_000)
    assert r_low.credit_rate == 0.165
    r_high = calculate_tax_credit(pension_savings_paid=1_000_000, irp_paid=0, comprehensive_income=46_000_000)
    assert r_high.credit_rate == 0.132


def test_over_total_contribution_limit_flag():
    r = calculate_tax_credit(pension_savings_paid=10_000_000, irp_paid=9_000_000, total_salary=50_000_000)
    assert r.over_contribution_limit is True  # 1,900만원 > 1,800만원


def test_raises_without_income_info():
    import pytest
    with pytest.raises(ValueError):
        calculate_tax_credit(pension_savings_paid=1_000_000, irp_paid=0)
