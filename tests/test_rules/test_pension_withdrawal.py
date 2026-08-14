from src.rules.pension_withdrawal import calculate_pension_withdrawal


def test_combines_all_three_results():
    r = calculate_pension_withdrawal(
        account_value=100_000_000,
        pension_payment_year=3,
        actual_receipt_year=5,
        age=60,
        tax_credited_principal_and_gains_withdrawn=10_000_000,
    )
    # withdrawal_limit: 100,000,000 / (11-3) * 1.2 = 15,000,000
    assert r.withdrawal_limit.limit_amount == 15_000_000
    assert r.withdrawal_limit.is_unlimited is False
    # retirement_tax_reduction: 5년차 -> 70% 납부
    assert r.retirement_tax_reduction.payment_ratio == 0.7
    # comprehensive_tax: 1000만원 <= 1500만원 -> 한도 이내, 60세는 5.5%
    assert r.comprehensive_tax.exceeds_threshold is False
    assert r.comprehensive_tax.separate_tax_rate == 0.055


def test_unlimited_after_year_11():
    r = calculate_pension_withdrawal(
        account_value=100_000_000,
        pension_payment_year=11,
        actual_receipt_year=21,
        age=75,
        tax_credited_principal_and_gains_withdrawn=20_000_000,
        is_lifetime_annuity=True,
    )
    assert r.withdrawal_limit.is_unlimited is True
    assert r.retirement_tax_reduction.payment_ratio == 0.5  # 21년차 이상
    assert r.comprehensive_tax.exceeds_threshold is True  # 2000만원 > 1500만원
    assert r.comprehensive_tax.optional_flat_rate_if_exceeded == 0.165
