import pytest
from src.rules.early_withdrawal import PlanType
from src.rules.investment_limit import (
    check_concentration_limit,
    check_product_eligibility,
    check_risk_asset_allocation,
    get_risk_tier,
    RiskTier,
)


def test_safe_asset_same_across_plan_types():
    for plan in (PlanType.DB, PlanType.DC, PlanType.IRP):
        assert get_risk_tier("예금·적금", plan) == RiskTier.SAFE


def test_domestic_listed_stock_db_only():
    # doc56/58 교차검증: 국내 상장주식은 DB만 투자가능(위험자산), DC/IRP는 투자금지
    assert get_risk_tier("국내상장주식", PlanType.DB) == RiskTier.RISKY
    assert get_risk_tier("국내상장주식", PlanType.DC) == RiskTier.FORBIDDEN
    assert get_risk_tier("국내상장주식", PlanType.IRP) == RiskTier.FORBIDDEN


def test_private_fund_and_dr_db_only():
    for product in ("사모펀드", "증권예탁증권(DR)"):
        assert get_risk_tier(product, PlanType.DB) == RiskTier.RISKY
        assert get_risk_tier(product, PlanType.DC) == RiskTier.FORBIDDEN
        assert get_risk_tier(product, PlanType.IRP) == RiskTier.FORBIDDEN


def test_unknown_product_raises():
    with pytest.raises(KeyError):
        get_risk_tier("존재하지않는상품", PlanType.DC)


def test_risky_asset_allocation_within_70_percent():
    r = check_risk_asset_allocation(PlanType.DC, 0.65)
    assert r.limit_ratio == 0.70
    assert r.within_limit is True
    assert r.is_tdf_exception is False


def test_risky_asset_allocation_exceeds_70_percent():
    r = check_risk_asset_allocation(PlanType.IRP, 0.85)
    assert r.within_limit is False


def test_tdf_qualified_allows_100_percent_dc_irp_but_70_for_db():
    dc = check_risk_asset_allocation(PlanType.DC, 0.95, is_tdf_qualified=True)
    irp = check_risk_asset_allocation(PlanType.IRP, 1.0, is_tdf_qualified=True)
    db = check_risk_asset_allocation(PlanType.DB, 0.95, is_tdf_qualified=True)
    assert dc.limit_ratio == 1.0 and dc.within_limit is True
    assert irp.limit_ratio == 1.0 and irp.within_limit is True
    assert db.limit_ratio == 0.70 and db.within_limit is False


def test_concentration_limit_same_issuer():
    db = check_concentration_limit("동일법인_증권", PlanType.DB, 0.10)
    dc = check_concentration_limit("동일법인_증권", PlanType.DC, 0.10)
    assert db.limit_ratio == 0.10 and db.within_limit is True
    assert dc.limit_ratio == 0.30 and dc.within_limit is True


def test_concentration_limit_forbidden_category():
    r = check_concentration_limit("사용자_이해관계인_발행증권", PlanType.DC, 0.01)
    assert r.eligible is False
    assert r.limit_ratio is None


def test_concentration_limit_db_no_affiliate_investment():
    # 사용자 계열회사·지분법적용사 발행증권: DB는 투자금지, DC/IRP는 한도 있음
    r = check_concentration_limit("사용자계열사_지분법적용사_발행증권", PlanType.DB, 0.05)
    assert r.eligible is False


def test_product_eligibility_forbidden_product():
    r = check_product_eligibility("국내상장주식", PlanType.DC)
    assert r.eligible is False
    assert r.risk_tier == RiskTier.FORBIDDEN


def test_product_eligibility_allowed_within_portfolio_limit():
    r = check_product_eligibility(
        "주식형·주식혼합형펀드", PlanType.IRP,
        current_risky_asset_ratio_after_purchase=0.60,
    )
    assert r.eligible is True


def test_product_eligibility_allowed_product_but_exceeds_portfolio_limit():
    r = check_product_eligibility(
        "주식형·주식혼합형펀드", PlanType.IRP,
        current_risky_asset_ratio_after_purchase=0.80,
    )
    assert r.eligible is False
    assert r.risk_tier == RiskTier.RISKY
