import pytest
from src.rules.early_withdrawal import PlanType
from src.rules.investment_limit import (
    check_concentration_limit,
    check_product_eligibility,
    check_risk_asset_allocation,
    classify_fund_category_risk_tier,
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


# ── classify_fund_category_risk_tier (search_funds 결과 -> RiskTier) ─────


def test_classify_government_bond_fund_as_safe():
    tier = classify_fund_category_risk_tier(
        "투자신탁, 증권(채권형), 개방형(중도환매가능), 추가형(추가납입가능), 종류형",
        "한국투자 퇴직연금 증권 자투자신탁 1호(국공채)",
    )
    assert tier == RiskTier.SAFE


def test_classify_credit_bond_fund_as_risky_despite_same_category_text():
    """국공채 펀드와 fund_category 텍스트가 동일한 크레딧(회사채) 펀드는 위험자산으로 분류한다.

    실측: "미래에셋퇴직플랜단기증권자투자신탁1호(채권)"(회사채 포함 크레딧물)와
    "한국투자 퇴직연금 증권 자투자신탁 1호(국공채)"는 fund_category가 둘 다
    "증권(채권형)"으로 동일해 카테고리 텍스트만으로는 구분되지 않는다. 펀드명의
    "국공채" 표시가 없으면 안전자산으로 오분류하지 않고 위험자산으로 본다.
    """
    tier = classify_fund_category_risk_tier(
        "투자신탁, 증권(채권형), 개방형(중도환매가능), 추가형(추가납입가능), 모자형, 종류형",
        "미래에셋퇴직플랜단기증권자투자신탁1호(채권)",
    )
    assert tier == RiskTier.RISKY


def test_classify_equity_fund_as_risky():
    tier = classify_fund_category_risk_tier(
        "투자신탁, 증권(주식형), 개방형(중도환매가능), 추가형(추가납입가능), 모자형, 종류형",
        "삼성퇴직연금KOSPI200증권자투자신탁 제1호[주식]",
    )
    assert tier == RiskTier.RISKY


def test_classify_mmf_as_safe():
    tier = classify_fund_category_risk_tier("투자신탁, MMF, 개방형", "단기 MMF 증권투자신탁")
    assert tier == RiskTier.SAFE


def test_classify_unknown_category_defaults_to_risky():
    """분류 근거가 불충분하면 안전 쪽(RISKY)으로 실패한다 — 위험자산을 안전자산으로
    오분류해 70% 한도 초과를 놓치는 사고를 만들지 않는다."""
    assert classify_fund_category_risk_tier(None, None) == RiskTier.RISKY
    assert classify_fund_category_risk_tier("알 수 없는 분류", "이상한 펀드") == RiskTier.RISKY
