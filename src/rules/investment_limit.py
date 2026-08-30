"""퇴직연금 적립금 운용방법·투자한도 판정기 — 근거: doc56(장내상품 매매 안내), doc58(적립금 운용 및 투자한도 안내)

핵심 구조 (doc56/58 원문 대조 완료, 상호 교차검증):
- 운용상품은 크게 세 단계로 분류된다.
  ① 안전자산: 한도 없이 100%까지 투자 가능 (원리금보장상품, 투자위험을 낮춘 상품)
  ② 위험자산: DB/DC/IRP 공통으로 적립금의 70%까지만 투자 가능
  ③ 투자금지상품: 제도 불문 전액 투자 불가
  단, TDF는 감독원장이 정한 조건(주식비중 80%(예상은퇴시점 이후 40%) 이내, 비적격채무증권
  자산총액 20%·채무증권투자액 50% 이내)을 충족하면 DC/IRP에 한해 100%까지 투자할 수 있다.
  이 특례는 DC/IRP 전용이며, DB는 예상은퇴시점을 특정할 수 없어 해당사항이 없다(doc58 chunk01).
- DB와 DC/IRP는 투자 가능한 상품 자체가 다르다. 대표적으로 **국내 상장주식은 DB만 직접투자
  가능하고 DC/IRP는 투자금지**다 (doc56 chunk02/03, doc58 chunk02 교차검증 완료 — 과거 이 규칙이
  반대로 잘못 파싱된 채 남아있던 이력이 있어 원문을 재확인함). 사모펀드, 증권예탁증권(DR)도 DB만
  가능하다.
- 위험자산 70%(또는 TDF 100%) 한도와 별개로, 발행자/계열기업군 단위 집중투자한도가 추가로 적용된다.
"""

from dataclasses import dataclass, field
from enum import Enum

from src.rules.early_withdrawal import PlanType

RISKY_ASSET_LIMIT = 0.70          # 위험자산 한도 (DB/DC/IRP 공통)
# TDF 적격 특례는 doc58 원문상 "DC/IRP에만 적용, DB는 예상은퇴시점을 특정할 수 없어 해당사항 없음".
# DB에 0.70을 두는 것은 특례를 인정한다는 뜻이 아니라, 특례가 없어 일반 위험자산 한도가 그대로
# 적용된다는 의미다 (RISKY_ASSET_LIMIT과 같은 값이므로 판정 결과도 동일하다).
TDF_QUALIFIED_LIMIT = {PlanType.DB: 0.70, PlanType.DC: 1.00, PlanType.IRP: 1.00}


class RiskTier(Enum):
    SAFE = "안전자산"       # 한도 없이 100%까지 투자 가능
    RISKY = "위험자산"      # 70% 한도 (TDF 조건충족 시 DC/IRP는 100%)
    FORBIDDEN = "투자금지"  # 전액 투자 불가


# 대표 상품유형 -> 제도별 RiskTier (doc56 chunk02/03, doc58 chunk01/02 종합)
PRODUCT_RISK_TIER: dict[str, dict[PlanType, RiskTier]] = {
    "예금·적금": {PlanType.DB: RiskTier.SAFE, PlanType.DC: RiskTier.SAFE, PlanType.IRP: RiskTier.SAFE},
    "GIC(원리금보장보험계약)": {PlanType.DB: RiskTier.SAFE, PlanType.DC: RiskTier.SAFE, PlanType.IRP: RiskTier.SAFE},
    "RP(환매조건부매수)": {PlanType.DB: RiskTier.SAFE, PlanType.DC: RiskTier.SAFE, PlanType.IRP: RiskTier.SAFE},
    "국채·통안채·정부보증채": {PlanType.DB: RiskTier.SAFE, PlanType.DC: RiskTier.SAFE, PlanType.IRP: RiskTier.SAFE},
    "원리금지급ELB·DLB": {PlanType.DB: RiskTier.SAFE, PlanType.DC: RiskTier.SAFE, PlanType.IRP: RiskTier.SAFE},
    "MMF": {PlanType.DB: RiskTier.SAFE, PlanType.DC: RiskTier.SAFE, PlanType.IRP: RiskTier.SAFE},
    "보증형실적배당보험": {PlanType.DB: RiskTier.FORBIDDEN, PlanType.DC: RiskTier.FORBIDDEN, PlanType.IRP: RiskTier.SAFE},  # IRP 限 (doc58)

    "지방채": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "투자적격 특수채·회사채": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "투자적격 기업어음": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "투자적격 해외채권": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "상장리츠·인프라펀드": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "적격 해외상장주식": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "주식형·주식혼합형펀드": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "하이일드펀드": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "특별자산펀드·혼합자산펀드": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "ELS·DLS(공모, 최대손실40%이내)": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "위험회피목적 파생상품": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},
    "TDF(조건충족)": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.RISKY, PlanType.IRP: RiskTier.RISKY},  # 한도는 별도 예외 처리 (TDF_QUALIFIED_LIMIT)

    # DB만 가능 — doc56 chunk02/03, doc58 chunk02 교차검증
    "국내상장주식": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.FORBIDDEN, PlanType.IRP: RiskTier.FORBIDDEN},
    "사모펀드": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.FORBIDDEN, PlanType.IRP: RiskTier.FORBIDDEN},
    "증권예탁증권(DR)": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.FORBIDDEN, PlanType.IRP: RiskTier.FORBIDDEN},

    "비상장주식": {PlanType.DB: RiskTier.FORBIDDEN, PlanType.DC: RiskTier.FORBIDDEN, PlanType.IRP: RiskTier.FORBIDDEN},
    "전환사채·신주인수권부사채·교환사채·후순위채권": {PlanType.DB: RiskTier.RISKY, PlanType.DC: RiskTier.FORBIDDEN, PlanType.IRP: RiskTier.FORBIDDEN},
    "투자비적격 증권·채권": {PlanType.DB: RiskTier.FORBIDDEN, PlanType.DC: RiskTier.FORBIDDEN, PlanType.IRP: RiskTier.FORBIDDEN},
    "ELS·DLS(최대손실40%초과)": {PlanType.DB: RiskTier.FORBIDDEN, PlanType.DC: RiskTier.FORBIDDEN, PlanType.IRP: RiskTier.FORBIDDEN},
}


def get_risk_tier(product_type: str, plan_type: PlanType) -> RiskTier:
    """상품유형과 제도(DB/DC/IRP)를 받아 안전자산/위험자산/투자금지 여부를 판정한다."""
    if product_type not in PRODUCT_RISK_TIER:
        raise KeyError(f"등록되지 않은 상품유형입니다: {product_type}")
    return PRODUCT_RISK_TIER[product_type][plan_type]


@dataclass
class RiskAssetAllocationResult:
    limit_ratio: float          # 적용된 위험자산 한도 (0.70 또는 TDF 100%/70%)
    within_limit: bool
    is_tdf_exception: bool


def check_risk_asset_allocation(
    plan_type: PlanType,
    risky_asset_ratio: float,
    is_tdf_qualified: bool = False,
) -> RiskAssetAllocationResult:
    """포트폴리오의 위험자산 비중이 한도 이내인지 판정한다.

    risky_asset_ratio: 0~1 사이, 전체 적립금 중 위험자산(RiskTier.RISKY, TDF 포함)이 차지하는 비중.
    is_tdf_qualified: 감독원장이 정한 조건을 충족한 TDF에 '전액' 투자하는 경우 True.
      이 경우 DC/IRP는 100%, DB는 70%까지 허용된다 (doc58).
    """
    if not 0.0 <= risky_asset_ratio <= 1.0:
        raise ValueError("risky_asset_ratio는 0~1 사이여야 합니다")

    limit = TDF_QUALIFIED_LIMIT[plan_type] if is_tdf_qualified else RISKY_ASSET_LIMIT
    return RiskAssetAllocationResult(
        limit_ratio=limit,
        within_limit=risky_asset_ratio <= limit,
        is_tdf_exception=is_tdf_qualified,
    )


# ── 집중투자한도 (doc56 chunk02/03, doc58 chunk03 종합) ─────────────────
# 카테고리별 (DB, DC, IRP) 한도. None은 "투자금지"를 의미.
CONCENTRATION_LIMITS: dict[str, dict[PlanType, float | None]] = {
    "동일법인_증권": {PlanType.DB: 0.10, PlanType.DC: 0.30, PlanType.IRP: 0.30},
    "동일법인_지방채특수채": {PlanType.DB: 0.50, PlanType.DC: 0.30, PlanType.IRP: 0.30},
    "동일계열기업군_증권": {PlanType.DB: 0.15, PlanType.DC: 0.40, PlanType.IRP: 0.40},
    "사용자계열사_지분법적용사_발행증권": {PlanType.DB: None, PlanType.DC: 0.20, PlanType.IRP: 0.30},
    "자산관리기관_보증_원리금보장상품": {PlanType.DB: None, PlanType.DC: None, PlanType.IRP: None},
    "사용자_이해관계인_발행증권": {PlanType.DB: None, PlanType.DC: None, PlanType.IRP: None},
}


@dataclass
class ConcentrationLimitResult:
    limit_ratio: float | None   # None이면 투자 자체가 금지
    eligible: bool
    within_limit: bool


def check_concentration_limit(
    category: str,
    plan_type: PlanType,
    proposed_ratio: float,
) -> ConcentrationLimitResult:
    """발행자/계열기업군 단위 집중투자한도를 판정한다.

    category: CONCENTRATION_LIMITS의 키 (예: '동일법인_증권').
    proposed_ratio: 해당 발행자/계열기업군 증권에 배정하려는 비중 (0~1).
    """
    if category not in CONCENTRATION_LIMITS:
        raise KeyError(f"등록되지 않은 집중투자한도 카테고리입니다: {category}")
    if not 0.0 <= proposed_ratio <= 1.0:
        raise ValueError("proposed_ratio는 0~1 사이여야 합니다")

    limit = CONCENTRATION_LIMITS[category][plan_type]
    if limit is None:
        return ConcentrationLimitResult(limit_ratio=None, eligible=False, within_limit=False)

    return ConcentrationLimitResult(
        limit_ratio=limit,
        eligible=True,
        within_limit=proposed_ratio <= limit,
    )


# ── 종합 판정 (③ 상품 Agent용 check_product_pension_eligibility 툴의 근간) ──

@dataclass
class ProductEligibilityResult:
    eligible: bool
    risk_tier: RiskTier
    reasons: list[str] = field(default_factory=list)


def check_product_eligibility(
    product_type: str,
    plan_type: PlanType,
    current_risky_asset_ratio_after_purchase: float | None = None,
    is_tdf_qualified: bool = False,
) -> ProductEligibilityResult:
    """특정 상품을 특정 제도(DB/DC/IRP) 계좌에서 매수 가능한지 종합 판정한다.

    current_risky_asset_ratio_after_purchase를 넘기면, 이 상품 매수를 반영한 이후의
    전체 포트폴리오 위험자산 비중까지 함께 확인한다 (넘기지 않으면 상품유형 자체의
    투자가능 여부만 판정).
    """
    tier = get_risk_tier(product_type, plan_type)
    reasons: list[str] = []

    if tier == RiskTier.FORBIDDEN:
        reasons.append(f"'{product_type}'은(는) {plan_type.value} 제도에서 투자가 금지된 상품입니다.")
        return ProductEligibilityResult(eligible=False, risk_tier=tier, reasons=reasons)

    if tier == RiskTier.RISKY and current_risky_asset_ratio_after_purchase is not None:
        allocation = check_risk_asset_allocation(
            plan_type, current_risky_asset_ratio_after_purchase, is_tdf_qualified,
        )
        if not allocation.within_limit:
            reasons.append(
                f"매수 후 위험자산 비중({current_risky_asset_ratio_after_purchase:.0%})이 "
                f"{plan_type.value} 한도({allocation.limit_ratio:.0%})를 초과합니다."
            )
            return ProductEligibilityResult(eligible=False, risk_tier=tier, reasons=reasons)

    reasons.append(f"'{product_type}'은(는) {plan_type.value} 제도에서 투자 가능합니다 ({tier.value}).")
    return ProductEligibilityResult(eligible=True, risk_tier=tier, reasons=reasons)


# ── search_funds(투자설명서 DB) 검색 결과의 자유서술형 fund_category를 위험자산
# 비중 판정에 쓸 RiskTier로 분류한다. PRODUCT_RISK_TIER는 "국내상장주식" 같은
# 정형 상품유형 키를 쓰는데, DB의 fund_category는 "투자신탁, 증권(채권형), 개방형..."
# 같은 자유텍스트라 그 키와 직접 매칭되지 않는다 — 여기서만 쓰는 별도 분류다.
#
# 국공채 벤치마크를 쓰는 채권형 펀드는 안전자산이지만, fund_category 텍스트만으로는
# 크레딧(회사채) 펀드와 구분되지 않는다(예: "미래에셋퇴직플랜단기증권자투자신탁1호(채권)"는
# 회사채 포함 크레딧물, "한국투자 퇴직연금 증권 자투자신탁 1호(국공채)"는 국공채 전용 —
# 둘 다 fund_category는 "증권(채권형)"으로 동일하다). 판정이 불확실하면 안전 쪽으로
# 실패한다(RISKY로 간주) — 안전자산을 위험자산으로 오분류해 한도를 더 엄격히 적용하는
# 부작용은 있어도, 위험자산을 안전자산으로 오분류해 70% 한도 초과 추천을 놓치는 사고는
# 만들지 않는다.
_SAFE_FUND_MARKERS = ("국공채", "MMF", "머니마켓")
_RISKY_FUND_MARKERS = ("주식형", "주식파생형", "주식-파생형", "주식혼합", "혼합채권형", "파생형", "리츠", "인프라")


def classify_fund_category_risk_tier(fund_category: str | None, fund_name: str | None = None) -> RiskTier:
    """search_funds 결과 한 건의 fund_category(+fund_name)를 SAFE/RISKY로 분류한다.

    투자금지(FORBIDDEN) 판정은 여기서 하지 않는다 — search_funds가 검색하는 투자설명서
    DB는 전부 공모펀드이고, PRODUCT_RISK_TIER에서 DC/IRP FORBIDDEN인 상품(사모펀드·
    국내상장주식 직접투자 등)은 애초에 이 DB에 없다.
    """
    haystack = f"{fund_name or ''} {fund_category or ''}"
    if any(marker in haystack for marker in _SAFE_FUND_MARKERS):
        return RiskTier.SAFE
    if any(marker in haystack for marker in _RISKY_FUND_MARKERS):
        return RiskTier.RISKY
    # 채권형인데 국공채 표시가 없으면 크레딧물일 가능성을 배제할 수 없어 위험자산으로 본다.
    if "채권형" in haystack:
        return RiskTier.RISKY
    return RiskTier.RISKY
