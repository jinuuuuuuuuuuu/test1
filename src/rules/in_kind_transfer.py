"""실물이전 판정기 — 근거: doc34 (실물이전 불가사유 안내, 25개 코드 + 99.기타)

doc34는 코드표 성격이 강해 원문 그대로 구조화 테이블로 옮기고, 불리언 조건으로 판정 가능한
항목은 checker 함수로, 상대기관 정보가 필요해 단순 불리언으로 못 미치는 항목(06, 17, 20)은
'directional'로 표시해 텍스트 안내만 반환한다 (RAG/상담원 판단 보조용).
"""

from dataclasses import dataclass

TRANSFER_BLOCK_CODES: dict[str, dict] = {
    "01": {"name": "소규모 펀드 임의해지", "desc": "보유 펀드가 소규모 펀드(잔고 50억 미만) 임의해지 대상인 경우", "flag": "small_fund_forced_liquidation"},
    "02": {"name": "언번들계약", "desc": "자산관리기관과 운용관리기관이 서로 다른 플랜인 경우", "flag": "unbundled_contract"},
    "03": {"name": "사모펀드", "desc": "특정 투자자 대상으로만 모집하는 사모펀드인 경우", "flag": "private_fund"},
    "04": {"name": "MMF", "desc": "MMF인 경우", "flag": "is_mmf"},
    "05": {"name": "환매수수료 존재", "desc": "환매수수료가 부과되는 펀드인 경우", "flag": "has_redemption_fee"},
    "06": {"name": "상품제공수수료 존재", "desc": "상품제공수수료 계약이 있는 상품 — 이관은 가능하나 수관기관의 미협약 시 수관 불가", "directional": True},
    "07": {"name": "운용지시 진행중", "desc": "사전조회 시 매도지시 있으면 해당 상품 불가, 이전접수 시 매수/매도지시 있으면 계좌 이전접수 자체 불가", "flag": "pending_trade_instruction"},
    "08": {"name": "압류 및 질권", "desc": "압류 또는 질권이 설정된 계좌인 경우", "flag": "seized_or_pledged"},
    "09": {"name": "만기매칭형 펀드", "desc": "만기매칭형 펀드인 경우", "flag": "maturity_matching_fund"},
    "10": {"name": "지분증권/리츠", "desc": "지분증권 또는 리츠 — 권리 처리 문제로 실물이전 불가", "flag": "equity_or_reit"},
    "11": {"name": "환매조건부채권(RP)", "desc": "RP인 경우", "flag": "is_rp"},
    "12": {"name": "발행어음", "desc": "발행어음인 경우", "flag": "is_promissory_note"},
    "13": {"name": "금리연동형 보험", "desc": "자산관리회사가 보험사인 플랜에서 현금성자산 대용으로 매수한 금리연동형 보험", "flag": "rate_linked_insurance"},
    "14": {"name": "실적배당형 보험", "desc": "실적배당형 보험 (보험사의 이율보증형상품 GIC만 예외적으로 실물이전 가능)", "flag": "variable_insurance"},
    "15": {"name": "원금비보장 파생결합사채", "desc": "원리금보장 ELB·DLB 이외의 파생결합사채인 경우", "flag": "principal_at_risk_derivative_bond"},
    "16": {"name": "규약 미체결", "desc": "수관받는 금융기관에 DC 규약이 체결돼 있지 않은 경우", "flag": "no_dc_agreement_at_receiver"},
    "17": {"name": "상품라인업", "desc": "이관/수관 상대기관의 퇴직연금 상품라인업에 해당 상품이 없는 경우 (이관불가/수관불가 방향성 구분 필요)", "directional": True},
    "18": {"name": "한도초과", "desc": "저축은행예금 예금자보호한도(1억원, 주민번호 기준 저축은행별 합산) 초과인 경우", "flag": "exceeds_savings_bank_protection_limit"},
    "19": {"name": "자사상품 편입", "desc": "수관기관 자사의 원리금보장상품인 경우 (자사상품 수관 불가)", "flag": "own_company_product_at_receiver"},
    "20": {"name": "사용자/가입자부담금 미분리", "desc": "재원(사용자/가입자부담금)이 구분관리되지 않는 상품 — 이관 자체를 막지는 않으며 미구분으로 제공, 수관 여부는 수관기관 판단", "directional": True},
    "21": {"name": "만기(상환)", "desc": "보유 상품의 만기가 이미 도래한 경우", "flag": "already_matured"},
    "22": {"name": "환매불가", "desc": "환매가 불가한 상품인 경우 (예: 러시아/동유럽 기초자산 펀드·ETF, ELB·DLB 환매불가기간)", "flag": "redemption_restricted"},
    "23": {"name": "실물이전불가(디폴트옵션)", "desc": "디폴트옵션 상품인 경우 — 실물이전 자체가 불가", "flag": "is_default_option_product"},
    "24": {"name": "상장투자회사", "desc": "상장투자회사(맥쿼리인프라 등)인 경우", "flag": "listed_investment_company"},
    "25": {"name": "상품협약(위탁계약) 미체결", "desc": "수관기관과 해당 상품 제공기관 간 상품협약(위탁계약)이 체결되지 않은 경우", "flag": "no_product_agreement_at_receiver"},
}

# checker 함수가 직접 판정 가능한 코드만 추출 (directional 제외)
_BOOLEAN_CHECKABLE = {v["flag"]: code for code, v in TRANSFER_BLOCK_CODES.items() if not v.get("directional")}


@dataclass
class TransferEligibilityResult:
    eligible: bool
    blocking_codes: list[str]              # 확정적으로 판정된 불가사유 코드
    needs_manual_review_codes: list[str]   # directional이라 추가 정보(상대기관 확인 등)가 필요한 코드


def check_transfer_eligibility(**flags: bool) -> TransferEligibilityResult:
    """실물이전 가능 여부를 판정한다.

    flags에는 TRANSFER_BLOCK_CODES의 'flag' 값(예: is_mmf, private_fund, seized_or_pledged 등)을
    키로 하는 불리언을 넘긴다. True인 항목만 불가사유로 집계한다.
    directional 항목(06, 17, 20)은 불리언만으로 확정 판정이 안 되므로, 항상
    needs_manual_review_codes에 안내용으로 포함한다.
    """
    blocking = []
    for flag_name, value in flags.items():
        if value and flag_name in _BOOLEAN_CHECKABLE:
            blocking.append(_BOOLEAN_CHECKABLE[flag_name])

    manual_review = [code for code, v in TRANSFER_BLOCK_CODES.items() if v.get("directional")]

    return TransferEligibilityResult(
        eligible=len(blocking) == 0,
        blocking_codes=sorted(blocking),
        needs_manual_review_codes=manual_review,
    )


def get_code_info(code: str) -> dict:
    """코드로 불가사유 상세(이름/설명)를 조회한다."""
    if code not in TRANSFER_BLOCK_CODES:
        raise KeyError(f"알 수 없는 실물이전 불가사유 코드입니다: {code}")
    return TRANSFER_BLOCK_CODES[code]
