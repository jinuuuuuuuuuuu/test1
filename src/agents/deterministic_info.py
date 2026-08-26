"""Deterministic answer generation for high-risk pension information questions.

분류(어느 카테고리인지)와 답변 생성(그 카테고리에서 실제 숫자·조건을 인용하는 것)의 역할을
분리한다 — 근거: "세액공제"+"얼마"만 있으면 무조건 한도질문으로 오분류하던 사고, 그리고
같은 클래스(주제어+느슨한 동반어)의 트리거가 "언제"/"시점"/"상품"/"세율" 등에서도 반복
재발할 위험이 있다는 판단(2026-08-25 재점검).

- candidate_category(): 1단계, 순수 키워드 규칙. 주제어(예: "세액공제")만 보고 "이 카테고리일
  수도 있다"는 느슨한 후보만 낸다. 결정적(같은 입력엔 항상 같은 결과)이고 빠르다.
- DETERMINISTIC_CATEGORIES: 후보로 나올 수 있는 전체 카테고리 목록 — router.py가
  RouterDecision.deterministic_category의 허용값을 정의할 때 이 목록을 그대로 쓴다.
- deterministic_response_for(category, question): 2단계, router의 LLM이 후보를 검증/확정한
  카테고리를 받아 실제 정형 답변(숫자·조건)을 만든다. 여기서부터는 결정론 그대로 유지한다 —
  숫자를 LLM에게 맡기면 학습 지식으로 틀린 값을 지어내는 사고가 실측된 바 있다.

왜 분류까지 전부 규칙으로 하지 않는가: 키워드 매칭은 "얼마"/"언제"/"시점"/"상품"/"세율"처럼
그 자체로 의미가 넓은 동반어를 요구하면 반드시 오탐(다른 의도의 질문을 잘못 걸러냄)을
만든다 — 이게 원래 사고의 근본 원인이었다. 반대로 분류를 통째로 LLM에게 맡기면(동반어 없이
전체 판단) 오탐은 줄지만 판단이 확률적이라 같은 입력에도 흔들릴 수 있고 실패 원인을 코드로
특정하기 어렵다. candidate_category()가 "주제어 존재 여부"라는 가장 안정적인 신호만
규칙으로 담당하고, 그 후보가 맞는지 최종 확정은 router의 LLM(이미 있는 호출에 필드만
추가돼 비용 증가 없음)에게 맡기는 하이브리드로 양쪽의 실패 모드를 서로 보완한다.
"""

from __future__ import annotations

from typing import Literal, Optional

from src.agents.state import RetrievedItem
from src.rules.comprehensive_tax import (
    ANNUAL_THRESHOLD,
    SEPARATE_TAXATION_RATE_OVER_THRESHOLD,
)
from src.rules.default_option import NOTICE_DELAY_DAYS_EXISTING, WAIT_DAYS_AFTER_NOTICE
from src.rules.in_kind_transfer import TRANSFER_BLOCK_CODES
from src.rules.retirement_tax_reduction import get_deferred_retirement_tax_rate
from src.rules.tax_credit import (
    COMBINED_CREDIT_LIMIT,
    CREDIT_RATE_HIGH,
    CREDIT_RATE_LOW,
    INCOME_THRESHOLD_COMPREHENSIVE,
    INCOME_THRESHOLD_SALARY,
    PENSION_SAVINGS_ONLY_LIMIT,
    TOTAL_CONTRIBUTION_LIMIT,
)
from src.rules.withdrawal_limit import (
    SIX_YEAR_EXCEPTION_CUTOFF,
    SIX_YEAR_EXCEPTION_START,
    UNLIMITED_FROM_YEAR,
)

DeterministicCategory = Literal[
    "세액공제_계산_입력부족",
    "세액공제_한도",
    "세금혜택_개요",
    "중도인출_일반",
    "디폴트옵션_자동매수",
    "실물이전_불가사유",
    "연금수령한도",
    "퇴직소득세감면",
    "연금소득세_종합과세",
    "해당없음",
]

# router.py가 프롬프트/RouterDecision의 Literal 정의에 그대로 재사용한다.
DETERMINISTIC_CATEGORIES: tuple[str, ...] = (
    "세액공제_계산_입력부족",
    "세액공제_한도",
    "세금혜택_개요",
    "중도인출_일반",
    "디폴트옵션_자동매수",
    "실물이전_불가사유",
    "연금수령한도",
    "퇴직소득세감면",
    "연금소득세_종합과세",
    "해당없음",
)


def candidate_categories(question: str) -> list[str]:
    """1단계: 주제어만 보고 느슨한 후보 목록을 낸다 (동반어 요구 없음).

    router의 LLM이 이 후보들 중 실제로 맞는 카테고리를 확정한다(또는 전부 기각하고
    "해당없음"). 여기서 여러 개가 동시에 후보로 나올 수 있다(예: "세액공제" 관련 두 카테고리).
    """
    text = _compact(question)
    candidates: list[str] = []

    if "세액공제" in text:
        candidates.append("세액공제_계산_입력부족")
        candidates.append("세액공제_한도")
    if any(word in text for word in ("세금혜택", "세제혜택", "절세혜택", "세금상혜택")):
        candidates.append("세금혜택_개요")
    if "중도인출" in text:
        candidates.append("중도인출_일반")
    if "디폴트옵션" in text:
        candidates.append("디폴트옵션_자동매수")
    if "실물이전" in text:
        candidates.append("실물이전_불가사유")
    if "연금수령한도" in text or ("연금" in text and "한도" in text):
        candidates.append("연금수령한도")
    if any(word in text for word in ("퇴직소득세", "이연퇴직소득세")):
        candidates.append("퇴직소득세감면")
    if any(word in text for word in ("연금소득세", "종합과세", "분리과세")):
        candidates.append("연금소득세_종합과세")

    return candidates


def deterministic_response_for(
    category: str, question: str
) -> Optional[tuple[str, list[RetrievedItem]]]:
    """2단계: router가 확정한 카테고리로 실제 정형 답변을 만든다.

    category가 "해당없음"이거나 핸들러가 없으면 None — 호출측(info_agent)은 LLM+툴 경로로
    진행해야 한다.
    """
    handler = _CATEGORY_HANDLERS.get(category)
    if handler is None:
        return None
    return handler(question)


def _compact(text: str) -> str:
    import re

    return re.sub(r"\s+", "", text or "")


def _won(amount: int) -> str:
    if amount % 10_000 == 0:
        return f"{amount // 10_000:,}만원"
    return f"{amount:,}원"


def _pct(rate: float) -> str:
    return f"{rate * 100:g}%"


def _context(source: str, content: str) -> list[RetrievedItem]:
    return [{"source": source, "content": content, "node": "info_agent"}]


def _tax_credit_calculation_missing_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc41 세액공제 계산 입력값 규칙"
    content = (
        f"세액공제액 계산에는 연금저축 납입액, IRP 납입액, 총급여 또는 종합소득금액이 필요합니다. "
        f"세액공제 대상 납입한도는 연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
        f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
        f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다."
    )
    draft = (
        "세액공제 금액은 납입액과 소득구간이 함께 있어야 계산할 수 있습니다.\n\n"
        "현재 질문에는 실제 계산에 필요한 입력값이 부족하므로, 세액공제액을 임의로 산출하지 않겠습니다.\n\n"
        "현재 답변 가능한 범위는 다음과 같습니다.\n"
        "- 세액공제 대상 한도는 연금저축 단독 연 600만원, 연금저축+IRP 합산 연 900만원입니다.\n"
        "- 세액공제율은 총급여 5,500만원 이하 또는 종합소득금액 4,500만원 이하이면 16.5%, "
        "그 초과 구간이면 13.2%입니다.\n\n"
        "정확한 계산을 위해 다음 정보를 한 번에 알려주세요.\n"
        "1. 올해 연금저축에 납입한 금액은 얼마인가요?\n"
        "2. 올해 IRP에 납입한 금액은 얼마인가요?\n"
        "3. 직장인이라면 총급여, 개인사업자라면 종합소득금액은 얼마인가요?\n"
        "4. 이미 회사 DC/IRP 추가납입 등 다른 연금계좌 납입액이 있다면 함께 알려주세요."
    )
    return draft, _context(source, content)


def _tax_benefit_overview_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc38~doc41 연금계좌 세금혜택 규칙"
    content = (
        f"연금계좌의 세금혜택은 납입 시 세액공제, 운용 중 과세이연, 연금수령 시 저율 과세, "
        f"퇴직금을 연금으로 받을 때 이연퇴직소득세 감면으로 정리됩니다. "
        f"세액공제 대상 납입한도는 연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
        f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
        f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다. "
        f"사적연금소득은 연 {_won(ANNUAL_THRESHOLD)} 초과 여부가 종합과세 판단 기준이며, "
        f"초과 시 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세를 선택할 수 있습니다. "
        f"연금소득세율은 만 55세 이상 70세 미만 5.5%, 70세 이상 80세 미만 4.4%, 80세 이상 3.3%입니다. "
        f"종신연금은 연령과 무관하게 3.3%입니다. "
        f"퇴직금을 연금으로 받을 때 이연퇴직소득세는 연금실제수령연차 1~10년차 30%, "
        f"11~20년차 40%, 21년차 이상 50% 감면됩니다."
    )
    draft = (
        "연금계좌의 세금혜택은 크게 네 가지로 볼 수 있습니다.\n\n"
        "1. 납입할 때 세액공제\n"
        "- 연금저축 단독 세액공제 대상 한도는 연 600만원입니다.\n"
        "- 연금저축과 IRP를 합치면 세액공제 대상 한도는 연 900만원입니다.\n"
        "- 세액공제율은 총급여 5,500만원 이하 또는 종합소득금액 4,500만원 이하이면 16.5%, "
        "그 초과 구간이면 13.2%입니다.\n\n"
        "2. 운용 중 과세이연\n"
        "- 계좌 안에서 발생한 운용수익에 대해 매년 바로 과세하지 않고, 나중에 인출할 때 과세하는 구조입니다.\n\n"
        "3. 연금으로 받을 때 낮은 세율 적용\n"
        "- 세액공제 받은 납입금과 운용수익을 연금으로 받으면 연령에 따라 5.5%, 4.4%, 3.3% 세율이 적용될 수 있습니다.\n"
        "- 종신연금은 연령과 무관하게 3.3%입니다.\n"
        "- 사적연금소득이 연 1,500만원을 초과하면 종합과세 또는 16.5% 분리과세 선택 문제가 생길 수 있습니다.\n\n"
        "4. 퇴직금을 연금으로 받을 때 퇴직소득세 감면\n"
        "- 퇴직금을 연금으로 받으면 이연퇴직소득세가 연금실제수령연차에 따라 30%, 40%, 50% 감면될 수 있습니다.\n\n"
        "정리하면, 연금계좌는 납입 시점에는 세액공제, 운용 중에는 과세이연, 수령 시점에는 저율 과세 또는 "
        "퇴직소득세 감면을 기대할 수 있는 구조입니다."
    )
    return draft, _context(source, content)


def _tax_credit_limit_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc41 세액공제 규칙"
    content = (
        f"연금저축+IRP 합산 납입한도는 연 {_won(TOTAL_CONTRIBUTION_LIMIT)}입니다. "
        f"세액공제 대상 납입한도는 연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        "연금저축 600만원과 IRP 900만원을 따로 더한 1,500만원이 세액공제 대상 한도가 아니라, "
        "연금저축 납입액을 포함한 IRP 합산 900만원이 세액공제 대상 한도입니다. "
        f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
        f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
        f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다. "
        f"최대 세액공제액은 {_won(COMBINED_CREDIT_LIMIT)} x {_pct(CREDIT_RATE_LOW)} = 148만 5천원, "
        f"{_won(COMBINED_CREDIT_LIMIT)} x {_pct(CREDIT_RATE_HIGH)} = 118만 8천원입니다."
    )
    draft = (
        "연금저축과 IRP를 합쳐서 볼 때 핵심은 **세액공제 대상 한도는 합산 900만원**이라는 점입니다.\n\n"
        "- 연금저축만 납입하는 경우: 세액공제 대상은 연 600만원까지\n"
        "- 연금저축 + IRP를 함께 납입하는 경우: 두 계좌 합산 세액공제 대상은 연 900만원까지\n"
        "- 단순 납입 가능 한도: 연금저축과 IRP 합산 연 1,800만원까지\n\n"
        "따라서 연금저축 600만원과 IRP 900만원을 각각 따로 더해 1,500만원까지 세액공제되는 구조가 "
        "아니라, **IRP 한도 900만원은 연금저축 납입액을 포함한 합산 한도**입니다.\n\n"
        "세액공제율은 소득 기준에 따라 달라집니다.\n"
        "- 총급여 5,500만원 이하 또는 종합소득금액 4,500만원 이하: 16.5%\n"
        "- 총급여 5,500만원 초과 또는 종합소득금액 4,500만원 초과: 13.2%\n\n"
        "합산 한도 900만원을 모두 채웠다면 최대 세액공제액은 16.5% 적용 시 148만 5천원, "
        "13.2% 적용 시 118만 8천원입니다."
    )
    return draft, _context(source, content)


def _early_withdrawal_general_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc46~doc50 중도인출 규칙"
    content = (
        "DB는 중도인출이 허용되지 않습니다. DC와 IRP는 법정 사유가 있으면 중도인출 대상 제도입니다. "
        "가능 사유는 6개월 이상 요양, 개인회생 또는 파산선고, 무주택자 전월세보증금, "
        "무주택자 주택구입, 재난피해입니다. 요양은 DC에서 직전 1년 의료비가 직전년도 연간임금총액의 "
        "12.5%를 초과해야 하며, IRP에는 이 비율 기준이 적용되지 않습니다. 개인회생·파산은 결정일 또는 "
        "선고일로부터 5년 이내 요건이 있습니다. 전월세보증금과 주택구입은 잔금지급일 또는 소유권 이전 "
        "등기일로부터 1개월 이내 신청 요건이 있습니다. 재난피해는 피해발생일로부터 3개월 이내가 원칙입니다."
    )
    draft = (
        "IRP는 중도인출이 가능한 제도이지만, 아무 때나 인출할 수 있는 것은 아니고 법정 사유가 필요합니다.\n\n"
        "대표적인 중도인출 사유는 다음과 같습니다.\n"
        "- 6개월 이상 요양\n"
        "- 개인회생 또는 파산선고\n"
        "- 무주택자 전월세보증금\n"
        "- 무주택자 주택구입\n"
        "- 재난피해\n\n"
        "참고로 DB형은 중도인출이 허용되지 않고, DC와 IRP가 중도인출 대상 제도입니다. "
        "각 사유별로 신청기한이나 추가 요건이 다릅니다. 예를 들어 개인회생·파산은 5년 이내 요건, "
        "전월세보증금·주택구입은 1개월 이내 신청 요건, 재난피해는 3개월 이내 신청 요건이 문제될 수 있습니다."
    )
    return draft, _context(source, content)


_EXISTING_MEMBER_WORDS = ("기존가입자", "기존 가입자", "기존가입", "만기가 된", "만기된", "만기 도래")
_NEW_MEMBER_WORDS = ("신규가입자", "신규 가입자", "신규가입", "최초 가입", "처음 가입", "신규 납입")


def _default_option_auto_purchase_response(question: str) -> tuple[str, list[RetrievedItem]]:
    """기존/신규가입자가 질문에 이미 특정돼 있으면 그 케이스만, 아니면 역질문+두 케이스를 함께 낸다.

    질문에 이미 없는 정보(정보 손실)는 아니지만, 특정된 사용자에게 무관한 케이스까지 섞어
    답하면 실제로 필요한 정보를 찾기 어려워진다 — 실물이전 사고와 같은 클래스(개인화 미반영)
    의 더 가벼운 변종.
    """
    source = "doc29 디폴트옵션 자동매수 규칙"
    content = (
        f"기존가입자는 상품 만기일로부터 4주({NOTICE_DELAY_DAYS_EXISTING}일) 후 통지하고, "
        f"통지 후 2주({WAIT_DAYS_AFTER_NOTICE}일) 대기 뒤 자동매수합니다. "
        f"신규가입자는 최초 부담금 납입 다음 영업일 통지하고, 통지 후 2주({WAIT_DAYS_AFTER_NOTICE}일) "
        "대기 뒤 자동매수합니다. 동일 상품 반복 만기 등 연속성이 유지되는 경우에는 통지·대기 없이 즉시 적용됩니다."
    )

    is_existing = any(word in question for word in _EXISTING_MEMBER_WORDS)
    is_new = any(word in question for word in _NEW_MEMBER_WORDS)

    if is_existing and not is_new:
        draft = (
            "기존가입자 기준으로 안내드리면, 디폴트옵션 자동매수는 상품 만기일로부터 4주(28일) 후 "
            "통지하고, 그 뒤 2주(14일) 대기한 뒤 이뤄집니다.\n\n"
            "다만 대기 중 전액을 다른 상품으로 이동해 연속성이 끊기면 다음 만기분부터 다시 통지와 대기 절차를 거칠 수 있습니다."
        )
    elif is_new and not is_existing:
        draft = (
            "신규가입자 기준으로 안내드리면, 디폴트옵션 자동매수는 최초 부담금 납입 다음 영업일에 "
            "통지하고, 그 뒤 2주(14일) 대기한 뒤 이뤄집니다."
        )
    else:
        draft = (
            "기존가입자인지 신규가입자인지에 따라 자동매수 시점이 달라 정확히 안내드리려면 "
            "어느 쪽에 해당하시는지 알려주시면 좋습니다. 다만 일반적으로는 다음과 같습니다.\n\n"
            "- 기존가입자: 상품 만기일로부터 4주(28일) 후 통지, 그 뒤 2주(14일) 대기 후 자동매수\n"
            "- 신규가입자: 최초 부담금 납입 다음 영업일 통지, 그 뒤 2주(14일) 대기 후 자동매수\n"
            "- 동일 상품 반복 만기처럼 연속성이 유지되는 경우: 통지·대기 없이 즉시 적용\n\n"
            "다만 대기 중 전액을 다른 상품으로 이동해 연속성이 끊기면 다음 만기분부터 다시 통지와 대기 절차를 거칠 수 있습니다."
        )
    return draft, _context(source, content)


def _in_kind_transfer_block_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc34 실물이전 불가사유 코드"
    definite_codes = [code for code, info in TRANSFER_BLOCK_CODES.items() if not info.get("directional")]
    manual_codes = [code for code, info in TRANSFER_BLOCK_CODES.items() if info.get("directional")]
    content = "확정 불가사유: " + "; ".join(
        f"{code}. {TRANSFER_BLOCK_CODES[code]['name']} - {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in definite_codes
    )
    content += " 추가 확인 필요 사유: " + "; ".join(
        f"{code}. {TRANSFER_BLOCK_CODES[code]['name']} - {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in manual_codes
    )
    # ⚠️ 답변 목록을 손으로 고른 하드코딩 리스트로 만들면 안 된다 — 실제로 그렇게 돼 있던
    # 동안 "21. 만기(상환)"이 목록에서 빠져, 만기 때문에 이전이 막힌 사용자에게 근거(content)
    # 에는 있는 사유가 답변 본문에서는 안 보이는 사고가 났다. 근거와 답변이 같은 원천
    # (TRANSFER_BLOCK_CODES)에서 나오도록 해서 둘이 어긋날 수 없게 한다.
    definite_lines = "\n".join(
        f"- {code}. {TRANSFER_BLOCK_CODES[code]['name']}: {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in definite_codes
    )
    manual_lines = "\n".join(
        f"- {code}. {TRANSFER_BLOCK_CODES[code]['name']}: {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in manual_codes
    )
    draft = (
        "퇴직연금 실물이전이 제한될 수 있는 상품·상황은 다음과 같습니다.\n\n"
        f"[실물이전 불가사유]\n{definite_lines}\n\n"
        f"[상대 금융기관 확인이 필요한 사유]\n{manual_lines}\n\n"
        "보유하신 상품이 어디에 해당하는지 알려주시면(예: MMF, 사모펀드, 만기 도래 여부, "
        "디폴트옵션 상품 여부) 실제 이전 가능 여부를 더 정확히 확인해 드릴 수 있습니다."
    )
    return draft, _context(source, content)


def _withdrawal_limit_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc39 연금수령한도 규칙"
    content = (
        "연금수령한도 = 연금계좌 평가액 ÷ (11 - 연금수령연차) × 120%입니다. "
        f"연금수령연차 {UNLIMITED_FROM_YEAR}년차 이상부터는 한도 자체가 사라집니다. "
        f"2013.3.1 이전 가입한 연금계좌는 {SIX_YEAR_EXCEPTION_START}년차부터 기산하는 특례가 있습니다. "
        "연금수령 요건은 가입기간 5년 이상, 만 55세 이후, 한도 이내 인출입니다."
    )
    draft = (
        "연금수령한도는 다음 공식으로 계산합니다.\n\n"
        "연금수령한도 = 연금계좌 평가액 ÷ (11 - 연금수령연차) × 120%\n\n"
        "다만 연금수령연차 11년차 이상부터는 한도가 없어져 전액 인출해도 연금수령으로 인정될 수 있습니다. "
        "또 2013.3.1 이전 가입한 연금계좌는 1년차가 아니라 6년차부터 기산하는 특례가 있습니다.\n\n"
        "구체적인 금액 계산을 하려면 연금계좌 평가액과 현재 연금수령연차가 필요합니다."
    )
    return draft, _context(source, content)


def _retirement_tax_reduction_response(question: str) -> tuple[str, list[RetrievedItem]]:
    r1 = get_deferred_retirement_tax_rate(1)
    r11 = get_deferred_retirement_tax_rate(11)
    r21 = get_deferred_retirement_tax_rate(21)
    source = "doc39~doc40 이연퇴직소득세 감면 규칙"
    content = (
        f"연금실제수령연차 1~10년차는 이연퇴직소득세의 {_pct(r1.payment_ratio)}를 납부하고 "
        f"{_pct(r1.reduction_ratio)}를 감면합니다. 11~20년차는 {_pct(r11.payment_ratio)}를 납부하고 "
        f"{_pct(r11.reduction_ratio)}를 감면합니다. 21년차 이상은 {_pct(r21.payment_ratio)}를 납부하고 "
        f"{_pct(r21.reduction_ratio)}를 감면합니다. 연금외수령은 감면 없이 전액 납부합니다."
    )
    draft = (
        "퇴직금을 연금으로 수령하면 이연퇴직소득세가 연차에 따라 감면됩니다.\n\n"
        "- 연금실제수령연차 1~10년차: 이연퇴직소득세의 70% 납부, 30% 감면\n"
        "- 연금실제수령연차 11~20년차: 60% 납부, 40% 감면\n"
        "- 연금실제수령연차 21년차 이상: 50% 납부, 50% 감면\n\n"
        "주의할 점은 여기서 쓰는 기준이 '연금수령연차'가 아니라 실제로 인출한 해만 세는 "
        "'연금실제수령연차'라는 점입니다. 연금외수령이면 감면 없이 이연퇴직소득세 전액을 납부합니다."
    )
    return draft, _context(source, content)


def _pension_income_tax_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc38 연금소득 종합과세 규칙"
    content = (
        f"사적연금소득은 연 {_won(ANNUAL_THRESHOLD)} 초과 여부가 종합과세 판단 기준입니다. "
        f"초과 시 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세를 선택할 수 있습니다. "
        "1,500만원 판정에는 세액공제 받은 납입금과 운용수익 재원만 포함하고, 세액공제 안 받은 원금과 퇴직금은 제외합니다. "
        "1,500만원 이내 연금소득세율은 만 55세 이상 70세 미만 5.5%, 70세 이상 80세 미만 4.4%, 80세 이상 3.3%입니다."
    )
    draft = (
        "사적연금소득의 종합과세 여부는 연 1,500만원 초과 여부가 기준입니다.\n\n"
        "- 연 1,500만원 이내: 연령별 연금소득세율로 분리과세\n"
        "- 연 1,500만원 초과: 종합과세 또는 16.5% 분리과세 선택 가능\n\n"
        "다만 이 1,500만원 기준에는 세액공제 받은 납입금과 운용수익 재원만 포함됩니다. "
        "세액공제 받지 않은 원금과 퇴직금 재원은 이 판정에서 제외됩니다.\n\n"
        "1,500만원 이내일 때 연금소득세율은 만 55세 이상 70세 미만 5.5%, "
        "70세 이상 80세 미만 4.4%, 80세 이상 3.3%입니다."
    )
    return draft, _context(source, content)


_CATEGORY_HANDLERS = {
    "세액공제_계산_입력부족": _tax_credit_calculation_missing_response,
    "세액공제_한도": _tax_credit_limit_response,
    "세금혜택_개요": _tax_benefit_overview_response,
    "중도인출_일반": _early_withdrawal_general_response,
    "디폴트옵션_자동매수": _default_option_auto_purchase_response,
    "실물이전_불가사유": _in_kind_transfer_block_response,
    "연금수령한도": _withdrawal_limit_response,
    "퇴직소득세감면": _retirement_tax_reduction_response,
    "연금소득세_종합과세": _pension_income_tax_response,
}
