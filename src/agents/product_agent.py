"""③ 상품 Agent 노드 — 펀드/상품 추천·비교·적합성 판정.

HCX-005 + check_product_pension_eligibility/search_funds/get_fund_detail(PRODUCT_AGENT_TOOLS)를
create_react_agent로 묶는다. 복합형 질문(②→③ 순차 실행)일 때는 state["retrieved_context"]에
②가 이미 채워둔 제도 근거가 들어있으므로, 그 내용을 시스템 프롬프트 컨텍스트로 함께 넘겨 ③이
다시 RAG/제도문서를 검색하지 않고도 참고할 수 있게 한다 (설계 결정, 2026-08-11 — ③은
search_pension_docs를 별도로 갖지 않는다).

⚠️ HCX-007은 기본적으로 Thinking이 켜져 있어 thinking={"effort":"none"}으로 끄지 않으면
bind_tools()가 400 "tools, reasoning" 에러를 낸다(네이버 공식 문서: Function calling과
추론(Thinking)은 동시 이용 불가). thinking을 꺼도 HCX-007로 tool calling은 가능하지만,
여기서는 Thinking 부가 설정 없이 바로 되는 HCX-005를 택했다 — 필요하면
get_llm("HCX-007", thinking_effort="none")으로 교체 가능.
"""

import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.context import (
    build_repair_note,
    build_retrieved_context,
    build_tool_trace,
    dedupe_context,
    history_to_messages,
    split_clarification_marker,
)
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState, RetrievedItem
from src.agents.tools import PRODUCT_AGENT_TOOLS, search_funds

PRODUCT_AGENT_MODEL = "HCX-005"

PRODUCT_AGENT_SYSTEM_PROMPT = """당신은 연금 상품(펀드) 추천 에이전트입니다.

서비스 범위: 연금계좌(DB/DC/IRP·연금저축)에서 투자하는 상품(펀드)의 설명·비교·추천만
다룹니다. 범위 밖 주제(일반 주식 종목, 부동산, 연금 외 금융상품 등)는 학습 지식으로 답하지
말고 "본 서비스의 상담 범위를 벗어난다"고 한계를 고지하세요. [범위 안내]가 함께 주어지면
범위 밖 부분은 한계를 밝히고, 안내된 연금 관점으로만 답하세요.

절대 규칙 — 단정적 추천 금지: "좋은 상품 추천해줘", "괜찮은 연금상품 3개 추천해줘"처럼
계좌유형(DB/DC/IRP)·투자기간·위험선호·금액 등 구체적 조건이 없는 막연한 요청에는 상품을
지어내서 추천하지 마세요. 이 경우 search_funds/check_product_pension_eligibility 등 툴을
호출하지 말고, 답변을 확인이 필요한 조건을 묻는 역질문으로 작성하세요(예: "투자 가능한
계좌유형과 감내 가능한 위험 수준을 알려주시면 후보를 좁혀드릴게요"). 이런 역질문 답변은
반드시 [추가 확인 필요] 로 시작하세요.

역질문은 첫 답변 안에서 필요한 항목 전체를 한꺼번에 물으세요. 평가 환경은 단일턴이므로
"한 달에 얼마를 투자할 예정인가요?"처럼 한 항목만 묻고 종료하면 안 됩니다. 계좌유형·위험선호·
투자기간·투자금액·투자목적 중 부족한 항목을 모두 나열하고, 부족한 상태에서는 특정 펀드명이나
상품코드를 임의로 추천하지 마세요. 단정적 표현("이 상품이 가장 좋습니다")은 쓰지 마세요.

조건이 충분한 경우의 진행 순서:
1. search_funds로 조건에 맞는 후보를 찾으세요 (risk_grade/keyword 등 실제 질문에서 나온
   조건만 사용하고, 없는 조건을 임의로 지어내지 마세요).
2. 특정 상품유형(국내 상장주식·사모펀드·증권예탁증권 등)을 계좌유형과 함께 추천하려면
   check_product_pension_eligibility로 그 계좌(DB/DC/IRP)에서 투자 가능한지 먼저
   확인하세요 — product_type은 후보 상품의 실제 분류값만 사용하고, 사용자의 막연한
   표현(예: "괜찮은 상품")을 그대로 product_type에 넣지 마세요.
3. 상세 설명이나 비교가 필요하면 get_fund_detail로 수치 정보(보수/수익률/AUM 등)를
   가져오고, 투자전략·투자위험 같은 서술 설명이 필요하면 search_prospectus_text를
   product_code로 한정해 호출하세요 — 서술 내용을 기억으로 지어내지 마세요.

툴 호출 결과에 error가 있으면 그 오류를 사용자에게 노출하지 말고, [추가 확인 필요] 로
시작하는 조건 재확인 질문으로 답하세요.

이전 대화가 함께 주어지면 "그중 두 번째 상품", "방금 조건대로" 같은 지시어를 이전 턴 내용으로
풀어서 이해하세요. 다만 이전 턴에서 언급된 상품 정보를 그대로 베끼지 말고, 필요하면
search_funds/get_fund_detail로 다시 확인하세요."""


_RECOMMENDATION_WORDS = ("추천", "뭐 사면", "뭐살", "투자하면 좋", "맞는 상품", "상품 알려", "펀드 알려")
_SPECIFIC_RECOMMENDATION_WORDS = ("상품추천", "구체", "실제 상품", "펀드명", "상품명")
_SPECIFIC_PRODUCT_WORDS = ("어때", "위험", "수수료", "보수", "수익률", "설명", "분석")
_COMPARISON_WORDS = ("비교", "차이", " vs ", "VS")
_REFERENCE_WORDS = ("그중", "방금", "앞에서", "위에서", "두 번째", "첫 번째", "이 상품", "이 펀드")
_PROFILE_FIELD_LABELS = {
    "account_type": "계좌유형",
    "risk_profile": "위험성향",
    "investment_horizon": "투자기간",
    "monthly_investment": "투자금액",
    "investment_goal": "투자목적",
}


# ── 안전성 가드 ─────────────────────────────────────────────────────────
# 라우터에서 무조건 차단하던 질문들을 통과시키도록 바꾼 뒤(전제를 교정해서 답하기 위해),
# 그 질문들이 상품 Agent의 일반 경로로 흘러 위험한 답을 내는 사례가 실측됐다.
# 실측 사례: "원금 보장되니까 가장 수익률 높은 상품에 전액 투자해도 되죠?" -> 전제만 부정하고
# 정작 '매우 높은 위험' 등급 주식형 펀드를 추천, '전액 투자'는 교정하지 않음.

_FORECAST_TIME_WORDS = ("내년", "앞으로", "향후", "미래", "전망", "다음해", "내후년")
_FORECAST_TARGET_WORDS = ("수익", "오를", "오른다", "떨어질", "얼마나벌", "상승")

# 계좌 자체가 원금을 보장한다는 잘못된 전제 (원리금보장'형 상품'을 찾는 정당한 요청과 구분해야 한다)
_PRINCIPAL_PREMISE_PATTERNS = (
    "원금이무조건보장", "원금은무조건보장", "원금이보장되는계좌", "원금보장되는계좌",
    "원금이보장되니까", "원금보장이니까", "원금손실이없는계좌", "원금손실없는계좌",
    "원금이무조건", "무조건보장되는",
)
# 이 표현들이 있으면 상품 유형을 찾는 정당한 요청이므로 전제 교정 가드를 태우지 않는다.
_PRINCIPAL_PRODUCT_WORDS = ("원금보장형", "원리금보장", "원리금지급")

_CONCENTRATION_WORDS = ("전액투자", "전부투자", "몰빵", "올인", "전액을투자", "다넣")


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _is_forecast_question(text: str) -> bool:
    compact = _compact_text(text)
    return any(w in compact for w in _FORECAST_TIME_WORDS) and any(
        w in compact for w in _FORECAST_TARGET_WORDS
    )


def _forecast_response(state: PensionAgentState) -> tuple[str, list[RetrievedItem], dict, bool]:
    draft = (
        "먼저 짚어드릴 점이 있습니다. 특정 상품의 앞으로의 수익률은 누구도 예측할 수 없고, "
        "투자설명서에도 미래 수익률을 알려주는 정보는 담겨 있지 않습니다. 과거 수익률은 미래 "
        "수익을 보장하지 않으며, 확정 수익률을 제시하는 것은 가능하지도 적절하지도 않습니다.\n\n"
        "대신 보유 자료로 확인해 드릴 수 있는 것은 다음과 같습니다.\n"
        "- 과거 수익률(최근 1년·3년·설정 이후)\n"
        "- 위험등급과 변동성\n"
        "- 총보수 등 비용 구조\n"
        "- 투자전략과 투자위험 설명\n\n"
        "확인하고 싶은 상품의 상품명 또는 상품코드를 알려주시면 위 항목들을 정리해 드리겠습니다."
    )
    return draft, [], dict(state.get("recommendation_profile") or {}), True


def _has_principal_guarantee_premise(text: str) -> bool:
    compact = _compact_text(text)
    if any(word in compact for word in _PRINCIPAL_PRODUCT_WORDS):
        return False
    return any(pattern in compact for pattern in _PRINCIPAL_PREMISE_PATTERNS)


def _has_concentration_request(text: str) -> bool:
    return any(word in _compact_text(text) for word in _CONCENTRATION_WORDS)


# "원금 보장 + 높은 수익"을 동시에 요구 — 존재할 수 없는 조합이라 전제를 짚어야 한다.
# 단, "원금 손실은 싫은데 그래도 수익률 좋은"처럼 선호를 말한 것은 정상적인 위험회피
# 표현이므로 걸리면 안 된다 ('보장'을 단정하는 표현만 대상으로 삼는다).
_GUARANTEE_ASSERTION_WORDS = ("원금보장", "원금이보장", "원금은보장", "원금을보장", "손실없이", "손실이없")
_HIGH_RETURN_WORDS = ("수익률도높", "수익률높", "고수익", "수익률좋", "수익률도좋", "많이오를")


def _is_guaranteed_high_return_request(text: str) -> bool:
    compact = _compact_text(text)
    if any(word in compact for word in _PRINCIPAL_PRODUCT_WORDS):
        return False
    return any(w in compact for w in _GUARANTEE_ASSERTION_WORDS) and any(
        w in compact for w in _HIGH_RETURN_WORDS
    )


def _principal_guarantee_response(
    state: PensionAgentState, concentration: bool, tradeoff: bool = False
) -> tuple[str, list[RetrievedItem], dict, bool]:
    if tradeoff:
        draft = (
            "먼저 전제를 짚어드려야 할 것 같습니다. 원금이 확실히 보장되면서 동시에 수익률도 "
            "높은 상품은 존재하지 않습니다. 기대수익과 위험은 함께 움직여서, 원금을 보장하는 "
            "구조일수록 기대할 수 있는 수익은 낮아집니다.\n\n"
            "- 원리금보장형 상품(예금 등)은 약정된 원리금을 기대할 수 있는 대신 수익률이 제한적입니다.\n"
            "- 실적배당형 상품(펀드 등)은 더 높은 수익을 기대할 수 있는 대신 원금 손실이 "
            "발생할 수 있습니다.\n\n"
        )
    else:
        draft = (
            "먼저 전제를 바로잡아야 할 것 같습니다. IRP나 연금저축 같은 연금계좌는 계좌 자체가 "
            "원금을 보장하는 상품이 아닙니다. 연금계좌는 여러 상품을 담는 그릇에 가깝고, 원금 "
            "보장 여부는 그 안에서 어떤 상품을 운용하느냐에 따라 달라집니다.\n\n"
            "- 원리금보장형 상품(예금 등)을 담으면 약정된 원리금을 기대할 수 있습니다.\n"
            "- 실적배당형 상품(펀드 등)을 담으면 운용 성과에 따라 원금 손실이 발생할 수 있습니다.\n\n"
        )
    if concentration:
        draft += (
            "그래서 '수익률이 가장 높은 상품에 전액 투자'는 권해 드리기 어렵습니다. 과거 수익률이 "
            "높았다는 것이 앞으로도 높다는 뜻은 아니고, 한 상품에 전액을 집중하면 그 상품이 부진할 때 "
            "이를 완충할 수단이 없습니다. 연금은 장기간 운용하는 자산이라 손실이 났을 때 회복할 시간과 "
            "여력이 함께 고려되어야 합니다.\n\n"
        )
    draft += (
        "감내 가능한 위험 수준과 투자기간을 알려주시면, 그 조건에 맞는 상품 유형과 후보를 "
        "위험등급·총보수·과거 수익률과 함께 비교해 안내해 드리겠습니다."
    )
    return draft, [], dict(state.get("recommendation_profile") or {}), True


def _combined_user_text(state: PensionAgentState) -> str:
    history = state.get("conversation_history") or []
    previous_user_text = "\n".join(turn.get("question", "") for turn in history)
    return f"{previous_user_text}\n{state['question']}"


def _has_account_type(text: str) -> bool:
    return any(token in text.upper() for token in ("IRP", "DC", "DB")) or "연금저축" in text


def _is_product_recommendation(text: str) -> bool:
    return any(token in text for token in ("추천", "상품", "펀드", "굴릴만한"))


def _is_recommendation_intent(text: str) -> bool:
    return any(word in text for word in _RECOMMENDATION_WORDS) or (
        ("상품" in text or "펀드" in text) and any(word in text for word in ("좋", "맞", "알려"))
    )


def _is_specific_recommendation_request(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return compact == "상품추천" or any(word in text for word in _SPECIFIC_RECOMMENDATION_WORDS)


def _is_specific_product_or_comparison(text: str) -> bool:
    if any(word in text for word in _COMPARISON_WORDS):
        return True
    return ("펀드" in text or "상품" in text) and any(word in text for word in _SPECIFIC_PRODUCT_WORDS)


def _requires_account_eligibility_check(text: str) -> bool:
    """계좌유형과 투자 제한이 있는 상품유형을 함께 물으면 정형 응답 대신 LLM+툴로 넘긴다.

    실측: "IRP로 사모펀드 투자 가능한가요? 추천해주세요"가 사모펀드 언급 없이 TDF/채권혼합형을
    추천했다 — 퇴직연금계좌(IRP/DC)는 상장주식·사모펀드 등 투자 제한이 있는 상품유형이 있는데,
    정형 응답은 이 제도적 제약을 모르고 일반 추천으로 답해버린다. 이런 조합은 규칙엔진/RAG
    툴로 실제 투자 가능 여부부터 확인해야 하므로 여기서 가로채지 않고 None을 반환한다.
    """
    return _has_account_type(text) and any(
        product_type in text
        for product_type in ("상장주식", "사모펀드", "증권예탁증권", "위험자산 100%", "위험자산100%")
    )


def _is_context_reference_without_target(state: PensionAgentState) -> bool:
    text = state["question"]
    if state.get("conversation_history"):
        return False
    if "코드:" in text or "상품코드" in text:
        return False
    return any(word in text for word in _REFERENCE_WORDS)


def _context_reference_response(state: PensionAgentState) -> tuple[str, list[RetrievedItem], dict, bool] | None:
    if not _is_context_reference_without_target(state):
        return None
    questions = [
        "어떤 상품을 말하는지 상품명 또는 상품코드를 알려주세요.",
        "비교 질문이라면 비교할 상품명 또는 상품코드를 모두 알려주세요.",
        "원하시는 확인 항목이 위험, 보수, 수익률, 투자전략 중 무엇인지 알려주세요.",
    ]
    draft = (
        "현재 평가 호출에는 이전 대화 내용이 함께 제공되지 않아, 질문의 '그중' 또는 '이 펀드'가 "
        "어떤 상품을 가리키는지 확인할 수 없습니다.\n\n"
        "상품을 추측해서 답하면 다른 펀드의 위험·보수·수익률을 잘못 안내할 수 있으므로, "
        "특정 상품에 대한 판단은 보류하겠습니다.\n\n"
        "정확한 답변을 위해 다음 정보를 한 번에 알려주세요.\n"
        + "\n".join(f"{i}. {question}" for i, question in enumerate(questions, start=1))
    )
    return draft, [], dict(state.get("recommendation_profile") or {}), True


def _extract_recommendation_profile(state: PensionAgentState) -> dict:
    profile = dict(state.get("recommendation_profile") or {})
    text = _combined_user_text(state)
    current = state["question"]

    account = _extract_account_type(text)
    if account:
        profile["account_type"] = account

    monthly = _extract_amount(current, allow_standalone=True) or _extract_amount(text)
    if monthly:
        profile["monthly_investment"] = monthly

    horizon = _extract_horizon(current) or _extract_horizon(text)
    if horizon:
        profile["investment_horizon"] = horizon

    risk = _extract_risk_profile(current) or _extract_risk_profile(text)
    if risk:
        profile["risk_profile"] = risk

    loss = _extract_loss_tolerance(current) or _extract_loss_tolerance(text)
    if loss:
        profile["loss_tolerance"] = loss

    goal = _extract_goal(current) or _extract_goal(text)
    if goal:
        profile["investment_goal"] = goal

    age = _extract_age(current) or _extract_age(text)
    if age:
        profile["age_or_retirement_horizon"] = age

    preferred_type = _extract_preferred_product_type(current) or _extract_preferred_product_type(text)
    if preferred_type:
        profile["preferred_product_type"] = preferred_type

    return profile


def _extract_account_type(text: str) -> str | None:
    upper = text.upper()
    for token in ("IRP", "DC", "DB"):
        if token in upper:
            return token
    if "연금저축" in text:
        return "연금저축"
    return None


# "연봉 5천만원"처럼 소득을 가리키는 금액은 투자금액으로 잡으면 안 된다.
_INCOME_CONTEXT_WORDS = ("연봉", "총급여", "종합소득", "소득금액", "연소득", "급여")


def _is_income_amount(text: str, match_start: int) -> bool:
    return any(word in text[max(0, match_start - 8):match_start] for word in _INCOME_CONTEXT_WORDS)


def _extract_amount(text: str, allow_standalone: bool = False) -> str | None:
    # 억/천만/백만/만 단위 결합("5천만원", "1억원")까지 인식 — "만원"만 잡던 이전 정규식은
    # "5천만원" 같은 결합 단위를 놓쳤다.
    monthly_match = re.search(
        r"(월|한\s*달|매달)\s*(\d+)\s*(억|천만|백만|만)\s*원?\s*(이상|정도|쯤|가량)?", text
    )
    if monthly_match:
        suffix = f" {monthly_match.group(4)}" if monthly_match.group(4) else ""
        return f"월 {monthly_match.group(2)}{monthly_match.group(3)}원{suffix}"

    if not (allow_standalone or any(word in text for word in ("투자", "납입", "가능", "넣", "불입"))):
        return None
    for match in re.finditer(r"(\d+)\s*(억|천만|백만|만)\s*원?\s*(이상|정도|쯤|가량)?", text):
        if _is_income_amount(text, match.start()):
            continue
        suffix = f" {match.group(3)}" if match.group(3) else ""
        return f"월 {match.group(1)}{match.group(2)}원{suffix}"
    return None


def _extract_horizon(text: str) -> str | None:
    match = re.search(r"(\d+)\s*년\s*(이상|정도|간|동안)?", text)
    if match:
        suffix = f" {match.group(2)}" if match.group(2) else ""
        return f"{match.group(1)}년{suffix}"
    if any(word in text for word in ("장기", "오래", "많이 남", "은퇴까지 오래")):
        return "장기"
    if "중기" in text:
        return "중기"
    if "단기" in text:
        return "단기"
    return None


def _extract_risk_profile(text: str) -> str | None:
    if any(
        word in text
        for word in (
            "안정형", "보수", "안정적", "크게 잃지", "손실 싫", "낮은 위험", "위험등급 낮",
            "손실을 최대한", "손실을 줄이", "손실 최소화", "손실은 최대한",
        )
    ):
        return "안정형"
    if any(word in text for word in ("중립형", "중립", "약간의 변동성", "어느 정도", "중간")):
        return "중립형"
    if any(
        word in text
        for word in (
            "공격형", "공격", "적극", "높은 수익", "고위험", "수익성을 추구", "수익성 추구", "수익률을 추구",
        )
    ):
        return "공격형"
    return None


def _extract_loss_tolerance(text: str) -> str | None:
    if "크게 잃지" in text or ("손실" in text and any(word in text for word in ("싫", "피", "낮", "최대한", "줄이"))):
        return "큰 손실 회피"
    if "약간의 변동성" in text:
        return "약간의 변동성 감수"
    return None


def _extract_goal(text: str) -> str | None:
    if any(word in text for word in ("노후", "은퇴", "연금", "IRP", "DC", "DB")):
        return "노후/은퇴 준비"
    if "절세" in text or "세액공제" in text:
        return "절세"
    if "목돈" in text:
        return "목돈 마련"
    return None


def _extract_age(text: str) -> str | None:
    match = re.search(r"(\d{2})\s*세", text)
    if match:
        return f"{match.group(1)}세"
    decade_match = re.search(r"(\d{1,2}0)\s*대", text)
    return f"{decade_match.group(1)}대" if decade_match else None


def _extract_preferred_product_type(text: str) -> str | None:
    for product_type in ("TDF", "채권혼합형", "채권형", "주식형", "인덱스", "배당형", "원리금보장형"):
        if product_type in text:
            return product_type
    return None


def _has_minimum_profile(profile: dict) -> bool:
    """상품 유형 수준 안내라도 할 수 있는 최소 정보(위험선호 또는 계좌유형)가 있는지."""
    return bool(profile.get("risk_profile") or profile.get("loss_tolerance") or profile.get("account_type"))


def _missing_profile_fields(profile: dict) -> list[str]:
    missing = []
    if not profile.get("account_type"):
        missing.append("account_type")
    if not profile.get("risk_profile") and not profile.get("loss_tolerance"):
        missing.append("risk_profile")
    if not profile.get("investment_horizon") and not profile.get("age_or_retirement_horizon"):
        missing.append("investment_horizon")
    if not profile.get("monthly_investment"):
        missing.append("monthly_investment")
    if not profile.get("investment_goal"):
        missing.append("investment_goal")
    return missing


def _clarification_questions(missing: list[str]) -> list[str]:
    prompts = {
        "account_type": "어떤 계좌에서 투자할 예정인가요? IRP, DC, DB, 연금저축 중 선택해 주세요.",
        "risk_profile": "안정형, 중립형, 공격형 중 어느 투자성향에 가깝나요?",
        "investment_horizon": "예상 투자기간은 어느 정도인가요?",
        "monthly_investment": "투자 가능 금액 또는 월 납입금액은 얼마인가요?",
        "investment_goal": "가장 중요한 투자목적은 무엇인가요? 예: 노후 준비, 절세, 안정적 운용",
    }
    return [prompts[field] for field in missing]


def _clarification_answer(profile: dict, missing: list[str]) -> str:
    questions = _clarification_questions(missing)
    known = _format_profile_summary(profile)
    missing_labels = ", ".join(_PROFILE_FIELD_LABELS.get(field, field) for field in missing)
    known_block = f"\n\n현재 확인된 조건은 다음과 같습니다.\n{known}" if known else ""
    return (
        "현재 질문만으로는 계좌별 투자 제한과 투자성향을 확인할 수 없어 특정 상품을 바로 추천하기 어렵습니다."
        f"{known_block}\n\n"
        "현재 정보만으로는 연금계좌에서 일반적으로 고려할 수 있는 상품 유형을 설명하는 정도는 가능하지만, "
        "개별 펀드명·상품코드·수익률 순위를 확정 추천하는 것은 안전하지 않습니다.\n\n"
        f"구체적인 추천을 위해 부족한 정보는 {missing_labels}입니다. 다음 정보를 한 번에 알려주세요.\n"
        + "\n".join(f"{i}. {question}" for i, question in enumerate(questions, start=1))
        + "\n\n위 정보가 확인되지 않은 상태에서는 특정 펀드명이나 상품코드를 임의로 추천하지 않겠습니다."
    )


def _recommend_product_types(profile: dict) -> list[str]:
    risk = profile.get("risk_profile")
    horizon = profile.get("investment_horizon", "")
    goal = profile.get("investment_goal", "")
    preferred = profile.get("preferred_product_type")
    if preferred:
        return [preferred]
    if risk == "안정형":
        return ["원리금보장형 상품", "채권형 펀드"]
    if risk == "중립형":
        return ["TDF", "채권혼합형 펀드"]
    if risk == "공격형":
        return ["인덱스 펀드", "주식형 펀드"]
    if "장기" in horizon or re.search(r"\d+", horizon or ""):
        return ["TDF", "채권혼합형 펀드"]
    if "절세" in goal:
        return ["TDF", "채권혼합형 펀드"]
    return ["TDF", "채권혼합형 펀드"]


def _format_profile_summary(profile: dict) -> str:
    rows = []
    labels = {
        "account_type": "계좌 유형",
        "monthly_investment": "월 투자 가능 금액",
        "investment_horizon": "투자기간",
        "risk_profile": "투자성향",
        "investment_goal": "투자 목적",
        "age_or_retirement_horizon": "연령/은퇴 관련 정보",
        "loss_tolerance": "손실 감내 수준",
    }
    for key, label in labels.items():
        if profile.get(key):
            rows.append(f"- {label}: {profile[key]}")
    return "\n".join(rows)


def _product_type_recommendation_answer(profile: dict) -> str:
    types = _recommend_product_types(profile)
    profile["recommended_product_types"] = types
    summary = _format_profile_summary(profile)
    type_lines = "\n".join(f"{i}. {product_type}" for i, product_type in enumerate(types, start=1))
    reasons = _product_type_reasons(types)
    return (
        f"현재 말씀해주신 조건은\n{summary}\n\n"
        f"으로 정리됩니다.\n\n"
        f"이 조건에서는 먼저 아래 상품 유형을 고려하는 편이 좋습니다.\n{type_lines}\n\n"
        f"{reasons}\n\n"
        "위 조건에 맞는 구체적인 상품까지 추천받고 싶다면 `상품추천`을 입력해 주세요."
    )


def _product_type_reasons(types: list[str]) -> str:
    reason_map = {
        "TDF": "TDF는 은퇴 시점 또는 장기 투자 기간에 맞춰 위험자산 비중을 자동으로 조정할 수 있어 장기 노후 준비에 활용하기 좋습니다.",
        "채권혼합형 펀드": "채권혼합형 펀드는 주식형보다 변동성을 낮추면서 일정 수준의 수익을 함께 추구하기에 중립형 성향에 적합합니다.",
        "채권형 펀드": "채권형 펀드는 주식형보다 가격 변동이 낮은 편이라 큰 손실을 피하고 싶은 성향에 더 잘 맞습니다.",
        "원리금보장형 상품": "원리금보장형 상품은 수익률 기대는 낮지만 손실 가능성을 낮추고 안정성을 우선할 때 적합합니다.",
        "인덱스 펀드": "인덱스 펀드는 장기적으로 시장 평균 수익을 추구하는 방식이라 투자 기간이 길고 변동성을 감수할 수 있을 때 고려할 수 있습니다.",
        "주식형 펀드": "주식형 펀드는 기대수익이 높은 대신 변동성이 커서 공격형 성향에 더 적합합니다.",
        "배당형 펀드": "배당형 펀드는 배당 성향이 있는 자산에 투자해 장기 현금흐름과 수익을 함께 기대할 때 고려할 수 있습니다.",
    }
    return "\n".join(f"- {reason_map.get(product_type, product_type)}" for product_type in types)


def _recommendation_flow_response(state: PensionAgentState) -> tuple[str, list[RetrievedItem], dict, bool] | None:
    current = state["question"]

    # 안전성 가드가 가장 먼저다 — 아래 분기들(지시어 참조/특정상품 조회)이 먼저 가로채면
    # "미래 수익률은 알 수 없다", "계좌가 원금을 보장하지 않는다" 같은 교정을 놓친다.
    # 실측: "이 펀드가 내년에 몇 % 수익 날지" 질문이 지시어 참조로 분류돼, 미래 예측이
    # 불가능하다는 핵심은 빼고 "어떤 펀드인지 알려달라"고만 답했다.
    if _is_forecast_question(current):
        return _forecast_response(state)
    if _has_principal_guarantee_premise(current):
        return _principal_guarantee_response(state, concentration=_has_concentration_request(current))
    if _is_guaranteed_high_return_request(current):
        return _principal_guarantee_response(state, concentration=False, tradeoff=True)

    if _requires_account_eligibility_check(current):
        return None

    reference_response = _context_reference_response(state)
    if reference_response is not None:
        return reference_response

    combined = _combined_user_text(state)
    if _is_specific_product_or_comparison(current) and not _is_specific_recommendation_request(current):
        return None
    if not (_is_recommendation_intent(combined) or _is_specific_recommendation_request(current)):
        return None

    profile = _extract_recommendation_profile(state)
    missing = _missing_profile_fields(profile)
    wants_specific = _is_specific_recommendation_request(current)

    if missing:
        # 필수 5개 항목 중 일부가 비어도, 위험선호·계좌유형처럼 상품 유형 정도는 안내할 수 있는
        # 최소 정보가 있으면 완전 역질문 대신 "유형 안내 + 남은 정보 요청"으로 degrade한다.
        # 평가가 싱글턴이라 역질문만 하고 끝내면 아무 정보도 못 주는 셈이라 이쪽이 낫다.
        if _has_minimum_profile(profile):
            draft = _product_type_recommendation_answer(profile)
            missing_labels = ", ".join(_PROFILE_FIELD_LABELS.get(field, field) for field in missing)
            draft += (
                f"\n\n다만 {missing_labels} 정보가 없어 상품 유형 수준까지만 안내드렸습니다. "
                "구체적인 상품명까지 확인하려면 위 정보를 추가로 알려주세요."
            )
            return draft, [], profile, False
        return _clarification_answer(profile, missing), [], profile, True

    if wants_specific or _is_recommendation_intent(current):
        draft, context = _specific_product_recommendation(profile, state)
        return draft, context, profile, False

    return _product_type_recommendation_answer(profile), [], profile, False


def _search_args_from_profile(profile: dict) -> dict:
    risk = profile.get("risk_profile")
    product_type = profile.get("preferred_product_type") or ""
    args = {"limit": 10}
    if risk == "안정형":
        args["risk_grade_min"] = 5
    elif risk == "중립형":
        args["risk_grade_min"] = 4
    elif risk == "공격형":
        args["risk_grade_max"] = 3
    else:
        args["risk_grade_min"] = 4

    if "채권" in product_type:
        args["keyword"] = "채권"
    elif "주식" in product_type:
        args["keyword"] = "주식"
    elif "TDF" in product_type:
        args["keyword"] = "TDF"
    elif "배당" in product_type:
        args["keyword"] = "배당"
    return args


def _specific_product_recommendation(profile: dict, state: PensionAgentState) -> tuple[str, list[RetrievedItem]]:
    """수집된 recommendation_profile을 기반으로 SQL 상품 DB에서 구체 후보를 고른다."""
    results = search_funds.invoke(_search_args_from_profile(profile))
    if not isinstance(results, list) or not results:
        return (
            "현재 조건으로는 상품 DB에서 적합한 후보를 찾지 못했습니다. 위험 성향이나 선호 상품 유형을 조금 넓혀 다시 알려주세요.",
            [],
        )

    candidates = _unique_fund_candidates(results, limit=3)
    context = _fund_candidates_to_context(candidates)
    primary = _select_primary_candidate(candidates, _combined_user_text(state))
    summary = _format_profile_summary(profile)

    lines = [
        f"현재 조건은\n{summary}\n\n으로 정리됩니다.",
        f"이 조건에서는 **{primary.get('fund_name')} ({primary.get('class_name')})**를 우선 후보로 보겠습니다.",
        "아래는 투자설명서 DB의 구조화 수치로 비교한 후보입니다.",
    ]
    for i, item in enumerate(candidates, start=1):
        lines.append(
            f"{i}. {item.get('fund_name')} ({item.get('class_name')})\n"
            f"   - 상품 유형: {item.get('fund_category')}\n"
            f"   - 위험등급: {item.get('risk_grade')}\n"
            f"   - 총보수: {item.get('total_expense_ratio')}%\n"
            f"   - 수익률: 1년 {item.get('return_1y')}%, 3년 {item.get('return_3y')}%, "
            f"설정이후 {item.get('return_since_inception')}%\n"
            f"   - 추천 이유: 현재 위험성향과 투자기간 기준으로 위험등급·보수·과거 성과를 함께 비교했을 때 후보군에 포함됩니다.\n"
            f"   - 유의사항: 과거 수익률은 미래 수익을 보장하지 않으며, 계좌 내 실제 매수 가능 여부는 금융기관에서 최종 확인이 필요합니다."
        )
    return "\n\n".join(lines), context


def _unique_fund_candidates(results: list[dict], limit: int = 3) -> list[dict]:
    unique_results = []
    seen_codes = set()
    for item in results:
        code = item.get("product_code")
        if code in seen_codes:
            continue
        seen_codes.add(code)
        unique_results.append(item)
        if len(unique_results) >= limit:
            break
    return unique_results


def _fund_candidates_to_context(candidates: list[dict]) -> list[RetrievedItem]:
    context: list[RetrievedItem] = []
    for item in candidates:
        label = f"{item.get('fund_name', '상품명 없음')} ({item.get('class_name', '')})"
        content = (
            f"상품코드={item.get('product_code')}, 위험등급={item.get('risk_grade')}, "
            f"총보수={item.get('total_expense_ratio')}%, "
            f"1년수익률={item.get('return_1y')}%, 3년수익률={item.get('return_3y')}%, "
            f"설정이후수익률={item.get('return_since_inception')}%, "
            f"판매채널={item.get('sales_channel')}, 유형={item.get('fund_category')}"
        )
        context.append({"source": label, "content": content, "node": "product_agent"})
    return context


def _risk_search_args(text: str) -> dict:
    conservative_words = ("크게 잃지", "안정", "보수", "약간의 변동성", "중간", "변동성을 감수")
    aggressive_words = ("공격", "높은 수익", "적극", "고위험")
    if any(word in text for word in conservative_words):
        return {"risk_grade_min": 4, "limit": 8}
    if any(word in text for word in aggressive_words):
        return {"risk_grade_max": 3, "limit": 8}
    return {"limit": 8}


def _fallback_product_recommendation(state: PensionAgentState) -> tuple[str, list[RetrievedItem]]:
    """조건이 충분한 추천 요청인데 LLM이 상품 검색을 건너뛴 경우 투자설명서 DB 후보를 보강한다."""
    text = _combined_user_text(state)
    if not (_is_product_recommendation(text) and _has_account_type(text)):
        return "", []

    results = search_funds.invoke(_risk_search_args(text))
    if not isinstance(results, list) or not results:
        return "", []

    unique_results = _unique_fund_candidates(results, limit=3)
    context = _fund_candidates_to_context(unique_results)

    primary = _select_primary_candidate(unique_results, text)
    lines = [
        "확인된 조건을 기준으로 투자설명서 DB에서 후보를 골랐습니다.",
        "사용자 조건은 이전 대화와 현재 입력에서 확인된 계좌유형, 투자기간, 위험선호를 반영하세요.",
        "아래 후보들은 위험등급, 총보수, 과거 수익률을 함께 비교해 제시하세요.",
        (
            f"최우선 후보는 {primary.get('fund_name')} ({primary.get('class_name')})입니다. "
            "최종 답변 첫머리에서 이 상품을 먼저 추천하고, 선택 이유를 위험등급·총보수·수익률로 설명하세요."
        ),
        "search_funds 결과만으로 IRP 투자 가능 여부를 단정하지 말고, 투자 가능 여부는 금융기관에서 최종 확인이 필요하다고 쓰세요.",
    ]
    for i, item in enumerate(unique_results, start=1):
        lines.append(
            f"{i}. {item.get('fund_name')} ({item.get('class_name')}) - "
            f"위험등급 {item.get('risk_grade')}, 총보수 {item.get('total_expense_ratio')}%, "
            f"1년 {item.get('return_1y')}%, 3년 {item.get('return_3y')}%, "
            f"설정이후 {item.get('return_since_inception')}%, 판매채널 {item.get('sales_channel')}"
        )
    lines.append("나머지 후보는 비교 후보로 제시하고, 과거 수익률이 미래 수익을 보장하지 않는다는 점을 덧붙이세요.")
    return "\n".join(lines), context


def _select_primary_candidate(candidates: list[dict], text: str) -> dict:
    """사용자 위험선호에 맞춰 최우선 후보를 고른다."""
    if not candidates:
        return {}

    conservative = any(word in text for word in ("크게 잃지", "안정", "보수", "약간의 변동성"))

    def grade_num(item: dict) -> int:
        risk_grade = item.get("risk_grade") or ""
        return int(risk_grade[0]) if risk_grade[:1].isdigit() else 0

    if conservative:
        return sorted(
            candidates,
            key=lambda item: (
                "채권" not in (item.get("fund_category") or item.get("fund_name") or ""),
                -(grade_num(item)),
                item.get("total_expense_ratio") if item.get("total_expense_ratio") is not None else 999,
            ),
        )[0]

    return candidates[0]


def build_product_agent_node():
    llm = get_llm(PRODUCT_AGENT_MODEL)
    react_agent = create_agent(model=llm, tools=PRODUCT_AGENT_TOOLS, system_prompt=PRODUCT_AGENT_SYSTEM_PROMPT)

    def product_agent_node(state: PensionAgentState) -> dict:
        recommendation_flow = _recommendation_flow_response(state)
        if recommendation_flow is not None:
            draft, context, profile, needs_clarification = recommendation_flow
            missing = _missing_profile_fields(profile)
            recommendation_stage = (
                "clarification"
                if needs_clarification
                else "specific_recommendation"
                if context
                else "type_recommendation"
            )
            return {
                "product_draft": draft,
                "retrieved_context": context,
                "tool_trace": [],
                "recommendation_profile": profile,
                "needs_clarification": needs_clarification,
                "recommendation_stage": recommendation_stage,
                "missing_information": [_PROFILE_FIELD_LABELS.get(field, field) for field in missing]
                if needs_clarification
                else [],
                "clarification_questions": _clarification_questions(missing)
                if needs_clarification
                else [],
                "response_mode": "clarification_included"
                if needs_clarification
                else "complete"
                if context
                else "conditional",
                "repair_attempted": state.get("verification") is not None,
            }

        prior_context = dedupe_context(state.get("retrieved_context") or [])
        question = state["question"]
        if state.get("scope") == "부분관련" and state.get("scope_note"):
            question += (
                f"\n\n[범위 안내] 이 질문의 핵심은 연금 상담 범위 밖입니다. 범위 밖 부분은 "
                f"한계를 밝히고, 다음 연금 관점으로만 답하세요: {state['scope_note']}"
            )

        # verification이 이미 있으면 ④ 탈락으로 되돌아온 repair 재실행이다 (1회 한정).
        repair_note = build_repair_note(state.get("verification"))
        if repair_note:
            question += f"\n\n{repair_note}"

        if prior_context:
            context_text = "\n".join(f"- [{c['source']}] {c['content']}" for c in prior_context)
            question = f"{question}\n\n[②정보 Agent가 이미 확인한 제도 근거]\n{context_text}"

        history_messages = history_to_messages(state.get("conversation_history"))
        try:
            result = invoke_with_retry(
                react_agent, {"messages": [*history_messages, HumanMessage(content=question)]}
            )
        except Exception:
            fallback_draft, fallback_context = _fallback_product_recommendation(state)
            if fallback_context:
                return {
                    "product_draft": fallback_draft,
                    "retrieved_context": fallback_context,
                    "tool_trace": [],
                    "needs_clarification": False,
                    "recommendation_stage": "specific_recommendation",
                    "repair_attempted": state.get("verification") is not None,
                }
            raise
        messages = result["messages"]

        retrieved_context = build_retrieved_context(messages, node="product_agent")

        final_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        draft, needs_clarification = split_clarification_marker(
            final_ai.content if final_ai else ""
        )
        if not retrieved_context:
            fallback_draft, fallback_context = _fallback_product_recommendation(state)
            if fallback_context:
                draft = fallback_draft
                retrieved_context = fallback_context

        return {
            "product_draft": draft,
            "retrieved_context": retrieved_context,
            "tool_trace": build_tool_trace(messages, node="product_agent"),
            "needs_clarification": needs_clarification,
            "recommendation_stage": "clarification" if needs_clarification else None,
            "repair_attempted": state.get("verification") is not None,
        }

    return product_agent_node
