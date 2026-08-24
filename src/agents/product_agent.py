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
계좌유형(DB/DC/IRP)·투자기간·위험선호 등 핵심 조건이 없는 막연한 요청에는 상품을
지어내서 추천하지 마세요. 이 경우 search_funds/check_product_pension_eligibility 등 툴을
호출하지 말고, 답변을 확인이 필요한 조건을 묻는 역질문으로 작성하세요(예: "투자 가능한
계좌유형과 감내 가능한 위험 수준을 알려주시면 후보를 좁혀드릴게요"). 이런 역질문 답변은
반드시 [추가 확인 필요] 로 시작하세요.

역질문은 첫 답변 안에서 필요한 항목 전체를 한꺼번에 물으세요. 평가 환경은 단일턴이므로
"한 달에 얼마를 투자할 예정인가요?"처럼 한 항목만 묻고 종료하면 안 됩니다. 계좌유형·위험선호·
투자기간·투자금액·투자목적 중 부족한 항목을 모두 나열하고, 부족한 상태에서는 특정 펀드명이나
상품코드를 임의로 추천하지 마세요. 단정적 표현("이 상품이 가장 좋습니다")은 쓰지 마세요.
다만 계좌유형·위험선호·투자기간이 확인됐다면 투자금액이나 목적이 명시되지 않았다는 이유만으로
추천을 전부 보류하지 마세요. 확인된 조건으로 후보를 제시하되 빠진 조건에 따라 결과가 달라질 수
있음을 밝히세요.

원금보장과 높은 수익을 동시에 요구하거나 미래에 가장 많이 오를 상품을 확답해 달라는 요청은
거절문만 출력하지 마세요. 두 조건을 동시에 보장하거나 미래수익률을 정확히 예측할 수 없다는 점을
바로잡고, 원리금보장형과 실적배당형의 차이 또는 과거 수익률·위험등급·보수로 비교 가능한 범위를
설명하세요.

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
    """계좌와 상품유형의 투자 가능 여부를 툴로 먼저 확인해야 하는 요청인지 판정한다."""
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


def _extract_amount(text: str, allow_standalone: bool = False) -> str | None:
    amount_token = r"\d[\d,]*(?:\.\d+)?\s*(?:억\s*원|억원|천만\s*원|천만원|백만\s*원|백만원|만\s*원|만원|원)"
    monthly_match = re.search(
        rf"(월|한 달|매달)\s*({amount_token})\s*(이상|정도|쯤|가량)?",
        text,
    )
    if monthly_match:
        amount = re.sub(r"\s+", "", monthly_match.group(2))
        suffix = f" {monthly_match.group(3)}" if monthly_match.group(3) else ""
        return f"월 {amount}{suffix}"
    amount_match = re.search(rf"({amount_token})\s*(이상|정도|쯤|가량)?", text)
    if amount_match and (
        allow_standalone or any(word in text for word in ("투자", "납입", "가능", "넣", "불입"))
    ):
        amount = re.sub(r"\s+", "", amount_match.group(1))
        suffix = f" {amount_match.group(2)}" if amount_match.group(2) else ""
        # 독립적인 후속 응답("20만원")은 월 납입액으로 해석하되, "퇴직금 3억원",
        # "IRP 5천만원" 같은 일시금은 임의로 월 금액으로 바꾸지 않는다.
        if allow_standalone and re.fullmatch(rf"\s*{amount_token}\s*(?:이상|정도|쯤|가량)?\s*", text):
            return f"월 {amount}{suffix}"
        return f"{amount}{suffix}"
    return None


def _extract_horizon(text: str) -> str | None:
    match = re.search(r"(\d+)\s*년\s*(이상|정도|간|동안)?", text)
    if match:
        suffix = f" {match.group(2)}" if match.group(2) else ""
        return f"{match.group(1)}년{suffix}"
    if "장기" in text or ("은퇴" in text and any(word in text for word in ("오래", "한참", "많이 남"))):
        return "장기"
    if "중기" in text:
        return "중기"
    if "단기" in text:
        return "단기"
    return None


def _extract_risk_profile(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text)
    if any(word in text for word in ("안정형", "보수", "안정적", "안전한", "크게 잃지", "손실 싫", "낮은 위험", "위험등급 낮")) or any(
        word in compact for word in ("손실최대한줄", "손실을최대한줄", "손실최소", "원금손실은절대싫")
    ):
        return "안정형"
    if any(word in text for word in ("중립형", "중립", "약간의 변동성", "어느 정도", "중간")):
        return "중립형"
    if any(word in text for word in ("공격형", "공격", "적극", "높은 수익", "고위험", "수익성", "수익 추구")) or any(
        word in compact for word in ("주식형비중최대한", "주식형비중을최대한", "수익을추구", "수익률높", "수익률도높")
    ):
        return "공격형"
    return None


def _extract_loss_tolerance(text: str) -> str | None:
    if "크게 잃지" in text or ("손실" in text and any(word in text for word in ("싫", "피", "낮", "줄", "적", "최소", "절대"))):
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
    match = re.search(r"(\d{2})\s*(세|대)", text)
    return f"{match.group(1)}{match.group(2)}" if match else None


def _extract_preferred_product_type(text: str) -> str | None:
    for product_type in ("TDF", "채권혼합형", "채권형", "국내 상장주식형", "상장주식형", "주식형", "인덱스", "배당형", "원리금보장형"):
        if product_type in text:
            return product_type
    return None


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


def _financial_limit_note(question: str) -> str:
    compact = re.sub(r"\s+", "", question or "")
    guarantee_terms = ("원금보장", "원금손실은절대싫", "원금이무조건")
    high_return_terms = ("수익률좋", "수익률높", "수익률도높", "고수익", "높은수익", "확정수익")
    if any(term in compact for term in guarantee_terms) and any(
        term in compact for term in high_return_terms
    ):
        return (
            "\n\n먼저 원금 보장과 높은 수익률을 동시에 확정하는 펀드는 없습니다. "
            "원리금보장형은 손실 위험을 낮추는 대신 기대수익이 제한되고, 실적배당형 펀드는 "
            "더 높은 수익을 기대할 수 있지만 원금 손실 가능성이 있습니다."
        )
    if any(term in compact for term in ("정확히몇%", "제일많이오를", "가장많이오를", "무조건오를")):
        return (
            "\n\n미래 수익률이나 가장 많이 오를 상품은 사전에 확정할 수 없습니다. "
            "대신 과거 수익률, 위험등급, 보수와 투자전략을 같은 기준으로 비교할 수 있습니다."
        )
    return ""


def _clarification_answer(profile: dict, missing: list[str], question: str = "") -> str:
    questions = _clarification_questions(missing)
    known = _format_profile_summary(profile)
    missing_labels = ", ".join(_PROFILE_FIELD_LABELS.get(field, field) for field in missing)
    known_block = f"\n\n현재 확인된 조건은 다음과 같습니다.\n{known}" if known else ""
    constraint_note = _financial_limit_note(question)
    type_note = ""
    if profile.get("risk_profile") or profile.get("loss_tolerance") or profile.get("investment_horizon"):
        types = ", ".join(_recommend_product_types(profile))
        type_note = (
            f"\n\n현재 확인된 조건만 놓고 보면 우선 살펴볼 상품 유형은 **{types}**입니다. "
            "이는 조건부 유형 안내이며 특정 펀드의 확정 추천은 아닙니다."
        )
    return (
        "현재 질문만으로는 계좌별 투자 제한과 투자성향을 확인할 수 없어 특정 상품을 바로 추천하기 어렵습니다."
        f"{constraint_note}{known_block}{type_note}\n\n"
        "현재 정보만으로는 연금계좌에서 일반적으로 고려할 수 있는 상품 유형을 설명하는 정도는 가능하지만, "
        "개별 펀드명·상품코드·수익률 순위를 확정 추천하는 것은 안전하지 않습니다.\n\n"
        f"구체적인 추천을 위해 부족한 정보는 {missing_labels}입니다. 다음 정보를 한 번에 알려주세요.\n"
        + "\n".join(f"{i}. {question}" for i, question in enumerate(questions, start=1))
        + "\n\n위 정보가 확인되지 않은 상태에서는 특정 펀드명이나 상품코드를 임의로 추천하지 않겠습니다."
    )


def _recommend_product_types(profile: dict) -> list[str]:
    risk = profile.get("risk_profile")
    if not risk and profile.get("loss_tolerance") == "큰 손실 회피":
        risk = "안정형"
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
    reference_response = _context_reference_response(state)
    if reference_response is not None:
        return reference_response

    current = state["question"]
    combined = _combined_user_text(state)
    if _requires_account_eligibility_check(current):
        return None
    if _is_specific_product_or_comparison(current) and not _is_specific_recommendation_request(current):
        return None
    if not (_is_recommendation_intent(combined) or _is_specific_recommendation_request(current)):
        return None

    profile = _extract_recommendation_profile(state)
    missing = _missing_profile_fields(profile)
    blocking_missing = [
        field for field in missing if field in ("account_type", "risk_profile", "investment_horizon")
    ]
    wants_specific = _is_specific_recommendation_request(current)

    if blocking_missing:
        return _clarification_answer(profile, missing, current), [], profile, True

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
    # 결정론 추천 경로는 CLOVA 호출이 필요 없다. API 키가 없는 로컬 테스트/개발 환경에서도
    # 이 경로를 실행할 수 있도록 ReAct agent는 실제로 필요할 때 한 번만 만든다.
    react_agent = None

    def product_agent_node(state: PensionAgentState) -> dict:
        nonlocal react_agent
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

        if react_agent is None:
            llm = get_llm(PRODUCT_AGENT_MODEL)
            react_agent = create_agent(
                model=llm,
                tools=PRODUCT_AGENT_TOOLS,
                system_prompt=PRODUCT_AGENT_SYSTEM_PROMPT,
            )

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
