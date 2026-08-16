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

from src.agents.context import build_retrieved_context, history_to_messages
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState
from src.agents.tools import PRODUCT_AGENT_TOOLS, search_funds

PRODUCT_AGENT_MODEL = "HCX-005"

PRODUCT_AGENT_SYSTEM_PROMPT = """당신은 연금 상품(펀드) 추천 에이전트입니다.

절대 규칙 — 단정적 추천 금지: "좋은 상품 추천해줘", "괜찮은 연금상품 3개 추천해줘"처럼
계좌유형(DB/DC/IRP)·투자기간·위험선호·금액 등 구체적 조건이 없는 막연한 요청에는 상품을
지어내서 추천하지 마세요. 이 경우 search_funds/check_product_pension_eligibility 등 툴을
호출하지 말고, 답변을 확인이 필요한 조건을 묻는 역질문으로 작성하세요(예: "투자 가능한
계좌유형과 감내 가능한 위험 수준을 알려주시면 후보를 좁혀드릴게요").

멀티턴 규칙:
- [이전 대화]와 [현재 사용자 입력]을 함께 보고 계좌유형·투자기간·위험선호가 이미 확인됐는지
  판단하세요. 이전 턴에서 물어본 조건을 현재 사용자가 답했다면, 같은 조건을 다시 묻지 말고
  그 조건으로 추천을 진행하세요.
- 반대로 계좌유형이 아직 확인되지 않았다면 search_funds를 호출하지 말고 반드시 계좌유형을
  먼저 물어보세요. 퇴직연금/IRP에서는 계좌유형별 투자제한이 달라 상품 추천의 필수 조건입니다.
- 이전 Agent 답변에 나온 추천 후보는 확정 근거가 아닙니다. 실제 후보는 현재 확인된 조건으로
  search_funds를 다시 호출해 가져온 결과만 사용하세요.
- "크게 잃지 않으면서", "안정적으로", "약간의 변동성 감수"처럼 보수적 성향이 확인되면
  위험등급 4~6등급 후보를 우선 검색하세요.
- search_funds 결과만으로 "IRP에서 투자 가능"이라고 단정하지 마세요. search_funds는
  투자설명서의 상품 수치 후보를 찾는 도구이고, 계좌별 투자 가능 여부는 별도 확인이 필요합니다.
  판매채널에 개인연금/퇴직연금이 있으면 그 판매채널만 사실로 언급하세요.

조건이 충분한 경우의 진행 순서:
1. search_funds로 조건에 맞는 후보를 찾으세요 (risk_grade/keyword 등 실제 질문에서 나온
   조건만 사용하고, 없는 조건을 임의로 지어내지 마세요).
2. 특정 상품유형(국내 상장주식·사모펀드·증권예탁증권 등)을 계좌유형과 함께 추천하려면
   check_product_pension_eligibility로 그 계좌(DB/DC/IRP)에서 투자 가능한지 먼저
   확인하세요 — product_type은 후보 상품의 실제 분류값만 사용하고, 사용자의 막연한
   표현(예: "괜찮은 상품")을 그대로 product_type에 넣지 마세요.
3. 상세 설명이나 비교가 필요하면 get_fund_detail로 전체 정보를 가져오세요.

툴 호출 결과에 error가 있으면 그 오류를 사용자에게 노출하지 말고, 조건을 다시 확인하는
질문으로 답하세요.

이전 대화가 함께 주어지면 "그중 두 번째 상품", "방금 조건대로" 같은 지시어를 이전 턴 내용으로
풀어서 이해하세요. 다만 이전 턴에서 언급된 상품 정보를 그대로 베끼지 말고, 필요하면
search_funds/get_fund_detail로 다시 확인하세요."""

_RECOMMENDATION_WORDS = ("추천", "뭐 사면", "뭐살", "투자하면 좋", "맞는 상품", "상품 알려", "펀드 알려")
_SPECIFIC_RECOMMENDATION_WORDS = ("상품추천", "구체", "실제 상품", "펀드명", "상품명")
_SPECIFIC_PRODUCT_WORDS = ("어때", "위험", "수수료", "보수", "수익률", "설명", "분석")
_COMPARISON_WORDS = ("비교", "차이", " vs ", "VS")


def _history_text(state: PensionAgentState) -> str:
    history = state.get("conversation_history") or []
    return "\n".join(f"{turn.get('question', '')}\n{turn.get('answer', '')}" for turn in history)


def _combined_user_text(state: PensionAgentState) -> str:
    history = state.get("conversation_history") or []
    previous_user_text = "\n".join(turn.get("question", "") for turn in history)
    return f"{previous_user_text}\n{state['question']}"


def _has_account_type(text: str) -> bool:
    return any(token in text.upper() for token in ("IRP", "DC", "DB"))


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
    monthly_match = re.search(r"(월|한 달|매달)\s*(\d+)\s*(만\s*원|만원|원)\s*(이상|정도|쯤|가량)?", text)
    if monthly_match:
        suffix = f" {monthly_match.group(4)}" if monthly_match.group(4) else ""
        return f"월 {monthly_match.group(2)}만원{suffix}" if "만" in monthly_match.group(3) else monthly_match.group(0)
    amount_match = re.search(r"(\d+)\s*(만\s*원|만원)\s*(이상|정도|쯤|가량)?", text)
    if amount_match and (
        allow_standalone or any(word in text for word in ("투자", "납입", "가능", "넣", "불입"))
    ):
        suffix = f" {amount_match.group(3)}" if amount_match.group(3) else ""
        return f"월 {amount_match.group(1)}만원{suffix}"
    return None


def _extract_horizon(text: str) -> str | None:
    match = re.search(r"(\d+)\s*년\s*(이상|정도|간|동안)?", text)
    if match:
        suffix = f" {match.group(2)}" if match.group(2) else ""
        return f"{match.group(1)}년{suffix}"
    if "장기" in text:
        return "장기"
    if "중기" in text:
        return "중기"
    if "단기" in text:
        return "단기"
    return None


def _extract_risk_profile(text: str) -> str | None:
    if any(word in text for word in ("안정형", "보수", "안정적", "크게 잃지", "손실 싫", "낮은 위험", "위험등급 낮")):
        return "안정형"
    if any(word in text for word in ("중립형", "중립", "약간의 변동성", "어느 정도", "중간")):
        return "중립형"
    if any(word in text for word in ("공격형", "공격", "적극", "높은 수익", "고위험")):
        return "공격형"
    return None


def _extract_loss_tolerance(text: str) -> str | None:
    if "크게 잃지" in text or "손실" in text and any(word in text for word in ("싫", "피", "낮")):
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
    return f"{match.group(1)}세" if match else None


def _extract_preferred_product_type(text: str) -> str | None:
    for product_type in ("TDF", "채권혼합형", "채권형", "주식형", "인덱스", "배당형", "원리금보장형"):
        if product_type in text:
            return product_type
    return None


def _missing_profile_fields(profile: dict) -> list[str]:
    missing = []
    if not profile.get("monthly_investment"):
        missing.append("monthly_investment")
    if not profile.get("risk_profile") and not profile.get("loss_tolerance"):
        missing.append("risk_profile")
    if not profile.get("investment_horizon") and not profile.get("age_or_retirement_horizon"):
        missing.append("investment_horizon")
    if not profile.get("investment_goal"):
        missing.append("investment_goal")
    return missing


def _next_profile_question(missing: list[str]) -> str:
    prompts = {
        "monthly_investment": "추천을 위해 몇 가지만 먼저 확인할게요.\n한 달에 어느 정도 금액을 투자할 예정인가요?",
        "risk_profile": "투자할 때 어느 정도의 가격 변동까지 감수할 수 있나요?\n예를 들어 안정형 / 중립형 / 공격형 중 가장 가까운 성향을 알려주세요.",
        "investment_horizon": "예상 투자 기간은 어느 정도인가요?\n예를 들어 3년 이하, 5년 이상, 은퇴 전까지처럼 알려주세요.",
        "investment_goal": "투자 목적은 무엇에 가장 가까운가요?\n예를 들어 노후 준비, 절세, 목돈 마련, 안정적 현금흐름 중에서 알려주세요.",
    }
    return prompts[missing[0]]


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
    reasons = _product_type_reasons(types, profile)
    return (
        f"현재 말씀해주신 조건은\n{summary}\n\n"
        f"으로 정리됩니다.\n\n"
        f"이 조건에서는 먼저 아래 상품 유형을 고려하는 편이 좋습니다.\n{type_lines}\n\n"
        f"{reasons}\n\n"
        "위 조건에 맞는 구체적인 상품까지 추천받고 싶다면 `상품추천`을 입력해 주세요."
    )


def _product_type_reasons(types: list[str], profile: dict) -> str:
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


def _recommendation_flow_response(state: PensionAgentState) -> tuple[str, list[dict], dict, bool] | None:
    current = state["question"]
    combined = _combined_user_text(state)
    if _is_specific_product_or_comparison(current) and not _is_specific_recommendation_request(current):
        return None
    if not (_is_recommendation_intent(combined) or _is_specific_recommendation_request(current)):
        return None

    profile = _extract_recommendation_profile(state)
    missing = _missing_profile_fields(profile)
    wants_specific = _is_specific_recommendation_request(current)

    if missing:
        return _next_profile_question(missing), [], profile, True

    if wants_specific:
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


def _specific_product_recommendation(profile: dict, state: PensionAgentState) -> tuple[str, list[dict]]:
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


def _fund_candidates_to_context(candidates: list[dict]) -> list[dict]:
    context = []
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


def _fallback_product_recommendation(state: PensionAgentState) -> tuple[str, list[dict]]:
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
                "recommendation_profile": profile,
                "needs_clarification": needs_clarification,
                "recommendation_stage": recommendation_stage,
            }

        prior_context = state.get("retrieved_context") or []
        question = state["question"]
        if prior_context:
            context_text = "\n".join(f"- [{c['source']}] {c['content']}" for c in prior_context)
            question = f"{question}\n\n[②정보 Agent가 이미 확인한 제도 근거]\n{context_text}"

        history_messages = history_to_messages(state.get("conversation_history"))
        try:
            result = invoke_with_retry(
                react_agent,
                {"messages": [*history_messages, HumanMessage(content=question)]},
            )
        except Exception:
            fallback_draft, fallback_context = _fallback_product_recommendation(state)
            if fallback_context:
                return {
                    "product_draft": fallback_draft,
                    "retrieved_context": fallback_context,
                }
            raise

        messages = result["messages"]

        retrieved_context = build_retrieved_context(messages, node="product_agent")

        final_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        draft = final_ai.content if final_ai else ""
        if not retrieved_context:
            fallback_draft, fallback_context = _fallback_product_recommendation(state)
            if fallback_context:
                draft = fallback_draft
                retrieved_context = fallback_context

        return {
            "product_draft": draft,
            "retrieved_context": retrieved_context,
        }

    return product_agent_node
