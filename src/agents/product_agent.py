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
from src.rules.early_withdrawal import PlanType
from src.rules.investment_limit import RISKY_ASSET_LIMIT, RiskTier, classify_fund_category_risk_tier
from src.storage.queries import (
    find_lower_cost_pension_class,
    get_pension_class_detail,
    normalize_pension_account_type,
)

PRODUCT_AGENT_MODEL = "HCX-005"

PRODUCT_AGENT_SYSTEM_PROMPT = """당신은 연금 상품(펀드) 추천 에이전트입니다.

서비스 범위: 연금계좌(DB/DC/IRP·연금저축)에서 투자하는 상품(펀드)의 설명·비교·추천만
다룹니다. 범위 밖 주제(일반 주식 종목, 부동산, 연금 외 금융상품 등)는 학습 지식으로 답하지
말고 "본 서비스의 상담 범위를 벗어난다"고 한계를 고지하세요. [범위 안내]가 함께 주어지면
범위 밖 부분은 한계를 밝히고, 안내된 연금 관점으로만 답하세요.

절대 규칙 — 개별 상품 데이터로 제도 일반론을 만들지 마세요: 당신의 툴은 개별 펀드의
수치·서술을 조회할 뿐이며, 제도·규정 문서를 검색할 수단이 없습니다. 따라서 특정 상품을
지목하지 않은 제도·규정 질문(예: "연금저축 펀드 환매 제한기간이 있나요", "펀드 환매수수료가
뭔가요")에는 임의의 펀드를 조회해 그 데이터로 답하지 마세요. 펀드 3개의 환매 규정을 보고
"연금저축 펀드는 일반적으로 환매 제한이 없다"처럼 전체에 대한 결론을 내리는 것은 근거
없는 일반화입니다 — 조회한 상품에만 해당하는 사실을 전체 제도의 규칙인 것처럼 말하면
안 됩니다.
이런 질문을 받으면 툴을 호출하지 말고, [추가 확인 필요] 표시를 포함해 (1) 제도 일반 규정은
이 경로에서 확인이 어렵다는 한계를 밝히고, (2) 특정 상품의 환매 규정·수수료를 알고 싶다면
상품명이나 상품코드를 알려달라고 요청하세요.

절대 규칙 — 단정적 추천 금지: "좋은 상품 추천해줘", "괜찮은 연금상품 3개 추천해줘"처럼
계좌유형(DB/DC/IRP)·투자기간·위험선호·금액 등 구체적 조건이 없는 막연한 요청에는 상품을
지어내서 추천하지 마세요. 이 경우 search_funds/check_product_pension_eligibility 등 툴을
호출하지 말고, 답변을 확인이 필요한 조건을 묻는 역질문으로 작성하세요(예: "투자 가능한
계좌유형과 감내 가능한 위험 수준을 알려주시면 후보를 좁혀드릴게요"). 이런 역질문 답변은
반드시 [추가 확인 필요] 표시를 포함하세요 (위치는 어디든 상관없습니다).

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

툴 호출 결과에 error가 있으면 그 오류를 사용자에게 노출하지 말고, [추가 확인 필요] 표시를
포함한 조건 재확인 질문으로 답하세요.

이전 대화가 함께 주어지면 "그중 두 번째 상품", "방금 조건대로" 같은 지시어를 이전 턴 내용으로
풀어서 이해하세요. 다만 이전 턴에서 언급된 상품 정보를 그대로 베끼지 말고, 필요하면
search_funds/get_fund_detail로 다시 확인하세요."""


_RECOMMENDATION_WORDS = (
    "추천", "뭐 사면", "뭐살", "투자하면 좋", "맞는 상품", "상품 알려", "펀드 알려",
    "골라주세요", "골라줘", "골라달라", "투자하고 싶", "투자하고싶",
)
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

_CLARIFICATION_BLOCKED_MARKERS = (
    "시장 상황이 바뀐다면",
    "미국·해외 증시",
    "원/달러 환율",
    "금리가 하락하면",
    "금리가 상승하면",
)

# 거시 지표를 근거로 특정 자산유형이 유리/불리하다고 단정하는 문장 — 조건이 아직
# 부족한 clarification 상태에서는 나올 수 없는 종류의 주장이다(계좌유형·위험성향도
# 모르는데 "지금이 매수 적기"라고 말할 근거가 없다).
#
# ⚠️ _CLARIFICATION_BLOCKED_MARKERS는 죽은 코드였던 _what_if_scenario_block이 만들던
# 정확한 문구("시장 상황이 바뀐다면?", "미국·해외 증시" 등)에만 맞춰져 있어, 같은 뜻의
# 다른 표현("요즘 증시가 좋아서 주식형이 유리합니다", "금리 인하 기조라 채권형이
# 매력적입니다")은 전혀 못 걸렀다(직접 재현 확인: 5개 paraphrase 전부 통과). 어휘
# 하드코딩이 아니라 "거시 지표 명사 + 조건부 유리·전망 서술"이라는 문장 구조로 잡는다.
_MACRO_INDICATOR_RE = re.compile(r"시장|경기|증시|금리|환율|주가|달러|인플레이션|물가")
_FAVORABLE_OUTLOOK_RE = re.compile(
    r"유리|불리|매력적|적기|좋을|좋은|좋다|긍정적|부정적|호재|악재|상승세|하락세|"
    r"기대됩니다|추천됩니다|늘리는게|늘리는 게|줄이는게|줄이는 게"
)

_INVESTMENT_LIMIT_SOURCE = "doc56~doc58 적립금 운용 및 투자한도 규칙"
_INVESTMENT_LIMIT_CONTENT = (
    "위험자산은 DB/DC/IRP 공통으로 적립금의 70%까지만 투자 가능하며, "
    "주식형·주식혼합형펀드는 위험자산으로 분류됩니다. "
    "TDF는 감독원장이 정한 조건을 충족하면 DC/IRP에 한해 100%까지 투자할 수 있습니다."
)

# 연금계좌 가입대상 — 소득 유무로 갈린다. 원문(연금저축계좌·IRP 세액공제 안내 —
# 연금계좌 종류와 가입대상)의 핵심 문장을 그대로 옮긴다.
_ACCOUNT_ELIGIBILITY_SOURCE = "연금저축계좌·IRP 세액공제 안내 — 연금계좌 종류와 가입대상"
_ACCOUNT_ELIGIBILITY_CONTENT = (
    "연금계좌는 연금저축과 IRP, 두 종류다. 연금저축은 누구나 가입할 수 있다. "
    "소득이 없어도 가입이 가능하지만 직장인, 자영업자 등 종합소득이 있어야 세액공제 "
    "혜택을 볼 수 있다. IRP는 직장인, 자영업자, 직역연금가입자 등 가입대상이 정해져 있다."
)

_SCENARIO_RULE_SOURCE = "상품 시나리오 규칙 — 후보 속성 기반 점검"
_SCENARIO_RULE_CONTENT = (
    "상품 후보의 구조화 속성에 주식형이 확인되면 주식형 가격 변동 위험을 점검합니다. "
    "채권형이 확인되면 채권형 상품의 가격 변동과 신용위험을 점검합니다. "
    "상품명이나 유형에서 USD 등 외화 노출 가능성이 확인되면 통화 관련 조건 확인 필요성을 점검합니다."
)


def _is_product_flow_turn(text: str) -> bool:
    """이 발화가 상품 추천 흐름에 속하는지 — 이력을 이어받을지 판단하는 기준.

    상품 추천은 여러 턴에 걸쳐 조건(계좌유형·위험성향·기간·금액)을 채우는 흐름이라
    이력이 필요하다. 하지만 주제가 다른 턴까지 합치면 그 턴의 수치가 추천 조건으로
    새어 들어간다.
    """
    if _is_product_recommendation(text) or _is_recommendation_intent(text):
        return True
    # 조건만 답하는 후속 턴("안정형이야", "IRP에서요", "월 30만원")도 흐름의 일부다.
    # ⚠️ 어휘 목록("년", "만원" 등)만으로 판정하면 "2026년에 인출하면 어떻게 되나요?"
    # 같은 제도 질문까지 흐름으로 오인한다(실측). 조건 답변은 질문이 아니라 짧은 값
    # 응답이라는 성질을 함께 본다 — 물음표가 없고 짧을 때만 흐름으로 인정한다.
    is_short_answer = len(text.strip()) <= 20 and "?" not in text
    return is_short_answer and (
        _has_account_type(text)
        or any(word in text for word in ("안정", "중립", "공격", "위험", "만원", "년", "개월", "노후", "은퇴", "절세", "목돈"))
    )


def _should_carry_product_history(state: PensionAgentState) -> bool:
    """현재 발화가 이전 추천 흐름의 후속 답변인지 판단한다.

    새 추천 요청("IRP에서 S&P500 ETF 같은 상품 추천해줘")은 그 자체로 완결된 새 질의일 수
    있다. 이때 직전 질문의 "안전한/안정형" 같은 조건을 재사용하면 history contamination이
    생긴다. 반면 "상품추천", "20년이야", "월 30만원"처럼 이전 역질문에 답하는 짧은 발화는
    기존 profile을 이어받아야 한다.
    """
    current = state["question"].strip()
    compact = re.sub(r"\s+", "", current)
    if any(word in current for word in _REFERENCE_WORDS):
        return True
    if compact in ("상품추천", "구체적인상품추천", "구체상품추천"):
        return True
    history = state.get("conversation_history") or []
    has_previous_product_flow = any(_is_product_flow_turn(turn.get("question", "")) for turn in history)
    if (
        has_previous_product_flow
        and "?" not in current
        and not _is_recommendation_intent(current)
        and (
            _has_account_type(current)
            or _extract_amount(current, allow_standalone=True)
            or _extract_horizon(current)
            or _extract_risk_profile(current)
            or _extract_goal(current)
        )
    ):
        return True
    is_short_answer = len(current) <= 30 and "?" not in current
    if is_short_answer and not _is_recommendation_intent(current):
        return _is_product_flow_turn(current)
    return False


def _combined_user_text(state: PensionAgentState) -> str:
    """현재 질문 + **상품 흐름에 속한** 이전 턴만 합친다.

    ⚠️ 예전에는 conversation_history 전체를 무차별로 합쳤다. 그러면 직전에 물어본
    제도·세제 질문의 수치가 상품 추천 프로필로 새어 들어간다 — 실측: "만 74세 연금
    세율" 질문 다음에 "IRP에서 S&P500 ETF 살 수 있나요"를 물었더니 추천 조건에
    "투자기간 2026년, 나이 74세"가 섞여 나왔다. 평가는 문항당 단발 호출이라 특히
    위험하다(다른 문항의 조건이 넘어오면 안 된다).

    나이·날짜만 골라 빼는 방식은 쓰지 않는다 — 그 다음엔 금액이, 그 다음엔 계좌유형이
    새는 식으로 반복된다. 애초에 "같은 흐름의 턴인가"로 걸러야 한다.
    """
    if not _should_carry_product_history(state):
        return state["question"]

    history = state.get("conversation_history") or []
    relevant = [
        turn.get("question", "")
        for turn in history
        if _is_product_flow_turn(turn.get("question", ""))
    ]
    previous_user_text = "\n".join(relevant)
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


def _extract_product_code(text: str) -> str | None:
    match = re.search(r"KR[0-9A-Z]+", (text or "").upper())
    return match.group(0) if match else None


def _extract_class_code(text: str) -> str | None:
    raw = (text or "").upper()
    match = re.search(r"(?<![A-Z0-9])([A-Z]-[A-Z0-9]{1,5})(?![A-Z0-9])", raw)
    if match:
        return match.group(1)
    compact = re.sub(r"\s+", "", raw)
    match = re.search(r"(?<![A-Z0-9])([A-Z]-[A-Z0-9]{1,5})(?![A-Z0-9])", compact)
    if match:
        return match.group(1)
    match = re.search(r"\(([^()]*)\)\s*$", text or "")
    if match:
        inner = match.group(1).strip()
        if re.fullmatch(r"[A-Za-z]-?[A-Za-z0-9]{0,5}", inner):
            return inner
    return None


def _asks_alternative_recommendation(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return any(
        marker in compact
        for marker in (
            "다른상품",
            "다른펀드",
            "비슷한상품",
            "비슷한펀드",
            "유사상품",
            "유사펀드",
            "대안",
            "비교",
        )
    )


def _asks_cost_topic(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return any(
        marker in compact
        for marker in (
            "보수",
            "수수료",
            "비용",
            "총보수",
            "합성총보수",
            "저렴",
            "싼",
            "더낮은클래스",
            "낮은클래스",
            "클래스비교",
        )
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


_COST_METRIC_LABELS = {
    "synthetic_total_expense_ratio": "합성총보수·비용",
    "total_expense_ratio": "총보수·비용",
}


def _format_pct(value: object) -> str:
    try:
        return f"{float(value):g}%"
    except (TypeError, ValueError):
        return "확인 필요"


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.1f}"


def _basis_suffix(item: dict, field: str, label: str) -> str:
    value = item.get(field)
    return f" ({label} {value} 기준)" if value else ""


def _format_return_line(item: dict) -> str:
    return (
        f"   - 수익률{_basis_suffix(item, 'return_asof_date', '수익률기준일')}: "
        f"1년 {item.get('return_1y')}%, 3년 {item.get('return_3y')}%, "
        f"설정이후 {item.get('return_since_inception')}%"
    )


def _format_expense_line(item: dict) -> str:
    return (
        f"   - 총보수·비용{_basis_suffix(item, 'prospectus_effective_date', '투자설명서 효력발생일')}: "
        f"{_format_pct(item.get('total_expense_ratio'))}"
    )


def _format_aum_line(item: dict, *, indent: str = "   ") -> str:
    value = item.get("aum_krw_million")
    if value is None:
        return ""
    return f"{indent}- 시장잔고: {_format_number(value)}백만원{_basis_suffix(item, 'aum_base_date', '잔고 기준일')}"


def _format_candidate_metric_block(item: dict, *, include_type: bool) -> str:
    lines = []
    if include_type:
        lines.append(f"   - 상품 유형: {item.get('fund_category')}")
    lines.append(f"   - 위험등급: {item.get('risk_grade')}")
    lines.append(_format_expense_line(item))
    aum_line = _format_aum_line(item)
    if aum_line:
        lines.append(aum_line)
    lines.append(_format_return_line(item))
    return "\n".join(lines)


def _explicit_product_context_response(
    state: PensionAgentState,
) -> tuple[str, list[RetrievedItem], dict, bool] | None:
    """명시된 product/class/account는 현재 상품 맥락으로 잠근다.

    특정 판매클래스를 들고 "이 상품"을 묻는 경우 기존 추천 검색으로 빠지면 Core는
    다른 후보 3개를 말하고 Guardian은 사용자가 명시한 상품의 비용을 말하는 충돌이 생긴다.
    단, "비슷한 다른 상품 추천/비교"처럼 대안 추천을 명시한 경우는 기존 recommender에 맡긴다.
    """
    question = state["question"]
    if _asks_alternative_recommendation(question):
        return None

    product_code = _extract_product_code(question)
    class_code = _extract_class_code(question)
    account_type = _extract_account_type(question)
    if not (product_code and class_code and account_type):
        return None

    normalized_account = normalize_pension_account_type(account_type)
    detail = get_pension_class_detail(product_code, class_code, normalized_account)
    if not detail:
        return None

    fund_name = detail.get("fund_name") or product_code
    canonical_class = detail.get("class_code") or class_code
    metric_lines = []
    if detail.get("synthetic_total_expense_ratio") is not None:
        metric_lines.append(
            f"- 합성총보수·비용{_basis_suffix(detail, 'prospectus_effective_date', '투자설명서 효력발생일')}: "
            f"{_format_pct(detail.get('synthetic_total_expense_ratio'))}"
        )
    if detail.get("total_expense_ratio") is not None:
        metric_lines.append(
            f"- 총보수·비용{_basis_suffix(detail, 'prospectus_effective_date', '투자설명서 효력발생일')}: "
            f"{_format_pct(detail.get('total_expense_ratio'))}"
        )
    if detail.get("cost_3y_per_10m_krw") is not None:
        metric_lines.append(f"- 1,000만원 3년 총비용 예시: {detail.get('cost_3y_per_10m_krw')}천원")

    lines = [
        f"말씀하신 상품은 **{fund_name} ({canonical_class})**로 확인됩니다.",
        "",
        "**확인된 상품 정보**",
        f"- 상품코드: {product_code}",
        f"- 계좌 유형: {normalized_account}",
        f"- 판매채널: {detail.get('channel') or '확인 필요'}",
    ]
    if detail.get("risk_grade"):
        lines.append(f"- 위험등급: {detail['risk_grade']}")
    if detail.get("fund_category"):
        lines.append(f"- 상품 유형: {detail['fund_category']}")
    lines.extend(metric_lines)
    aum_line = _format_aum_line(detail, indent="")
    if aum_line:
        lines.append(aum_line)

    if detail.get("investment_objective"):
        lines.append("")
        lines.append("**투자 목적**")
        lines.append(str(detail["investment_objective"]))
    if detail.get("investment_strategy"):
        lines.append("")
        lines.append("**투자 전략**")
        lines.append(str(detail["investment_strategy"]))

    if _asks_cost_topic(question):
        lower = find_lower_cost_pension_class(product_code, canonical_class, normalized_account)
        lines.append("")
        lines.append("**동일 펀드 클래스 비용 확인**")
        if lower.found and lower.eligibility_type == "STANDARD":
            label = _COST_METRIC_LABELS.get(lower.comparison_metric, lower.comparison_metric)
            lines.append(
                f"동일한 연금계좌 유형에서 {label} 기준으로 더 낮은 클래스가 확인됩니다. "
                f"현재 {lower.current_class_code} 클래스는 {_format_pct(lower.current_value)}, "
                f"{lower.target_class_code} 클래스는 {_format_pct(lower.target_value)}입니다."
            )
        else:
            lines.append("동일한 연금계좌 유형에서 바로 비교 가능한 더 낮은 비용의 STANDARD 클래스는 확인되지 않았습니다.")

    content = (
        f"상품코드={product_code}, 클래스={canonical_class}, 계좌유형={normalized_account}, "
        f"판매채널={detail.get('channel')}, 위험등급={detail.get('risk_grade')}, "
        f"유형={detail.get('fund_category')}, "
        f"총보수·비용={detail.get('total_expense_ratio')}%, "
        f"합성총보수·비용={detail.get('synthetic_total_expense_ratio')}%, "
        f"투자설명서효력발생일={detail.get('prospectus_effective_date')}, "
        f"시장잔고={detail.get('aum_krw_million')}백만원, 잔고기준일={detail.get('aum_base_date')}, "
        f"dataset_version={detail.get('dataset_version')}, dataset_status={detail.get('dataset_status')}"
    )
    context = [{"source": f"{fund_name} ({canonical_class})", "content": content, "node": "product_agent"}]
    profile = dict(state.get("recommendation_profile") or {})
    profile["account_type"] = account_type
    return "\n".join(lines), context, profile, False


def _extract_recommendation_profile(state: PensionAgentState) -> dict:
    profile = dict(state.get("recommendation_profile") or {}) if _should_carry_product_history(state) else {}
    text = _combined_user_text(state)
    current = state["question"]

    account = _extract_account_type(text)
    if account:
        profile["account_type"] = account
    elif _has_no_income_status(text):
        # 소득이 없는 신분이면 가입 가능한 계좌가 연금저축 하나로 좁혀진다 — 되물을
        # 필요 없이 근거로 확정할 수 있다(IRP/DC/DB는 가입대상이 정해져 있다).
        profile["account_type"] = "연금저축"
        profile["no_income_status"] = True

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


# 근로·사업소득이 없음을 드러내는 표현. 연금계좌는 **가입 자격이 소득 유무로 갈리므로**
# (연금저축은 누구나, IRP는 직장인·자영업자·직역연금가입자, DB/DC는 재직 근로자),
# 이 신호가 있으면 계좌유형을 되물을 게 아니라 가능한 것을 짚어줘야 한다.
_NO_INCOME_STATUS_MARKERS = (
    "대학생", "대학교", "대학 입학", "대학입학", "신입생", "학생", "고등학생", "중학생",
    "취업 준비", "취업준비", "취준", "무직", "소득이 없", "소득 없", "수입이 없", "수입 없",
    "직장이 없", "일을 안", "백수", "전업주부", "주부", "미성년",
)
# 위 표현이 있어도 소득이 있음을 함께 밝히면 제외한다 — "학생인데 아르바이트로 소득이
# 있어요"처럼 예외가 실재한다.
_HAS_INCOME_MARKERS = (
    "직장인", "회사원", "재직", "근로소득", "사업소득", "자영업", "프리랜서",
    "소득이 있", "소득 있", "월급", "연봉", "아르바이트", "알바",
)


def _has_no_income_status(text: str) -> bool:
    """질문에 "소득이 없는 신분"(학생·무직 등)이 드러나는지 판정한다.

    ⚠️ 계좌유형을 단순히 "사용자에게 물어볼 빈칸"으로만 다루면, 대학생에게 "IRP, DC,
    DB, 연금저축 중 선택해 주세요"처럼 **답할 수 없는 질문**을 던지게 된다(실측:
    "이제 막 대학 입학한 학생인데 노후 대비 상품 추천해줘"). 근거 문서에 답이 이미
    있다 — "연금저축은 누구나 가입할 수 있다. 소득이 없어도 가입이 가능하지만
    ... IRP는 직장인, 자영업자, 직역연금가입자 등 가입대상이 정해져 있다."
    """
    if any(marker in text for marker in _HAS_INCOME_MARKERS):
        return False
    return any(marker in text for marker in _NO_INCOME_STATUS_MARKERS)


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


# 위험선호를 누그러뜨리는 표현. "약간 공격적으로"를 그대로 공격형(1~3등급)으로 읽으면
# 사용자가 스스로 단 완충 표현을 무시하고 가장 위험한 구간을 추천하게 된다 —
# 실측 V11("약간 공격적으로 투자하고 싶은데")이 이 경로였다. 완충 표현이 붙으면
# 한 단계 안쪽(중립형)으로 당긴다.
_RISK_HEDGE_MARKERS = ("약간", "조금", "살짝", "다소", "어느 정도", "적당히", "너무 위험하지")

# ⚠️ "보수"는 쓰지 않는다 — **총보수(수수료)**에도 걸려 오판을 만든다.
# 실측 V23("합리적인 수준의 총보수를 가진 펀드가 뭔가요?")이 위험성향 "안정형"으로
# 확정돼 5등급 이상만 후보에 올랐다. 수수료를 물었는데 위험등급이 좁혀지는 것은
# 사용자가 의도한 바가 아니다. 보수적 성향은 "보수적"으로만 받는다.
_CONSERVATIVE_MARKERS = (
    "안정형", "보수적", "안정적", "안전한", "안전하게", "크게 잃지", "손실 싫",
    "낮은 위험", "위험등급 낮", "원금 지키", "잃고 싶지",
)
_NEUTRAL_MARKERS = ("중립형", "중립", "약간의 변동성", "어느 정도", "중간")
_AGGRESSIVE_MARKERS = ("공격형", "공격", "적극", "높은 수익", "고위험")


def _extract_risk_profile(text: str) -> str | None:
    """질문에서 위험선호를 읽는다. 못 읽으면 None(→ 역질문)이다.

    ⚠️ 여기서 확정한 값은 곧바로 위험등급 필터로 번역된다(_search_args_from_profile).
    즉 이 함수의 오판은 곧 "사용자가 원하지 않은 상품군만 보여주는" 결과가 되므로,
    애매하면 확정하지 말고 None을 내 역질문으로 보내는 편이 낫다.
    """
    hedged = any(marker in text for marker in _RISK_HEDGE_MARKERS)

    if any(word in text for word in _CONSERVATIVE_MARKERS):
        return "안정형"
    if any(word in text for word in _AGGRESSIVE_MARKERS):
        # "약간 공격적" → 공격형(1~3등급)이 아니라 중립형으로 완화한다.
        return "중립형" if hedged else "공격형"
    if any(word in text for word in _NEUTRAL_MARKERS):
        return "중립형"
    # "위험은 조금만 감수하고" — 완충 표현 + 위험 언급은 중립형으로 읽는다.
    # 이 표현을 놓치면 사용자가 이미 말한 성향을 다시 되묻게 된다(실측 V10).
    if hedged and any(word in text for word in ("위험", "리스크", "변동")):
        return "중립형"
    return None


def _extract_loss_tolerance(text: str) -> str | None:
    if "크게 잃지" in text or ("손실" in text and any(word in text for word in ("싫", "피", "낮"))):
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
    if any(word in text for word in ("미국 주식", "미국주식", "해외주식", "해외 주식", "S&P", "s&p", "S&P500", "에스앤피")):
        return "해외주식형 펀드"
    if "국내주식" in text or "국내 주식" in text:
        return "국내주식형 펀드"
    for product_type in ("TDF", "채권혼합형", "채권형", "혼합형", "주식형", "인덱스", "배당형", "원리금보장형"):
        if product_type in text:
            return "혼합형 펀드" if product_type == "혼합형" else product_type
    return None


def _missing_profile_fields(profile: dict) -> list[str]:
    missing = []
    if not profile.get("account_type"):
        missing.append("account_type")
    if not profile.get("risk_profile") and not profile.get("loss_tolerance"):
        missing.append("risk_profile")
    if not profile.get("investment_horizon") and not profile.get("age_or_retirement_horizon"):
        missing.append("investment_horizon")
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


def _grounded_account_constraint(profile: dict) -> tuple[str, list[RetrievedItem]]:
    """clarification mode에서도 DB/Rule 근거가 있는 계좌 제약만 짧게 안내한다."""
    account = profile.get("account_type")

    if profile.get("no_income_status"):
        # account_type이 이미 "연금저축"으로 확정돼 있으므로(profile 조립 단계에서)
        # 아래 IRP/DC 위험자산 분기와는 겹치지 않는다 — 여기서 먼저 반환한다.
        section = (
            "확인된 계좌 제약은 다음과 같습니다.\n"
            "- 소득이 없는 신분이면 연금저축만 가입할 수 있습니다. IRP는 직장인·자영업자·"
            "직역연금가입자 등 가입대상이 정해져 있어 해당하지 않습니다. DC·DB는 재직 "
            "근로자의 퇴직연금이라 역시 대상이 아닙니다.\n"
            "- 다만 연금저축은 소득이 없으면 납입해도 세액공제 혜택은 받을 수 없습니다 — "
            "종합소득(근로소득·사업소득 등)이 있어야 세액공제가 적용됩니다."
        )
        return section, [
            {
                "source": _ACCOUNT_ELIGIBILITY_SOURCE,
                "content": _ACCOUNT_ELIGIBILITY_CONTENT,
                "node": "product_agent",
            }
        ]

    preferred = profile.get("preferred_product_type") or ""
    if account not in {"IRP", "DC"}:
        return "", []
    if not any(marker in preferred for marker in ("주식형", "해외주식형", "국내주식형", "TDF")):
        return "", []

    section = (
        "확인된 계좌 제약은 다음과 같습니다.\n"
        "- IRP/DC에서 주식형·주식혼합형펀드는 위험자산으로 분류되어 위험자산 한도 확인이 필요합니다."
    )
    return section, [
        {
            "source": _INVESTMENT_LIMIT_SOURCE,
            "content": _INVESTMENT_LIMIT_CONTENT,
            "node": "product_agent",
        }
    ]


def _is_unfounded_market_outlook_line(line: str) -> bool:
    """거시 지표를 근거로 특정 자산유형이 유리하다고 단정하는 문장인지 판정한다.

    clarification 상태(계좌유형·위험성향도 아직 모름)에서는 이런 조건부 시장 전망을
    말할 근거가 없다 — RAG 문서에도 없는 미래 예측이라 grounding 위반이다.
    """
    return bool(_MACRO_INDICATOR_RE.search(line)) and bool(_FAVORABLE_OUTLOOK_RE.search(line))


def _apply_clarification_policy(answer: str) -> str:
    """clarification 답변에서 근거 없는 시장/what-if 블록이 남으면 최종 조립 전에 제거한다."""
    lines = answer.splitlines()
    kept: list[str] = []
    skipping_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("**시장 상황이 바뀐다면"):
            skipping_section = True
            continue
        if skipping_section and (not stripped or stripped.startswith("**")):
            skipping_section = False
            if not stripped:
                continue
        if skipping_section:
            continue
        if any(marker in line for marker in _CLARIFICATION_BLOCKED_MARKERS):
            continue
        if _is_unfounded_market_outlook_line(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _clarification_response(profile: dict, missing: list[str]) -> tuple[str, list[RetrievedItem]]:
    questions = _clarification_questions(missing)
    known = _format_profile_summary(profile)
    missing_labels = ", ".join(_PROFILE_FIELD_LABELS.get(field, field) for field in missing)
    known_block = f"\n\n현재 확인된 조건은 다음과 같습니다.\n{known}" if known else ""
    account_constraint, account_context = _grounded_account_constraint(profile)
    # account_constraint(무소득 신분 등 제약 고지)가 이미 계좌 가입대상을 설명했다면
    # _general_guidance_block의 account_type 일반론과 내용이 겹친다 — 중복을 피하려고
    # account_constraint가 비어 있을 때만 일반 기준을 추가한다.
    general_guidance, guidance_context = (
        _general_guidance_block(profile, missing) if not account_constraint else ("", [])
    )
    context = [*account_context, *guidance_context]
    guidance_block = f"{general_guidance}\n\n" if general_guidance else ""
    answer = (
        "현재 질문만으로는 특정 상품을 바로 추천하기 어렵습니다."
        f"{known_block}\n\n"
        f"{guidance_block}"
        "조건에 맞는 상품 후보를 비교하려면 추천 결과를 실제로 바꾸는 정보가 더 필요합니다.\n\n"
        f"{account_constraint}"
        f"{'\n\n' if account_constraint else ''}"
        f"구체적인 추천을 위해 부족한 정보는 {missing_labels}입니다. 다음 정보를 한 번에 알려주세요.\n"
        + "\n".join(f"{i}. {question}" for i, question in enumerate(questions, start=1))
        + "\n\n이 정보가 확인되면 상품 DB에서 투자 가능한 후보를 확인한 뒤, 상품별 근거를 바탕으로 비교하겠습니다."
    )
    return _apply_clarification_policy(answer), context


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


# 위험성향을 실제 검색 조건(위험등급)으로 번역한 결과. _search_args_from_profile과
# 반드시 같은 값이어야 한다 — 보여주는 해석과 실제 필터가 어긋나면 고지가 오히려
# 오해를 만든다. (위험등급은 숫자가 클수록 안전하다: 1등급=매우 높은 위험, 6등급=매우 낮은 위험)
_RISK_GRADE_DISCLOSURE = {
    "안정형": "위험등급 5~6등급",
    "중립형": "위험등급 4등급 이상",
    "공격형": "위험등급 1~3등급",
}


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
        "preferred_product_type": "관심 상품 유형",
    }
    for key, label in labels.items():
        if not profile.get(key):
            continue
        value = profile[key]
        # "안정적인 걸 원해요" 같은 모호한 표현을 코드가 위험등급으로 확정하는데,
        # 그 해석을 알리지 않으면 사용자는 왜 그 상품들만 나왔는지 알 수 없다.
        # 해석 결과를 밝혀, 사용자가 다르게 생각했다면 바로잡을 수 있게 한다.
        if key == "risk_profile" and value in _RISK_GRADE_DISCLOSURE:
            value = f"{value}({_RISK_GRADE_DISCLOSURE[value]}으로 해석)"
        rows.append(f"- {label}: {value}")
    return "\n".join(rows)


def _general_guidance_block(profile: dict, missing: list[str]) -> tuple[str, list[RetrievedItem]]:
    """역질문 답변에도 지금 근거로 말할 수 있는 일반 기준을 담는다.

    ## 왜 필요한가

    조건이 부족해 역질문할 때, 예전에는 질문 목록만 던지고 끝났다("부족한 정보는
    계좌유형, 투자기간입니다. 알려주세요"). 사용자는 아무 답도 못 받은 채 되묻기만
    당한 셈이라 "정보한계 대응"·"요구사항 충족" 평가지표에 불리하다.

    이 서비스는 단일턴 평가라 실제 멀티턴처럼 답을 받은 뒤 이어갈 수 없다 —
    그래서 역질문 자체를 답변에서 빼는 대신, **한 응답 안에 "지금 답할 수 있는
    일반 기준" + "부족한 조건과 역질문"을 함께 넣는다.**

    ## 왜 여기서 새 근거를 만들지 않는가

    이 함수는 새로운 사실을 조사하지 않는다 — 이미 확보해둔 근거 상수만 재사용한다
    (_ACCOUNT_ELIGIBILITY_CONTENT, _RISK_GRADE_DISCLOSURE는 _search_args_from_profile의
    실제 검색 조건과 동일한 값). 없는 내용을 조립해 채우면 이 프로젝트가 하루 종일
    고쳐온 바로 그 할루시네이션 문제를 역질문 경로에 새로 만드는 셈이다.

    ## 무엇을 넣는가 (알려진 조건에 따라 달라진다)

    - account_type이 없으면: 연금저축·IRP 가입대상 차이(계좌선택_가이드와 같은 사실)
    - risk_profile이 확정됐으면: 그 성향이 검색에서 어느 위험등급대로 해석되는지
      (_format_profile_summary가 조건 요약에 붙이는 것과 같은 고지)
    - 그 외에는 아무것도 추가하지 않는다 — 근거 없는 일반론보다 침묵이 낫다.
    """
    lines: list[str] = []
    context: list[RetrievedItem] = []

    if "account_type" in missing:
        lines.append(
            "- 계좌 종류: 연금저축은 소득이 없어도 누구나 가입할 수 있지만(세액공제를 "
            "받으려면 종합소득 필요), IRP는 직장인·자영업자·직역연금가입자 등 가입대상이 "
            "정해져 있습니다. DC·DB는 재직 중인 회사를 통해 가입하는 퇴직연금제도입니다."
        )
        context.append(
            {
                "source": _ACCOUNT_ELIGIBILITY_SOURCE,
                "content": _ACCOUNT_ELIGIBILITY_CONTENT,
                "node": "product_agent",
            }
        )

    risk_profile = profile.get("risk_profile")
    if risk_profile in _RISK_GRADE_DISCLOSURE:
        lines.append(f"- 투자성향({risk_profile}): {_RISK_GRADE_DISCLOSURE[risk_profile]} 상품 위주로 검토됩니다.")

    if not lines:
        return "", []

    section = "현재 조건으로 일반적으로 말씀드릴 수 있는 내용은 다음과 같습니다.\n" + "\n".join(lines)
    return section, context


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
        "해외주식형 펀드": "해외주식형 펀드는 장기 성장자산에 투자할 수 있지만 주식시장 변동과 환율 영향을 함께 받습니다.",
        "국내주식형 펀드": "국내주식형 펀드는 국내 증시 상승의 수혜를 기대할 수 있지만 국내 경기와 증시 변동에 민감합니다.",
        "혼합형 펀드": "혼합형 펀드는 주식과 채권을 함께 담아 한쪽 자산에만 집중하는 위험을 낮추려는 목적에 맞습니다.",
        "배당형 펀드": "배당형 펀드는 배당 성향이 있는 자산에 투자해 장기 현금흐름과 수익을 함께 기대할 때 고려할 수 있습니다.",
    }
    return "\n".join(f"- {reason_map.get(product_type, product_type)}" for product_type in types)


def _conditional_recommendation_guidance(profile: dict) -> str:
    """조건이 부족할 때도 개별 상품 대신 상품 유형 수준의 안전한 방향을 제시한다."""
    account = profile.get("account_type")
    risk = profile.get("risk_profile")
    loss = profile.get("loss_tolerance")
    horizon = profile.get("investment_horizon") or profile.get("age_or_retirement_horizon")
    preferred = profile.get("preferred_product_type")

    lines = ["다만 투자성향과 투자기간에 따라 일반적으로 다음 방향을 고려할 수 있습니다."]
    if preferred:
        lines.append(
            f"- 관심 상품 유형이 {preferred}라면, 해당 유형이 계좌 제한과 본인 위험성향에 맞는지 먼저 확인하는 편이 좋습니다."
        )
    if risk == "안정형" or loss:
        lines.append(
            "- 안정적인 성향이라면 채권형·원리금보장형처럼 변동성이 상대적으로 낮은 상품의 비중을 높이는 방향을 검토할 수 있습니다."
        )
    elif risk == "중립형":
        lines.append(
            "- 중립형 성향이라면 TDF나 채권혼합형처럼 주식과 채권이 분산된 상품을 우선 검토할 수 있습니다."
        )
    elif risk == "공격형":
        lines.append(
            "- 공격형 성향이고 투자기간이 충분히 길다면 주식형·인덱스형처럼 성장자산 비중이 높은 상품을 검토할 수 있습니다."
        )
    else:
        lines.extend(
            [
                "- 투자기간이 길고 공격적인 성향이라면 주식 등 성장자산 비중이 상대적으로 높은 상품이나 TDF를 검토할 수 있습니다.",
                "- 투자기간이 길지만 중립적인 성향이라면 주식과 채권이 분산된 TDF·혼합형 상품을 검토할 수 있습니다.",
                "- 은퇴가 가깝거나 안정적인 성향이라면 채권형·원리금보장형 등 변동성이 상대적으로 낮은 상품의 비중을 높이는 방향을 고려할 수 있습니다.",
            ]
        )
    if horizon and "장기" in horizon:
        lines.append("- 장기 투자라면 단기 수익률보다 위험자산 비중, 비용, 리밸런싱 방식이 더 중요해질 수 있습니다.")
    if account in ("IRP", "DC"):
        lines.append("- IRP/DC에는 위험자산 투자 제한이 있으므로 실제 상품 선정 전 계좌 내 투자 가능 범위를 확인해야 합니다.")
    return "\n".join(lines) + "\n\n"


# _what_if_scenario_block/_scenario_kind는 여기 있었다 — 사용자가 말한 profile만 보고
# "시장 상황이 바뀐다면?" 시나리오를 만들었는데, 실제 검색된 후보(candidates)의 속성과
# 무관했다. 예: "해외주식형"을 요청했지만 실제 후보가 국내 KOSPI 펀드뿐이어도
# "환노출형 상품이라면 원/달러 환율..." 같은 근거 없는 서술이 붙을 수 있었다.
# d69440b에서 모든 호출부를 _candidate_scenario_notes(실제 후보의 fund_name/
# fund_category에 있는 속성만 반영)로 교체하며 의도적으로 끊겼고, _CLARIFICATION_
# BLOCKED_MARKERS로 옛 문구가 실수로 남아도 걸러내는 방어까지 걸려 있다. 호출부가
# 전혀 없는 채로 남아있던 죽은 코드라 제거한다 — 다시 연결하면 이 문제가 재현된다.


def _candidate_scenario_notes(candidates: list[dict]) -> tuple[str, list[RetrievedItem]]:
    """상품 후보의 확인된 속성과 승인된 scenario rule이 모두 있을 때만 추가 점검을 붙인다."""
    joined = " ".join(
        " ".join(str(item.get(key) or "") for key in ("fund_name", "fund_category"))
        for item in candidates
    )
    notes: list[str] = []
    if "주식형" in joined:
        notes.append(
            "- 후보 중 주식형으로 분류된 상품은 주식형 자산 가격 변동에 따라 평가금액이 변동할 수 있습니다."
        )
    if "채권형" in joined:
        notes.append(
            "- 후보 중 채권형으로 분류된 상품은 채권 가격 변동과 발행자 신용위험을 함께 확인해야 합니다."
        )
    if any(marker in joined.upper() for marker in ("USD", "UH", "H]")):
        notes.append(
            "- 후보명이나 유형에서 외화 관련 표기가 확인되는 상품은 환헤지 여부와 통화 관련 조건을 상품 설명에서 확인해야 합니다."
        )
    if not notes:
        return "", []

    section = (
        "**상품 속성 기반 시나리오 점검**\n"
        "아래 내용은 실제 후보 상품의 구조화 속성과 승인된 시나리오 규칙이 함께 확인된 경우만 표시합니다.\n"
        + "\n".join(notes)
    )
    return section, [
        {
            "source": _SCENARIO_RULE_SOURCE,
            "content": _SCENARIO_RULE_CONTENT,
            "node": "product_agent",
        }
    ]


def _recommendation_flow_response(state: PensionAgentState) -> tuple[str, list[RetrievedItem], dict, bool] | None:
    reference_response = _context_reference_response(state)
    if reference_response is not None:
        return reference_response

    explicit_product_response = _explicit_product_context_response(state)
    if explicit_product_response is not None:
        return explicit_product_response

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
        draft, context = _clarification_response(profile, missing)
        return draft, context, profile, True

    if wants_specific or _is_recommendation_intent(combined):
        draft, context = _specific_product_recommendation(profile, state)
        return draft, context, profile, False

    return _product_type_recommendation_answer(profile), [], profile, False


# 펀드명에 해외 투자대상을 드러내는 표시가 있는지 판정한다. search_funds가 돌려주는
# FundSearchResult에는 benchmark 필드가 없어(DB 스키마 자체가 그 필드를 반환하지
# 않음), 펀드명이 사실상 유일하게 확인 가능한 신호다.
_OVERSEAS_FUND_NAME_MARKERS = (
    "미국", "나스닥", "S&P", "글로벌", "해외", "아시아", "유럽", "신흥국", "중국",
    "일본", "베트남", "인도", "선진국",
)


def _is_overseas_fund(fund_name: str | None) -> bool:
    return any(marker in (fund_name or "") for marker in _OVERSEAS_FUND_NAME_MARKERS)


def _requested_overseas_but_candidates_are_domestic(profile: dict, candidates: list[dict]) -> bool:
    """해외주식형을 요청했는데 후보가 실제로는 국내 상품인지 확인한다.

    ⚠️ search_funds의 keyword 매칭은 "주식"처럼 넓은 문자열만 보고 국내/해외를
    구분하지 못한다. 실측: DB 100개 펀드 중 해외 투자대상이 이름에 드러나는 펀드는
    2개뿐이고(그중 1개는 채권형), keyword="주식"으로 검색하면 KOSPI 추종 국내주식형
    펀드가 대량으로 섞여 나온다. "미국 주식 투자하는 상품 추천해줘"에 삼성퇴직연금
    KOSPI200(국내 지수)을 "이 조건에서는 이 상품을" 이라고 제시하면, 사용자는 국내
    자산을 해외 자산으로 오인해 원치 않는 국가·통화 노출을 갖게 된다.
    """
    if "해외" not in (profile.get("preferred_product_type") or ""):
        return False
    return candidates and not any(_is_overseas_fund(c.get("fund_name")) for c in candidates)


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

    overseas_mismatch = _requested_overseas_but_candidates_are_domestic(profile, candidates)
    lines = [f"현재 조건은\n{summary}\n\n으로 정리됩니다."]
    if overseas_mismatch:
        lines.append(
            "다만 해외주식형 상품을 요청하셨는데, 투자설명서 DB에서 해외 투자대상이 "
            "확인되는 후보를 찾지 못했습니다. 아래 후보는 참고용으로 국내 상품 위주의 "
            "결과이며, 실제로는 해외주식형이 아닐 수 있습니다. 해외 상품이 꼭 필요하시면 "
            "가입 금융기관에 별도로 문의해 주세요."
        )
    else:
        lines.append(
            f"이 조건에서는 **{primary.get('fund_name')} ({primary.get('class_name')})**를 우선 후보로 보겠습니다."
        )
    lines.append("아래는 투자설명서 DB의 구조화 수치로 비교한 후보입니다.")
    for i, item in enumerate(candidates, start=1):
        lines.append(
            f"{i}. {item.get('fund_name')} ({item.get('class_name')})\n"
            f"{_format_candidate_metric_block(item, include_type=True)}\n"
            f"   - 추천 이유: 현재 위험성향과 투자기간 기준으로 위험등급·보수·과거 성과를 함께 비교했을 때 후보군에 포함됩니다.\n"
            f"   - 유의사항: 과거 수익률은 미래 수익을 보장하지 않으며, 계좌 내 실제 매수 가능 여부는 금융기관에서 최종 확인이 필요합니다."
        )
    scenario_section, scenario_context = _candidate_scenario_notes(candidates)
    if scenario_section:
        lines.append(scenario_section)
        context.extend(scenario_context)

    limit_note, limit_context = _risk_asset_limit_note(candidates, profile)
    if limit_note:
        lines.append(limit_note)
        context.extend(limit_context)

    return "\n\n".join(lines), context


def _risk_asset_limit_note(candidates: list[dict], profile: dict) -> tuple[str, list[RetrievedItem]]:
    """추천 후보 전부를 매수하면 위험자산 70%(TDF 조건충족 시 DC/IRP 100%) 한도를 넘는지 확인한다.

    ⚠️ check_product_pension_eligibility 툴은 존재하지만 이 결정론 추천 경로에서는 호출되지
    않는다 — 프롬프트로만 "먼저 확인하라"고 지시했는데, 실측(501문항)에서 상품형 62건 중
    이 툴이 실제로 호출된 건 2건(3%)뿐이었다. search_funds가 반환하는 100개 펀드는 전부
    공모펀드라 PRODUCT_RISK_TIER의 FORBIDDEN 상품(사모펀드·개별주식 직접투자)은 애초에
    이 DB에 없으므로 투자금지 상품 추천 사고는 나지 않지만, 후보 3개가 전부 위험자산이면
    그 자체로 위험자산 100% 추천이 되는 경로는 이 결정론 경로 어디에도 막는 코드가 없었다.

    후보를 임의로 바꾸지는 않는다 — 사용자가 명시한 위험성향(예: 공격형)을 무시하고
    안전자산으로 바꿔치기하면 다른 방향의 오답이 된다. 대신 한도 초과 사실과 안전자산
    비중 확대 필요성을 답변에 명시해, 사용자가 판단할 수 있게 한다.
    """
    account_type = profile.get("account_type")
    if account_type not in ("DB", "DC", "IRP"):
        return "", []

    plan = PlanType(account_type)
    tiers = [classify_fund_category_risk_tier(c.get("fund_category"), c.get("fund_name")) for c in candidates]
    if not tiers or any(t == RiskTier.SAFE for t in tiers):
        return "", []

    # TDF 조건충족(감독원장 기준)을 실측으로 확인할 방법이 이 경로엔 없으므로, "TDF"를
    # 선호 상품유형으로 명시한 경우에만 완화된 한도(DC/IRP 100%)를 적용한다.
    is_tdf_preference = "TDF" in (profile.get("preferred_product_type") or "")
    limit_ratio = (
        1.00 if is_tdf_preference and plan in (PlanType.DC, PlanType.IRP) else RISKY_ASSET_LIMIT
    )
    if limit_ratio >= 1.00:
        return "", []

    content = (
        f"{plan.value} 제도의 위험자산 투자한도는 적립금의 {limit_ratio:.0%}까지입니다. "
        "추천 후보 전부가 위험자산으로 분류되어, 이 후보들만으로 포트폴리오를 구성하면 "
        "한도를 초과할 수 있습니다."
    )
    note = (
        f"⚠️ 위 후보는 모두 위험자산으로 분류됩니다. {plan.value} 제도는 위험자산 투자한도가 "
        f"적립금의 **{limit_ratio:.0%}**까지이므로, 이 후보들만으로 포트폴리오를 채우면 한도를 "
        "초과할 수 있습니다. 안전자산(예금·국공채 펀드 등)을 일부 포함해 비중을 조절하는 것을 "
        "권장합니다."
    )
    return note, [{"source": "doc58 퇴직연금 적립금 운용 및 투자한도 안내", "content": content, "node": "product_agent"}]


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
            f"투자설명서효력발생일={item.get('prospectus_effective_date')}, "
            f"1년수익률={item.get('return_1y')}%, 3년수익률={item.get('return_3y')}%, "
            f"설정이후수익률={item.get('return_since_inception')}%, "
            f"수익률기준일={item.get('return_asof_date')}, "
            f"시장잔고={item.get('aum_krw_million')}백만원, 잔고기준일={item.get('aum_base_date')}, "
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
    """조건이 충분한 추천 요청인데 LLM이 상품 검색을 건너뛴 경우 투자설명서 DB 후보를 보강한다.

    ⚠️ 진입 조건은 반드시 "실제 추천 의도"여야 한다. 예전에는 _is_product_recommendation
    ("추천"/"상품"/"펀드" 중 하나만 있어도 True) + _has_account_type만 봤는데, 그러면
    "연금저축 펀드 환매 제한기간이 있나요?"처럼 추천과 무관한 제도 질문도("펀드"+"연금저축")
    조건을 통과해 임의의 펀드 3개를 근거로 끌어왔다. 더 나쁜 것은 이 폴백이 LLM이 툴을
    호출하지 않았을 때(retrieved_context가 빈 경우) 발동한다는 점이다 — LLM이 "제도 질문이라
    상품 데이터로 답할 수 없다"는 지시를 올바르게 따를수록 오히려 이 폴백에 덮여버리는
    역설이 생긴다. 그래서 추천 의도를 명시적으로 요구한다.

    ⚠️ 반환값은 LLM에게 다시 넘기는 지시문이 아니라 **최종 사용자 답변 그 자체**다.
    호출부(build_product_agent_node) 두 곳 모두 이 draft를 LLM 재호출 없이 바로
    product_draft로 쓴다. 예전 버전은 "~하세요/~설명하세요" 같은 지시문 스타일로
    작성돼 있었다 — 정상 경로(react_agent가 성공)에서는 이 함수가 아예 호출되지
    않아 501문항 실측에는 드러나지 않았지만, react_agent가 예외를 던지는 경로(CLOVA
    API 간헐적 400 오류 등, 이미 문서화된 실측 사례)에서는 이 지시문이 그대로 사용자
    에게 노출된다. _specific_product_recommendation과 같은 완성된 서술형으로 쓴다.
    """
    text = _combined_user_text(state)
    if not (_is_recommendation_intent(text) or _is_specific_recommendation_request(text)):
        return "", []
    if not _has_account_type(text):
        return "", []

    results = search_funds.invoke(_risk_search_args(text))
    if not isinstance(results, list) or not results:
        return "", []

    unique_results = _unique_fund_candidates(results, limit=3)
    context = _fund_candidates_to_context(unique_results)

    profile = _extract_recommendation_profile(state)
    primary = _select_primary_candidate(unique_results, text)
    overseas_mismatch = _requested_overseas_but_candidates_are_domestic(profile, unique_results)
    lines = ["확인된 조건을 기준으로 투자설명서 DB에서 후보를 비교했습니다."]
    if overseas_mismatch:
        lines.append(
            "다만 해외주식형 상품을 요청하셨는데, 투자설명서 DB에서 해외 투자대상이 "
            "확인되는 후보를 찾지 못했습니다. 아래 후보는 참고용으로 국내 상품 위주의 "
            "결과이며, 실제로는 해외주식형이 아닐 수 있습니다. 해외 상품이 꼭 필요하시면 "
            "가입 금융기관에 별도로 문의해 주세요."
        )
    else:
        lines.append(
            f"이 조건에서는 **{primary.get('fund_name')} ({primary.get('class_name')})**를 "
            "우선 후보로 보겠습니다."
        )
    lines.append("아래는 투자설명서 DB의 구조화 수치로 비교한 후보입니다.")
    for i, item in enumerate(unique_results, start=1):
        lines.append(
            f"{i}. {item.get('fund_name')} ({item.get('class_name')})\n"
            f"{_format_candidate_metric_block(item, include_type=False)}\n"
            f"   - 판매채널: {item.get('sales_channel')}"
        )
    lines.append(
        "과거 수익률은 미래 수익을 보장하지 않으며, 계좌 내 실제 매수 가능 여부는 "
        "금융기관에서 최종 확인이 필요합니다."
    )

    limit_note, limit_context = _risk_asset_limit_note(unique_results, profile)
    if limit_note:
        lines.append(limit_note)
        context.extend(limit_context)

    return "\n\n".join(lines), context


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
                    "product_fallback_used": True,
                }
            # ⚠️ 여기서 raise하면 그래프 전체가 죽어 API가 500을 반환하고 그 문항은
            # 무응답으로 0점 처리된다 (실측: 501문항 평가에서 5건, 전부 CLOVA 간헐적
            # 400 "Unsupported function" 오류가 재시도 예산을 다 쓴 뒤 폴백까지 조건
            # 불충분(계좌유형 등)으로 실패한 경우). "지연보다 무응답이 압도적으로
            # 비싸다"(llm.py)는 이 프로젝트의 원칙에 따라, 여기서는 절대 죽지 않고
            # 최소한 계좌유형을 되묻는 역질문으로 응답한다 — 정보가 부족해 정형 추천을
            # 못 한다는 사실 자체가 사용자에게 유용한 답이다.
            missing = _missing_profile_fields(_extract_recommendation_profile(state))
            fallback_questions = _clarification_questions(missing) or [
                "어떤 계좌에서 투자할 예정인가요? IRP, DC, DB, 연금저축 중 선택해 주세요."
            ]
            draft = (
                "일시적인 오류로 상품 검색을 완료하지 못했습니다. "
                "정확한 추천을 위해 다음 정보를 알려주시면 다시 확인해드리겠습니다.\n\n"
                + "\n".join(f"- {q}" for q in fallback_questions)
            )
            return {
                "product_draft": draft,
                "retrieved_context": [],
                "tool_trace": [],
                "needs_clarification": True,
                "recommendation_stage": "clarification",
                "missing_information": [_PROFILE_FIELD_LABELS.get(f, f) for f in missing],
                "clarification_questions": fallback_questions,
                "response_mode": "clarification_included",
                "repair_attempted": state.get("verification") is not None,
                "product_fallback_used": True,
            }
        messages = result["messages"]

        retrieved_context = build_retrieved_context(messages, node="product_agent")

        final_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        draft, needs_clarification = split_clarification_marker(
            final_ai.content if final_ai else ""
        )
        if needs_clarification:
            draft = _apply_clarification_policy(draft)
        fallback_used = False
        if not retrieved_context:
            fallback_draft, fallback_context = _fallback_product_recommendation(state)
            if fallback_context:
                draft = fallback_draft
                retrieved_context = fallback_context
                fallback_used = True

        # ⚠️ response_mode를 안 채우면 Guardian이 이 경로를 절대 못 탄다 — 같은 결함이
        # info_agent.py의 LLM 자유 응답 경로에도 있었다(실측: response_mode=None이라
        # grounded=True/requirements_met=True를 다 통과해도 Guardian 노드 자체에
        # 못 갔다). 같은 원인이므로 같은 방식으로 고친다.
        return {
            "product_draft": draft,
            "retrieved_context": retrieved_context,
            "tool_trace": build_tool_trace(messages, node="product_agent"),
            "needs_clarification": needs_clarification,
            "response_mode": "clarification_included" if needs_clarification else "complete",
            "recommendation_stage": "clarification" if needs_clarification else None,
            "repair_attempted": state.get("verification") is not None,
            "product_fallback_used": fallback_used,
        }

    return product_agent_node
