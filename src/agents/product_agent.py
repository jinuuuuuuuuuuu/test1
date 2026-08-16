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


def _combined_user_text(state: PensionAgentState) -> str:
    history = state.get("conversation_history") or []
    previous_user_text = "\n".join(turn.get("question", "") for turn in history)
    return f"{previous_user_text}\n{state['question']}"


def _has_account_type(text: str) -> bool:
    return any(token in text.upper() for token in ("IRP", "DC", "DB"))


def _is_product_recommendation(text: str) -> bool:
    return any(token in text for token in ("추천", "상품", "펀드", "굴릴만한"))


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

    unique_results = []
    seen_codes = set()
    for item in results:
        code = item.get("product_code")
        if code in seen_codes:
            continue
        seen_codes.add(code)
        unique_results.append(item)
        if len(unique_results) >= 3:
            break

    context = []
    for item in unique_results:
        label = f"{item.get('fund_name', '상품명 없음')} ({item.get('class_name', '')})"
        content = (
            f"상품코드={item.get('product_code')}, 위험등급={item.get('risk_grade')}, "
            f"총보수={item.get('total_expense_ratio')}%, "
            f"1년수익률={item.get('return_1y')}%, 3년수익률={item.get('return_3y')}%, "
            f"설정이후수익률={item.get('return_since_inception')}%, "
            f"판매채널={item.get('sales_channel')}, 유형={item.get('fund_category')}"
        )
        context.append({"source": label, "content": content, "node": "product_agent"})

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
