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

from src.agents.context import (
    build_repair_note,
    build_retrieved_context,
    build_tool_trace,
    dedupe_context,
    history_to_messages,
    split_clarification_marker,
)
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState
from src.agents.tools import PRODUCT_AGENT_TOOLS

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

역질문은 꼭 필요한 것만 최소로 물으세요. 위험선호·투자기간 등 핵심 조건이 파악됐다면(이전
대화에서 이미 답한 것 포함) 더 되묻지 말고 추천을 진행하세요 — 이때 지금까지 확인된 조건을
먼저 정리해 보여주고, 아직 확인 안 된 조건(예: 계좌유형)은 "IRP·DC 등 계좌유형에 따라 매수
가능 여부가 달라질 수 있습니다"처럼 추천의 전제·유의사항으로 명시하는 조건부 추천으로
답하세요. 단정적 표현("이 상품이 가장 좋습니다")은 쓰지 마세요.

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


def build_product_agent_node():
    llm = get_llm(PRODUCT_AGENT_MODEL)
    react_agent = create_agent(model=llm, tools=PRODUCT_AGENT_TOOLS, system_prompt=PRODUCT_AGENT_SYSTEM_PROMPT)

    def product_agent_node(state: PensionAgentState) -> dict:
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
        result = invoke_with_retry(
            react_agent, {"messages": [*history_messages, HumanMessage(content=question)]}
        )
        messages = result["messages"]

        retrieved_context = build_retrieved_context(messages, node="product_agent")

        final_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        draft, needs_clarification = split_clarification_marker(
            final_ai.content if final_ai else ""
        )

        return {
            "product_draft": draft,
            "retrieved_context": retrieved_context,
            "tool_trace": build_tool_trace(messages, node="product_agent"),
            "needs_clarification": needs_clarification,
            "repair_attempted": state.get("verification") is not None,
        }

    return product_agent_node
