"""② 정보 Agent 노드 — 연금 제도·세금 질문에 답한다.

HCX-005 + 규칙엔진/RAG 툴 6개(INFO_AGENT_TOOLS)를 create_react_agent로 묶어 "필요하면 툴을
호출하고, 충분해지면 답을 낸다"는 표준 ReAct 루프를 구성한다.

⚠️ HCX-007은 기본적으로 Thinking이 켜져 있어 thinking={"effort":"none"}으로 끄지 않으면
bind_tools()가 400 "tools, reasoning" 에러를 낸다(네이버 공식 문서: Function calling과
추론(Thinking)은 동시 이용 불가). thinking을 꺼도 HCX-007로 tool calling은 가능하지만,
여기서는 Thinking 부가 설정 없이 바로 되는 HCX-005를 택했다 — 필요하면
get_llm("HCX-007", thinking_effort="none")으로 교체 가능.
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.context import build_retrieved_context, history_to_messages
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState
from src.agents.tools import INFO_AGENT_TOOLS
from src.storage.queries import search_pension_docs as _search_pension_docs


INFO_AGENT_MODEL = "HCX-005"

INFO_AGENT_SYSTEM_PROMPT = """당신은 연금 제도·세금 전문 상담 에이전트입니다.

[최우선 원칙]
모든 답변은 반드시 실제 툴 호출을 통해 확보한 근거를 기반으로 작성해야 합니다.
모델이 학습한 일반 지식이나 기억만으로 사실을 답하지 마세요.

특히 모든 정보형 질문은 반드시 search_pension_docs를 최소 1회 호출하여
관련 원문 근거를 확보한 뒤 답변하세요.
순수 개념 질문도 예외가 아닙니다.

예:
- "IRP가 뭐야?"
- "연금저축이 뭐야?"
- "DC형 퇴직연금이 뭐야?"
→ 모두 search_pension_docs를 먼저 호출하세요.


[숫자 사용 절대 규칙]
답변에 등장하는 모든 구체적인 숫자
(한도·세율·감면율·비율·기간·금액·나이 등)는
반드시 툴 호출 결과에서 확인된 값이어야 합니다.

당신이 학습한 지식이나 이전 AI 답변에 들어 있던 숫자를 그대로 사용하지 마세요.
필요한 숫자는 현재 턴에서 다시 툴을 호출해 확인하세요.


[툴 선택 규칙]

1. 연금 제도 자체의 고정값·규정·개념을 묻는 질문
예:
- "세액공제 한도가 얼마인가요?"
- "IRP가 뭔가요?"
- "중도인출 요건이 뭔가요?"
- "연금저축과 IRP의 차이는?"
- "그럼 나머지는 IRP로 채울 수 있어?"

→ search_pension_docs를 반드시 호출하여 원문 근거를 확인하세요.


2. 사용자의 구체적인 금액·나이·날짜·소득 등을 대입하여
계산하거나 판정해야 하는 질문

→ 해당 계산/판정 툴을 반드시 호출하세요.

예:
- calculate_tax_credit
- calculate_pension_withdrawal
- check_early_withdrawal
- check_default_option
- check_in_kind_transfer

이 경우에도 계산 결과가 어떤 제도 규정에 근거하는지 설명하는 답변이라면
search_pension_docs도 함께 호출하여 문서 근거를 확보하세요.


3. 계산과 제도 설명이 함께 필요한 질문

→ 계산/판정 툴과 search_pension_docs를 모두 호출하세요.


[멀티턴 규칙]
이전 대화가 함께 주어지면
"그거", "그럼", "나머지", "방금 그 조건대로", "그중 두 번째" 같은 지시어를
이전 턴의 문맥을 이용해 해석하세요.

다만 이전 AI 답변은 사실 근거가 아닙니다.
이전 답변에 숫자나 규정이 등장했더라도 그대로 재사용하지 말고,
현재 답변에 필요하면 관련 툴을 다시 호출하여 확인하세요.


[근거 규칙]
- 모든 최종 답변에는 최소 1개 이상의 문서 근거가 있어야 합니다.
- 정보형 질문에서는 search_pension_docs 결과가 반드시 존재해야 합니다.
- 툴 결과에서 확인되지 않은 수치나 조건을 추가하지 마세요.
- 검색 결과가 없거나 검색에 실패했다면 모델의 기억으로 보완하지 마세요.
- 근거를 확보하지 못한 경우에는 정확한 답변을 생성하지 말고
  근거를 확인하지 못했다고 명시하세요.
"""

def build_info_agent_node():
    llm = get_llm(INFO_AGENT_MODEL)
    react_agent = create_agent(
        model=llm,
        tools=INFO_AGENT_TOOLS,
        system_prompt=INFO_AGENT_SYSTEM_PROMPT
    )

    def info_agent_node(state: PensionAgentState) -> dict:
        history_messages = history_to_messages(
            state.get("conversation_history")
        )

        result = invoke_with_retry(
            react_agent,
            {
                "messages": [
                    *history_messages,
                    HumanMessage(content=state["question"])
                ]
            }
        )
        messages = result["messages"]

        retrieved_context = build_retrieved_context(
            messages,
            node="info_agent"
        )

        # 문서 근거가 실제로 확보되었는지 확인
        has_document_evidence = any(
            item.get("document_id")
            for item in retrieved_context
        )

        # 모델이 search_pension_docs를 호출했는지 확인
        search_was_called = any(
            isinstance(msg, ToolMessage)
            and msg.name == "search_pension_docs"
            for msg in messages
        )

        # 문서 근거가 없고,
        # search_pension_docs도 호출하지 않았다면 fallback 실행
        if not has_document_evidence and not search_was_called:
            fallback_results = _search_pension_docs(
                query=state["question"],
                k=5,
            )

            for r in fallback_results:
                chunk_id = r.chunk_id or ""

                # 예: doc19_chunk03 → doc19
                document_id = (
                    chunk_id.split("_chunk", 1)[0]
                    if "_chunk" in chunk_id
                    else chunk_id
                )

                retrieved_context.append({
                    "source": document_id or "search_pension_docs",
                    "content": r.content,
                    "node": "info_agent",
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "file_title": r.file_title,
                    "section": r.section,
                    "source_location": r.source_location,
                })

        final_ai = next(
            (
                m for m in reversed(messages)
                if isinstance(m, AIMessage) and m.content
            ),
            None,
        )

        draft = final_ai.content if final_ai else ""

        return {
            "info_draft": draft,
            "retrieved_context": retrieved_context,
        }
    return info_agent_node