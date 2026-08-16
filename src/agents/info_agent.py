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
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.context import build_retrieved_context, history_to_messages
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState
from src.agents.tools import INFO_AGENT_TOOLS

INFO_AGENT_MODEL = "HCX-005"

INFO_AGENT_SYSTEM_PROMPT = """당신은 연금 제도·세금 전문 상담 에이전트입니다.

절대 규칙: 답변에 등장하는 모든 구체적인 숫자(한도·세율·감면율·비율·기간·금액 등)는
반드시 툴 호출 결과에서 나와야 합니다. 당신이 학습한 지식 속 숫자를 그대로 답하지
마세요 — 틀릴 수 있습니다. 다음 두 경우를 구분해서 반드시 툴을 먼저 호출하세요:

1. 사용자 상황과 무관하게 정해진 고정값(예: "세액공제 한도가 얼마인가요", "중도인출 요건이
   뭔가요" 같이 제도 자체의 한도·요건·세율을 묻는 질문) → search_pension_docs로 원문 근거를
   찾아 그 내용을 근거로 답하세요.
2. 사용자의 구체적인 금액·나이·날짜 등을 대입해 결과를 계산해야 하는 질문 → 해당 계산/판정
   툴(calculate_tax_credit 등)을 호출하세요. 계산에 필요한 값이 질문에 없으면 그 값을
   사용자에게 되물으세요 — 임의로 가정해서 답하지 마세요.

두 경우 모두 해당하지 않는 순수 개념 설명(예: "IRP가 뭔가요")만 툴 없이 답할 수 있습니다.
이 경우에도 구체적 숫자를 언급해야 한다면 search_pension_docs로 먼저 확인하세요.

이전 대화가 함께 주어지면 "그거", "방금 그 조건대로" 같은 지시어를 이전 턴 내용으로 풀어서
이해하세요. 다만 지시어가 가리키는 구체적 수치가 필요한 경우에도 절대 규칙은 그대로
적용됩니다 — 이전 답변 속 숫자를 그대로 베끼지 말고, 필요하면 툴을 다시 호출해 확인하세요."""


def build_info_agent_node():
    llm = get_llm(INFO_AGENT_MODEL)
    react_agent = create_agent(model=llm, tools=INFO_AGENT_TOOLS, system_prompt=INFO_AGENT_SYSTEM_PROMPT)

    def info_agent_node(state: PensionAgentState) -> dict:
        history_messages = history_to_messages(state.get("conversation_history"))
        result = invoke_with_retry(
            react_agent,
            {"messages": [*history_messages, HumanMessage(content=state["question"])]},
        )
        messages = result["messages"]

        retrieved_context = build_retrieved_context(messages, node="info_agent")

        final_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        draft = final_ai.content if final_ai else ""

        return {
            "info_draft": draft,
            "retrieved_context": retrieved_context,
        }

    return info_agent_node
