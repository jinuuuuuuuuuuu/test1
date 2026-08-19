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

from src.agents.context import (
    build_repair_note,
    build_retrieved_context,
    build_tool_trace,
    history_to_messages,
    split_clarification_marker,
)
from src.agents.deterministic_info import deterministic_info_response
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState, RetrievedItem, ToolCallRecord
from src.agents.tools import INFO_AGENT_TOOLS, search_pension_docs

INFO_AGENT_MODEL = "HCX-005"

INFO_AGENT_SYSTEM_PROMPT = """당신은 연금 제도·세금 전문 상담 에이전트입니다.

서비스 범위: 연금 제도(DB/DC/IRP·연금저축, 디폴트옵션·실물이전·중도인출 등)와 연금
세제(세액공제·연금소득세·퇴직소득세 등)에 대해서만 답합니다. 범위 밖 주제(일반 사업소득
절세, 주식·부동산, 일반 상식 등)는 학습 지식으로 답하지 말고 "본 서비스의 상담 범위를
벗어난다"고 한계를 고지하세요. [범위 안내]가 함께 주어지면 범위 밖 부분은 한계를 밝히고,
안내된 연금 관점으로만 답하세요.

절대 규칙: 답변에 등장하는 모든 구체적인 숫자(한도·세율·감면율·비율·기간·금액 등)는
반드시 툴 호출 결과에서 나와야 합니다. 당신이 학습한 지식 속 숫자를 그대로 답하지
마세요 — 틀릴 수 있습니다. 다음 두 경우를 구분해서 반드시 툴을 먼저 호출하세요:

1. 사용자 상황과 무관하게 정해진 고정값(예: "세액공제 한도가 얼마인가요", "중도인출 요건이
   뭔가요" 같이 제도 자체의 한도·요건·세율을 묻는 질문) → search_pension_docs로 원문 근거를
   찾아 그 내용을 근거로 답하세요.
2. 사용자의 구체적인 금액·나이·날짜 등을 대입해 결과를 계산해야 하는 질문 → 해당 계산/판정
   툴(calculate_tax_credit 등)을 호출하세요. 계산에 필요한 값이 질문에 없으면 그 값을
   사용자에게 되물으세요 — 임의로 가정해서 답하지 마세요. 평가 환경은 단일턴이므로 조건이
   부족하면 한 가지 값만 묻지 말고, 현재 제공된 정보·현재 답변 가능한 일반 기준·부족한 입력값
   전체·입력값별 구체적 역질문을 첫 답변 안에 모두 포함하세요. 이렇게 조건이 부족해 되묻는
   답변은 반드시 [추가 확인 필요] 로 시작하세요.

두 경우 모두 해당하지 않는 순수 개념 설명(예: "IRP가 뭔가요")만 툴 없이 답할 수 있습니다.
이 경우에도 구체적 숫자를 언급해야 한다면 search_pension_docs로 먼저 확인하세요.

search_pension_docs가 빈 결과([])를 반환하면 그 내용은 보유 자료에 없는 것입니다 —
학습한 지식으로 메워서 답하지 말고, "보유 자료로는 확인되지 않습니다"라고 한계를
고지하세요.

이전 대화가 함께 주어지면 "그거", "방금 그 조건대로" 같은 지시어를 이전 턴 내용으로 풀어서
이해하세요. 다만 지시어가 가리키는 구체적 수치가 필요한 경우에도 절대 규칙은 그대로
적용됩니다 — 이전 답변 속 숫자를 그대로 베끼지 말고, 필요하면 툴을 다시 호출해 확인하세요."""


def _missing_context_response(state: PensionAgentState) -> tuple[str, list[RetrievedItem]] | None:
    question = state["question"]
    if state.get("conversation_history"):
        return None
    if not any(word in question for word in ("그거", "그중", "방금", "앞에서", "위에서", "다시 계산")):
        return None

    draft = (
        "현재 평가 호출에는 이전 대화 내용이 함께 제공되지 않아, 질문이 가리키는 조건이나 대상을 "
        "확인할 수 없습니다.\n\n"
        "이전 조건을 추측해서 계산하거나 판단하면 잘못된 세액공제액, 인출 가능 여부, 상품 정보를 "
        "안내할 수 있으므로 확정 답변은 보류하겠습니다.\n\n"
        "정확한 답변을 위해 다음 정보를 한 번에 알려주세요.\n"
        "1. 다시 계산하거나 확인할 대상이 무엇인가요? 예: 세액공제, 연금수령한도, 중도인출 가능 여부\n"
        "2. 계산 질문이라면 필요한 금액, 나이, 계좌유형, 소득금액을 함께 알려주세요.\n"
        "3. 상품 질문이라면 상품명 또는 상품코드를 알려주세요.\n"
        "4. 이전 답변의 특정 항목을 묻는 것이라면 해당 항목의 내용을 질문에 함께 적어주세요."
    )
    return draft, []


def _doc_search_context(question: str) -> tuple[list[RetrievedItem], list[ToolCallRecord]]:
    """LLM이 RAG 호출을 건너뛴 경우에도 정보형 질문은 한 번 직접 검색해 근거를 확보한다."""
    results = search_pension_docs.invoke({"query": question, "k": 5})
    if not isinstance(results, list) or not results:
        return [], [
            {
                "node": "info_agent",
                "tool": "search_pension_docs",
                "args": f'query="{question}", k=5',
                "result": "검색 결과 없음 (보유 문서에 관련 내용 없음)",
            }
        ]

    items: list[RetrievedItem] = []
    for r in results:
        chunk_id = r.get("chunk_id", "")
        document_id = chunk_id.split("_chunk", 1)[0] if "_chunk" in chunk_id else chunk_id
        label = r.get("file_title") or "search_pension_docs"
        section = r.get("section")
        source = f"{label} — {section}" if section else label
        items.append(
            {
                "source": source,
                "content": r.get("content", ""),
                "node": "info_agent",
                "chunk_id": chunk_id,
                "document_id": document_id,
                "file_title": r.get("file_title", ""),
                "section": section or "",
                "source_location": r.get("source_location", ""),
            }
        )
    titles = "; ".join(item["source"] for item in items[:3])
    return items, [
        {
            "node": "info_agent",
            "tool": "search_pension_docs",
            "args": f'query="{question}", k=5',
            "result": f"{len(items)}건 검색: {titles}",
        }
    ]


def build_info_agent_node():
    llm = get_llm(INFO_AGENT_MODEL)
    react_agent = create_agent(model=llm, tools=INFO_AGENT_TOOLS, system_prompt=INFO_AGENT_SYSTEM_PROMPT)

    def info_agent_node(state: PensionAgentState) -> dict:
        missing_context = _missing_context_response(state)
        if missing_context is not None:
            draft, retrieved_context = missing_context
            return {
                "info_draft": draft,
                "retrieved_context": retrieved_context,
                "tool_trace": [],
                "needs_clarification": True,
                "missing_information": ["이전 대화 문맥", "계산 또는 확인 대상", "필요 입력값"],
                "clarification_questions": [
                    "다시 계산하거나 확인할 대상이 무엇인가요?",
                    "계산에 필요한 금액, 나이, 계좌유형, 소득금액을 함께 알려주세요.",
                    "상품 질문이라면 상품명 또는 상품코드를 알려주세요.",
                ],
                "response_mode": "clarification_included",
                "repair_attempted": state.get("verification") is not None,
                "deterministic_info": False,
            }

        deterministic = deterministic_info_response(state["question"])
        if deterministic is not None:
            draft, retrieved_context = deterministic
            deterministic_needs_clarification = "정확한 계산을 위해 다음 정보를 한 번에 알려주세요" in draft
            return {
                "info_draft": draft,
                "retrieved_context": retrieved_context,
                "tool_trace": [],
                "needs_clarification": deterministic_needs_clarification,
                "missing_information": ["연금저축 납입액", "IRP 납입액", "총급여 또는 종합소득금액"]
                if deterministic_needs_clarification
                else [],
                "clarification_questions": [
                    "올해 연금저축에 납입한 금액은 얼마인가요?",
                    "올해 IRP에 납입한 금액은 얼마인가요?",
                    "직장인 총급여 또는 개인사업자 종합소득금액은 얼마인가요?",
                ]
                if deterministic_needs_clarification
                else [],
                "response_mode": "clarification_included" if deterministic_needs_clarification else "complete",
                "repair_attempted": state.get("verification") is not None,
                "deterministic_info": True,
            }

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

        history_messages = history_to_messages(state.get("conversation_history"))
        result = invoke_with_retry(
            react_agent, {"messages": [*history_messages, HumanMessage(content=question)]}
        )
        messages = result["messages"]

        retrieved_context = build_retrieved_context(messages, node="info_agent")
        tool_trace = build_tool_trace(messages, node="info_agent")

        final_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        draft, needs_clarification = split_clarification_marker(
            final_ai.content if final_ai else ""
        )
        if not needs_clarification and not retrieved_context:
            forced_context, forced_trace = _doc_search_context(state["question"])
            retrieved_context = forced_context
            tool_trace = [*tool_trace, *forced_trace]

        return {
            "info_draft": draft,
            "retrieved_context": retrieved_context,
            "tool_trace": tool_trace,
            "needs_clarification": needs_clarification,
            "repair_attempted": state.get("verification") is not None,
            "deterministic_info": False,
        }

    return info_agent_node
