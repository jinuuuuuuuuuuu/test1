"""② 정보 Agent 노드 — 연금 제도·세금 질문에 답한다.

HCX-005 + 규칙엔진/RAG 툴 5개(INFO_AGENT_TOOLS)를 create_react_agent로 묶어 "필요하면 툴을
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
from src.agents.deterministic_info import (
    deterministic_response_for,
    in_kind_transfer_judgement_response,
)
from src.agents.in_kind_transfer_intent import has_in_kind_transfer_intent
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.query_rewrite import rewrite_search_queries
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
   답변에는 반드시 [추가 확인 필요] 표시를 포함하세요 (위치는 어디든 상관없습니다).

★ 1과 2는 배타적 선택지가 아닙니다 — 한 질문이 둘 다 요구하면 둘 다 호출하세요.
질문이 여러 항목을 물으면(예: "언제부터 인출 가능하고, 얼마까지 되고, 세금은 어떻게
되나요"), **항목마다 필요한 툴을 각각 호출한 뒤 전부 답하세요.** 한 툴만 부르고 나머지를
학습 지식이나 추측으로 메우면 안 됩니다. 답하기 전에 "질문이 요구한 항목을 빠짐없이
다뤘는가"를 확인하고, 아직 근거가 없는 항목이 남았으면 그 항목의 툴을 마저 호출하세요.
제도 규정과 계산이 함께 필요한 질문은 search_pension_docs로 규정을 확인하고 계산 툴로
수치를 확정하는 것이 정상 경로입니다.

실물이전 개별 판정은 질문에 상품 유형·상태가 명시된 경우에만 할 수 있습니다. 계좌 유형
IRP는 환매조건부채권(RP) 같은 상품 유형이 아니므로, "IRP 상품을 그대로 이전하는 방법"처럼
상품이 특정되지 않은 질문에서 RP·MMF·사모펀드 등의 상태를 임의로 가정하지 마세요. 이 경우
search_pension_docs로 확인 가능한 일반 기준만 안내하고, 제공 자료에 없는 신청 절차는 모른다고
밝히세요.

두 경우 모두 해당하지 않는 순수 개념 설명(예: "IRP가 뭔가요")만 툴 없이 답할 수 있습니다.
이 경우에도 구체적 숫자를 언급해야 한다면 search_pension_docs로 먼저 확인하세요.

search_pension_docs가 빈 결과([])를 반환하면 그 내용은 보유 자료에 없는 것입니다 —
학습한 지식으로 메워서 답하지 말고, "보유 자료로는 확인되지 않습니다"라고 한계를
고지하세요.

★ 근거로 쓰기 전에 "이 문서가 질문이 묻는 그 제도의 문서인가"를 먼저 확인하세요.
검색 결과에는 **이름이 비슷하지만 서로 다른 제도**의 문서가 섞여 옵니다. 대표적으로
"(구)개인연금저축"은 지금의 "연금저축"과 다른 폐지된 옛 제도이고, 한도·과세·요건이
전부 다릅니다(예: 중도해지 과세가 옛 제도는 이자소득세 15.4%, 현행 연금저축은
기타소득세 16.5%). 문서 제목이나 본문에 "(구)", "구 제도", "폐지", "종전" 같은
표시가 있으면 그 수치를 현행 제도 답변에 그대로 쓰지 마세요 — 근거에 있는 숫자라도
**다른 제도의 숫자면 오답**입니다. 사용자가 그 옛 제도를 명시적으로 물은 경우에만
쓰고, 그때도 "(구)개인연금저축 기준"임을 답변에 밝히세요.

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


def _search_with_rewrites(question: str) -> tuple[list[dict], list[str]]:
    """원문 검색 결과에 '제도 용어로 재작성한 질의'의 결과를 더해 돌려준다.

    사용자는 일상어로 묻고 문서는 제도 용어로 쓰여 있어, 원문을 그대로 임베딩하면
    엉뚱한 문서가 잡힌다(실측 S03 "전업주부인데 노후 대비..." -> MP 알림톡 FAQ,
    d=31.80). 그 빈틈을 LLM이 학습 지식으로 메우는 것이 할루시네이션의 최대
    발생원이었다 — 같은 질문에서 폐지된 세액공제 한도 400만원이 창작됐다.

    ⚠️ 원문 검색이 0건이면 재작성 결과도 쓰지 않는다. 재작성기는 무엇을 주든
    연금 검색어를 만들어내므로("오늘 점심 뭐 먹지" -> 5건), 이 가드가 없으면
    범위 밖 질문이 되살아나 "빈 리스트 = 보유 문서에 없음"이라는 계약이 깨진다.

    ⚠️ 원문 결과를 버리지 않고 **더한다**. 이미 제도 용어로 잘 물은 질문은
    재작성이 오히려 나빠진다(실측 10.65 -> 17.32).
    """
    base = search_pension_docs.invoke({"query": question, "k": 5})
    if not isinstance(base, list) or not base:
        return [], []

    merged: list[dict] = list(base)
    seen = {r.get("chunk_id") for r in merged}
    used_queries: list[str] = []
    for rewritten in rewrite_search_queries(question):
        extra = search_pension_docs.invoke({"query": rewritten, "k": 5})
        if not isinstance(extra, list) or not extra:
            continue
        used_queries.append(rewritten)
        for r in extra:
            if r.get("chunk_id") in seen:
                continue
            seen.add(r.get("chunk_id"))
            merged.append(r)

    # 거리(distance)가 작을수록 질문에 가깝다. 재작성 질의가 찾아온 더 정확한
    # 문서가 앞에 오도록 정렬한 뒤 상위 5건만 근거로 넘긴다.
    merged.sort(key=lambda r: r.get("distance", float("inf")))
    return merged[:5], used_queries


def _searched_with_raw_question(tool_trace: list[ToolCallRecord], question: str) -> bool:
    """LLM이 문서 검색에 질문 원문을 그대로 넘겼는지 판정한다.

    실측(501문항): search_pension_docs 호출 210건 중 83건(39%)이 원문 그대로였다.
    이 경우 일상어와 제도 용어의 간극 때문에 근거 품질이 떨어지기 쉬우므로,
    제도 용어로 재작성한 검색을 한 번 더 돌려 근거를 보강한다.
    """
    head = (question or "").strip("?. ")[:24]
    if len(head) < 8:
        return False
    return any(
        record.get("tool") == "search_pension_docs" and head in (record.get("args") or "")
        for record in tool_trace
    )


def _doc_search_context(question: str) -> tuple[list[RetrievedItem], list[ToolCallRecord]]:
    """LLM이 RAG 호출을 건너뛴 경우에도 정보형 질문은 한 번 직접 검색해 근거를 확보한다."""
    results, rewritten_queries = _search_with_rewrites(question)
    if not results:
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
    # 재작성 질의를 think_trace에 드러낸다 — 실제로 검색한 것과 기록이 다르면
    # "추론 논리성" 서사가 사실과 어긋난다.
    args = f'query="{question}", k=5'
    if rewritten_queries:
        args += f" (+제도용어 재작성: {'; '.join(rewritten_queries)})"
    return items, [
        {
            "node": "info_agent",
            "tool": "search_pension_docs",
            "args": args,
            "result": f"{len(items)}건 검색: {titles}",
        }
    ]


def _select_deterministic_response(
    category: str, question: str
) -> tuple[str, list[RetrievedItem]] | None:
    """라우터가 놓친 명시 상품형 실물이전 판정만 결정론 경로로 보완한다.

    상품 상태가 없는 일반 실물이전 절차 질문에는 None을 반환한다. 그런 질문에서 LLM이
    RP·MMF 등의 Boolean 플래그를 추정해 개별 판정 도구를 부르는 경로를 열지 않는다.
    """
    deterministic = deterministic_response_for(category, question) if category != "해당없음" else None
    if deterministic is not None:
        return deterministic
    if has_in_kind_transfer_intent(question):
        return in_kind_transfer_judgement_response(question)
    return None


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

        category = state.get("deterministic_category", "해당없음")
        deterministic = _select_deterministic_response(category, state["question"])
        if deterministic is not None:
            draft, retrieved_context = deterministic
            deterministic_source_limited = "calculation_basis=not_defined_in_source" in "\n".join(
                item.get("content", "") for item in retrieved_context
            )
            deterministic_needs_clarification = any(
                marker in draft
                for marker in (
                    "정확한 계산을 위해 다음 정보를 한 번에 알려주세요",
                    "추가로 필요한 정보는 다음과 같습니다",
                    "현재 질문만으로는 적용할 세금 계산 방식을 확정할 수 없습니다",
                )
            )
            response_mode = (
                "conditional"
                if deterministic_source_limited
                else "clarification_included" if deterministic_needs_clarification else "complete"
            )
            return {
                "info_draft": draft,
                "retrieved_context": retrieved_context,
                "tool_trace": [],
                "needs_clarification": deterministic_needs_clarification and not deterministic_source_limited,
                "missing_information": ["추가 확인 필요 입력값 또는 DB에 정의되지 않은 계산 기준"]
                if deterministic_needs_clarification and not deterministic_source_limited
                else [],
                "clarification_questions": [
                    "답변에 적힌 추가 확인 항목을 한 번에 알려주세요.",
                ]
                if deterministic_needs_clarification and not deterministic_source_limited
                else [],
                "response_mode": response_mode,
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
        # 근거가 하나도 없으면 역질문 여부와 무관하게 한 번은 직접 검색한다.
        #
        # ⚠️ 예전에는 `not needs_clarification` 조건이 함께 걸려 있어, LLM이 답변에
        # [추가 확인 필요] 마커만 붙이면 이 안전망을 통째로 건너뛰었다. 그런데 위
        # 시스템 프롬프트는 조건이 부족한 질문에도 "현재 답변 가능한 일반 기준"을
        # 함께 쓰라고 지시한다 — 즉 역질문 답변에도 사실 서술이 들어가도록 설계돼
        # 있다. 안전망만 "역질문이면 근거가 필요 없다"고 가정한 셈이라, 되묻기 직전에
        # 쓴 일반 기준이 근거 0건인 채로 나갔다.
        #
        # 실측(501문항): 근거 0건 + 툴 호출 0건인데 ④가 "근거에 없다"고 확정한 수치가
        # 답변에 남은 문항 6건(no.86/99/104/140/321/483)이 **전부** 이 경로였다.
        # no.86 "55세 미만인데 연금 받을 수 있나요"는 사적연금 서비스인데 국민연금
        # 조기노령연금 수치(10년·6%·30%)를 지어냈고, no.483은 연금수령한도 계산에
        # 필요한 값이 질문에 다 있는데도 검색조차 없이 연령별 세율표를 창작했다.
        if not retrieved_context:
            forced_context, forced_trace = _doc_search_context(state["question"])
            retrieved_context = forced_context
            tool_trace = [*tool_trace, *forced_trace]
        elif _searched_with_raw_question(tool_trace, state["question"]):
            # 근거는 있지만 LLM이 **질문 원문을 그대로** 검색어로 쓴 경우다.
            # 사용자는 일상어로 묻고 문서는 제도 용어로 쓰여 있어, 이 조합은 엉뚱한
            # 문서를 끌어오기 쉽다(실측 S03: "전업주부인데 노후 대비..." -> MP 알림톡
            # FAQ). 제도 용어로 재작성한 질의 결과를 **더해서** 근거를 보강한다.
            #
            # 실측(501문항): 근거는 있는데 grounded=False인 32건 중 9건이 이 경로였고,
            # 표본 5건에서 3건이 뚜렷이 개선됐다(25.9 -> 10.8 등), 나빠진 건 없었다.
            # 원문 결과를 버리지 않고 합치므로 이미 좋은 검색은 그대로 유지된다.
            extra_context, extra_trace = _doc_search_context(state["question"])
            if extra_context:
                retrieved_context = [*retrieved_context, *extra_context]
                tool_trace = [*tool_trace, *extra_trace]

        # ⚠️ response_mode를 안 채우면 Guardian(파수꾼)이 절대 작동하지 않는다 —
        # _guardian_route_possible이 response_mode=="complete"를 요구하는데, 이 LLM
        # 자유 응답 경로(결정론도 clarification도 아닌 일반 답변)는 이 키를 아예 채운
        # 적이 없었다. 실측: "IRP 실물이전 절차 알려줘"는 grounded=True/
        # requirements_met=True까지 통과하고도 response_mode=None이라 Guardian 노드
        # 자체를 못 탔다(guardian_result가 아예 None으로 남음, NO_CANDIDATE조차 아님).
        return {
            "info_draft": draft,
            "retrieved_context": retrieved_context,
            "tool_trace": tool_trace,
            "needs_clarification": needs_clarification,
            "response_mode": "clarification_included" if needs_clarification else "complete",
            "repair_attempted": state.get("verification") is not None,
            "deterministic_info": False,
        }

    return info_agent_node
