"""ReAct 에이전트가 남긴 ToolMessage 목록을 think_trace/답변에 쓸 깔끔한 근거(RetrievedItem)
목록으로 변환한다.

기존에는 툴 결과를 통째로 str()해서 하나의 근거 항목에 욱여넣었다 — search_pension_docs처럼
결과가 여러 개(청크 리스트)인 툴은 그 리스트 전체(마크다운 표 포함)가 "근거 1건"으로 뭉쳐
들어가는 문제가 있었다. 여기서는 검색류 툴의 결과를 항목별로 풀어서 각각 하나의 근거로
만들고, source에 원문 파일명/섹션 같은 사람이 읽을 수 있는 값을 넣는다.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.state import RetrievedItem

_DOC_SEARCH_TOOLS = {"search_pension_docs"}
_FUND_LIST_TOOLS = {"search_funds"}
_FUND_DETAIL_TOOLS = {"get_fund_detail"}

_MAX_SNIPPET_CHARS = 400

# 히스토리 한 턴(특히 답변)이 통째로 다시 프롬프트에 들어가면 턴이 쌓일수록 토큰이 급격히
# 불어난다 — 과거 답변은 요약 스니펫 정도로만 자른다. chat.py가 애초에 턴 개수도 제한한다.
_MAX_HISTORY_ANSWER_CHARS = 300


def _truncate(text: str, limit: int = _MAX_SNIPPET_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit].rstrip() + "…"


def history_to_messages(history: list[dict] | None) -> list:
    """이전 대화(question/answer 쌍)를 HumanMessage/AIMessage 시퀀스로 변환.

    create_agent 기반 ReAct 노드(info_agent/product_agent)가 "그거", "두 번째 상품은?" 같은
    후속 질문을 실제 메시지 문맥으로 풀어낼 수 있도록, 텍스트 요약이 아니라 메시지 형태로 넘긴다.
    """
    messages = []
    for turn in history or []:
        question = turn.get("question")
        answer = turn.get("answer")
        if question:
            messages.append(HumanMessage(content=question))
        if answer:
            messages.append(AIMessage(content=_truncate(answer, _MAX_HISTORY_ANSWER_CHARS)))
    return messages


def format_conversation_history(history: list[dict] | None) -> str:
    """router/generator처럼 메시지 리스트가 아니라 텍스트 프롬프트를 조립하는 노드용 포맷."""
    if not history:
        return ""
    lines = []
    for i, turn in enumerate(history, 1):
        question = turn.get("question", "")
        answer = _truncate(turn.get("answer", ""), _MAX_HISTORY_ANSWER_CHARS)
        lines.append(f"{i}. 사용자: {question}\n   답변: {answer}")
    return "\n".join(lines)


def _parse_json(raw: Any):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def build_retrieved_context(messages: list, node: str) -> list[RetrievedItem]:
    """ReAct 에이전트 실행 결과(result["messages"])에서 ToolMessage만 골라 근거 목록으로 변환."""
    items: list[RetrievedItem] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        tool_name = msg.name or "unknown_tool"
        raw = msg.content

        # 1. 연금 문서 RAG 검색 결과
        if tool_name in _DOC_SEARCH_TOOLS:
            results = _parse_json(raw)

            if isinstance(results, list) and results:
                for r in results:
                    chunk_id = r.get("chunk_id", "")
                    section = r.get("section", "")
                    source_location = r.get("source_location", "")
                    file_title = r.get("file_title", "")

                    # 예: doc19_chunk03 -> doc19
                    document_id = (
                        chunk_id.split("_chunk", 1)[0]
                        if "_chunk" in chunk_id
                        else chunk_id
                    )

                    items.append({
                        "source": document_id or tool_name,
                        "content": _truncate(r.get("content", "")),
                        "node": node,
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "file_title": file_title,
                        "section": section,
                        "source_location": source_location,
                    })

            else:
                items.append({
                    "source": tool_name,
                    "content": "(검색 결과 없음)",
                    "node": node,
                })

            continue

        # 2. 펀드 목록 검색 결과
        if tool_name in _FUND_LIST_TOOLS:
            results = _parse_json(raw)

            if isinstance(results, list) and results:
                for r in results:
                    label = (
                        f"{r.get('fund_name', tool_name)} "
                        f"({r.get('class_name', '')})"
                    )

                    items.append({
                        "source": label,
                        "content": _truncate(
                            json.dumps(r, ensure_ascii=False)
                        ),
                        "node": node,
                    })

            else:
                items.append({
                    "source": tool_name,
                    "content": "(검색 결과 없음)",
                    "node": node,
                })

            continue

        # 3. 펀드 상세 조회 결과
        if tool_name in _FUND_DETAIL_TOOLS:
            result = _parse_json(raw)

            if isinstance(result, dict) and result.get("found"):
                label = result.get("master", {}).get(
                    "fund_name",
                    tool_name,
                )

                items.append({
                    "source": label,
                    "content": _truncate(
                        json.dumps(result, ensure_ascii=False)
                    ),
                    "node": node,
                })

            else:
                items.append({
                    "source": tool_name,
                    "content": "(해당 상품코드 없음)",
                    "node": node,
                })

            continue

        # 4. 계산/판정 툴
        items.append({
            "source": tool_name,
            "content": _truncate(str(raw)),
            "node": node,
        })

    return items
