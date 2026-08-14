"""ReAct 에이전트가 남긴 ToolMessage 목록을 think_trace/답변에 쓸 깔끔한 근거(RetrievedItem)
목록으로 변환한다.

기존에는 툴 결과를 통째로 str()해서 하나의 근거 항목에 욱여넣었다 — search_pension_docs처럼
결과가 여러 개(청크 리스트)인 툴은 그 리스트 전체(마크다운 표 포함)가 "근거 1건"으로 뭉쳐
들어가는 문제가 있었다. 여기서는 검색류 툴의 결과를 항목별로 풀어서 각각 하나의 근거로
만들고, source에 원문 파일명/섹션 같은 사람이 읽을 수 있는 값을 넣는다.
"""

import json
from typing import Any

from langchain_core.messages import ToolMessage

from src.agents.state import RetrievedItem

_DOC_SEARCH_TOOLS = {"search_pension_docs"}
_FUND_LIST_TOOLS = {"search_funds"}
_FUND_DETAIL_TOOLS = {"get_fund_detail"}

_MAX_SNIPPET_CHARS = 400


def _truncate(text: str, limit: int = _MAX_SNIPPET_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit].rstrip() + "…"


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

        if tool_name in _DOC_SEARCH_TOOLS:
            results = _parse_json(raw)
            if isinstance(results, list) and results:
                for r in results:
                    label = r.get("file_title") or tool_name
                    section = r.get("section")
                    source = f"{label} — {section}" if section else label
                    items.append({"source": source, "content": _truncate(r.get("content", "")), "node": node})
            else:
                items.append({"source": tool_name, "content": "(검색 결과 없음)", "node": node})
            continue

        if tool_name in _FUND_LIST_TOOLS:
            results = _parse_json(raw)
            if isinstance(results, list) and results:
                for r in results:
                    label = f"{r.get('fund_name', tool_name)} ({r.get('class_name', '')})"
                    items.append(
                        {"source": label, "content": _truncate(json.dumps(r, ensure_ascii=False)), "node": node}
                    )
            else:
                items.append({"source": tool_name, "content": "(검색 결과 없음)", "node": node})
            continue

        if tool_name in _FUND_DETAIL_TOOLS:
            result = _parse_json(raw)
            if isinstance(result, dict) and result.get("found"):
                label = result.get("master", {}).get("fund_name", tool_name)
                items.append(
                    {"source": label, "content": _truncate(json.dumps(result, ensure_ascii=False)), "node": node}
                )
            else:
                items.append({"source": tool_name, "content": "(해당 상품코드 없음)", "node": node})
            continue

        # 계산/판정 툴: 결과가 이미 간결한 dict라 그대로 근거로 남기되 길이만 방어적으로 자른다.
        items.append({"source": tool_name, "content": _truncate(str(raw)), "node": node})

    return items
