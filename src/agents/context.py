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

from src.agents.state import RetrievedItem, ToolCallRecord

_DOC_SEARCH_TOOLS = {"search_pension_docs"}
_FUND_LIST_TOOLS = {"search_funds"}
_FUND_DETAIL_TOOLS = {"get_fund_detail"}
_PROSPECTUS_TEXT_TOOLS = {"search_prospectus_text"}

# 근거(retrieved_context)는 ④검증이 초안을 대조하는 기준이자 ⑤생성이 쓸 수 있는 유일한
# 숫자 공급원이고, 평가 API의 retrieved_context 필드로도 그대로 나간다 — 여기서 내용이
# 잘리면 초안의 맞는 숫자가 "근거 없음"으로 오판되고 잘린 JSON은 파싱조차 안 된다.
# 실측 크기(RAG 청크 ≤600자, 계산툴 dict ~1KB, get_fund_detail 전체 JSON 수 KB)보다
# 충분히 큰 값으로, 비정상 폭주를 막는 방어선 용도로만 둔다.
_MAX_DOC_EVIDENCE_CHARS = 2_000
_MAX_TOOL_EVIDENCE_CHARS = 6_000

# 히스토리 한 턴(특히 답변)이 통째로 다시 프롬프트에 들어가면 턴이 쌓일수록 토큰이 급격히
# 불어난다 — 과거 답변은 요약 스니펫 정도로만 자른다. chat.py가 애초에 턴 개수도 제한한다.
_MAX_HISTORY_ANSWER_CHARS = 300


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit].rstrip() + "…"


CLARIFICATION_MARKER = "[추가 확인 필요]"


def split_clarification_marker(draft: str) -> tuple[str, bool]:
    """②③ 초안이 역질문 마커로 시작하면 (마커를 뗀 본문, True)를 반환한다.

    ②③은 조건 불충분으로 사용자에게 되물을 때 초안을 CLARIFICATION_MARKER로 시작하라는
    지시를 받는다 — 구조화 출력 없이(HCX-005 지원 여부 미확정) 역질문 여부를 결정론적으로
    식별하기 위한 규약이다. 이 플래그가 서면 ③ 스킵 + ④의 요구사항 검증 면제 + ⑤의 답변
    보충 금지가 걸린다 (역질문을 ④⑤가 "요구사항 미충족"으로 교정해 추천을 되살리는 경로 차단).
    """
    stripped = (draft or "").lstrip()
    if stripped.startswith(CLARIFICATION_MARKER):
        return stripped[len(CLARIFICATION_MARKER):].lstrip(), True
    return draft or "", False


def build_repair_note(verification: dict | None) -> str | None:
    """④검증 탈락 사유를 ②③ 재실행(1회 한정 repair)용 지시문으로 만든다. 통과·미검증이면 None."""
    if not verification:
        return None
    problems = []
    if verification.get("grounded") is False:
        details = verification.get("issues") or []
        problems.append(("근거 미비 — " + " / ".join(details)) if details else "근거 미비")
    if verification.get("requirements_met") is False:
        missing = verification.get("missing_requirements") or []
        problems.append(("누락 항목 — " + " / ".join(missing)) if missing else "요구사항 누락")
    if not problems:
        return None
    return (
        "[재작성 지시] 직전 초안이 검증에서 탈락했습니다: " + " · ".join(problems) + ". "
        "근거 없는 수치는 반드시 툴을 호출해 근거를 확보하거나 답변에서 제거하고, "
        "누락된 항목은 툴을 호출해 보완하세요."
    )


def dedupe_context(items: list[RetrievedItem]) -> list[RetrievedItem]:
    """(source, content)가 같은 근거의 중복을 등장 순서를 유지하며 제거한다.

    repair 재실행 시 retrieved_context의 operator.add 리듀서가 1차 실행분과 같은 근거를
    다시 누적하므로, 근거를 읽는 쪽(④⑤·think_trace)에서 항상 이 함수를 거쳐야 한다.
    """
    seen: set[tuple[str, str]] = set()
    result: list[RetrievedItem] = []
    for item in items:
        key = (item["source"], item["content"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def merge_drafts(info_draft: str | None, product_draft: str | None) -> str:
    """②③ 초안을 ④검증·⑤생성이 쓸 하나의 초안으로 합친다.

    복합형(②→③ 순차 실행)에서는 두 초안이 모두 존재하므로 반드시 둘 다 포함해야 한다 —
    `info_draft or product_draft`처럼 하나만 고르면 ③상품 Agent의 답변이 검증과 최종
    답변에서 통째로 빠진다 (2026-08 실측 버그, 이 함수로 교체).
    """
    if info_draft and product_draft:
        return f"[제도·세제 관련 답변]\n{info_draft}\n\n[상품 관련 답변]\n{product_draft}"
    return info_draft or product_draft or ""


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


# think_trace는 사람이 읽는 요약이라 근거 본문만큼 길 필요가 없다.
_MAX_TRACE_VALUE_CHARS = 120
_MAX_TRACE_RESULT_CHARS = 220


def _format_args(args: dict) -> str:
    """툴 호출 인자를 'k=v, k=v' 한 줄로 요약한다. 값이 길면 개별로 자른다."""
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else f'"{value}"'
        parts.append(f"{key}={_truncate(text, _MAX_TRACE_VALUE_CHARS)}")
    return ", ".join(parts)


def _summarize_tool_result(tool_name: str, raw) -> str:
    """툴 결과를 서사에 쓸 한 줄로 요약한다 (근거 본문은 retrieved_context가 따로 보관)."""
    parsed = _parse_json(raw)

    if tool_name in _DOC_SEARCH_TOOLS:
        if isinstance(parsed, list) and parsed:
            titles = "; ".join(filter(None, (r.get("file_title") for r in parsed[:3])))
            return _truncate(f"{len(parsed)}건 검색: {titles}", _MAX_TRACE_RESULT_CHARS)
        return "검색 결과 없음 (보유 문서에 관련 내용 없음)"

    if tool_name in _PROSPECTUS_TEXT_TOOLS:
        if isinstance(parsed, list) and parsed:
            labels = "; ".join(f"{r.get('fund_name', '')}({r.get('section', '')})" for r in parsed[:3])
            return _truncate(f"{len(parsed)}건 검색: {labels}", _MAX_TRACE_RESULT_CHARS)
        return "검색 결과 없음"

    if tool_name in _FUND_LIST_TOOLS:
        if isinstance(parsed, list) and parsed:
            labels = "; ".join(
                f"{r.get('fund_name', '')}({r.get('class_name', '')})" for r in parsed[:3]
            )
            return _truncate(f"{len(parsed)}건 후보: {labels}", _MAX_TRACE_RESULT_CHARS)
        return "조건에 맞는 후보 없음"

    if tool_name in _FUND_DETAIL_TOOLS:
        if isinstance(parsed, dict) and parsed.get("found"):
            name = parsed.get("master", {}).get("fund_name", "")
            classes = len(parsed.get("classes") or [])
            return _truncate(f"{name} 상세 조회 (판매클래스 {classes}개)", _MAX_TRACE_RESULT_CHARS)
        return "해당 상품코드 없음"

    # 계산/판정 툴: 결과 dict가 이미 간결하므로 그대로 자른다.
    return _truncate(str(raw), _MAX_TRACE_RESULT_CHARS)


def build_tool_trace(messages: list, node: str) -> list[ToolCallRecord]:
    """ReAct 실행 메시지에서 (툴, 인자, 결과요약)을 호출 순서대로 뽑아낸다.

    build_retrieved_context는 ToolMessage의 '내용'만 근거로 남기고 어떤 인자로 호출했는지는
    버리기 때문에, 서사형 think_trace를 만들려면 AIMessage.tool_calls까지 함께 봐야 한다.
    tool_call_id로 호출과 결과를 짝지으며, 짝을 못 찾은 결과도 누락 없이 남긴다.
    """
    records: dict[str, dict] = {}
    order: list[str] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", None) or []:
                key = call.get("id") or f"{call.get('name')}#{len(order)}"
                records[key] = {"tool": call.get("name") or "unknown_tool", "args": call.get("args") or {}}
                order.append(key)
        elif isinstance(msg, ToolMessage):
            key = msg.tool_call_id
            if key in records:
                records[key]["result"] = _summarize_tool_result(msg.name or "", msg.content)
            else:
                orphan = f"orphan#{len(order)}"
                records[orphan] = {
                    "tool": msg.name or "unknown_tool",
                    "args": {},
                    "result": _summarize_tool_result(msg.name or "", msg.content),
                }
                order.append(orphan)

    return [
        {
            "node": node,
            "tool": records[key]["tool"],
            "args": _format_args(records[key]["args"]),
            "result": records[key].get("result", "(결과 미수신)"),
        }
        for key in order
    ]


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
                    chunk_id = r.get("chunk_id", "")
                    document_id = chunk_id.split("_chunk", 1)[0] if "_chunk" in chunk_id else chunk_id
                    items.append(
                        {
                            "source": source,
                            "content": _truncate(r.get("content", ""), _MAX_DOC_EVIDENCE_CHARS),
                            "node": node,
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "file_title": r.get("file_title", ""),
                            "section": section or "",
                            "source_location": r.get("source_location", ""),
                        }
                    )
            else:
                items.append({"source": tool_name, "content": "(검색 결과 없음)", "node": node})
            continue

        if tool_name in _FUND_LIST_TOOLS:
            results = _parse_json(raw)
            if isinstance(results, list) and results:
                for r in results:
                    label = f"{r.get('fund_name', tool_name)} ({r.get('class_name', '')})"
                    items.append(
                        {
                            "source": label,
                            "content": _truncate(json.dumps(r, ensure_ascii=False), _MAX_TOOL_EVIDENCE_CHARS),
                            "node": node,
                        }
                    )
            else:
                items.append({"source": tool_name, "content": "(검색 결과 없음)", "node": node})
            continue

        if tool_name in _PROSPECTUS_TEXT_TOOLS:
            results = _parse_json(raw)
            if isinstance(results, list) and results:
                for r in results:
                    source = f"{r.get('fund_name', '')} 투자설명서 — {r.get('section', '')}"
                    items.append(
                        {"source": source, "content": _truncate(r.get("content", ""), _MAX_DOC_EVIDENCE_CHARS), "node": node}
                    )
            else:
                items.append({"source": tool_name, "content": "(검색 결과 없음)", "node": node})
            continue

        if tool_name in _FUND_DETAIL_TOOLS:
            result = _parse_json(raw)
            if isinstance(result, dict) and result.get("found"):
                label = result.get("master", {}).get("fund_name", tool_name)
                items.append(
                    {
                        "source": label,
                        "content": _truncate(json.dumps(result, ensure_ascii=False), _MAX_TOOL_EVIDENCE_CHARS),
                        "node": node,
                    }
                )
            else:
                items.append({"source": tool_name, "content": "(해당 상품코드 없음)", "node": node})
            continue

        # 계산/판정 툴: 결과 dict를 그대로 근거로 남긴다 (한도는 비정상 폭주 방어용).
        items.append({"source": tool_name, "content": _truncate(str(raw), _MAX_TOOL_EVIDENCE_CHARS), "node": node})

    return items
