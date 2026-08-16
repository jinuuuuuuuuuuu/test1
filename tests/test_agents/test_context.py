"""context.py의 초안 병합(merge_drafts)과 근거 변환(build_retrieved_context) 검증.

특히 두 가지 회귀를 막는다:
- 복합형에서 초안 하나만 골라 상품 답변이 유실되는 버그 (merge_drafts로 수정)
- 근거를 400자로 잘라 계산툴/펀드상세 JSON이 깨진 채 ④검증·⑤생성·평가 API에 쓰이던 버그
"""

import json

from langchain_core.messages import ToolMessage

from src.agents.context import (
    CLARIFICATION_MARKER,
    build_repair_note,
    build_retrieved_context,
    dedupe_context,
    merge_drafts,
    split_clarification_marker,
)

# ── merge_drafts ─────────────────────────────────────────────────────────


def test_merge_drafts_keeps_both_drafts_in_composite():
    merged = merge_drafts("세액공제는 900만원까지 됩니다.", "OO펀드를 후보로 볼 수 있습니다.")
    assert "세액공제는 900만원까지 됩니다." in merged
    assert "OO펀드를 후보로 볼 수 있습니다." in merged
    assert "[제도·세제 관련 답변]" in merged
    assert "[상품 관련 답변]" in merged


def test_merge_drafts_info_only():
    assert merge_drafts("정보 답변", None) == "정보 답변"


def test_merge_drafts_product_only():
    assert merge_drafts(None, "상품 답변") == "상품 답변"


def test_merge_drafts_empty():
    assert merge_drafts(None, None) == ""
    assert merge_drafts("", "") == ""


# ── split_clarification_marker ───────────────────────────────────────────


def test_marker_draft_is_detected_and_stripped():
    draft = f"{CLARIFICATION_MARKER} 투자 가능한 계좌유형과 위험 수준을 알려주세요."
    body, needs = split_clarification_marker(draft)
    assert needs is True
    assert body == "투자 가능한 계좌유형과 위험 수준을 알려주세요."


def test_marker_with_leading_whitespace_is_detected():
    body, needs = split_clarification_marker(f"  \n{CLARIFICATION_MARKER}\n계좌유형이 무엇인가요?")
    assert needs is True
    assert body == "계좌유형이 무엇인가요?"


def test_normal_draft_is_not_marked():
    body, needs = split_clarification_marker("세액공제 한도는 900만원입니다.")
    assert needs is False
    assert body == "세액공제 한도는 900만원입니다."


def test_empty_draft():
    assert split_clarification_marker("") == ("", False)


# ── build_repair_note ────────────────────────────────────────────────────


def test_repair_note_none_when_verification_passed():
    assert build_repair_note({"grounded": True, "requirements_met": True}) is None
    assert build_repair_note(None) is None


def test_repair_note_includes_failure_details():
    note = build_repair_note({
        "grounded": False,
        "issues": ["500만원은 근거에 없음"],
        "requirements_met": False,
        "missing_requirements": ["수령 시기별 세율"],
    })
    assert note is not None
    assert "[재작성 지시]" in note
    assert "500만원은 근거에 없음" in note
    assert "수령 시기별 세율" in note


# ── dedupe_context ───────────────────────────────────────────────────────


def test_dedupe_context_removes_repair_duplicates_keeping_order():
    a = {"source": "매뉴얼", "content": "한도 900만원", "node": "info_agent"}
    b = {"source": "펀드A", "content": "수익률 5%", "node": "product_agent"}
    assert dedupe_context([a, b, a, b, a]) == [a, b]


# ── build_retrieved_context: 근거가 잘리지 않아야 한다 ────────────────────


def _tool_msg(name: str, content: str) -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id="t1")


def test_doc_search_chunk_over_400_chars_is_not_truncated():
    # 임베딩 분할 한도(600자)까지의 RAG 청크가 통째로 근거에 남아야 한다.
    content = "연금수령한도 규정 " * 60  # 약 600자
    raw = json.dumps([{"file_title": "업무매뉴얼", "section": "인출", "content": content}], ensure_ascii=False)
    items = build_retrieved_context([_tool_msg("search_pension_docs", raw)], node="info_agent")
    assert len(items) == 1
    assert items[0]["content"] == content
    assert "…" not in items[0]["content"]


def test_calc_tool_result_over_400_chars_is_not_truncated():
    # calculate_pension_withdrawal처럼 1KB급 dict 결과가 잘리면 안 된다.
    result = {f"field_{i}": f"값 {i} " * 10 for i in range(20)}  # 약 1.5KB
    raw = str(result)
    items = build_retrieved_context([_tool_msg("calculate_pension_withdrawal", raw)], node="info_agent")
    assert len(items) == 1
    assert items[0]["content"] == raw


def test_fund_detail_json_stays_parseable():
    # 펀드상세(마스터+전체 클래스)는 수 KB JSON — 잘리면 유효한 JSON이 아니게 된다.
    detail = {
        "found": True,
        "master": {"fund_name": "OO국공채펀드", "investment_strategy": "국공채 중심 " * 100},
        "classes": [{"class_name": f"C-P{i}", "total_expense_ratio": 0.5 + i} for i in range(8)],
    }
    raw = json.dumps(detail, ensure_ascii=False)
    assert len(raw) > 400
    items = build_retrieved_context([_tool_msg("get_fund_detail", raw)], node="product_agent")
    assert len(items) == 1
    parsed = json.loads(items[0]["content"])  # 파싱 실패하면 절단이 재발한 것
    assert parsed["master"]["fund_name"] == "OO국공채펀드"
    assert len(parsed["classes"]) == 8


def test_pathological_tool_output_is_still_capped():
    # 방어선: 비정상적으로 거대한 출력(6,000자 초과)만 잘린다.
    raw = "x" * 20_000
    items = build_retrieved_context([_tool_msg("calculate_tax_credit", raw)], node="info_agent")
    assert len(items[0]["content"]) < 20_000
    assert items[0]["content"].endswith("…")
