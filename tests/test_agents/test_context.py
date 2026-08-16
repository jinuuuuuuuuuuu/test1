import json

from langchain_core.messages import ToolMessage

from src.agents.context import build_retrieved_context


def test_fund_tool_result_is_not_truncated():
    long_tail = "투자전략상세" * 120
    raw = [
        {
            "product_code": "P001",
            "fund_name": "테스트 펀드",
            "class_name": "C-P",
            "risk_grade": "4등급",
            "strategy": long_tail,
        }
    ]

    context = build_retrieved_context(
        [ToolMessage(content=json.dumps(raw, ensure_ascii=False), name="search_funds", tool_call_id="1")],
        node="product_agent",
    )

    assert context[0]["content"].endswith(f'"strategy": "{long_tail}"}}')
    assert "…" not in context[0]["content"]


def test_calculation_tool_result_is_not_truncated():
    raw = {"detail": "계산근거" * 150, "tax_credit": 1485000}

    context = build_retrieved_context(
        [ToolMessage(content=json.dumps(raw, ensure_ascii=False), name="calculate_tax_credit", tool_call_id="1")],
        node="info_agent",
    )

    assert "1485000" in context[0]["content"]
    assert context[0]["content"].endswith("}")
    assert "…" not in context[0]["content"]
