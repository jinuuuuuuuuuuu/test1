"""src/agents/tools.py의 9개 LangChain 툴 wrapper 테스트.

@tool로 감싼 함수는 일반 함수로 직접 호출할 수 없고 .invoke(dict)로 호출한다
(LangChain StructuredTool 규약). 반환값은 전부 JSON 직렬화 가능한 순수 dict여야 한다.

search_funds/get_fund_detail/search_pension_docs는 실제로 적재된
data/processed/prospectus.db, data/processed/chroma_docs에 의존하므로, 해당 산출물이
없는 환경(다른 개발자 로컬 등)에서는 skip 된다.
"""

import os

import pytest

from src.agents.tools import (
    INFO_AGENT_TOOLS,
    PRODUCT_AGENT_TOOLS,
    calculate_pension_withdrawal,
    calculate_tax_credit,
    check_default_option,
    check_early_withdrawal,
    check_in_kind_transfer,
    check_product_pension_eligibility,
    get_fund_detail,
    search_funds,
    search_pension_docs,
)
from src.storage.queries import DEFAULT_CHROMA_DIR, DEFAULT_DB_PATH

_HAS_PROSPECTUS_DB = os.path.exists(DEFAULT_DB_PATH)
_HAS_CHROMA_DOCS = os.path.exists(DEFAULT_CHROMA_DIR)
_HAS_REAL_API_KEY = not os.environ.get("CLOVASTUDIO_API_KEY", "").startswith("dummy-") and os.environ.get(
    "CLOVASTUDIO_API_KEY"
)


def test_tool_registries_have_expected_members():
    assert {t.name for t in INFO_AGENT_TOOLS} == {
        "calculate_tax_credit",
        "calculate_pension_withdrawal",
        "check_early_withdrawal",
        "check_default_option",
        "check_in_kind_transfer",
        "search_pension_docs",
    }
    assert {t.name for t in PRODUCT_AGENT_TOOLS} == {
        "check_product_pension_eligibility",
        "search_funds",
        "get_fund_detail",
    }


def test_calculate_tax_credit_tool():
    out = calculate_tax_credit.invoke({
        "pension_savings_paid": 6_000_000,
        "irp_paid": 3_000_000,
        "total_salary": 50_000_000,
    })
    assert isinstance(out, dict)
    assert out["credit_rate"] == 0.165
    assert out["tax_credit_amount"] == 1_485_000


def test_calculate_pension_withdrawal_tool():
    out = calculate_pension_withdrawal.invoke({
        "account_value": 100_000_000,
        "pension_payment_year": 3,
        "actual_receipt_year": 5,
        "age": 60,
        "tax_credited_principal_and_gains_withdrawn": 10_000_000,
    })
    assert out["withdrawal_limit"]["limit_amount"] == 15_000_000
    assert out["retirement_tax_reduction"]["payment_ratio"] == 0.7
    assert out["comprehensive_tax"]["exceeds_threshold"] is False


def test_check_early_withdrawal_tool_medical():
    out = check_early_withdrawal.invoke({
        "reason": "요양",
        "plan_type": "DC",
        "request_date": "2026-06-01",
        "medical_expense_last_year": 5_000_000,
        "prior_year_annual_wage": 30_000_000,
        "treatment_end_date": "2026-05-15",
    })
    assert out["eligible"] is True


def test_check_early_withdrawal_tool_db_blocked():
    out = check_early_withdrawal.invoke({
        "reason": "재난피해",
        "plan_type": "DB",
        "request_date": "2026-06-01",
        "damage_date": "2026-05-01",
    })
    assert out["eligible"] is False
    assert "DB" in out["reason"]


def test_check_default_option_tool_optin():
    out = check_default_option.invoke({
        "mode": "옵트인가능여부",
        "current_holdings_count": 0,
    })
    assert out["eligible"] is True


def test_check_default_option_tool_schedule():
    out = check_default_option.invoke({
        "mode": "자동매수일정",
        "base_date": "2026-01-01",
        "is_new_participant": True,
    })
    assert out["notice_date"] == "2026-01-02"
    assert out["waited"] is True


def test_check_in_kind_transfer_tool():
    out = check_in_kind_transfer.invoke({"private_fund": True})
    assert out["eligible"] is False
    assert "03" in out["blocking_codes"]
    assert set(out["needs_manual_review_codes"]) == {"06", "17", "20"}


def test_check_in_kind_transfer_tool_all_clear():
    out = check_in_kind_transfer.invoke({})
    assert out["eligible"] is True
    assert out["blocking_codes"] == []


def test_check_product_pension_eligibility_tool_forbidden():
    out = check_product_pension_eligibility.invoke({
        "product_type": "국내상장주식",
        "plan_type": "DC",
    })
    assert out["eligible"] is False
    assert out["risk_tier"] == "투자금지"


def test_check_product_pension_eligibility_tool_unknown_type_does_not_raise():
    # 실제 채팅 테스트에서 "괜찮은 연금상품 3개 추천해줘" 같은 막연한 질문에 LLM이 존재하지
    # 않는 product_type을 지어내 호출했을 때 그래프 전체가 죽었던 크래시의 회귀 테스트.
    out = check_product_pension_eligibility.invoke({
        "product_type": "괜찮은상품",
        "plan_type": "DC",
    })
    assert isinstance(out, dict)
    assert "error" in out


def test_check_product_pension_eligibility_tool_allowed():
    out = check_product_pension_eligibility.invoke({
        "product_type": "국내상장주식",
        "plan_type": "DB",
    })
    assert out["eligible"] is True


@pytest.mark.skipif(not _HAS_PROSPECTUS_DB, reason="data/processed/prospectus.db가 아직 없습니다")
def test_search_funds_tool_returns_list():
    out = search_funds.invoke({"risk_grade_min": 4, "limit": 3})
    assert isinstance(out, list)
    assert len(out) <= 3
    for item in out:
        assert "product_code" in item and "class_name" in item


@pytest.mark.skipif(not _HAS_PROSPECTUS_DB, reason="data/processed/prospectus.db가 아직 없습니다")
def test_get_fund_detail_tool_known_and_unknown_code():
    known = search_funds.invoke({"limit": 1})[0]["product_code"]
    out = get_fund_detail.invoke({"product_code": known})
    assert out["found"] is True
    assert out["master"]["product_code"] == known
    assert isinstance(out["classes"], list) and len(out["classes"]) >= 1

    missing = get_fund_detail.invoke({"product_code": "KR_NOT_EXIST"})
    assert missing["found"] is False


@pytest.mark.skipif(
    not (_HAS_CHROMA_DOCS and _HAS_REAL_API_KEY),
    reason="data/processed/chroma_docs 및 실제 CLOVASTUDIO_API_KEY가 있어야 임베딩 검색이 가능합니다",
)
def test_search_pension_docs_tool_returns_relevant_chunks():
    out = search_pension_docs.invoke({"query": "연금저축계좌 중도인출 사유", "k": 3})
    assert isinstance(out, list)
    assert len(out) <= 3
    assert all("chunk_id" in item and "content" in item for item in out)
