"""think_trace 서사화 검증 — 툴 호출 추적(build_tool_trace)과 서사 조립(_format_think_trace).

둘 다 순수 포맷팅이라 LLM/네트워크 없이 검증한다 (대회 평가지표 "추론 논리성" 대응).
"""

import json
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

from src.agents.context import build_tool_trace
from src.agents.generator import _format_think_trace

# ── build_tool_trace ─────────────────────────────────────────────────────


def test_pairs_tool_calls_with_results_in_order():
    messages = [
        HumanMessage(content="연금저축 세액공제 한도?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_pension_docs", "args": {"query": "세액공제 한도", "k": 5}, "id": "c1"}],
        ),
        ToolMessage(
            content=json.dumps([{"file_title": "IRP 세액공제 안내", "content": "한도 900만원"}], ensure_ascii=False),
            name="search_pension_docs",
            tool_call_id="c1",
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "calculate_tax_credit", "args": {"irp_paid": 3000000}, "id": "c2"}],
        ),
        ToolMessage(content="{'credit': 495000}", name="calculate_tax_credit", tool_call_id="c2"),
        AIMessage(content="900만원까지 공제됩니다."),
    ]
    trace = build_tool_trace(messages, node="info_agent")

    assert [t["tool"] for t in trace] == ["search_pension_docs", "calculate_tax_credit"]
    assert 'query="세액공제 한도"' in trace[0]["args"]
    assert "k=5" in trace[0]["args"]
    assert "1건 검색" in trace[0]["result"]
    assert "IRP 세액공제 안내" in trace[0]["result"]
    assert "495000" in trace[1]["result"]
    assert all(t["node"] == "info_agent" for t in trace)


def test_empty_search_result_is_reported_as_no_evidence():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "search_pension_docs", "args": {"query": "401k"}, "id": "c1"}]),
        ToolMessage(content="[]", name="search_pension_docs", tool_call_id="c1"),
    ]
    trace = build_tool_trace(messages, node="info_agent")
    assert "보유 문서에 관련 내용 없음" in trace[0]["result"]


def test_no_tool_calls_yields_empty_trace():
    assert build_tool_trace([AIMessage(content="IRP는 개인형 퇴직연금입니다.")], node="info_agent") == []


def test_orphan_tool_message_is_not_dropped():
    # tool_call_id 짝을 못 찾아도 호출 사실은 기록에 남아야 한다.
    messages = [ToolMessage(content="{'ok': 1}", name="check_default_option", tool_call_id="unknown")]
    trace = build_tool_trace(messages, node="info_agent")
    assert len(trace) == 1
    assert trace[0]["tool"] == "check_default_option"


# ── _format_think_trace ──────────────────────────────────────────────────


def _composite_state() -> dict:
    return {
        "question": "명퇴수당을 연금계좌에 넣으면 세금감면이 어마어마한가요? 상품도 추천해주세요",
        "intent": ["정보형", "상품형"],
        "scope": "범위내",
        "is_safe": True,
        "tool_trace": [
            {"node": "info_agent", "tool": "search_pension_docs", "args": 'query="퇴직소득세 감면"', "result": "3건 검색: 연금 인출 가이드"},
            {"node": "product_agent", "tool": "search_funds", "args": "risk_grade_min=4", "result": "2건 후보: OO국공채(C-P)"},
            {"node": "info_agent", "tool": "calculate_pension_withdrawal", "args": "age=55", "result": "{'감면율': 0.3}"},
        ],
        "info_draft": "감면율은 30%입니다.",
        "product_draft": "OO국공채를 후보로 볼 수 있습니다.",
        "retrieved_context": [
            {"source": "연금 인출 가이드 — 감면", "content": "연금수령 시 30% 감면", "node": "info_agent"},
        ],
        "verification": {
            "grounded": True,
            "issues": [],
            "l0_suspect_numbers": ["30%"],
            "unsupported_numbers_confirmed": [],
            "premise_issues": ["'세금감면이 어마어마하다'는 과장"],
            "requirements_met": True,
            "missing_requirements": [],
        },
        "repair_attempted": True,
    }


def test_narrative_has_ordered_sections():
    trace = _format_think_trace(_composite_state())
    for marker in ("[① 질문 분류]", "② 정보 Agent", "③ 상품 Agent", "[④ 검증]", "[⑤ 최종 답변 조립]"):
        assert marker in trace
    # 시간순: ① → ② → ③ → ④ → ⑤
    assert trace.index("[① 질문 분류]") < trace.index("② 정보 Agent") < trace.index("[④ 검증]")


def test_narrative_describes_plan_and_tool_calls():
    trace = _format_think_trace(_composite_state())
    assert "복합형" in trace
    assert "순차 실행" in trace  # 실행 계획 문장
    assert 'search_pension_docs(query="퇴직소득세 감면") → 3건 검색' in trace


def test_narrative_marks_repair_rerun_segment():
    # 같은 노드가 다른 노드 뒤에 다시 나오면 "재실행" 구간으로 표시된다.
    trace = _format_think_trace(_composite_state())
    assert "(④검증 후 재실행)" in trace
    assert "1회 재실행한 결과를 반영" in trace


def test_narrative_reports_l0_and_premise_findings():
    trace = _format_think_trace(_composite_state())
    assert "L0 결정론적 수치 대조" in trace
    assert "L1 근거 부합: 통과" in trace
    assert "어마어마하다'는 과장" in trace


def test_narrative_flags_agent_that_called_no_tools():
    # 실측된 실패 유형: 툴 없이 학습 지식으로 답한 경우가 서사에 드러나야 한다.
    state = {
        "intent": ["정보형"],
        "scope": "부분관련",
        "scope_note": "개인사업자의 연금계좌 세액공제 관점",
        "is_safe": True,
        "tool_trace": [],
        "info_draft": "노란우산공제 연 500만원까지 가능합니다.",
        "retrieved_context": [],
        "verification": {
            "grounded": False,
            "issues": ["근거가 0건인데 초안에 구체적 수치가 있습니다: 500만원"],
            "l0_suspect_numbers": ["500만원"],
            "unsupported_numbers_confirmed": [],
            "requirements_met": True,
        },
    }
    trace = _format_think_trace(state)
    assert "툴 호출 없이 답변 작성 (근거 미확보)" in trace
    assert "L1 근거 부합: 불합격" in trace
    assert "사용한 근거 없음" in trace
    assert "개인사업자의 연금계좌 세액공제 관점" in trace


def test_narrative_reports_clarification_mode():
    state = {
        "intent": ["상품형"],
        "scope": "범위내",
        "is_safe": True,
        "tool_trace": [],
        "product_draft": "투자 가능한 계좌유형을 알려주시면 후보를 좁혀드릴게요.",
        "retrieved_context": [],
        "needs_clarification": True,
        "verification": {
            "grounded": True,
            "requirements_met": True,
            "clarification_mode": True,
            "l0_suspect_numbers": [],
        },
    }
    trace = _format_think_trace(state)
    assert "검증 면제" in trace
    assert "첫 답변에 정보한계와 필요한 역질문 전체를 포함" in trace


# ── 폴백 경로의 근거 출처 표기 (F-7) ──────────────────────────────────
#
# ③이 폴백을 쓰면 근거는 실재하지만 tool_trace에는 안 잡힌다(코드가 DB를 직접 조회하므로).
# 그대로 두면 같은 trace에 "툴 호출 없이 답변 작성 (근거 미확보)"와 "근거 N건 사용"이
# 동시에 나와 서로 모순된다 — 심사자가 답변과 trace를 대조하면 바로 드러나는 결함이다.

def _fallback_state(**overrides):
    state = {
        "question": "IRP에 넣을 펀드 추천해줘",
        "intent": ["상품형"],
        "scope": "범위내",
        "is_safe": True,
        "product_draft": "추천 상품은 다음과 같습니다...",
        "retrieved_context": [{"source": "미래에셋퇴직플랜단기(C)", "content": "위험등급 5"}],
        "tool_trace": [],
        "verification": {"grounded": True, "issues": [], "requirements_met": True},
    }
    state.update(overrides)
    return state


def test_fallback_trace_states_evidence_origin():
    """폴백이 근거를 만들었으면 그 사실을 밝힌다 — '근거 미확보'라고 쓰면 안 된다."""
    from src.agents.generator import _format_think_trace

    trace = _format_think_trace(_fallback_state(product_fallback_used=True))

    assert "근거 미확보" not in trace
    assert "폴백" in trace
    assert "투자설명서 DB를 직접 조회" in trace


def test_deterministic_info_trace_states_mapped_evidence_origin():
    """정형 정보 답변은 근거를 함께 반환하므로 '근거 미확보'로 쓰면 안 된다."""
    from src.agents.generator import _format_think_trace

    state = {
        "question": "전세 중도인출 세금은?",
        "intent": ["정보형"],
        "scope": "범위내",
        "is_safe": True,
        "info_draft": "세금은 재원별로 다릅니다.",
        "deterministic_info": True,
        "retrieved_context": [{"source": "doc40", "content": "16.5%"}],
        "tool_trace": [],
        "verification": {"grounded": True, "issues": [], "requirements_met": True},
    }

    trace = _format_think_trace(state)

    assert "정형 규칙 핸들러 실행" in trace
    assert "사전 매핑된 근거" in trace
    assert "근거 미확보" not in trace


def test_guardian_trace_separates_core_and_guardian_evidence():
    """파수꾼 근거는 Core 근거와 분리해 표시해야 후단 계층임이 드러난다."""
    from src.agents.generator import _format_think_trace

    state = {
        "question": "전세보증금 중도인출 필요서류 알려줘",
        "intent": ["정보형"],
        "scope": "범위내",
        "is_safe": True,
        "info_draft": "필요서류는 다음과 같습니다.",
        "deterministic_info": True,
        "retrieved_context": [
            {"source": "doc48 중도인출 무주택 전월세보증금 필요서류", "content": "서류", "node": "info_agent"}
        ],
        "guardian_result": {
            "enabled": True,
            "candidate_id": "housing_deposit_documents",
            "topic": "withdrawal_tax",
        },
        "guardian_evidence": [
            {"source": "doc38~doc40 주택 관련 중도인출 재원별 과세 규칙", "content": "세금", "node": "guardian"}
        ],
        "tool_trace": [],
        "verification": {"grounded": True, "issues": [], "requirements_met": True},
        "response_mode": "complete",
    }

    trace = _format_think_trace(state)

    assert "Core 근거 1건 사용: doc48 중도인출 무주택 전월세보증금 필요서류" in trace
    assert "파수꾼 근거 1건 사용: doc38~doc40 주택 관련 중도인출 재원별 과세 규칙" in trace
    assert "근거 2건 사용: doc48" not in trace


def test_withdrawal_trace_records_intentional_guardian_off_reason():
    from src.agents.generator import _format_think_trace

    trace = _format_think_trace({
        "question": "퇴직금 일부는 중도인출하고 나머지는 연금으로 받으면 세금 어떻게 돼?",
        "intent": ["정보형"],
        "scope": "범위내",
        "is_safe": True,
        "response_mode": "complete",
        "withdrawal_context": {
            "source_type": "RETIREMENT_PAY",
            "source_type_source": "explicit_user",
            "receipt_mode": "SPLIT",
            "receipt_mode_source": "deterministic_rule",
            "locked_fields": ["source_type", "receipt_mode"],
        },
        "verification": {"grounded": True, "requirements_met": True},
        "guardian_result": {"enabled": False, "disabled_reason": "EXPLICIT_USER_TOPIC"},
    })

    assert "중도인출 문맥 고정" in trace
    assert "파수꾼 체크 미실행: EXPLICIT_USER_TOPIC" in trace


def test_fallback_trace_marks_draft_replacement():
    """폴백은 LLM 초안을 폐기하고 대체한다 — 그 사실도 trace에 남긴다."""
    from src.agents.generator import _format_think_trace

    trace = _format_think_trace(_fallback_state(product_fallback_used=True))

    assert "초안을 폐기" in trace
    assert "③ 폴백 조회 결과" in trace


def test_no_tool_no_evidence_still_warns():
    """폴백이 아닌 진짜 '툴 미호출 + 근거 없음'은 기존 경고를 유지한다."""
    from src.agents.generator import _format_think_trace

    trace = _format_think_trace(
        _fallback_state(product_fallback_used=False, retrieved_context=[])
    )

    assert "툴 호출 없이 답변 작성 (근거 미확보)" in trace
    assert "폴백" not in trace


def test_normal_tool_path_unaffected():
    """툴을 정상 호출한 경로는 폴백 문구가 붙지 않는다."""
    from src.agents.generator import _format_think_trace

    trace = _format_think_trace(_fallback_state(
        product_fallback_used=False,
        tool_trace=[{"node": "product_agent", "tool": "search_funds",
                     "args": "risk_grade_max=3", "result": "2건"}],
    ))

    assert "search_funds" in trace
    assert "폴백" not in trace
    assert "근거 미확보" not in trace
