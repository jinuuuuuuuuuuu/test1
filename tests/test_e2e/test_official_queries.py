"""대회 공식 참고 질의 E2E 회귀 테스트 (요강 p7).

실제 모델을 호출하므로 기본은 skip이고 RUN_LIVE_AGENT_TESTS=1일 때만 돈다
(tests/test_agents/test_graph.py의 기존 라이브 테스트와 같은 패턴).

    RUN_LIVE_AGENT_TESTS=1 pytest tests/test_e2e -v

## 왜 필요한가

단위 테스트 276개가 전부 통과하는 상태에서 대회 공식 질의 "솔로몬 국공채
단기·중장기·장기, 뭐가 달라요?"가 범위외로 오분류돼 3/3 거부되고 있었다(F-2).
파이프라인 전체를 공식 질의로 통과시키는 테스트가 하나도 없었기 때문이다.
제출 직전 회귀를 잡는 것이 이 파일의 목적이다.

## 무엇을 단언하는가

답변 문구를 고정하지 않는다. LLM 출력은 실행마다 달라지므로 "600만원이
포함돼야 한다" 식으로 못박으면 멀쩡한 답변에도 테스트가 깨지고, 결국 아무도
돌리지 않는 테스트가 된다.

대신 **지금까지 실측으로 확인한 결함이 재발하면 반드시 걸리는 조건**만 본다:
  - 답할 수 있는 질문을 거부하지 않았는가 (F-2)
  - 출처가 내부 인덱스 "[근거 N]"으로 새지 않았는가 (F-4)
  - 근거를 썼으면 출처를 표기했는가 (요강: 모든 답변에 근거 문서 표시)
  - 조건이 부족한 추천 요청에 단정 추천을 하지 않았는가 (요강: 단정 추천 금지)
"""

import os
import re

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_AGENT_TESTS") != "1",
    reason="실제 API 네트워크 호출이 필요하므로 RUN_LIVE_AGENT_TESTS=1일 때만 실행합니다",
)

# ⑤가 범위외 판정 시 내보내는 정형 거절문의 핵심 문구. 답할 수 있는 질문에
# 이 문구가 나오면 F-2 재발이다.
REFUSAL_MARKER = "상담 범위"

# 내부 근거 인덱스가 답변에 그대로 노출되는 패턴 (F-4).
EVIDENCE_PLACEHOLDER_RE = re.compile(r"\[\s*근거\s*\d+\s*\]")

# 요강 p7 참고 질의 set 5건.
OFFICIAL_QUERIES = {
    "C1_제도": "DC와 DB, 퇴직금이 정해지는 방식이랑 운용 주체가 어떻게 다른가요?",
    "C2_세제": "연금저축이랑 IRP에 넣으면 세액공제 얼마까지 되나요? 다 합쳐서요.",
    "C3_종합": (
        "명퇴하는 교사예요. 명퇴수당을 연금계좌에 넣으면 세금 감면이 "
        "어마어마하다던데, 절세법만 알려주세요."
    ),
    "C4_상품비교": "솔로몬 국공채 단기 · 중장기 · 장기, 뭐가 달라요? 안정적인 걸 원해요.",
    "C5_조건부추천": "좋은 연금 상품 하나 추천해 주세요.",
}


@pytest.fixture(scope="module")
def graph():
    from src.agents.graph import build_graph

    return build_graph()


def _run(graph, question_id: str, question: str) -> dict:
    from src.agents.text import normalize_user_text

    return graph.invoke({
        "question_id": question_id,
        "question": normalize_user_text(question),
        "conversation_history": [],
        "recommendation_profile": {},
    })


@pytest.mark.parametrize("label", sorted(OFFICIAL_QUERIES))
def test_official_query_is_answered_not_refused(graph, label):
    """공식 질의는 전부 답변 가능해야 한다 — 하나라도 거절문이 나오면 회귀다.

    F-2 실측: C4가 scope=범위외로 오분류돼 3/3 거부됐다. 해당 상품 4건이
    fund_master에 실재하는데도 "상담 범위를 벗어나 답변드리기 어렵습니다"를
    반환했다. 요강 평가지표상 답을 틀리는 것과 못 하는 것은 똑같이 감점이다.
    """
    result = _run(graph, label, OFFICIAL_QUERIES[label])
    answer = result.get("answer") or ""

    assert answer, f"{label}: 답변이 비어 있다"
    assert REFUSAL_MARKER not in answer, (
        f"{label}: 답할 수 있는 질문에 범위외 거절문이 나왔다 (scope={result.get('scope')})\n"
        f"답변: {answer[:200]}"
    )


@pytest.mark.parametrize("label", sorted(OFFICIAL_QUERIES))
def test_official_query_has_no_evidence_placeholder(graph, label):
    """내부 근거 인덱스가 사용자 답변에 노출되면 안 된다.

    F-4 실측: LLM 생성 답변 7건 중 5건이 "참고 근거: [근거 1]; [근거 2]"처럼
    출처명 대신 내부 표기를 그대로 내보냈다. 심사자에게 [근거 1]은 아무
    정보도 아니다 (요강: 모든 답변에 근거 문서 표시할 것).
    """
    answer = _run(graph, label, OFFICIAL_QUERIES[label]).get("answer") or ""

    leaked = EVIDENCE_PLACEHOLDER_RE.findall(answer)
    assert not leaked, f"{label}: 내부 근거 표기가 노출됐다 {leaked}"


@pytest.mark.parametrize("label", sorted(OFFICIAL_QUERIES))
def test_answer_cites_sources_when_evidence_used(graph, label):
    """근거를 사용했으면 출처를 표기해야 한다.

    근거가 0건인 답변(역질문, 순수 개념 설명, 한계 고지)은 표기 대상이 아니므로
    제외한다 — 없는 출처를 지어내는 것이 더 나쁘다.
    """
    result = _run(graph, label, OFFICIAL_QUERIES[label])
    context = result.get("retrieved_context") or []
    answer = result.get("answer") or ""

    if not context:
        pytest.skip(f"{label}: 근거 0건 답변이라 출처 표기 대상이 아님")

    assert "참고 근거:" in answer or "출처:" in answer, (
        f"{label}: 근거 {len(context)}건을 쓰고도 출처 표기가 없다"
    )


def test_underspecified_recommendation_asks_back_instead_of_recommending(graph):
    """조건이 없는 추천 요청에는 단정 추천 대신 확인 질문을 해야 한다.

    요강 주제2가 명시한 요구사항이다 — "확인조건을 먼저 제시하고 상황별 결론
    제공(단정적 추천 지양)".
    """
    result = _run(graph, "C5_조건부추천", OFFICIAL_QUERIES["C5_조건부추천"])
    answer = result.get("answer") or ""

    assert result.get("needs_clarification") is True, (
        f"조건 불충분 추천 요청인데 역질문 상태가 아니다 "
        f"(stage={result.get('recommendation_stage')})\n답변: {answer[:200]}"
    )
    assert "?" in answer, "확인이 필요한 조건을 되묻지 않았다"


def test_product_comparison_uses_prospectus_evidence(graph):
    """상품 비교 질의는 투자설명서 DB 근거로 답해야 한다.

    F-2가 고친 경로를 직접 지킨다. 거부하지 않는 것만으로는 부족하고,
    실제로 해당 상품 데이터를 조회했는지까지 봐야 회귀를 잡을 수 있다.
    """
    result = _run(graph, "C4_상품비교", OFFICIAL_QUERIES["C4_상품비교"])

    assert result.get("scope") != "범위외", "보유 상품 질의를 범위외로 판정했다"
    assert "상품형" in (result.get("intent") or []), (
        f"상품 비교 질의인데 상품형으로 분류되지 않았다 (intent={result.get('intent')})"
    )
    assert result.get("retrieved_context"), "상품 근거를 하나도 확보하지 못했다"
