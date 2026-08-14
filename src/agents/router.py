"""① 라우터/가드레일 노드 — 질문을 정보형/상품형(복합형은 둘 다)으로 다중분류하고
안전성을 사전 필터링한다.

복합형일 때는 항상 ②정보 Agent → ③상품 Agent 순차 실행이다 (병렬 아님) — ③이 ②의
retrieved_context를 State에서 그대로 읽어 쓸 수 있어야 하기 때문 (설계 결정, 2026-08-11).

⚠️ Structured Outputs는 네이버 공식 문서 기준 HCX-007에서만 지원된다(HCX-DASH-002는
"Unsupported function" 에러). HCX-007은 기본적으로 Thinking이 켜져 있어 Structured
Outputs와 동시 사용이 안 되므로 thinking_effort="none"으로 꺼서 써야 한다.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState

ROUTER_MODEL = "HCX-007"

ROUTER_SYSTEM_PROMPT = """당신은 연금 상담 AI의 질문 분류 게이트입니다. 사용자 질문을 보고 다음을 판단하세요.

1. intent (해당하는 것 전부 선택, 복합형이면 둘 다):
   - 정보형: 연금 제도, 세금(세액공제/연금소득세/퇴직소득세), 인출·이전 규정 등 지식성 질문
   - 상품형: 특정 펀드/상품 추천, 비교, 매수 가능 여부 등 상품 관련 질문
   (예: "IRP에 넣을 펀드 추천해줘"는 상품 추천이지만 계좌 유형별 투자한도 판단이 필요하므로
   상품형만으로 충분합니다 — 투자한도는 상품 Agent가 자체 툴로 확인합니다.)

2. is_safe: 아래에 해당하는 경우에만 False로 표시하세요. 애매하면 True로 두고 넘기세요 —
   과도한 차단이 더 큰 문제입니다.
   - 확정 수익률/원금 보장을 요구하거나 암시하는 질문
   - 탈세(소득 은닉, 허위서류 등 명백히 불법인 방법)를 구체적으로 묻는 질문
   - 본 서비스 범위를 벗어나는 개인정보·명의도용 관련 요청

   ⚠️ 절세(세액공제·과세이연·연금수령 시기 조정처럼 세법이 허용하는 합법적 세금 최적화)는
   이 서비스의 핵심 주제이며 전혀 위험하지 않습니다 — "절세법 알려주세요", "세금 어떻게
   줄여요?" 같은 질문에 is_safe=False를 주면 안 됩니다. 질문에 "세금감면이 어마어마하다던데"
   처럼 과장되거나 잘못된 전제가 섞여 있어도 그 자체로는 차단 사유가 아닙니다 — 그 전제를
   바로잡아 답하면 되는 정상적인 정보형/종합 질문입니다.
"""


class RouterDecision(BaseModel):
    """사용자 질문의 의도 분류와 안전성 판정 결과."""

    intent: list[Literal["정보형", "상품형"]] = Field(
        description="해당하는 의도 전부. 정보+상품이 모두 필요하면 둘 다 포함(복합형)."
    )
    is_safe: bool = Field(description="질문이 안전 가이드라인을 위반하지 않으면 True")
    safety_reason: Optional[str] = Field(default=None, description="is_safe=False일 때만 사유를 적는다")


def build_router_node():
    llm = get_llm(ROUTER_MODEL, thinking_effort="none").with_structured_output(RouterDecision)

    def router_node(state: PensionAgentState) -> dict:
        decision: RouterDecision = invoke_with_retry(llm, [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": state["question"]},
        ])
        return {
            "intent": decision.intent,
            "is_safe": decision.is_safe,
            "safety_reason": decision.safety_reason,
        }

    return router_node
