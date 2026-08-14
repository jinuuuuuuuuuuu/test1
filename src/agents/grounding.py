"""④ 검증/Grounding 노드 — ②③이 만든 답변 초안을 4단계로 검증한다 (설명회 "검증 명세서
L0~L3" 체계를 반영):
  - L1 근거부합(grounded): 근거에 없는 수치·단정적 주장을 하지 않았는가
  - L1 전제교정(premise_issues): 질문에 잘못된 전제·유도성 주장이 섞여 있는데 그대로
    받아들이지 않았는가 (대회 평가지표 "정확성"의 핵심 항목)
  - L3 요구사항충족(requirements_met/missing_requirements): 질문이 요구한 항목을
    빠짐없이 답했는가 (대회 평가지표 "요구사항 충족"에 직접 대응)

새로 추론하거나 계산하는 무거운 에이전트가 아니라, "초안이 근거를 벗어난 주장을 하지
않는지 + 질문에 다 답했는지"만 가볍게 체크하는 모델 호출이다 (설계 원칙, 세션 초반 합의 —
④는 별도 무거운 추론 에이전트로 만들지 않는다).

⚠️ Structured Outputs는 네이버 공식 문서 기준 HCX-007에서만 지원된다(HCX-DASH-002는
"Unsupported function" 에러로 애초에 구조화 출력을 못 한다). HCX-007은 기본적으로 Thinking이
켜져 있어 Structured Outputs와 동시 사용이 안 되므로 thinking_effort="none"으로 꺼서 써야 한다.

⚠️ 이전 버전은 원본 질문(state["question"])을 프롬프트에 아예 넘기지 않아서 "질문에 다
답했는지"를 구조적으로 검증할 수 없었다 — 이번에 질문을 포함하도록 고쳤다.
"""

from typing import List

from pydantic import BaseModel, Field

from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState

GROUNDING_MODEL = "HCX-007"

GROUNDING_SYSTEM_PROMPT = """당신은 연금 상담 AI의 답변 검증기입니다. [질문], [초안 답변],
[근거]를 보고 세 가지를 확인하세요. 당신 스스로 새로운 사실을 추가하거나 재계산하지 마세요 —
아래 판정만 내리면 됩니다.

1. grounded / issues: 초안에 근거로 뒷받침되지 않는 수치·단정적 주장이 있으면 grounded=False로
   표시하고 issues에 어떤 문장이 문제인지 적으세요. 근거가 필요 없는 순수 설명형 질문이면
   grounded=True로 둡니다.

2. premise_issues: 질문 자체에 사실과 다르거나 과장된 전제("세금 감면이 어마어마하다던데" 같은
   유도성 표현, 잘못된 제도 이해 등)가 섞여 있는데 초안이 그걸 그대로 받아들이고 넘어갔다면,
   어떤 전제를 바로잡아야 하는지 premise_issues에 적으세요. 문제 없으면 빈 리스트로 둡니다.

3. requirements_met / missing_requirements: 질문이 요구한 항목(여러 개를 동시에 물었다면 그
   전부)을 초안이 빠짐없이 다뤘는지 확인하세요. 하나라도 빠졌다면 requirements_met=False로
   표시하고 missing_requirements에 빠진 항목을 구체적으로 적으세요."""


class GroundingResult(BaseModel):
    """답변 초안이 근거·질문 요구사항과 부합하는지에 대한 검증 결과."""

    grounded: bool = Field(description="초안이 근거와 일치하거나 근거가 불필요한 설명이면 True")
    issues: List[str] = Field(default_factory=list, description="근거와 어긋나는 부분 목록 (없으면 빈 리스트)")
    premise_issues: List[str] = Field(
        default_factory=list, description="질문의 잘못된/과장된 전제 중 초안이 바로잡지 않은 것 (없으면 빈 리스트)"
    )
    requirements_met: bool = Field(description="질문이 요구한 항목을 모두 다뤘으면 True")
    missing_requirements: List[str] = Field(
        default_factory=list, description="질문에서 요구했는데 초안이 빠뜨린 항목 (없으면 빈 리스트)"
    )


def build_grounding_node():
    llm = get_llm(GROUNDING_MODEL, thinking_effort="none").with_structured_output(GroundingResult)

    def grounding_node(state: PensionAgentState) -> dict:
        draft = state.get("info_draft") or state.get("product_draft") or ""
        context = state.get("retrieved_context") or []
        context_text = "\n".join(f"- [{c['source']}] {c['content']}" for c in context) or "(근거 없음)"

        result: GroundingResult = invoke_with_retry(llm, [
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[질문]\n{state['question']}\n\n"
                    f"[초안 답변]\n{draft}\n\n"
                    f"[근거]\n{context_text}"
                ),
            },
        ])

        return {
            "verification": {
                "grounded": result.grounded,
                "issues": result.issues,
                "premise_issues": result.premise_issues,
                "requirements_met": result.requirements_met,
                "missing_requirements": result.missing_requirements,
            }
        }

    return grounding_node
