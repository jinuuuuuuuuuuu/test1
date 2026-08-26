"""④ 검증/Grounding 노드 — ②③이 만든 답변 초안을 검증한다 (설명회 "검증 명세서
L0~L3" 체계를 반영):
  - L0 결정론적 검사(src/agents/verification.py): 초안의 수치를 근거 원문과 코드로 대조해
    의심 목록을 만들고, LLM 판정 위에 오버라이드를 건다 — 근거 0건 + 수치 존재면 LLM이
    뭐라 하든 grounded=False. (프롬프트 순종에만 의존하다 실제로 뚫린 사례가 있어 추가됨.)
  - L1 근거부합(grounded): 근거에 없는 수치·단정적 주장을 하지 않았는가 — LLM은 L0 의심
    목록을 근거 번호와 대조·확인하는 기계적 과제를 받는다
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

from src.agents.context import dedupe_context, merge_drafts
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState
from src.agents.verification import (
    apply_clarification_override,
    apply_l0_overrides,
    find_unsupported_numbers,
)

GROUNDING_MODEL = "HCX-007"

GROUNDING_SYSTEM_PROMPT = """당신은 연금 상담 AI의 답변 검증기입니다. [질문], [초안 답변],
[근거](번호 매김), [코드 검사: 근거에 없는 것으로 보이는 수치]를 보고 세 가지를 확인하세요.
당신 스스로 새로운 사실을 추가하거나 재계산하지 마세요 — 아래 판정만 내리면 됩니다.

1. grounded / issues / unsupported_numbers_confirmed:
   [코드 검사] 목록의 각 수치를 [근거 N] 원문과 하나씩 대조하세요. 정말 어떤 근거에도 없는
   수치면 unsupported_numbers_confirmed에 그대로 넣고, 표기 차이일 뿐 실제로는 근거에 있는
   값(예: 900만원 vs 9백만 원, 0.5% vs 0.50%)이면 제외하세요. 목록에 없더라도 근거로
   뒷받침되지 않는 단정적 주장·수치가 초안에 있으면 issues에 적으세요.
   grounded=True는 초안의 모든 구체적 수치와 단정적 주장이 근거 항목으로 뒷받침될 때만
   가능합니다. 초안에 구체적 수치가 하나도 없는 순수 개념 설명일 때도 grounded=True로 둘 수
   있지만, 근거가 0건인데 초안에 구체적 수치가 있으면 반드시 grounded=False입니다.
   예외: [질문]이나 이전 사용자 발화에 등장한 수치(예: "월 30만원 투자 가능")를 초안이
   그대로 되받아 정리한 것은 근거 위반이 아닙니다.

   ⚠️ 근거 적격성도 함께 보세요 — 수치가 근거 안에 있더라도, 그 근거가 질문 대상에 대해
   답할 자격이 있어야 합니다. 특히 개별 상품(특정 펀드) 데이터를 근거로 제도·상품군 전체에
   대한 일반적 결론을 내렸다면 grounded=False이고, issues에 그 사실을 적으세요.
   예: 질문은 "연금저축 펀드 환매 제한기간이 있나요"(제도 일반)인데 근거는 특정 펀드 3개의
   환매 시각 규정뿐인 상황에서 초안이 "연금저축 펀드는 일반적으로 환매 제한이 없다"고
   결론내린 경우 — 조회된 3개 펀드에 대한 사실일 뿐 전체에 대한 근거가 아니므로 위반입니다.
   반대로 초안이 "조회한 ○○펀드의 경우"처럼 근거 범위를 명시해 한정했다면 위반이 아닙니다.

   ⚠️ 근거 문장의 **의미가 뒤집히지 않았는지**도 보세요. 수치가 근거에 있어도 그 수치가
   붙은 조건·방향을 반대로 옮겼다면 위반입니다 — 수치 대조만으로는 걸러지지 않는 유형이라
   여기서 잡아야 합니다. 특히 다음을 확인하세요:
   - 허용/의무 뒤집기: 근거의 "기간 제한 없음", "~까지 적용", "~할 수 있다"를 초안이
     "반드시 ~해야 한다", "~년 이상 의무"로 바꿔 쓰지 않았는가
     (실측 사례: 근거 "연금수령기간: 기간제한 없음(연금수령한도는 10년차까지 적용)"을
     초안이 "의무적으로 10년 이상 연금으로 수령해야 한다"로 뒤집어 서술)
   - 가능/불가 뒤집기, 상한/하한 뒤집기(이하↔이상), 대상 뒤집기(A만 해당↔A는 제외)
   이런 왜곡을 발견하면 grounded=False로 두고 issues에 어떤 근거를 어떻게 뒤집었는지 적으세요.

2. premise_issues: 질문 자체에 사실과 다르거나 과장된 전제("세금 감면이 어마어마하다던데" 같은
   유도성 표현, 잘못된 제도 이해 등)가 섞여 있는데 초안이 그걸 그대로 받아들이고 넘어갔다면,
   어떤 전제를 바로잡아야 하는지 premise_issues에 적으세요. 문제 없으면 빈 리스트로 둡니다.

3. requirements_met / missing_requirements: 질문이 요구한 항목(여러 개를 동시에 물었다면 그
   전부)을 초안이 빠짐없이 다뤘는지 확인하세요. 하나라도 빠졌다면 requirements_met=False로
   표시하고 missing_requirements에 빠진 항목을 구체적으로 적으세요."""


class GroundingResult(BaseModel):
    """답변 초안이 근거·질문 요구사항과 부합하는지에 대한 검증 결과."""

    grounded: bool = Field(description="초안의 모든 구체적 수치·단정 주장이 근거로 뒷받침되면 True")
    issues: List[str] = Field(default_factory=list, description="근거와 어긋나는 부분 목록 (없으면 빈 리스트)")
    unsupported_numbers_confirmed: List[str] = Field(
        default_factory=list,
        description="[코드 검사] 의심 수치 목록 중 실제로 어떤 근거에도 없다고 확인된 것 (표기 차이로 근거에 있으면 제외)",
    )
    premise_issues: List[str] = Field(
        default_factory=list, description="질문의 잘못된/과장된 전제 중 초안이 바로잡지 않은 것 (없으면 빈 리스트)"
    )
    requirements_met: bool = Field(description="질문이 요구한 항목을 모두 다뤘으면 True")
    missing_requirements: List[str] = Field(
        default_factory=list, description="질문에서 요구했는데 초안이 빠뜨린 항목 (없으면 빈 리스트)"
    )


def build_grounding_node():
    # ⚠️ router.py와 동일한 이유로 method="json_schema" 명시 필요 (실측: 미지정 시 간헐적
    # 400 "Unsupported function"/40009).
    llm = get_llm(GROUNDING_MODEL, thinking_effort="none").with_structured_output(
        GroundingResult, method="json_schema"
    )

    def grounding_node(state: PensionAgentState) -> dict:
        # 복합형에서는 info_draft/product_draft가 둘 다 있으므로 반드시 병합해서 검증한다.
        draft = merge_drafts(state.get("info_draft"), state.get("product_draft"))
        context = dedupe_context(state.get("retrieved_context") or [])
        context_text = "\n".join(
            f"[근거 {i}] [{c['source']}] {c['content']}" for i, c in enumerate(context, 1)
        ) or "(근거 없음)"

        # L0: 초안의 수치를 근거 원문과 기계적으로 대조해 의심 목록을 만든다 — ④ LLM은 이
        # 목록을 하나씩 확인하는 과제를 받고, 최종 판정은 apply_l0_overrides가 강제한다.
        # 사용자 발화(현재 질문 + 이전 턴 질문)의 수치는 초안이 되받아 써도 근거 위반이
        # 아니다 — 예: 역질문 흐름에서 "월 30만원·10년 이상" 조건 요약. 이전 턴의 '답변'은
        # 지원 근거로 넣지 않는다.
        history = state.get("conversation_history") or []
        user_texts = [state["question"], *(turn.get("question", "") for turn in history)]
        suspects = find_unsupported_numbers(
            draft, [c["content"] for c in context], user_texts=user_texts
        )
        suspects_text = "\n".join(f"- {n}" for n in suspects) or "(없음)"

        clarification = bool(state.get("needs_clarification"))
        type_recommendation = state.get("recommendation_stage") == "type_recommendation"
        clarification_note = (
            "\n\n[참고] 초안은 답변 조건이 불충분해 첫 답변 안에 정보한계와 필요한 역질문을 "
            "포함한 초안입니다. 부족한 정보를 묻는 것 자체를 요구사항 누락으로 판정하지 마세요."
            if clarification
            else ""
        )

        result: GroundingResult = invoke_with_retry(llm, [
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[질문]\n{state['question']}\n\n"
                    f"[초안 답변]\n{draft}\n\n"
                    f"[근거]\n{context_text}\n\n"
                    f"[코드 검사: 근거에 없는 것으로 보이는 수치]\n{suspects_text}"
                    f"{clarification_note}"
                ),
            },
        ])

        verification = apply_l0_overrides(
            {
                "grounded": result.grounded,
                "issues": result.issues,
                "unsupported_numbers_confirmed": result.unsupported_numbers_confirmed,
                "premise_issues": result.premise_issues,
                "requirements_met": result.requirements_met,
                "missing_requirements": result.missing_requirements,
            },
            suspects=suspects,
            has_evidence=bool(context),
        )
        if clarification or type_recommendation:
            # 역질문 초안은 요구사항 검증을 코드로 면제한다 (프롬프트 지시만으로는 ④가
            # "추천 누락"으로 판정 → ⑤가 추천을 되살리는 경로를 막을 수 없다).
            verification = apply_clarification_override(verification)
        return {"verification": verification}

    return grounding_node
