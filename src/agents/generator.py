"""⑤ 생성기 노드 — 초안·근거·검증결과를 받아 최종 답변을 조립하고, 대회 평가 API 스키마의
think_trace를 포맷팅한다. HCX-005 사용.

is_safe=False(①가드레일에서 차단된 경우)는 모델을 호출하지 않고 바로 정형 거절 응답을 만든다.
"""

from src.agents.context import dedupe_context, format_conversation_history, merge_drafts
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState

GENERATOR_MODEL = "HCX-005"

GENERATOR_SYSTEM_PROMPT = """당신은 연금 상담 AI의 최종 답변 작성자입니다. [질문], [초안],
[근거], [검증결과]를 참고해 사용자에게 보여줄 자연스러운 한국어 답변을 작성하세요.

- grounded=False이면, [근거]에 등장하지 않는 퍼센트(%)·금액·기간 등 숫자를 답변에 **하나도**
  쓰지 마세요. 초안에 그런 숫자가 있어도 절대 베끼지 마세요 — [근거]를 다시 보고 거기 있는
  숫자만 쓰거나, 숫자 없이 "구체적인 공제 비율은 계좌·수령 방식에 따라 달라 확인이
  필요합니다"처럼 일반론으로만 답하세요. 이건 문체 문제가 아니라 규칙입니다: 답변에 쓸 숫자
  하나하나에 대해 "이 숫자가 [근거] 원문에 그대로 있는가?"를 자문하고, 없으면 삭제하세요.
- 검증결과의 premise_issues(질문의 잘못된/과장된 전제)가 있으면, 답변 시작 부분에서 그
  전제를 짚고 바로잡은 뒤 본 답변으로 넘어가세요 (예: "말씀하신 것만큼 크지는 않고,
  정확히는 ~입니다").
- 검증결과의 missing_requirements(질문에서 요구했는데 초안이 빠뜨린 항목)가 있으면, 근거에
  그 내용이 있는 경우 답변에 추가하세요. 근거로 확인이 안 되는 항목이면 "~는 제공된 자료로
  확인이 어렵습니다"처럼 한계를 명시하세요 — 답을 지어내지 마세요.
- [이전 대화]가 함께 주어지면 자연스러운 대화 흐름을 유지하되(중복 설명 반복 지양), 숫자·사실은
  이번 턴의 [근거]에 있는 것만 쓰세요 — 이전 답변에 등장했던 숫자라도 이번 [근거]에 없으면
  다시 쓰지 마세요.
- 근거 문서 표시(필수): [근거]를 사용해 답했다면, 핵심 수치·규정을 언급한 문장에
  (출처: 출처명)을 붙이고, 답변 마지막 줄에 "참고 근거: 출처명1; 출처명2" 목록을 적으세요.
  출처명은 [근거 N] 대괄호 안의 출처를 그대로 쓰고, [근거]에 없는 출처를 지어내지 마세요.
  근거를 전혀 사용하지 않은 답변(역질문, 순수 개념 설명, 한계 고지)에는 출처 표기를
  생략합니다."""


def _format_think_trace(state: PensionAgentState) -> str:
    context = dedupe_context(state.get("retrieved_context") or [])
    context_lines = "\n".join(f"  - [{c['source']}] {c['content']}" for c in context) or "  (없음)"
    verification = state.get("verification") or {}
    return (
        f"[①분류] intent={state.get('intent')}, scope={state.get('scope')}, is_safe={state.get('is_safe')}\n"
        f"[②③근거 {len(context)}건] needs_clarification={state.get('needs_clarification')}, "
        f"repair_attempted={state.get('repair_attempted')}\n{context_lines}\n"
        f"[④검증] grounded={verification.get('grounded')}, issues={verification.get('issues')}, "
        f"premise_issues={verification.get('premise_issues')}, "
        f"requirements_met={verification.get('requirements_met')}, "
        f"missing_requirements={verification.get('missing_requirements')}"
    )


def build_generator_node():
    llm = get_llm(GENERATOR_MODEL)

    def generator_node(state: PensionAgentState) -> dict:
        if state.get("is_safe") is False:
            reason = state.get("safety_reason") or "요청하신 내용은 안내해드리기 어렵습니다."
            return {
                "answer": f"죄송하지만 해당 질문에는 답변드릴 수 없습니다. ({reason})",
                "think_trace": f"[①가드레일 차단] {reason}",
            }

        if state.get("scope") == "범위외":
            # 연금과 접점이 없는 질문 — LLM 호출 없이 정형 한계 고지로 답한다. LLM에 맡기면
            # 범위 밖 주제를 학습 지식으로 답해버리는 실측 사례가 있어 결정론적으로 처리한다.
            note = state.get("scope_note")
            return {
                "answer": (
                    "죄송하지만 문의하신 내용은 본 서비스의 상담 범위(연금 제도 · 연금 세제 · "
                    "연금계좌에서 투자 가능한 상품)를 벗어나 정확한 답변을 드리기 어렵습니다. "
                    "연금 제도(DB/DC/IRP · 연금저축), 세액공제 등 연금 세제, 연금 상품에 대해 "
                    "질문해 주시면 도움을 드리겠습니다."
                ),
                "think_trace": f"[①분류] scope=범위외({note}) — 정형 한계 고지 응답",
            }

        # 복합형에서는 info_draft/product_draft가 둘 다 있으므로 반드시 병합해서 조립한다.
        draft = merge_drafts(state.get("info_draft"), state.get("product_draft"))
        verification = state.get("verification") or {}
        context = dedupe_context(state.get("retrieved_context") or [])
        context_text = "\n".join(
            f"[근거 {i}] [{c['source']}] {c['content']}" for i, c in enumerate(context, 1)
        ) or "(근거 없음)"
        history_text = format_conversation_history(state.get("conversation_history"))

        prompt = (
            (f"[이전 대화]\n{history_text}\n\n" if history_text else "")
            + f"[질문]\n{state['question']}\n\n"
            f"[초안]\n{draft}\n\n"
            f"[근거]\n{context_text}\n\n"
            f"[검증결과]\n"
            f"- grounded: {verification.get('grounded')}, issues: {verification.get('issues')}\n"
            f"- premise_issues(질문의 잘못된 전제): {verification.get('premise_issues')}\n"
            f"- requirements_met: {verification.get('requirements_met')}, "
            f"missing_requirements: {verification.get('missing_requirements')}"
        )
        if state.get("needs_clarification"):
            prompt += (
                "\n\n[참고] 초안은 답변에 필요한 조건이 불충분해 사용자에게 되묻는 "
                "역질문입니다. 근거를 이용해 답이나 추천을 만들어 보충하지 말고, 어떤 조건이 "
                "왜 필요한지가 잘 드러나도록 역질문을 자연스럽게 다듬어 전달만 하세요."
            )
        response = invoke_with_retry(llm, [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        return {
            "answer": response.content,
            "think_trace": _format_think_trace(state),
        }

    return generator_node
