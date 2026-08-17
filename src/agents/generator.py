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


_NODE_LABELS = {
    "info_agent": "② 정보 Agent — 제도·세제 조사",
    "product_agent": "③ 상품 Agent — 상품 확인",
}


def _classification_lines(state: PensionAgentState) -> list[str]:
    intent = state.get("intent") or []
    if len(intent) > 1:
        intent_label = "복합형(제도·세제 + 상품)"
    elif intent:
        intent_label = intent[0]
    else:
        intent_label = "분류 실패"
    safety = "차단" if state.get("is_safe") is False else "통과"

    lines = [
        "[① 질문 분류]",
        f"- 의도: {intent_label} / 서비스 범위: {state.get('scope') or '미판정'} / 안전성: {safety}",
    ]
    if state.get("scope_note"):
        lines.append(f"- 범위 판단: {state['scope_note']}")
    return lines


def _plan_sentence(state: PensionAgentState) -> str:
    """①의 분류 결과가 어떤 실행 경로로 이어졌는지 한 문장으로 설명한다."""
    if state.get("is_safe") is False:
        return "안전 가이드라인 위반으로 ②③④를 건너뛰고 정형 거절 응답을 반환"
    if state.get("scope") == "범위외":
        return "연금 상담 범위를 벗어나 ②③④를 건너뛰고 정형 한계 고지를 반환"
    intent = state.get("intent") or []
    if "정보형" in intent and "상품형" in intent:
        return "②정보 Agent로 제도·세제 근거를 먼저 확보한 뒤, 그 근거를 넘겨 ③상품 Agent를 순차 실행"
    if "상품형" in intent:
        return "상품 질의이므로 ③상품 Agent를 곧바로 실행"
    if "정보형" in intent:
        return "제도·세제 질의이므로 ②정보 Agent를 실행"
    return "의도 분류에 실패해 검색 툴을 가진 ②정보 Agent로 폴백"


def _tool_trace_lines(state: PensionAgentState) -> list[str]:
    """툴 호출 기록을 노드별 구간으로 묶어 실행 순서대로 서술한다."""
    trace = state.get("tool_trace") or []
    lines: list[str] = []

    # 연속된 같은 노드 기록을 한 구간으로 묶는다 — repair 재실행은 뒤에 다시 붙으므로
    # 자연히 별도 구간("재실행")으로 드러난다.
    segments: list[tuple[str, list]] = []
    for record in trace:
        if segments and segments[-1][0] == record["node"]:
            segments[-1][1].append(record)
        else:
            segments.append((record["node"], [record]))

    seen_nodes: set[str] = set()
    for node, records in segments:
        label = _NODE_LABELS.get(node, node)
        suffix = " (④검증 후 재실행)" if node in seen_nodes else ""
        seen_nodes.add(node)
        lines.append(f"[{label}{suffix}]")
        for i, record in enumerate(records, 1):
            call = f"{record['tool']}({record['args']})"
            lines.append(f"  {i}. {call} → {record['result']}")

    # 툴을 한 번도 호출하지 않고 답을 낸 노드는 기록이 없어 위 구간에 안 나타난다 —
    # 근거 없는 답변의 신호이므로 명시한다 (실측된 실패 유형).
    for node, draft_key in (("info_agent", "info_draft"), ("product_agent", "product_draft")):
        if state.get(draft_key) and node not in seen_nodes:
            lines.append(f"[{_NODE_LABELS[node]}]")
            lines.append("  - 툴 호출 없이 답변 작성 (근거 미확보)")
    return lines


def _verification_lines(state: PensionAgentState) -> list[str]:
    verification = state.get("verification")
    if not verification:
        return ["[④ 검증] 수행하지 않음 (①에서 종료된 경로)"]

    suspects = verification.get("l0_suspect_numbers") or []
    confirmed = verification.get("unsupported_numbers_confirmed") or []
    lines = ["[④ 검증]"]
    if suspects:
        lines.append(
            f"  - L0 결정론적 수치 대조: 근거에서 확인되지 않는 수치 {len(suspects)}개 발견"
            f"({', '.join(suspects)}) → 그중 {len(confirmed)}개가 근거 부재로 확정"
            + (f"({', '.join(confirmed)})" if confirmed else "")
        )
    else:
        lines.append("  - L0 결정론적 수치 대조: 초안의 모든 수치가 근거/질문에서 확인됨")

    grounded = verification.get("grounded")
    issues = verification.get("issues") or []
    lines.append(
        f"  - L1 근거 부합: {'통과' if grounded else '불합격'}"
        + (f" — {'; '.join(issues)}" if issues else "")
    )

    premise = verification.get("premise_issues") or []
    lines.append(
        f"  - 전제 교정: {'; '.join(premise)}" if premise else "  - 전제 교정: 잘못된 전제 없음"
    )

    if verification.get("clarification_mode"):
        lines.append("  - 요구사항 충족: 검증 면제 (조건 불충분으로 역질문한 초안)")
    else:
        missing = verification.get("missing_requirements") or []
        lines.append(
            f"  - 요구사항 충족: {'충족' if verification.get('requirements_met') else '미충족'}"
            + (f" — 누락: {'; '.join(missing)}" if missing else "")
        )
    return lines


def _assembly_lines(state: PensionAgentState, context: list) -> list[str]:
    lines = ["[⑤ 최종 답변 조립]"]
    if context:
        sources = "; ".join(dict.fromkeys(c["source"] for c in context))
        lines.append(f"  - 근거 {len(context)}건 사용: {sources}")
    else:
        lines.append("  - 사용한 근거 없음 (근거가 필요한 수치는 답변에서 제외)")
    if state.get("needs_clarification"):
        lines.append("  - 조건 불충분 → 답을 만들지 않고 역질문을 다듬어 전달")
    if state.get("repair_attempted"):
        lines.append("  - ④검증 탈락으로 ②③을 1회 재실행한 결과를 반영 (재실행 한도 1회 소진)")
    return lines


def _format_think_trace(state: PensionAgentState) -> str:
    """대회 평가 스키마의 think_trace — "사고·추론·도구 사용 과정"을 시간순 서사로 조립한다.

    추가 LLM 호출 없이 State에 이미 있는 값(①분류, tool_trace, ④검증 결과)만으로 만든다.
    """
    context = dedupe_context(state.get("retrieved_context") or [])
    lines = _classification_lines(state)
    lines.append(f"- 실행 계획: {_plan_sentence(state)}")
    lines.extend(_tool_trace_lines(state))
    lines.extend(_verification_lines(state))
    lines.extend(_assembly_lines(state, context))
    lines.append("[참고: 근거 원문]")
    if context:
        lines.extend(f"  - [{c['source']}] {c['content']}" for c in context)
    else:
        lines.append("  (없음)")
    return "\n".join(lines)


def build_generator_node():
    llm = get_llm(GENERATOR_MODEL)

    def generator_node(state: PensionAgentState) -> dict:
        if state.get("is_safe") is False:
            reason = state.get("safety_reason") or "요청하신 내용은 안내해드리기 어렵습니다."
            return {
                "answer": f"죄송하지만 해당 질문에는 답변드릴 수 없습니다. ({reason})",
                "think_trace": "\n".join([
                    *_classification_lines(state),
                    f"- 차단 사유: {reason}",
                    f"- 실행 계획: {_plan_sentence(state)}",
                ]),
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
                "think_trace": "\n".join([
                    *_classification_lines(state),
                    f"- 실행 계획: {_plan_sentence(state)}",
                    "[⑤ 최종 답변 조립]",
                    "  - 보유 자료로 답할 수 없는 범위이므로 한계를 고지 (추측 답변 생성 금지)",
                ]),
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
