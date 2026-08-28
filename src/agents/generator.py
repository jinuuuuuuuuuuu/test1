"""⑤ 생성기 노드 — 초안·근거·검증결과를 받아 최종 답변을 조립하고, 대회 평가 API 스키마의
think_trace를 포맷팅한다. HCX-005 사용.

is_safe=False(①가드레일에서 차단된 경우)는 모델을 호출하지 않고 바로 정형 거절 응답을 만든다.
"""

from src.agents.context import dedupe_context, format_conversation_history, merge_drafts
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState
from src.agents.verification import (
    enforce_missing_requirements,
    enforce_premise_issues,
    replace_evidence_placeholders,
    split_premise_issues,
)

GENERATOR_MODEL = "HCX-005"

# ⚠️ 이 프롬프트의 지시 중 premise_issues 반영·missing_requirements 한계고지·출처 표기
# 세 가지는 _enforce_verification()이 출력에서 다시 검사해 코드로 확정한다. 프롬프트가
# 제대로 따라주면 코드는 개입하지 않으므로(이미 반영됐는지 먼저 확인한다) 둘은 상충하지
# 않고, LLM이 지시를 흘렸을 때만 코드가 메운다 — 프롬프트는 품질을, 코드는 하한을 담당한다.
GENERATOR_SYSTEM_PROMPT = """당신은 연금 상담 AI의 최종 답변 작성자입니다. [질문], [초안],
[근거], [검증결과]를 참고해 사용자에게 보여줄 자연스러운 한국어 답변을 작성하세요.

- grounded=False이면, [근거]에 등장하지 않는 퍼센트(%)·금액·기간 등 숫자를 답변에 **하나도**
  쓰지 마세요. 초안에 그런 숫자가 있어도 절대 베끼지 마세요 — [근거]를 다시 보고 거기 있는
  숫자만 쓰거나, 숫자 없이 "구체적인 공제 비율은 계좌·수령 방식에 따라 달라 확인이
  필요합니다"처럼 일반론으로만 답하세요. 이건 문체 문제가 아니라 규칙입니다: 답변에 쓸 숫자
  하나하나에 대해 "이 숫자가 [근거] 원문에 그대로 있는가?"를 자문하고, 없으면 삭제하세요.
- [검증결과]에 "근거에 없는 것으로 확정된 수치"가 있으면, 그 수치들은 **grounded 값과
  무관하게** 답변에 절대 쓰지 마세요. 이미 근거 원문과 대조해 없다고 확인된 값이므로
  다시 판단하지 말고 그대로 배제하면 됩니다. 초안에 있어도 베끼지 말고, 그 수치가 필요한
  자리는 근거에 있는 값으로 바꾸거나 "제공된 자료로는 확인이 어렵습니다"로 처리하세요.
- 검증결과의 premise_issues(질문의 잘못된/과장된 전제)가 있으면, 답변 시작 부분에서 그
  전제를 짚고 바로잡은 뒤 본 답변으로 넘어가세요 (예: "말씀하신 것만큼 크지는 않고,
  정확히는 ~입니다").
- 검증결과의 missing_requirements(질문에서 요구했는데 초안이 빠뜨린 항목)가 있으면, 근거에
  그 내용이 있는 경우 답변에 추가하세요. 근거로 확인이 안 되는 항목이면 "~는 제공된 자료로
  확인이 어렵습니다"처럼 한계를 명시하세요 — 답을 지어내지 마세요.
- 가능 여부를 묻는 질문("~할 수 있나요", "~되나요")에는 **사용자 질문의 방향에 맞춰**
  첫머리에서 예/아니오를 분명히 하세요. 불가하면 "아니요, ~는 불가합니다"로 시작합니다.
  "네, 불가능합니다"처럼 쓰면 논리적으로는 "맞습니다"라는 뜻이어도 사용자는 반대로 읽습니다.
  가능하면 "네, ~할 수 있습니다"로 시작하고, 조건부라면 "조건에 따라 다릅니다"로 시작한 뒤
  조건을 설명하세요.
- [이전 대화]가 함께 주어지면 자연스러운 대화 흐름을 유지하되(중복 설명 반복 지양), 숫자·사실은
  이번 턴의 [근거]에 있는 것만 쓰세요 — 이전 답변에 등장했던 숫자라도 이번 [근거]에 없으면
  다시 쓰지 마세요.
- 수치를 강조할 때는 단위까지 함께 감싸세요. "**16.5**%"가 아니라 "**16.5%**"입니다.
  숫자만 감싸면 단위가 떨어져 나가 "10%"가 "10"으로 나가는 일이 실제로 있었습니다 —
  비율인지 금액인지 알 수 없는 숫자는 사용자를 오도합니다. 강조하지 않아도 무방하니,
  강조한다면 반드시 단위를 포함하세요.
- 근거 문서 표시(필수): [근거]를 사용해 답했다면, 핵심 수치·규정을 언급한 문장에
  (출처: 출처명)을 붙이고, 답변 마지막 줄에 "참고 근거: 출처명1; 출처명2" 목록을 적으세요.
  출처명은 [근거 N] 대괄호 안의 출처를 그대로 쓰고, [근거]에 없는 출처를 지어내지 마세요.
  근거를 전혀 사용하지 않은 답변(역질문, 순수 개념 설명, 한계 고지)에는 출처 표기를
  생략합니다."""


_NODE_LABELS = {
    "info_agent": "② 정보 Agent — 제도·세제 조사",
    "product_agent": "③ 상품 Agent — 상품 확인",
}


def _append_reference_line(answer: str, context: list) -> str:
    if not context or "참고 근거:" in answer:
        return answer
    sources = "; ".join(dict.fromkeys(c["source"] for c in context))
    return f"{answer}\n\n참고 근거: {sources}"


def _enforce_verification(answer: str, verification: dict, context: list) -> str:
    """LLM 답변에 ④ 검증 결과를 코드로 강제 반영한다.

    ④에는 L0 코드 오버라이드를 깔아뒀으면서 ⑤에는 프롬프트 부탁만 있어, 검증이
    정확히 지적한 사항이 최종 답변에서 그대로 무시되는 경로가 있었다 (실측 4/4 위반:
    grounded=False인데 수치 잔존, missing_requirements가 있는데 한계 고지 없음).
    "반드시 지켜야 하는 것은 코드로 강제한다"는 verification.py의 원칙을 ⑤까지 연장한다.

    순서에 의미가 있다 — 전제 교정은 답변 맨 앞에 와야 하므로 마지막에 적용한다.
    """
    if not answer:
        return answer

    # ① 내부 인덱스 "[근거 1]"을 실제 출처명으로 치환 (요강: 모든 답변에 근거 문서 표시)
    answer = replace_evidence_placeholders(answer, context)

    missing = list(verification.get("missing_requirements") or [])
    premise_issues = list(verification.get("premise_issues") or [])

    # ④가 premise_issues에 "답변의 결함"을 적어 넣는 경우를 성격으로 재분류한다.
    # 문자열이 두 필드에 똑같이 중복될 때만 걸러내면(아래 로직) 표현이 다른 오분류는
    # 그대로 통과해, "먼저 질문에 담긴 전제를 짚고 넘어가겠습니다: 초안이 날짜에 직접
    # 답하지 않음"처럼 사용자가 하지도 않은 말을 전제로 지적하는 답변이 나간다.
    premise_issues, misfiled = split_premise_issues(premise_issues)
    missing.extend(item for item in misfiled if item not in missing)

    # ④가 같은 항목을 두 필드에 동시에 넣는 경우가 있다 (실측 S1: "2027년 개편안 확정
    # 내용"이 premise_issues와 missing_requirements 양쪽에 등장). 이때 겹치는 항목은
    # **한계 고지 쪽으로 넘긴다** — 자료에 없어서 답하지 못한 것을 "사실과 다르거나
    # 과장된 전제"라고 표현하면 부정확하고, 정보한계 대응이라는 실제 성격도 가려진다.
    if not verification.get("clarification_mode"):
        premise_issues = [p for p in premise_issues if p not in missing]
    else:
        # 역질문 초안은 요구사항 검증이 면제된 상태다(apply_clarification_override).
        # 의도적으로 답을 유보하고 되물은 답변에 "확인이 어렵다"를 덧붙이면 중복이 된다.
        missing = []

    # ② 답하지 못한 요구 항목을 한계로 명시 (요강: 정보한계 대응)
    answer = enforce_missing_requirements(answer, missing)

    # ③ 잘못된 전제를 바로잡지 않았으면 앞머리에 교정문을 붙인다 (요강: 정확성)
    answer = enforce_premise_issues(answer, premise_issues)

    # ④ 근거를 썼는데 출처 줄이 없으면 코드가 붙인다 (LLM에게 맡기면 누락된다 — 실측 K1)
    return _append_reference_line(answer, context)


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
    #
    # 단 ③이 폴백을 썼다면 "툴 호출 없음"과 "근거 N건 사용"이 같은 trace에 함께 나와
    # 서로 모순된다. 폴백은 LLM이 툴을 안 불렀을 때 코드가 투자설명서 DB를 직접 조회해
    # 초안을 대체하는 경로라, 근거는 실재하지만 tool_trace에는 잡히지 않는다 —
    # 근거의 출처를 사실대로 밝혀야 심사자가 답변과 trace를 대조했을 때 어긋나지 않는다.
    for node, draft_key in (("info_agent", "info_draft"), ("product_agent", "product_draft")):
        if not state.get(draft_key) or node in seen_nodes:
            continue
        lines.append(f"[{_NODE_LABELS[node]}]")
        if node == "product_agent" and state.get("product_fallback_used"):
            lines.append(
                "  - LLM이 상품 검색 툴을 호출하지 않아, 코드가 투자설명서 DB를 직접 조회해"
                " 후보를 구성 (폴백 경로 — 아래 근거는 이 조회 결과)"
            )
        elif node == "product_agent" and state.get("retrieved_context"):
            lines.append(
                "  - 정형 상품 정책/DB 조회 경로 실행 — 사전 매핑된 근거와 함께 응답 생성"
                " (툴 호출 없음)"
            )
        elif node == "info_agent" and state.get("deterministic_info") and state.get("retrieved_context"):
            lines.append(
                "  - 정형 규칙 핸들러 실행 — 사전 매핑된 근거와 함께 응답 생성"
                " (툴 호출 없음)"
            )
        else:
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
        origin = " (③ 폴백 조회 결과)" if state.get("product_fallback_used") else ""
        lines.append(f"  - 근거 {len(context)}건 사용{origin}: {sources}")
    else:
        lines.append("  - 사용한 근거 없음 (근거가 필요한 수치는 답변에서 제외)")
    if state.get("product_fallback_used"):
        lines.append("  - ③ LLM 초안을 폐기하고 폴백이 만든 상품 후보 답변으로 대체")
    if state.get("needs_clarification"):
        lines.append("  - 조건 불충분 → 첫 답변에 정보한계와 필요한 역질문 전체를 포함")
    if state.get("response_mode"):
        lines.append(f"  - 응답 모드: {state['response_mode']}")
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

        # ④가 이미 "근거에 없다"고 확정한 수치 목록을 ⑤에 그대로 넘긴다.
        # 이 목록 없이 "숫자를 근거와 대조하라"고만 시키면 ⑤가 그 대조를 처음부터 다시
        # 해야 하는데, ④는 L0(기계적 토큰 대조) + L1(LLM 확인)을 거쳐 답을 이미 갖고 있다.
        # 구체적인 금지 목록을 주는 편이 "알아서 검사하라"보다 훨씬 지키기 쉽다.
        confirmed_numbers = verification.get("unsupported_numbers_confirmed") or []
        unsupported_line = (
            f"- 근거에 없는 것으로 확정된 수치(절대 사용 금지): {confirmed_numbers}\n"
            if confirmed_numbers
            else ""
        )

        prompt = (
            (f"[이전 대화]\n{history_text}\n\n" if history_text else "")
            + f"[질문]\n{state['question']}\n\n"
            f"[초안]\n{draft}\n\n"
            f"[근거]\n{context_text}\n\n"
            f"[검증결과]\n"
            f"- grounded: {verification.get('grounded')}, issues: {verification.get('issues')}\n"
            f"{unsupported_line}"
            f"- premise_issues(질문의 잘못된 전제): {verification.get('premise_issues')}\n"
            f"- requirements_met: {verification.get('requirements_met')}, "
            f"missing_requirements: {verification.get('missing_requirements')}"
        )
        # ⚠️ 아래 조기 반환 경로들도 _enforce_verification을 거쳐야 한다. 실측 S1(2027년
        # 세제 개편안)은 "세금혜택_개요" 정형 응답으로 분류돼 이 경로로 빠졌는데, 정형
        # 답변이라 근거는 확실해도 **질문이 요구한 것에 답하지 못했다는 사실**은 그대로였다
        # (④가 missing_requirements로 정확히 지적함). 근거 신뢰도와 "요구에 답했는가"는
        # 별개 축이라, LLM 경로에만 강제를 걸면 정형 경로에 구멍이 남는다.
        if state.get("needs_clarification"):
            return {
                "answer": _enforce_verification(draft, verification, context),
                "think_trace": _format_think_trace(state),
            }
        if state.get("recommendation_stage") == "type_recommendation":
            return {
                "answer": _enforce_verification(draft, verification, context),
                "think_trace": _format_think_trace(state),
            }
        if state.get("deterministic_info"):
            return {
                "answer": _enforce_verification(draft, verification, context),
                "think_trace": _format_think_trace(state),
            }

        response = invoke_with_retry(llm, [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        return {
            "answer": _enforce_verification(response.content, verification, context),
            "think_trace": _format_think_trace(state),
        }

    return generator_node
