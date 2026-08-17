"""⑤ 생성기 노드 — 초안·근거·검증결과를 받아 최종 답변을 조립하고, 대회 평가 API 스키마의
think_trace를 포맷팅한다. HCX-005 사용.

is_safe=False(①가드레일에서 차단된 경우)는 모델을 호출하지 않고 바로 정형 거절 응답을 만든다.
"""

from src.agents.context import format_conversation_history
from src.agents.llm import get_llm
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
  다시 쓰지 마세요."""


def _format_think_trace(state: PensionAgentState) -> str:
    context = state.get("retrieved_context") or []
    verification = state.get("verification") or {}

    context_lines = []
    seen = set()

    for c in context:
        document_id = c.get("document_id", "")
        section = c.get("section", "")
        source_location = c.get("source_location", "")
        source = c.get("source", "unknown_tool")

        # ① search_pension_docs에서 가져온 문서 근거
        if document_id:
            key = (document_id, section, source_location)

            # 같은 문서/단락 중복 제거
            if key in seen:
                continue

            seen.add(key)

            line = f"  - 문서: {document_id}"

            if section:
                line += f"\n    단락: {section}"

            if source_location:
                line += f"\n    위치: {source_location}"

            context_lines.append(line)

        # ② 계산기/판정 툴 등 문서검색이 아닌 근거
        else:
            key = ("tool", source)

            if key in seen:
                continue

            seen.add(key)

            context_lines.append(
                f"  - 툴: {source}"
            )

    context_text = "\n".join(context_lines) or "  (없음)"

    return (
        f"[①분류] intent={state.get('intent')}, is_safe={state.get('is_safe')}\n"
        f"[②③근거 {len(seen)}건]\n"
        f"{context_text}\n"
        f"[④검증] grounded={verification.get('grounded')}, "
        f"issues={verification.get('issues')}, "
        f"premise_issues={verification.get('premise_issues')}, "
        f"requirements_met={verification.get('requirements_met')}, "
        f"missing_requirements={verification.get('missing_requirements')}"
    )

def build_generator_node():
    llm = get_llm(GENERATOR_MODEL)

    def generator_node(state: PensionAgentState) -> dict:
        # 1. 가드레일 차단
        if state.get("is_safe") is False:
            reason = (
                state.get("safety_reason")
                or "요청하신 내용은 안내해드리기 어렵습니다."
            )

            return {
                "answer": (
                    f"죄송하지만 해당 질문에는 답변드릴 수 없습니다. "
                    f"({reason})"
                ),
                "think_trace": f"[①가드레일 차단] {reason}",
            }

        # 2. Agent 초안 / 검증결과 / 근거 가져오기
        draft = (
            state.get("info_draft")
            or state.get("product_draft")
            or ""
        )

        verification = state.get("verification") or {}
        context = state.get("retrieved_context") or []

        # 역질문이면 바로 출력
        if state.get("needs_clarification"):
            return {
                "answer": draft,
                "think_trace": _format_think_trace(state),
            }
        
        # 3. 문서 근거가 실제로 존재하는지 확인
        has_document_evidence = any(
            c.get("document_id")
            for c in context
        )

        # 문서 근거가 없으면 최종 답변 생성 금지
        if not has_document_evidence:
            return {
                "answer": (
                    "현재 질문에 대해 확인 가능한 문서 근거를 확보하지 못해 "
                    "정확한 답변을 생성하지 않았습니다."
                ),
                "think_trace": _format_think_trace(state),
            }

        # 4. Grounding 검증 실패 시 최종 답변 생성 금지
        if verification.get("grounded") is False:
            return {
                "answer": (
                    "관련 문서는 확인했지만, 현재 확보된 근거만으로는 "
                    "답변의 모든 내용을 충분히 검증할 수 없습니다. "
                    "추가 근거 확인이 필요합니다."
                ),
                "think_trace": _format_think_trace(state),
            }

        # 5. Grounding에서 요구사항 미충족 판정 시 차단
        if verification.get("requirements_met") is False:
            return {
                "answer": (
                    "관련 근거는 확인했지만, 현재 확보된 정보만으로는 "
                    "질문의 요구사항을 충분히 충족하는 답변을 만들기 어렵습니다."
                ),
                "think_trace": _format_think_trace(state),
            }

        # 6. 최종 Generator에 전달할 근거 텍스트
        context_text = "\n".join(
            f"- [{c['source']}] {c['content']}"
            for c in context
        ) or "(근거 없음)"

        history_text = format_conversation_history(
            state.get("conversation_history")
        )

        # 7. 최종 답변 생성 프롬프트
        prompt = (
            (f"[이전 대화]\n{history_text}\n\n" if history_text else "")
            + f"[질문]\n{state['question']}\n\n"
            f"[초안]\n{draft}\n\n"
            f"[근거]\n{context_text}\n\n"
            f"[검증결과]\n"
            f"- grounded: {verification.get('grounded')}, "
            f"issues: {verification.get('issues')}\n"
            f"- premise_issues(질문의 잘못된 전제): "
            f"{verification.get('premise_issues')}\n"
            f"- requirements_met: {verification.get('requirements_met')}, "
            f"missing_requirements: "
            f"{verification.get('missing_requirements')}"
        )

        response = llm.invoke([
            {
                "role": "system",
                "content": GENERATOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ])

        # 8. 최종 응답
        return {
            "answer": response.content,
            "think_trace": _format_think_trace(state),
        }

    return generator_node
