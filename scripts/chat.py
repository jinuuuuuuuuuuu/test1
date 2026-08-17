"""터미널에서 연금 Agent와 직접 대화해보는 스크립트.

사용법: python scripts/chat.py (venv 활성화 상태에서 실행)
종료: 빈 줄 입력 또는 Ctrl+C

멀티턴 문맥: 이 스크립트가 대화 턴(question/answer)을 리스트로 들고 있다가 다음 질문 때
PensionAgentState["conversation_history"]로 함께 넘긴다. 그래프 자체(app)는 매번 새로
invoke()되는 싱글턴 호출이고 LangGraph checkpointer 같은 세션 저장소는 쓰지 않는다 — 대화
연속성은 순전히 이 리스트를 호출자가 계속 이어서 넘기는 방식으로만 유지된다.
"""

import os
import sys
import uuid

# 턴이 계속 쌓이면 프롬프트 토큰이 무한히 늘어나므로 최근 N턴만 유지한다.
MAX_HISTORY_TURNS = 6

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.agents.graph import build_graph  # noqa: E402
from src.agents.text import normalize_user_text  # noqa: E402


def main():
    print("연금 Agent 그래프를 준비하는 중...")
    app = build_graph()
    print("준비 완료. 질문을 입력하세요 (빈 줄 입력 시 종료)\n")

    conversation_history: list[dict] = []
    recommendation_profile: dict = {}

    while True:
        try:
            question = normalize_user_text(input("질문> "))
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            print("종료합니다.")
            break

        try:
            result = app.invoke({
                "question_id": str(uuid.uuid4())[:8],
                "question": question,
                "conversation_history": conversation_history,
                "recommendation_profile": recommendation_profile,
            })
        except Exception as e:
            print(f"\n[오류] {e}\n")
            continue

        answer = result.get("answer") or "(답변 없음)"

        print("\n" + "=" * 60)
        print("[답변]")
        print(answer)
        print("\n[think_trace]")
        print(result.get("think_trace") or "(없음)")
        print("=" * 60 + "\n")

        conversation_history.append({"question": question, "answer": answer})
        conversation_history[:] = conversation_history[-MAX_HISTORY_TURNS:]
        recommendation_profile = result.get("recommendation_profile") or recommendation_profile


if __name__ == "__main__":
    main()
