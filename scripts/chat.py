"""터미널에서 연금 Agent와 직접 대화해보는 스크립트.

사용법: ./.venv/Scripts/python.exe scripts/chat.py
종료: 빈 줄 입력 또는 Ctrl+C
"""

import os
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.agents.graph import build_graph  # noqa: E402


def main():
    print("연금 Agent 그래프를 준비하는 중...")
    app = build_graph()
    print("준비 완료. 질문을 입력하세요 (빈 줄 입력 시 종료)\n")

    while True:
        try:
            question = input("질문> ").strip()
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
            })
        except Exception as e:
            print(f"\n[오류] {e}\n")
            continue

        print("\n" + "=" * 60)
        print("[답변]")
        print(result.get("answer") or "(답변 없음)")
        print("\n[think_trace]")
        print(result.get("think_trace") or "(없음)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
