"""search_pension_docs 관련성 임계값(DEFAULT_MAX_DISTANCE) 캘리브레이션 스크립트.

범위내/범위외 대표 질의의 L2 거리 분포(낮을수록 관련)를 찍어본다 — 범위내 질의의 최대
거리와 범위외 질의의 최소 거리 사이에 임계값을 두면 된다. 실제 CLOVASTUDIO_API_KEY 필요.

⚠️ LangChain의 relevance score(0~1 정규화)는 이 컬렉션(비정규화 임베딩)에서 음수가 나와
쓸 수 없다 (실측 2026-08-16) — raw 거리(similarity_search_with_score) 기준으로 본다.
데이터/임베딩을 재적재하면 분포가 바뀌므로 반드시 다시 돌려볼 것.

사용법: python scripts/calibrate_relevance.py            # 기본 질의 세트
        python scripts/calibrate_relevance.py "질문1" "질문2"
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

DEFAULT_QUERIES = [
    # 범위내 — 임계값 이내여야 정상
    "연금저축 세액공제 한도가 얼마인가요",
    "IRP 중도인출 요건이 뭔가요",
    "디폴트옵션 자동매수는 언제 되나요",
    "실물이전이 안 되는 상품은?",
    # 범위외 — 임계값이 걸러야 정상
    "미국 401k를 한국으로 이전할 수 있나요",
    "부동산 양도소득세 계산 방법",
    "오늘 점심 메뉴 추천해줘",
    "삼성전자 주가 전망 알려줘",
]


def main():
    from langchain_chroma import Chroma

    from src.agents.llm import get_embeddings
    from src.storage.queries import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION, DEFAULT_MAX_DISTANCE

    queries = sys.argv[1:] or DEFAULT_QUERIES
    vectorstore = Chroma(
        collection_name=DEFAULT_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DEFAULT_CHROMA_DIR,
    )

    print(f"현재 임계값 DEFAULT_MAX_DISTANCE = {DEFAULT_MAX_DISTANCE} (거리가 이보다 크면 제외)")
    for query in queries:
        hits = vectorstore.similarity_search_with_score(query, k=5)
        print(f"\n질의: {query}")
        for doc, distance in hits:
            mark = "  " if distance <= DEFAULT_MAX_DISTANCE else "✗ "
            title = doc.metadata.get("file_title", "")
            print(f"  {mark}{distance:7.2f}  [{title}] {doc.page_content[:50]!r}")


if __name__ == "__main__":
    main()
