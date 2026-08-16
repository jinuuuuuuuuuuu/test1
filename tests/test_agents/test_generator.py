"""⑤생성기의 결정론적 분기(LLM 호출 없이 처리되는 경로) 검증 — 더미 API 키로 실행 가능."""

import os

os.environ.setdefault("CLOVASTUDIO_API_KEY", "dummy-key-for-wiring-test-only")

from src.agents.generator import build_generator_node


def test_unsafe_returns_canned_refusal_without_llm():
    node = build_generator_node()
    result = node({"question": "탈세 방법 알려줘", "is_safe": False, "safety_reason": "불법 행위 문의"})
    assert "답변드릴 수 없습니다" in result["answer"]
    assert "불법 행위 문의" in result["think_trace"]


def test_out_of_scope_returns_canned_limitation_without_llm():
    # 범위외 질문은 LLM 호출 없이(더미 키에서 네트워크 호출이 나면 이 테스트가 에러로 실패)
    # 정형 한계 고지로 답한다 — "정보한계 대응" 평가지표 경로.
    node = build_generator_node()
    result = node({
        "question": "삼성전자 주식 지금 사도 돼?",
        "is_safe": True,
        "scope": "범위외",
        "scope_note": "개별 주식 투자 판단은 연금 상담과 무관",
    })
    assert "상담 범위" in result["answer"]
    assert "연금" in result["answer"]
    assert "범위외" in result["think_trace"]
    assert "개별 주식 투자 판단은 연금 상담과 무관" in result["think_trace"]
