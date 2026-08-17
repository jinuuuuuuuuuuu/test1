"""사용자 입력 텍스트 정규화 유틸리티."""

import re

_KOREAN_SYLLABLE_SPACES = re.compile(r"(?<![가-힣])([가-힣])\s{2,}([가-힣])(?![가-힣])")
_MULTI_SPACES = re.compile(r"\s+")


def normalize_user_text(text: str) -> str:
    """오타성 공백을 줄여 Agent가 같은 의도로 이해하기 쉽게 만든다."""
    text = (text or "").strip()
    text = _KOREAN_SYLLABLE_SPACES.sub(r"\1\2", text)
    return _MULTI_SPACES.sub(" ", text).strip()
