from src.agents.text import normalize_user_text


def test_normalize_user_text_removes_accidental_korean_double_spaces():
    assert normalize_user_text("좋  은 연금상품 하나 추천해주세요") == "좋은 연금상품 하나 추천해주세요"


def test_normalize_user_text_collapses_regular_spaces():
    assert normalize_user_text("좋은  연금 상품") == "좋은 연금 상품"
