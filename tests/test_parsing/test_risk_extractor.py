"""투자위험 요약 표 추출 텍스트 파싱 검증 — PDF/API 불필요."""

from src.parsing.risk_extractor import extract_risk_text

PAGE = """구분 투자위험의 주요내용
주요투자
위험
원본손실위험
이 집합투자기구는 원본을 보장하지 않습니다.
주식가격 변동위험
주식의 가치는 급변할 수 있습니다.
매입방법
1. 15시 30분 이전 : 자금을 납입하는 영업일(D)로부터...
"""


def test_extracts_risk_table_until_terminator():
    text = extract_risk_text(PAGE)
    assert text is not None
    assert "원본손실위험" in text
    assert "주식가격 변동위험" in text
    assert "매입방법" not in text  # 다음 섹션에서 잘려야 한다
    assert "15시 30분" not in text


def test_marker_spacing_variant_is_matched():
    # 실측: KR5111420047은 "투자위험의 주요 내용"(공백 포함) 표기를 쓴다.
    text = extract_risk_text(PAGE.replace("투자위험의 주요내용", "투자위험의 주요 내용"))
    assert text is not None
    assert "원본손실위험" in text


def test_table_spanning_page_boundary_uses_next_page():
    page1 = "구분 투자위험의 주요내용\n원본손실위험\n이 집합투자기구는 원본을"
    page2 = " 보장하지 않습니다.\n환매방법\n환매 절차는..."
    text = extract_risk_text(page1, page2)
    assert text is not None
    assert "보장하지 않습니다" in text
    assert "환매 절차" not in text


def test_returns_none_without_marker():
    assert extract_risk_text("투자전략에 대한 설명입니다.") is None
