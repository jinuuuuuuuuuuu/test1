"""AUM 추출(요약 재무상태표 → 자본총계 최신 기) 텍스트 파싱 검증 — PDF/API 불필요."""

from src.parsing.aum_extractor import extract_aum_from_text

# 실제 KR510902511M 투자설명서 p40에서 가져온 형태
SAMPLE = """주1)이 집합투자기구는 회계감사인의 회계감사에 대한 적용이 면제되었습니다.
가. 요약재무정보 (단위: 백만원)
요약 재무상태표
항목 14기(24.12.22) 13기(23.12.22) 12기(22.12.22)
   운용자산 5,695 9,213 7,363
자산총계 5,711 9,235 7,381
부채총계 14 49 42
자본총계 5,697 9,185 7,339
요약 손익계산서
항목 14기(23.12.23 ~ 24.12.22)
운용수익 -1,942 2,031 -2,328
"""


def test_extracts_latest_period_net_assets():
    result = extract_aum_from_text(SAMPLE)
    assert result is not None
    assert result.aum_krw_million == 5697.0  # 최신 기(14기) 자본총계, 과거 기 값 아님
    assert result.base_date == "2024-12-22"
    assert result.period_label == "14기"
    assert "자본총계" in result.source_snippet


def test_returns_none_without_summary_section():
    assert extract_aum_from_text("연금저축 세액공제 한도는 900만원입니다.") is None


def test_returns_none_when_value_is_dash():
    text = SAMPLE.replace("자본총계 5,697 9,185 7,339", "자본총계 - - -")
    assert extract_aum_from_text(text) is None


def test_handles_soonjasan_variant():
    text = SAMPLE.replace("자본총계", "순자산총액")
    result = extract_aum_from_text(text)
    assert result is not None
    assert result.aum_krw_million == 5697.0


# 실제 KR5111420047 투자설명서 p42의 변형 양식: "제 N 기" 라벨 + 전체 날짜 + 원 단위
SAMPLE_WON_UNIT = """가.요약재무정보                              (단위 : 원)
요약재무정보
항       목
제 18 기 제 17 기 제 16 기
2025.04.16 2024.04.16 2023.04.16
운용자산 486,790,129,186  344,619,444,489  262,872,968,324
자산총계 486,790,504,657  344,619,785,508  262,874,679,383
부채총계 19,999,446,986  12,331,867,142  7,230,330,055
자본총계 466,791,057,671  332,287,918,366  255,644,349,328
운용수익 17,620,769,920  15,699,716,839  9,803,948,817
"""


def test_won_unit_variant_is_converted_to_million():
    result = extract_aum_from_text(SAMPLE_WON_UNIT)
    assert result is not None
    assert result.aum_krw_million == 466791.1  # 466,791,057,671원 → 백만원 환산
    assert result.base_date == "2025-04-16"
    assert result.period_label == "18기"


# 실제 KR5144420020: pypdf가 표를 역순 추출 — 값 3개가 라벨 앞, 날짜는 괄호 표기
SAMPLE_REVERSED = """요약재무정보
NH-Amundi 국채10년 인덱스 증권자투자신탁[채권] [단위 : 원]
424,167,622,451 408,509,880,501 323,559,868,824자본총계
424,168,172,451 408,510,430,501 323,560,418,824자산총계
( 2025.08.31 ) ( 2024.08.31 ) ( 2023.08.31 )
항       목
운용자산
재무상태표
제 9기제 10기제 11기
"""


def test_reversed_layout_picks_latest_by_date_order():
    result = extract_aum_from_text(SAMPLE_REVERSED)
    assert result is not None
    assert result.aum_krw_million == 424167.6  # 첫 값(2025.08.31 열) = 최신
    assert result.base_date == "2025-08-31"
    assert result.period_label == "11기"  # 연속 라벨 블록의 최대값


def test_reversed_layout_rejects_when_capital_exceeds_assets():
    # 자본총계 > 자산총계면 열 매핑이 틀렸다는 신호 — 값을 쓰지 않고 수기 확인에 맡긴다.
    broken = SAMPLE_REVERSED.replace("424,167,622,451", "924,167,622,451")
    assert extract_aum_from_text(broken) is None


def test_reversed_layout_rejects_non_monotonic_dates():
    broken = SAMPLE_REVERSED.replace("( 2024.08.31 )", "( 2026.08.31 )")
    assert extract_aum_from_text(broken) is None


def test_footnote_mention_without_table_header_is_rejected():
    # 실측(KR5111420047 p43): 각주 "주1) 요약재무정보 사항 중 …"가 마커로 오인되면
    # 뒤따르는 무관한 표의 숫자를 자본총계로 집을 수 있다 — "항목" 머리글이 없으면 거부.
    footnote = """주 1)  요약재무정보 사항 중 매매회전율이란 주식매매의 빈번한 정도를 나타내는 지표입니다.
<주식의 매매회전율> (단위:주,원,%)
자본 총계 486,790,504,657  제 18 기 2025.04.16
"""
    assert extract_aum_from_text(footnote) is None


def test_company_balance_sheet_without_period_label_is_rejected():
    # 운용사/신탁업자 '회사' 재무상태표에는 "N기" 회계기수 라벨이 없다 — 펀드 AUM으로
    # 오인하면 안 된다 (실측: 첫 샘플 p50 신탁업자 표, 자본총계 76,506 백만원).
    company_table = """다. 최근 2개 사업연도 요약 재무내용(백만원)
요약 재무상태표 요약 손익계산서
항   목 2023.12.31 2022.12.31
현금 및 현금성자산 2,640 22,074
자본총계 76,506 71,147
"""
    assert extract_aum_from_text(company_table) is None
