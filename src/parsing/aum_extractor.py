"""투자설명서 PDF에서 AUM(시장잔고)을 추출한다 — 대회 6축 중 마지막 축.

추출 위치: 제3부 "요약재무정보 > 요약 재무상태표"(단위: 백만원)의 자본총계(=순자산총액)
행, 최신 회계기수 열. 예 (KR510902511M, 실측):

    가. 요약재무정보 (단위: 백만원)
    요약 재무상태표
    항목 14기(24.12.22) 13기(23.12.22) 12기(22.12.22)
    ...
    자본총계 5,697 9,185 7,339      ← 최신 기(24.12.22)의 5,697백만원이 AUM

투자설명서는 실시간 잔고를 싣지 않으므로 이 값이 문서 기준의 공식 AUM이며, 기준일
(aum_base_date)을 함께 저장해 답변에서 "몇 기 결산 기준"임을 밝힐 수 있게 한다.
"""

import re
from dataclasses import dataclass
from typing import Optional

# 양식 A: "14기(24.12.22)" — 괄호 안이 그 회계기수의 결산일 (단위: 백만원이 일반적)
_PERIOD_RE = re.compile(r"(\d+)\s*기\s*\(\s*(\d{2})[.\s]*(\d{2})[.\s]*(\d{2})\s*\)")
# 양식 B (실측 KR5111420047): "제 18 기" 라벨과 "2025.04.16" 날짜가 별도 줄, 단위: 원
_PERIOD_LABEL_RE = re.compile(r"제?\s*(\d+)\s*기")
_FULL_DATE_RE = re.compile(r"(20\d{2})\.(\d{2})\.(\d{2})")
# "자본총계 5,697 9,185 7,339" (순자산총액 표기 변형 포함) — 첫 숫자가 최신 기 값
_NET_ASSET_ROW_RE = re.compile(r"(자본\s*총계|순자산\s*총액)\s+([\d,]+(?:\.\d+)?|-)")
# 양식 C (실측 KR5144420020): pypdf가 표를 역순으로 추출해 "424,167,622,451
# 408,509,880,501 323,559,868,824자본총계"처럼 값 3개가 라벨 앞에 온다. 값 순서는
# 괄호 날짜 "( 2025.08.31 ) ( 2024.08.31 ) ( 2023.08.31 )"의 방향으로 판정한다.
# 7자리(천만원, 원 단위) 미만 값은 매치하지 않는다 — 짧은 숫자 나열 오탐 방지.
_REVERSED_ROW_RE = re.compile(
    r"(\d[\d,]{6,})\s+(\d[\d,]{6,})\s+(\d[\d,]{6,})\s*(자본\s*총계|순자산\s*총액)"
)
_REVERSED_ASSET_RE = re.compile(r"(\d[\d,]{6,})\s+(\d[\d,]{6,})\s+(\d[\d,]{6,})\s*자산\s*총계")
_PAREN_DATE_RE = re.compile(r"\(\s*(20\d{2})\.(\d{2})\.(\d{2})\s*\)")
# "(단위 : 원)" — 백만원 표기("단위: 백만원")는 콜론 뒤가 '백'이라 매치되지 않는다
_UNIT_WON_RE = re.compile(r"단위\s*[:：]\s*원")

_SECTION_KEYWORDS = ("요약 재무상태표", "요약재무상태표", "요약재무정보")


@dataclass
class AumExtraction:
    aum_krw_million: float   # 최신 기 자본총계 (단위: 백만원)
    base_date: str           # 최신 기 결산일 "20YY-MM-DD"
    period_label: str        # 예: "14기"
    source_snippet: str      # 검증용 원문 발췌 (헤더 행 + 자본총계 행)


def find_summary_section(text: str) -> Optional[int]:
    """페이지 텍스트에서 요약 재무상태표 섹션 시작 위치를 찾는다. 없으면 None."""
    for keyword in _SECTION_KEYWORDS:
        idx = text.find(keyword)
        if idx >= 0:
            return idx
    return None


def extract_aum_from_text(text: str) -> Optional[AumExtraction]:
    """요약 재무상태표가 포함된 텍스트에서 최신 기 자본총계를 추출한다. 실패 시 None."""
    start = find_summary_section(text)
    if start is None:
        return None
    # 손익계산서가 나오기 전까지가 재무상태표 구간이다 (없으면 섹션 이후 1500자).
    window = text[start:]
    cut = window.find("손익계산서")
    window = window[: cut if cut > 0 else 1500]

    return _extract_standard(window) or _extract_reversed(window)


def _extract_standard(window: str) -> Optional[AumExtraction]:
    """양식 A/B: 헤더(항목 + 기수/날짜) 뒤에 '자본총계 값…' 행이 오는 일반 배치."""
    # 실제 요약표에는 "항목"(또는 "항   목") 열 머리글이 있다 — 각주("주1) 요약재무정보
    # 사항 중 …")가 마커로 오인되어 뒤따르는 무관한 표의 숫자를 집는 것을 막는다
    # (실측: KR5111420047 p43 각주 → p44 다른 표의 값을 자본총계로 오인).
    header = re.search(r"항\s*목", window)
    if not header:
        return None
    body = window[header.end():]

    row = _NET_ASSET_ROW_RE.search(body)
    if not row:
        return None
    value_text = row.group(2)
    if value_text == "-":
        return None

    # 양식 A: "14기(24.12.22)" / 양식 B: "제 18 기" + "2025.04.16" 별도 매치.
    # 회계기수·날짜는 "항목" 머리글 뒤(body)에서만 찾는다.
    period_a = _PERIOD_RE.search(body)
    if period_a:
        label = f"{period_a.group(1)}기"
        base_date = f"20{period_a.group(2)}-{period_a.group(3)}-{period_a.group(4)}"
        header_line = next(
            (line.strip() for line in body.splitlines() if _PERIOD_RE.search(line)), ""
        )
    else:
        label_m = _PERIOD_LABEL_RE.search(body)
        date_m = _FULL_DATE_RE.search(body)
        if not label_m or not date_m:
            return None
        label = f"{label_m.group(1)}기"
        base_date = f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}"
        header_line = f"{label_m.group(0)} {date_m.group(0)}"

    value = float(value_text.replace(",", ""))
    # 단위 판정: "(단위 : 원)"이면 백만원으로 환산. 표기가 없어도 10조(백만원 단위로
    # 10,000,000)를 넘는 값은 현실적으로 원 단위 표가 확실하므로 방어적으로 환산한다.
    if _UNIT_WON_RE.search(window) or value > 10_000_000:
        value = value / 1_000_000

    return AumExtraction(
        aum_krw_million=round(value, 1),
        base_date=base_date,
        period_label=label,
        source_snippet=f"{header_line} | {row.group(0)}",
    )


def _extract_reversed(window: str) -> Optional[AumExtraction]:
    """양식 C: pypdf가 표를 역순 추출해 값 3개가 자본총계 라벨보다 먼저 나오는 배치.

    괄호 날짜 3개의 방향(내림차순/오름차순)으로 어느 값이 최신 기인지 판정하고,
    같은 방식으로 뽑은 자산총계와 대소 검증(자본총계 ≤ 자산총계)까지 통과해야 값을
    반환한다 — 열 순서를 확신할 수 없으면 None(수기 확인)으로 남긴다.
    """
    row = _REVERSED_ROW_RE.search(window)
    if not row:
        return None
    dates = _PAREN_DATE_RE.findall(window)
    if len(dates) < 3:
        return None
    date_strs = ["-".join(d) for d in dates[:3]]
    if date_strs[0] > date_strs[1] > date_strs[2]:
        latest = 0
    elif date_strs[0] < date_strs[1] < date_strs[2]:
        latest = 2
    else:
        return None

    values = [float(row.group(i).replace(",", "")) for i in (1, 2, 3)]
    value = values[latest]

    asset_row = _REVERSED_ASSET_RE.search(window)
    if asset_row:
        assets = float(asset_row.group(latest + 1).replace(",", ""))
        if value > assets * 1.001:
            return None

    # 기수 라벨은 "제 9기제 10기제 11기"처럼 연속 블록으로 붙어 나온다 — 창 안의 무관한
    # "N기" 표현(각주 등)에 오염되지 않도록 연속 블록 안에서만 뽑아 최대값을 취한다.
    run = re.search(r"(?:제\s*\d+\s*기\s*){2,}", window)
    labels = [int(n) for n in _PERIOD_LABEL_RE.findall(run.group(0))] if run else []
    label = f"{max(labels)}기" if labels else ""

    if _UNIT_WON_RE.search(window) or value > 10_000_000:
        value = value / 1_000_000

    return AumExtraction(
        aum_krw_million=round(value, 1),
        base_date=date_strs[latest],
        period_label=label,
        source_snippet=f"(역순 표) {row.group(0)[:120]} | 날짜열 {date_strs}",
    )


def extract_aum_from_pdf(pdf_path: str) -> Optional[AumExtraction]:
    """PDF에서 요약 재무상태표 페이지를 찾아 AUM을 추출한다. 실패 시 None.

    반드시 **앞에서부터** 훑는다 — 펀드 자체의 요약재무정보(제3부 첫머리)는 그 뒤에 오는
    각주("주1) 요약재무정보 사항 중 …"), 정식 재무상태표(열이 뒤섞여 추출됨), 운용사·
    신탁업자 '회사' 재무 표보다 항상 앞에 있어서, 첫 유효 매치가 곧 정답이다 (역방향
    탐색은 각주/후속 표를 먼저 만나 오탐한 실측 사례 있음: KR5111420047).
    표가 페이지 경계에 걸릴 수 있어 해당 페이지 + 다음 페이지를 이어 붙여 파싱한다.
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    # 페이지 텍스트 추출이 페이지당 수백 ms로 비싸다 — 전체를 미리 뽑지 말고 지연 추출한다.
    cache: dict[int, str] = {}

    def page_text(i: int) -> str:
        if i not in cache:
            cache[i] = reader.pages[i].extract_text() or ""
        return cache[i]

    for i in range(total):
        if find_summary_section(page_text(i)) is None:
            continue
        combined = page_text(i) + "\n" + (page_text(i + 1) if i + 1 < total else "")
        result = extract_aum_from_text(combined)
        if result is not None:
            return result
    return None
