"""투자설명서 PDF에서 "투자위험" 요약 표 텍스트를 추출한다.

투자설명서 앞부분(간이 요약부)에 "구분 투자위험의 주요내용" 표가 있고, 여기에 그 펀드
고유의 핵심 위험(원본손실위험/주식가격 변동위험/금리 변동위험 등)이 서술돼 있다 —
"이 펀드 위험이 뭐예요" 같은 단일 상품 설명 질의의 근거로 쓰기에 가장 적합한 구간이다.
(문서 중반의 상세 위험 절은 법정 보일러플레이트 비중이 높아 요약 표만 쓴다.)
"""

import re
from typing import Optional

# 표 헤더 표기가 문서마다 다르다: "투자위험의 주요내용" / "투자위험의 주요 내용" 등
_RISK_TABLE_MARKER_RE = re.compile(r"투자위험의\s*주요\s*내용")
# 요약 위험표 다음에 이어지는 섹션들 — 첫 등장 지점에서 자른다.
_TERMINATORS = ("매입방법", "매입 방법", "환매방법", "환매 방법", "매입·환매", "기준가격")
_MAX_RISK_CHARS = 4_000


def extract_risk_text(page_text: str, next_page_text: str = "") -> Optional[str]:
    """페이지 텍스트에서 위험 요약 표 구간을 추출한다. 표가 페이지 경계에 걸치면 다음
    페이지를 이어 붙여 파싱한다. 마커가 없으면 None."""
    marker = _RISK_TABLE_MARKER_RE.search(page_text)
    if marker is None:
        return None
    window = page_text[marker.end():] + "\n" + next_page_text
    cut = len(window)
    for term in _TERMINATORS:
        t = window.find(term)
        if 0 <= t < cut:
            cut = t
    text = window[:cut][:_MAX_RISK_CHARS].strip()
    # 표 헤더 잔여물 제거
    text = re.sub(r"^(주요투자\s*위험|세부구분)\s*", "", text)
    return text or None


def extract_risk_summary_from_pdf(pdf_path: str) -> Optional[str]:
    """PDF 앞쪽에서 위험 요약 표를 찾아 추출한다. 실패 시 None."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    # 요약부는 문서 앞 ~15페이지 안에 있다 (뒤쪽 상세 절과 혼동하지 않도록 앞에서부터 탐색).
    limit = min(15, len(reader.pages))
    pages = [reader.pages[i].extract_text() or "" for i in range(limit)]
    for i in range(limit):
        result = extract_risk_text(pages[i], pages[i + 1] if i + 1 < limit else "")
        if result:
            return result
    return None
