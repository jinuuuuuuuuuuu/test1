"""투자설명서 PDF에서 "제3부 3.가. 연평균수익률" 표를 클래스 단위로 추출한다.

운용사마다 표 양식이 조금씩 다르다(확인된 변형):
  - 헤더 첫 열이 "기간"/"연도"/"종류" 등으로 다르고, 앞에 빈 여백 열이 끼기도 함
  - 클래스 행이 "종류A(수수료선취-오프라인)", "종류A\\n수수료선취-오프라인", "펀드ClassA" 등 표기가 다름
  - "최초설정일" 같은 열이 값 앞에 끼어들어 값의 열 위치가 고정적이지 않음
  - 클래스 행 바로 뒤에 "비교지수"/"수익률변동성(%)" 행이 따라붙기도 하고 없기도 함
그래서 값을 "라벨 다음 N번째 칸"으로 가정하지 않고, 헤더 행에서 "최근1년/최근3년/설정" 열이
실제로 몇 번째 칸인지 찾아 그 열 위치를 데이터 행에도 그대로 적용한다.
"나. 연도별 수익률" 표(헤더에 "년차"가 들어감, 컬럼 의미가 다름)와 섞이지 않도록 구분한다.
"""

import re
from dataclasses import dataclass
from typing import Optional

import fitz

CLASS_LABEL_RE = re.compile(r"^(?:종류|펀드\s*Class)\s*([A-Za-z0-9\-]+)")


@dataclass
class ClassReturns:
    class_code: str
    class_label: str
    return_1y: Optional[float] = None
    return_3y: Optional[float] = None
    return_since_inception: Optional[float] = None
    volatility_1y: Optional[float] = None
    volatility_3y: Optional[float] = None
    volatility_since_inception: Optional[float] = None


@dataclass
class _ColMap:
    idx_1y: int
    idx_3y: int
    idx_since: int
    ncols: int


def _cell_text(cell) -> str:
    return (cell or "").strip()


def _first_line(cell) -> str:
    return _cell_text(cell).split("\n")[0].strip()


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _to_float(text: str) -> Optional[float]:
    text = (text or "").replace("\n", "").strip()
    if text in ("", "-", "None", "NULL"):
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


_HEADER_FIRST_CELL = {"기간", "연도", "구분", "종류"}


def _detect_col_map(row) -> Optional[_ColMap]:
    if not any(_normalize(_cell_text(c)) in _HEADER_FIRST_CELL for c in row[:2]):
        return None
    normed = [_normalize(_cell_text(c)) for c in row]
    joined = "".join(normed)
    if "년차" in joined:
        return None
    idx_1y = next((i for i, c in enumerate(normed) if "최근1년" in c and "년차" not in c), None)
    idx_3y = next((i for i, c in enumerate(normed) if "최근3년" in c), None)
    # "최초설정일" 같은 열도 "설정"을 포함하므로, "최초"가 없는 "설정" 열(=설정일이후)만 고른다.
    idx_since = next((i for i, c in enumerate(normed) if "설정" in c and "최초" not in c), None)
    if idx_1y is None or idx_3y is None or idx_since is None:
        return None
    return _ColMap(idx_1y, idx_3y, idx_since, ncols=len(row))


def _row_is_annual_header(row) -> bool:
    joined = _normalize("".join(_cell_text(c) for c in row))
    return "최근1년차" in joined


def _find_label_index(row) -> Optional[int]:
    for idx, cell in enumerate(row):
        if CLASS_LABEL_RE.match(_first_line(cell)):
            return idx
    return None


def _row_first_nonempty(row) -> str:
    for cell in row:
        text = _first_line(cell)
        if text:
            return text
    return ""


def _value_at(row, idx: Optional[int], offset: int = 0) -> Optional[float]:
    if idx is None:
        return None
    idx = idx + offset
    if idx < 0 or idx >= len(row):
        return None
    return _to_float(row[idx])


def extract_class_returns(pdf_path: str) -> list[ClassReturns]:
    """'가. 연평균수익률' 표에서 클래스별 [최근1년, 최근3년, 설정일이후] 수익률·변동성을 추출."""
    doc = fitz.open(pdf_path)
    try:
        rows: list[tuple[list[str], _ColMap]] = []
        col_map: Optional[_ColMap] = None
        done = False

        for pno in range(doc.page_count):
            if done:
                break
            page = doc[pno]
            tabs = page.find_tables()
            for t in tabs.tables:
                if done:
                    break
                grid = t.extract()
                if not grid:
                    continue
                for row in grid:
                    if _row_is_annual_header(row):
                        done = True
                        break
                    cm = _detect_col_map(row)
                    if cm is not None:
                        col_map = cm
                        continue
                    if col_map is not None:
                        rows.append(([_cell_text(c) for c in row], col_map))
    finally:
        doc.close()

    return _parse_rows(rows)


def _parse_rows(rows: list[tuple[list[str], "_ColMap"]]) -> list[ClassReturns]:
    results: list[ClassReturns] = []
    i = 0
    n = len(rows)
    while i < n:
        row, cm = rows[i]
        idx = _find_label_index(row)
        if idx is None:
            i += 1
            continue
        code_m = CLASS_LABEL_RE.match(_first_line(row[idx]))
        code = code_m.group(1)

        offset = len(row) - cm.ncols

        cr = ClassReturns(class_code=code, class_label=_first_line(row[idx]))
        cr.return_1y = _value_at(row, cm.idx_1y, offset)
        cr.return_3y = _value_at(row, cm.idx_3y, offset)
        cr.return_since_inception = _value_at(row, cm.idx_since, offset)
        i += 1

        if i < n and _row_first_nonempty(rows[i][0]).startswith("비교지수"):
            i += 1
            if i < n and "변동성" in _row_first_nonempty(rows[i][0]):
                vrow, vcm = rows[i]
                voffset = len(vrow) - vcm.ncols
                cr.volatility_1y = _value_at(vrow, vcm.idx_1y, voffset)
                cr.volatility_3y = _value_at(vrow, vcm.idx_3y, voffset)
                cr.volatility_since_inception = _value_at(vrow, vcm.idx_since, voffset)
                i += 1

        results.append(cr)
    return results
