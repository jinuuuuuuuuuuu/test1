"""투자설명서 PDF 100개에서 AUM(시장잔고)을 추출해 prospectus.db fund_master에 적재한다.

- 추출 위치: 요약 재무상태표의 자본총계(최신 기), 단위 백만원 (src/parsing/aum_extractor.py)
- 기존 DB에 AUM 컬럼이 없으면 자동 추가 (schema.ensure_aum_columns)
- 검증 리포트(data/processed/aum_report.csv): 상품코드·펀드명·값·기준일·원문 스니펫·상태 —
  사람이 원문 대조 검수할 때 이 파일을 쓴다 (다른 필드들과 같은 검수 절차)
- ⚠️ load_prospectus_xlsm은 fund_master를 지우고 다시 채우므로, xlsm 재적재 후에는
  이 스크립트를 반드시 다시 실행해야 한다.

사용법: python scripts/extract_aum.py
"""

import csv
import os
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsing.aum_extractor import extract_aum_from_pdf, find_summary_section
from src.storage.schema import connect

DB_PATH = "data/processed/prospectus.db"
PROSPECTUS_ROOT = os.path.join("data", "raw", "prospectus")
REPORT_PATH = "data/processed/aum_report.csv"


def _find_pdf_dir() -> str:
    # 하위의 "투자설명서" 폴더 — 한글 폴더명이 NFD로 저장돼 있어 이름 비교 대신 유일한
    # 하위 디렉터리를 취한다.
    subdirs = [d for d in os.listdir(PROSPECTUS_ROOT) if os.path.isdir(os.path.join(PROSPECTUS_ROOT, d))]
    assert len(subdirs) == 1, f"예상 밖의 폴더 구조: {subdirs}"
    return os.path.join(PROSPECTUS_ROOT, subdirs[0])


def _classify_failure(pdf_path: str) -> tuple[str, str]:
    """추출 실패를 두 유형으로 구분한다 — 수기 검수 시 어디를 봐야 하는지 알려주기 위함.

    - PARSE_FAIL: 요약재무정보 섹션은 있으나 표 열이 뒤섞여 추출됨(pypdf 한계) 등으로
      기계 파싱 불가. 잘못된 연도의 값을 적재할 위험이 있어 일부러 값을 쓰지 않는다 —
      리포트의 페이지·스니펫을 보고 사람이 원문에서 최신 기 자본총계를 확인해야 한다.
    - NOT_FOUND: 문서에 펀드 자체의 요약재무정보가 없음 (예: "제4부를 참고" 안내만 있는
      문서). 이 경우 투자설명서만으로는 AUM을 얻을 수 없다.
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        idx = find_summary_section(text)
        if idx is not None:
            snippet = text[idx:idx + 250].replace("\n", " / ")
            return f"PARSE_FAIL(p{i + 1}): 표 형식 비정형 — 수기 확인 필요", snippet
    return "NOT_FOUND: 문서에 펀드 요약재무정보 없음 — 수기 확인 필요", ""


def main():
    pdf_root = _find_pdf_dir()
    conn = connect(DB_PATH)
    fund_names = {
        code: name for code, name in conn.execute("SELECT product_code, fund_name FROM fund_master")
    }

    rows = []
    ok = failed = 0
    for code in sorted(os.listdir(pdf_root)):
        code_dir = os.path.join(pdf_root, code)
        pdfs = [f for f in os.listdir(code_dir) if f.lower().endswith(".pdf")]
        if code not in fund_names or not pdfs:
            rows.append([code, "", "", "", "", "", "SKIP: DB에 없거나 PDF 없음"])
            failed += 1
            continue

        pdf_path = os.path.join(code_dir, pdfs[0])
        result = extract_aum_from_pdf(pdf_path)
        name = unicodedata.normalize("NFC", fund_names[code])
        if result is None:
            status, snippet = _classify_failure(pdf_path)
            rows.append([code, name, "", "", "", snippet, status])
            failed += 1
            print(f"  FAIL {code} {name} — {status}")
            continue

        with conn:
            conn.execute(
                "UPDATE fund_master SET aum_krw_million = ?, aum_base_date = ?, aum_period_label = ? "
                "WHERE product_code = ?",
                (result.aum_krw_million, result.base_date, result.period_label, code),
            )
        rows.append([
            code, name, result.aum_krw_million, result.base_date,
            result.period_label, result.source_snippet, "OK",
        ])
        ok += 1
        print(f"  OK {code} {result.aum_krw_million:>12,.0f}백만원 ({result.base_date}) {name[:30]}")

    conn.close()

    with open(REPORT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["product_code", "fund_name", "aum_krw_million", "aum_base_date", "period", "source_snippet", "status"])
        writer.writerows(rows)

    print(f"\n완료: 성공 {ok} / 실패 {failed} — 리포트: {REPORT_PATH}")


if __name__ == "__main__":
    main()
