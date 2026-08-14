"""투자설명서 18~33번 파일(16개)의 트래커 값 + 원본 PDF 전체 텍스트를 나란히 뽑아서
비교용 파일로 저장한다."""

import os
import openpyxl
import fitz

TRACKER_PATH = r"C:\Users\kevin\Downloads\[파싱]투자설명서.xlsm"
PROSPECTUS_DIR = r"C:\Users\kevin\pension-agent\data\raw\prospectus"

TARGET_CODES = [
    "KR5113420069", "KR5113450111", "KR5113450401", "KR5113470030",
    "KR5113470031", "KR5114420016", "KR5114420022", "KR5114420027",
    "KR5114420046", "KR5114450222", "KR5114450270", "KR5116501001",
    "KR5117420097", "KR5118201004", "KR5118420006", "KR5118420036",
]

wb = openpyxl.load_workbook(TRACKER_PATH, keep_vba=True, data_only=True)
ws = wb["투자설명서_파싱"]

headers = ['파일명','집합투자기구명칭','상품코드','집합투자업자명칭','작성기준일','증권신고서효력발생일',
           '투자위험등급','모집증권종류/총액','투자목적','투자전략','상품분류','판매클래스','판매방식',
           '총보수비용','3년총비용','수익률기준일','수익률1년','수익률3년','수익률설정이후','변동성1년',
           '변동성3년','변동성설정이후','비교지수','운용역성명','운용역생년','운용역직위','운용역경력',
           '동종운용사수익률','매입기준','환매기준','환매대금지급','환매수수료','과세특징','전환가능여부',
           '최초설정일','근거위치','검수메모','검수상태']

rows_by_code = {}
for row in ws.iter_rows(min_row=6, values_only=True):
    fname = row[0]
    if not fname:
        continue
    code = row[2]
    if code in TARGET_CODES:
        rows_by_code.setdefault(code, []).append(row)

code_to_path = {}
for root, dirs, files in os.walk(PROSPECTUS_DIR):
    for fn in files:
        for code in TARGET_CODES:
            if fn == f"R2_{code}.pdf":
                code_to_path[code] = os.path.join(root, fn)

os.makedirs("prospectus_check", exist_ok=True)

for code in TARGET_CODES:
    out_path = f"prospectus_check/{code}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"########## TRACKER DATA: {code} ##########\n")
        for row in rows_by_code.get(code, []):
            f.write(f"--- class={row[11]} ---\n")
            for i, h in enumerate(headers):
                val = row[i]
                if val is not None and str(val).strip() != "":
                    f.write(f"  [{h}] {val}\n")
        f.write("\n########## SOURCE PDF FULL TEXT ##########\n")
        path = code_to_path.get(code)
        if not path:
            f.write("*** FILE NOT FOUND ***\n")
            print(code, "NOT FOUND")
            continue
        doc = fitz.open(path)
        for i in range(doc.page_count):
            f.write(f"\n===== PAGE {i+1}/{doc.page_count} =====\n")
            f.write(doc[i].get_text())
        doc.close()
    print(code, "->", out_path)

print("Done:", len(TARGET_CODES), "files")
