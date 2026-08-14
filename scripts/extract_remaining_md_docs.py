import os
import openpyxl
import fitz

TRACKER_PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
DOCS_DIR = r"C:\Users\kevin\pension-agent\data\raw\docs\docs_renamed"
OUT_DIR = r"C:\Users\kevin\pension-agent\docs_check"

TARGET_DOCS = [4, 32, 37, 54]

wb = openpyxl.load_workbook(TRACKER_PATH, keep_vba=True, data_only=True)
ws = wb["파싱 결과"]

header = ['파일번호', '파일제목', '파일형식', '카테고리', '저장방식', 'section', '원문위치',
          'chunk_id', 'chunk_text', '표포함', '표구조화텍스트', '특이사항', '검수상태']

rows_by_doc = {}
for row in ws.iter_rows(min_row=6, values_only=True):
    fnum = row[0]
    if fnum is None:
        continue
    try:
        num = int(float(fnum))
    except (TypeError, ValueError):
        continue
    if num in TARGET_DOCS:
        rows_by_doc.setdefault(num, []).append(row)

for num in TARGET_DOCS:
    src_path = os.path.join(DOCS_DIR, f"doc{num}.pdf")
    out_path = os.path.join(OUT_DIR, f"doc{num}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"########## TRACKER DATA: doc{num} ##########\n")
        for row in rows_by_doc.get(num, []):
            f.write(f"--- {row[7]} (status={row[12]}) ---\n")
            for i, h in enumerate(header):
                val = row[i]
                if val is not None and str(val).strip() != "":
                    f.write(f"  [{h}] {val}\n")
        f.write("\n########## SOURCE FULL TEXT ##########\n")
        if not os.path.exists(src_path):
            f.write("*** FILE NOT FOUND ***\n")
            print(num, "NOT FOUND")
            continue
        doc = fitz.open(src_path)
        for i in range(doc.page_count):
            f.write(f"\n===== PAGE {i+1}/{doc.page_count} =====\n")
            f.write(doc[i].get_text())
        doc.close()
    print(num, "->", out_path)
