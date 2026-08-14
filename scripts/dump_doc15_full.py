import io
import openpyxl
import fitz

TRACKER = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
SRC = r"C:\Users\kevin\pension-agent\data\raw\docs\docs_renamed\doc15.pdf"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_doc15_full.txt"

wb = openpyxl.load_workbook(TRACKER, data_only=True)
ws = wb["파싱 결과"]

rows = []
for row in ws.iter_rows(min_row=6):
    fnum = row[0].value
    try:
        num = int(float(fnum))
    except (TypeError, ValueError):
        continue
    if num == 15:
        rows.append(row)

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("########## TRACKER: doc15 청크 목록 ##########\n")
    for row in rows:
        f.write(f"\n--- {row[7].value} (section={row[5].value!r}, 원문위치={row[6].value!r}, 표포함={row[9].value}) ---\n")
        f.write(str(row[8].value) + "\n")

    f.write("\n\n########## SOURCE FULL TEXT (원본 PDF, 페이지별) ##########\n")
    doc = fitz.open(SRC)
    for i in range(doc.page_count):
        f.write(f"\n===== PAGE {i+1}/{doc.page_count} =====\n")
        f.write(doc[i].get_text())
    doc.close()

print("done ->", OUT)
