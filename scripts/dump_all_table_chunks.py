import io
import re
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_all_table_chunks.txt"

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]

rows = []
for row in ws.iter_rows(min_row=6):
    if str(row[9].value).strip().upper() == "Y":
        rows.append(row)

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write(f"총 {len(rows)}건\n\n")
    for row in rows:
        cid = row[7].value
        m = re.match(r"doc(\d+)_chunk(\d+)", str(cid))
        docnum = m.group(1) if m else "?"
        f.write(f"===== {cid} (doc{docnum}) =====\n")
        f.write(f"[파일제목] {row[1].value}\n")
        f.write(f"[원문위치] {row[6].value}\n")
        f.write(f"[검수상태] {row[12].value}\n")
        f.write("--- chunk_text ---\n")
        f.write(str(row[8].value) + "\n\n")

print("done", len(rows))
