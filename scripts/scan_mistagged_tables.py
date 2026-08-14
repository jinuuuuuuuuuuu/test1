import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_mistagged_scan.txt"

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]

hits = []
for row in ws.iter_rows(min_row=6):
    cid = row[7].value
    if not cid:
        continue
    has_table_tag = str(row[9].value).strip().upper()
    text = str(row[8].value or "")
    if "[표]" in text and has_table_tag != "Y":
        hits.append((cid, has_table_tag, text[:200]))

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write(f"[표] 마커가 본문에 있는데 표포함≠Y인 청크: {len(hits)}건\n\n")
    for cid, tag, preview in hits:
        f.write(f"--- {cid} (표포함={tag}) ---\n{preview}\n\n")

print(len(hits))
