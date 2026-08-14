import io
import re
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_doc46_50_check.txt"

TARGET_DOCS = {"3", "6", "7", "12", "14", "27", "28", "29", "34", "46", "47", "48", "49", "50"}

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]

with io.open(OUT, "w", encoding="utf-8") as f:
    for row in ws.iter_rows(min_row=6):
        cid = row[7].value
        if not cid:
            continue
        m = re.match(r"doc(\d+)_chunk(\d+)", str(cid))
        if not m or m.group(1) not in TARGET_DOCS:
            continue
        f.write(f"===== {cid} (표포함={row[9].value}) =====\n")
        f.write(str(row[8].value)[:600] + "\n\n")
