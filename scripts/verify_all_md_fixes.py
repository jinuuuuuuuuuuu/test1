import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_verify_all_md_fixes.txt"

TARGETS = [
    "doc15_chunk01", "doc15_chunk02", "doc18_chunk05", "doc22_chunk01",
    "doc22_chunk02", "doc22_chunk03", "doc22_chunk04", "doc23_chunk05",
    "doc23_chunk07", "doc25_chunk05", "doc33_chunk04", "doc36_chunk05",
    "doc37_chunk03", "doc38_chunk02", "doc39_chunk06", "doc40_chunk01",
    "doc41_chunk03", "doc56_chunk02", "doc56_chunk03", "doc58_chunk02",
    "doc58_chunk03",
]

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]

rows = {}
for row in ws.iter_rows(min_row=6):
    cid = row[7].value
    if cid in TARGETS:
        rows[cid] = row

with io.open(OUT, "w", encoding="utf-8") as f:
    for cid in TARGETS:
        row = rows.get(cid)
        f.write(f"===== {cid} =====\n")
        if row is None:
            f.write("!!! NOT FOUND !!!\n\n")
            continue
        f.write(f"[검수상태] {row[12].value}\n")
        f.write(f"[특이사항] {row[11].value}\n")
        f.write("--- chunk_text ---\n")
        f.write(str(row[8].value) + "\n")
        f.write("--- 표구조화텍스트 ---\n")
        f.write(str(row[10].value) + "\n\n")

print("done", len(rows), "/", len(TARGETS))
