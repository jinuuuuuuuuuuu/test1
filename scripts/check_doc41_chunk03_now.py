import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_doc41_chunk03_now.txt"

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]

with io.open(OUT, "w", encoding="utf-8") as f:
    for row in ws.iter_rows(min_row=6):
        if row[7].value in ("doc41_chunk03", "doc41_chunk04"):
            f.write(f"===== {row[7].value} (표포함={row[9].value}, section={row[5].value!r}, 원문위치={row[6].value!r}) =====\n")
            f.write(str(row[8].value) + "\n\n")
