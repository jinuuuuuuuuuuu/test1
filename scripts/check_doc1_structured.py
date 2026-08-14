import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_doc1_structured_check.txt"

TARGETS = ["doc1_chunk02", "doc1_chunk03", "doc1_chunk05", "doc5_chunk01"]

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]

with io.open(OUT, "w", encoding="utf-8") as f:
    for row in ws.iter_rows(min_row=6):
        if row[7].value in TARGETS:
            f.write(f"===== {row[7].value} =====\n")
            f.write("--- 표구조화텍스트 (raw repr) ---\n")
            f.write(repr(row[10].value) + "\n\n")
            f.write("--- 표구조화텍스트 ---\n")
            f.write(str(row[10].value) + "\n\n")
