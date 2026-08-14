import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_doc8_31_check.txt"

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]

with io.open(OUT, "w", encoding="utf-8") as f:
    for row in ws.iter_rows(min_row=6):
        if row[7].value in ("doc8_chunk06", "doc31_chunk04"):
            f.write(f"===== {row[7].value} =====\n")
            f.write(f"[표포함] {row[9].value}\n")
            f.write("--- chunk_text ---\n")
            f.write(str(row[8].value) + "\n")
            f.write("--- 표구조화텍스트 ---\n")
            f.write(str(row[10].value) + "\n\n")
