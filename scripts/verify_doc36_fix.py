import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_doc36_fix_result.txt"

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]

for row in ws.iter_rows(min_row=6):
    if row[7].value == "doc36_chunk04":
        with io.open(OUT, "w", encoding="utf-8") as f:
            f.write("=== chunk_text ===\n")
            f.write(str(row[8].value) + "\n\n")
            f.write("=== 표구조화텍스트 ===\n")
            f.write(str(row[10].value) + "\n\n")
            f.write("=== 검수상태 ===\n")
            f.write(str(row[12].value) + "\n")
        break
