import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"

wb = openpyxl.load_workbook(PATH, keep_vba=True)
ws = wb["파싱 결과"]

for row in ws.iter_rows(min_row=6):
    if row[7].value in ("doc20_chunk03", "doc20_chunk04", "doc20_chunk05"):
        row[12].value = "검수완료"

wb.save(PATH)
print("done")
