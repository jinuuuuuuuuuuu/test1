import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"

wb = openpyxl.load_workbook(PATH, keep_vba=True, data_only=True)
ws = wb["파싱 결과"]
target_docs = {1, 2, 3, 4, 5, 7, 8, 9, 21, 24, 27, 28, 30, 31, 32, 37, 54, 56, 58}

lines = []
total = 0
for row in ws.iter_rows(min_row=6, values_only=True):
    fnum = row[0]
    if fnum is None:
        continue
    try:
        num = int(float(fnum))
    except (TypeError, ValueError):
        continue
    if num not in target_docs:
        continue
    total += 1
    status = row[12] or "(empty)"
    if not str(status).startswith("검수완료") and "추가확인" not in str(status) and "신규추가" not in str(status):
        lines.append(f"{row[7]} | doc{num} | status={status}")

with open("full_status_check.txt", "w", encoding="utf-8") as f:
    f.write(f"Total rows in scope: {total}\n")
    f.write("\n".join(lines) if lines else "ALL RESOLVED (except intentionally out-of-scope doc58_chunk01)")

print("done")
