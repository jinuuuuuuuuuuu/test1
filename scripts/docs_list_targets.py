import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\prospectus_check\_docs_targets.txt"

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]

docs = {}
for row in ws.iter_rows(min_row=6, values_only=True):
    fnum = row[0]
    if fnum is None:
        continue
    try:
        num = int(float(fnum))
    except (TypeError, ValueError):
        continue
    status = str(row[12] or "")
    title = row[1]
    fmt = row[2]
    d = docs.setdefault(num, {"title": title, "fmt": fmt, "total": 0, "todo": 0})
    d["total"] += 1
    if status == "검수전":
        d["todo"] += 1

with io.open(OUT, "w", encoding="utf-8") as f:
    for num in sorted(docs):
        d = docs[num]
        if d["todo"] > 0:
            f.write(f"doc{num} [{d['fmt']}] {d['title']} — todo {d['todo']}/{d['total']}\n")
