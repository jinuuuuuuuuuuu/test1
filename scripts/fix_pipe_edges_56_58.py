import re
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
TARGETS = ["doc56_chunk02", "doc56_chunk03", "doc58_chunk02", "doc58_chunk03"]

wb = openpyxl.load_workbook(PATH, keep_vba=True)
ws = wb["파싱 결과"]


def normalize(text):
    lines = text.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        is_sep = bool(re.fullmatch(r"\|?(\s*-{2,}\s*\|)+\s*-{0,}\s*\|?", stripped)) if "---" in stripped else False
        if "|" in line and (is_sep or (not stripped.startswith("|") or not stripped.endswith("|"))):
            if not stripped.startswith("|"):
                stripped = "| " + stripped
            if not stripped.endswith("|"):
                stripped = stripped + " |"
            out.append(stripped)
        else:
            out.append(line)
    return "\n".join(out)


for row in ws.iter_rows(min_row=6):
    if row[7].value in TARGETS:
        row[8].value = normalize(row[8].value)
        row[10].value = normalize(row[10].value)
        print(row[7].value, "normalized")

wb.save(PATH)
print("done")
