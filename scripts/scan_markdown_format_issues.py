import io
import re
import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_markdown_format_scan.txt"

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["파싱 결과"]


def looks_like_markdown_table(text):
    if not text:
        return False
    lines = text.split("\n")
    return any(re.match(r"^\s*\|.*\|\s*$", ln) for ln in lines) and any(
        re.match(r"^\s*\|[\s:-]+\|", ln) for ln in lines
    )


has_table_not_markdown = []
has_table_is_markdown = []

for row in ws.iter_rows(min_row=6):
    fnum = row[0].value
    if fnum is None:
        continue
    has_table = row[9].value  # 0-indexed col9 = 표포함
    if has_table != "Y":
        continue
    chunk_id = row[7].value
    structured = row[10].value or ""
    chunk_text = row[8].value or ""
    text_to_check = structured if structured.strip() else chunk_text
    if looks_like_markdown_table(text_to_check):
        has_table_is_markdown.append(chunk_id)
    else:
        has_table_not_markdown.append(chunk_id)

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write(f"표포함=Y 전체: {len(has_table_is_markdown) + len(has_table_not_markdown)}\n")
    f.write(f"이미 Markdown 표 형식: {len(has_table_is_markdown)}\n")
    f.write(f"Markdown 아님(수정 필요): {len(has_table_not_markdown)}\n\n")
    f.write("--- 수정 필요 목록 ---\n")
    for cid in has_table_not_markdown:
        f.write(f"{cid}\n")
