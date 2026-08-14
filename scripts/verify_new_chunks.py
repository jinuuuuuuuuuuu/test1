import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"

wb = openpyxl.load_workbook(PATH, keep_vba=True, data_only=True)
ws = wb["파싱 결과"]

target_ids = [
    "doc1_chunk06", "doc3_chunk06", "doc8_chunk06", "doc27_chunk11",
    "doc31_chunk06", "doc37_chunk06", "doc37_chunk07",
]
found = {}
for row in ws.iter_rows(min_row=6, values_only=True):
    cid = row[7]
    if cid in target_ids:
        found[cid] = {
            "text_len": len(row[8]) if row[8] else 0,
            "table_len": len(row[10]) if row[10] else 0,
            "status": row[12],
            "text_preview": (row[8][:60] if row[8] else ""),
        }

with open("chunk_check.txt", "w", encoding="utf-8") as f:
    for cid in target_ids:
        if cid in found:
            info = found[cid]
            f.write(f"{cid}: FOUND | text_len={info['text_len']} | table_len={info['table_len']} | status={info['status']}\n")
            f.write(f"  preview: {info['text_preview']}\n")
        else:
            f.write(f"{cid}: *** NOT FOUND ***\n")

print("done")
