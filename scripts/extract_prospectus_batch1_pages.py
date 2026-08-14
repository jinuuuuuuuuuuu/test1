"""1~33번 나머지 16개 파일의 1~12페이지만 추출 (요약정보 구간 — 컨텍스트 절약용)."""

import os
import fitz

PROSPECTUS_DIR = r"C:\Users\kevin\pension-agent\data\raw\prospectus"

TARGET_CODES = [
    "KR510902511M", "KR510902773M", "KR510902777M",
    "KR515302022M", "KR516702010M", "KR518101002M", "KR518101012M",
    "KR518102001M", "KR555202013M", "KR5110501016", "KR5110601022",
    "KR5111420047", "KR5111450067", "KR5113420012", "KR5113420013",
    "KR5113420015",
]

code_to_path = {}
for root, dirs, files in os.walk(PROSPECTUS_DIR):
    for fn in files:
        for code in TARGET_CODES:
            if fn == f"R2_{code}.pdf":
                code_to_path[code] = os.path.join(root, fn)

os.makedirs("prospectus_check", exist_ok=True)

for code in TARGET_CODES:
    path = code_to_path.get(code)
    out_path = f"prospectus_check/{code}_p1-12.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        if not path:
            f.write("*** FILE NOT FOUND ***\n")
            continue
        doc = fitz.open(path)
        end = min(12, doc.page_count)
        for i in range(end):
            f.write(f"\n===== PAGE {i+1}/{doc.page_count} =====\n")
            f.write(doc[i].get_text())
        doc.close()
    print(code, "->", out_path)

print("Done")
