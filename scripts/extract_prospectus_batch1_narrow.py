"""나머지 14개 파일의 1,3~9페이지만 추출 (요약정보 핵심구간, 유의사항 반복 페이지는 제외)."""

import os
import fitz

PROSPECTUS_DIR = r"C:\Users\kevin\pension-agent\data\raw\prospectus"

TARGET_CODES = [
    "KR510902777M", "KR515302022M", "KR516702010M", "KR518101002M",
    "KR518101012M", "KR518102001M", "KR555202013M", "KR5110501016",
    "KR5110601022", "KR5111420047", "KR5111450067", "KR5113420012",
    "KR5113420013", "KR5113420015",
]
PAGES = [1, 3, 4, 5, 6, 7, 8, 9]

code_to_path = {}
for root, dirs, files in os.walk(PROSPECTUS_DIR):
    for fn in files:
        for code in TARGET_CODES:
            if fn == f"R2_{code}.pdf":
                code_to_path[code] = os.path.join(root, fn)

os.makedirs("prospectus_check", exist_ok=True)

for code in TARGET_CODES:
    path = code_to_path.get(code)
    out_path = f"prospectus_check/{code}_narrow.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        if not path:
            f.write("*** FILE NOT FOUND ***\n")
            continue
        doc = fitz.open(path)
        for i in PAGES:
            if i - 1 >= doc.page_count:
                continue
            f.write(f"\n===== PAGE {i}/{doc.page_count} =====\n")
            f.write(doc[i - 1].get_text())
        doc.close()
    print(code, "->", out_path)

print("Done")
