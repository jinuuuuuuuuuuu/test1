import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\[파싱]투자설명서.xlsm"
OUT = r"C:\Users\kevin\pension-agent\prospectus_check\_verify_1_33_final.txt"

CODES_1_33 = [
    "KR514X450008", "KR510902511M", "KR510902773M", "KR510902777M", "KR515302022M",
    "KR516702010M", "KR518101002M", "KR518101012M", "KR518102001M", "KR555202013M",
    "KR5110501016", "KR5110601022", "KR5111420047", "KR5111450067", "KR5113420012",
    "KR5113420013", "KR5113420015", "KR5113420069", "KR5113450111", "KR5113450401",
    "KR5113470030", "KR5113470031", "KR5114420016", "KR5114420022", "KR5114420027",
    "KR5114420046", "KR5114450222", "KR5114450270", "KR5116501001", "KR5117420097",
    "KR5118201004", "KR5118420006", "KR5118420036",
]

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["투자설명서_파싱"]

rows_by_code = {}
for row in ws.iter_rows(min_row=6):
    code = row[2].value
    if code in CODES_1_33:
        rows_by_code.setdefault(code, []).append(row)

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write(f"대상 파일 수: {len(CODES_1_33)}\n")
    n_rows = 0
    n_not_done = 0
    for i, code in enumerate(CODES_1_33, start=1):
        rows = rows_by_code.get(code, [])
        if not rows:
            f.write(f"file#{i} {code}: *** NOT FOUND IN TRACKER ***\n")
            continue
        for r in rows:
            n_rows += 1
            status = r[37].value
            if not status or "검수완료" not in str(status):
                n_not_done += 1
                f.write(f"file#{i} {code} class={r[11].value}: 상태={status} <-- 미완료\n")
    f.write(f"\n총 {n_rows}개 행 확인, 미완료 {n_not_done}개\n")
    f.write("\n--- 상태별 집계 ---\n")
    from collections import Counter
    c = Counter()
    for code in CODES_1_33:
        for r in rows_by_code.get(code, []):
            c[str(r[37].value)] += 1
    for k, v in c.most_common():
        f.write(f"{k}: {v}\n")
