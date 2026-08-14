"""투자설명서 85개 파일의 목차(구조) + 1페이지(투자위험등급)를 훑어서
트래커([파싱]투자설명서.xlsm) 값과 자동 교차검증하고 이상 징후를 리포트한다.

전체 40페이지씩 읽는 대신, 구조 확인용으로 저비용 스캔:
- 1페이지: 투자위험등급 텍스트 추출 -> 트래커 값과 자동 대조
- 목차 페이지(보통 4페이지 부근, '목' '차' 포함): 제1~5부 섹션 존재 여부 확인
- 표준 양식에서 벗어난(페이지 수 이상치, 목차 없음 등) 파일 플래그
"""

import json
import os
import re

import fitz
import openpyxl

TRACKER_PATH = r"C:\Users\kevin\Downloads\[파싱]투자설명서.xlsm"
PROSPECTUS_DIR = r"C:\Users\kevin\pension-agent\data\raw\prospectus"
OUT_PATH = r"C:\Users\kevin\pension-agent\prospectus_toc_check.txt"

wb = openpyxl.load_workbook(TRACKER_PATH, keep_vba=True, data_only=True)
ws = wb["투자설명서_파싱"]

rows_by_file = {}
for row in ws.iter_rows(min_row=6, values_only=True):
    fname = row[0]
    if not fname:
        continue
    if row[37] != "추가확인":
        continue
    rows_by_file.setdefault(fname, []).append(
        {
            "class": row[11],
            "위험등급": row[6],
            "총보수": row[13],
            "3년비용": row[14],
        }
    )

# 파일명 -> 실제 경로 매핑
file_to_path = {}
for root, dirs, files in os.walk(PROSPECTUS_DIR):
    for fn in files:
        if fn in rows_by_file:
            file_to_path[fn] = os.path.join(root, fn)

missing = set(rows_by_file) - set(file_to_path)
results = []
risk_grade_pattern = re.compile(r"(\d)\s*등급\s*[\[\(]?\s*([^\]\)\n]{0,10})")

for fname, path in file_to_path.items():
    entry = {"file": fname, "path_found": True}
    try:
        doc = fitz.open(path)
        entry["page_count"] = doc.page_count

        page1_text = doc[0].get_text()
        m = risk_grade_pattern.search(page1_text)
        entry["page1_risk_grade_found"] = f"{m.group(1)}등급[{m.group(2).strip()}]" if m else None

        toc_page_idx = None
        toc_text = None
        for i in range(min(8, doc.page_count)):
            t = doc[i].get_text()
            if "목" in t and "차" in t and ("제 1 부" in t or "제1부" in t or "요약정보" in t):
                toc_page_idx = i
                toc_text = t
                break
        entry["toc_page"] = (toc_page_idx + 1) if toc_page_idx is not None else None
        if toc_text:
            sections_found = []
            for label in ["요약정보", "제 1 부", "제1부", "제 2 부", "제2부", "제 3 부", "제3부", "제 4 부", "제4부", "제 5 부", "제5부"]:
                if label in toc_text:
                    sections_found.append(label)
            entry["sections_in_toc"] = sorted(set(sections_found))
        else:
            entry["sections_in_toc"] = None

        tracker_grades = set(r["위험등급"] for r in rows_by_file[fname] if r["위험등급"])
        entry["tracker_risk_grades"] = list(tracker_grades)

        if entry["page1_risk_grade_found"] and tracker_grades:
            match = any(entry["page1_risk_grade_found"].split("[")[0] in g for g in tracker_grades)
            entry["risk_grade_match"] = match
        else:
            entry["risk_grade_match"] = None

        doc.close()
    except Exception as e:
        entry["error"] = str(e)
    results.append(entry)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Total target files: {len(rows_by_file)}\n")
    f.write(f"Files found on disk: {len(file_to_path)}\n")
    f.write(f"Files MISSING from disk: {sorted(missing)}\n\n")

    anomalies = []
    for e in results:
        flags = []
        if e.get("page_count") and not (45 <= e["page_count"] <= 75):
            flags.append(f"비정상 페이지수({e['page_count']})")
        if e.get("toc_page") is None:
            flags.append("목차 페이지 못 찾음")
        if e.get("risk_grade_match") is False:
            flags.append(f"위험등급 불일치(p1={e.get('page1_risk_grade_found')}, tracker={e.get('tracker_risk_grades')})")
        if "제2부" not in (e.get("sections_in_toc") or []) and "제 2 부" not in (e.get("sections_in_toc") or []):
            flags.append("목차에 제2부 없음")
        if "제3부" not in (e.get("sections_in_toc") or []) and "제 3 부" not in (e.get("sections_in_toc") or []):
            flags.append("목차에 제3부 없음")
        if flags:
            anomalies.append((e["file"], flags, e))

    f.write(f"=== 이상 징후 있는 파일: {len(anomalies)} / {len(results)} ===\n\n")
    for fname, flags, e in anomalies:
        f.write(f"{fname}: {', '.join(flags)}\n")
        f.write(f"  detail: {json.dumps(e, ensure_ascii=False)}\n")

    f.write(f"\n=== 정상(구조 이상 없음) 파일: {len(results) - len(anomalies)} ===\n")
    for e in results:
        if not any(e["file"] == a[0] for a in anomalies):
            f.write(f"{e['file']}: pages={e.get('page_count')}, risk_grade_p1={e.get('page1_risk_grade_found')}, tracker={e.get('tracker_risk_grades')}\n")

print("Done. Anomalies:", len(anomalies), "/ Total:", len(results))
