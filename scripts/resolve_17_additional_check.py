import openpyxl

PATH = r"C:\Users\kevin\Downloads\[파싱]투자설명서 (2).xlsm"

wb = openpyxl.load_workbook(PATH, keep_vba=True)
ws = wb["투자설명서_파싱_검수본"]

# (code, class) -> 확인 메모
CONFIRMED = {
    ("KR5147430065", "C"): "PyMuPDF find_tables()로 제3부 연평균수익률 표를 독립 재추출한 결과 이 펀드는 전 클래스 수익률이 원문 자체에 없음(신규설정) 재확인 — NULL 처리 정확함.",
    ("KR5147430065", "Ce"): "PyMuPDF find_tables()로 제3부 연평균수익률 표를 독립 재추출한 결과 이 펀드는 전 클래스 수익률이 원문 자체에 없음(신규설정) 재확인 — NULL 처리 정확함.",
    ("KR5153420022", "C"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=-1.5/3y=2.73/설정후=0.47로 기존값과 정확히 일치 확인.",
    ("KR5153420022", "C-e"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=-1.32/3y=2.92/설정후=2.19로 기존값과 정확히 일치 확인.",
    ("KR5153420063", "C-P"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=3.87/3y=3.1/설정후=2.07로 기존값과 정확히 일치 확인.",
    ("KR5153420063", "C-P2"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=3.88/3y=3.11/설정후=2.05로 기존값과 정확히 일치 확인.",
    ("KR5153420318", "C"): "PyMuPDF find_tables()로 제3부 연평균수익률 표를 독립 재추출한 결과 이 표에 'C' 클래스 자체가 없음(A-e/C-e/F/C-Pe/C-P2e만 존재) 재확인 — NULL 처리 정확함.",
    ("KR5153450209", "C"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=5.76/3y=8.1/설정후=6.1로 기존값과 정확히 일치 확인.",
    ("KR5153450209", "C-P2e"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=6.18/3y=8.53/설정후=9.67로 기존값과 정확히 일치 확인.",
    ("KR5153450250", "C"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=17.69/3y=12.23/설정후=5.85로 기존값과 정확히 일치 확인.",
    ("KR5153450250", "C-P2e"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=18.16/3y=12.68/설정후=9.5로 기존값과 정확히 일치 확인.",
    ("KR5153450268", "C1"): "PyMuPDF find_tables()로 제3부 연평균수익률 표를 독립 재추출한 결과 C1 클래스는 전 구간 값 없음(원문 '-') 재확인 — NULL 처리 정확함.",
    ("KR5153450268", "C-e"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=18.61/3y=7.5/설정후=5.12로 기존값과 정확히 일치 확인(최초설정일만 원문 요약정보에 미기재로 NULL 유지).",
    ("KR5153450431", "C"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=9.92/3y=12.09/설정후=4.52로 기존값과 정확히 일치 확인.",
    ("KR5153450431", "C-e"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=10.27/3y=12.45/설정후=4.65로 기존값과 정확히 일치 확인.",
    ("KR5153450658", "C1"): "PyMuPDF find_tables()로 제3부 연평균수익률 표를 독립 재추출한 결과 C1 클래스는 전 구간 값 없음(원문 '-') 재확인 — NULL 처리 정확함(최초설정일 2017-10-23은 기존대로 유지).",
    ("KR5153450658", "C-e"): "PyMuPDF find_tables()로 제3부 연평균수익률 표 독립 재추출 결과 1y=2.93/3y=4.31/설정후=2.62로 기존값과 정확히 일치 확인.",
}

updated = []
for row in ws.iter_rows(min_row=6):
    key = (row[2].value, row[11].value)
    if key in CONFIRMED:
        note = CONFIRMED[key]
        existing = row[36].value or ""
        row[36].value = (existing + " " + note).strip()
        row[37].value = "검수완료(교차검증완료)"
        updated.append(key)

wb.save(PATH)
print("updated:", len(updated))
for k in updated:
    print(" ", k)
