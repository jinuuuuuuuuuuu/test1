import openpyxl

PATH = r"C:\Users\kevin\Downloads\[파싱]투자설명서 (1).xlsm"

wb = openpyxl.load_workbook(PATH, keep_vba=True)
ws = wb["투자설명서_파싱_검수본"]

# (product_code, class_name) -> (수익률_1년, 수익률_3년, 수익률_설정후, 메모)
FIXES = {
    ("KR515302022M", "C1"): (35.52, 21.30, 7.90,
        "PyMuPDF find_tables()로 제3부 3.가.연평균수익률 표를 클래스 단위로 자동 재추출하여 확인: "
        "종류C1 최근1년=35.52%, 최근3년=21.30%, 설정일이후=7.90% (원문 p.50 표 직접 대조 완료)."),
    ("KR515302022M", "Ce"): (36.46, 22.03, 8.63,
        "PyMuPDF find_tables()로 제3부 3.가.연평균수익률 표를 클래스 단위로 자동 재추출하여 확인: "
        "종류Ce 최근1년=36.46%, 최근3년=22.03%, 설정일이후=8.63% (원문 p.50 표 직접 대조 완료)."),
    ("KR5153420318", "C-e"): (0.86, None, 3.01,
        "PyMuPDF find_tables()로 제3부 3.가.연평균수익률 표를 클래스 단위로 자동 재추출하여 확인: "
        "종류C-e 최근1년=0.86%, 설정일이후=3.01% (최근3년은 원문 자체에 값 없음, NULL 유지)."),
}

applied = []
for row in ws.iter_rows(min_row=6):
    code = row[2].value
    cls = row[11].value
    key = (code, cls)
    if key in FIXES:
        r1, r3, rsince, note = FIXES[key]
        row[16].value = r1 if r1 is not None else "NULL"
        row[17].value = r3 if r3 is not None else "NULL"
        row[18].value = rsince if rsince is not None else "NULL"
        row[36].value = ((row[36].value or "") + " " + note).strip()
        row[37].value = "검수완료(수정)"
        applied.append((row[0].row, key))

wb.save(PATH)
print("applied:", applied)
