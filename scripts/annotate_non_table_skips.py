import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"

NOTES = {
    "doc4_chunk03": "표포함=Y로 표시되어 있으나 원문 확인 결과 실제 표가 아니라 MIN() 계산식을 도식화한 박스임(원문 p.2 이미지로 시각 대조 확인). Markdown 표 변환 대상 아님.",
    "doc32_chunk03": "표포함=Y로 표시되어 있으나 원문 확인 결과 표가 아니라 Q&A 서술형 FAQ 내용임. Markdown 표 변환 대상 아님.",
    "doc39_chunk01": "표포함=Y로 표시되어 있으나 원문 확인 결과 표가 아니라 연금수령한도 계산식 1줄임. Markdown 표 변환 대상 아님.",
    "doc54_chunk04": "표포함=Y로 표시되어 있으나 원문 확인 결과 표가 아니라 알림톡 메시지 템플릿(안내문구+치환필드)임. Markdown 표 변환 대상 아님.",
    "doc54_chunk05": "표포함=Y로 표시되어 있으나 원문 확인 결과 표가 아니라 알림톡 메시지 템플릿(안내문구+치환필드)임. Markdown 표 변환 대상 아님.",
    "doc54_chunk06": "표포함=Y로 표시되어 있으나 원문 확인 결과 표가 아니라 알림톡 메시지 템플릿(안내문구+치환필드)임. Markdown 표 변환 대상 아님.",
}

wb = openpyxl.load_workbook(PATH, keep_vba=True)
ws = wb["파싱 결과"]

for row in ws.iter_rows(min_row=6):
    cid = row[7].value
    if cid in NOTES:
        existing = (row[11].value or "").strip()
        note = NOTES[cid]
        if note not in existing:
            row[11].value = (existing + " " + note).strip() if existing else note
        if not row[12].value or "검수" not in str(row[12].value):
            row[12].value = "검수완료"
        print(cid, "annotated")

wb.save(PATH)
print("done")
