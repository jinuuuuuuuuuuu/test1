import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"

wb = openpyxl.load_workbook(PATH, keep_vba=True)
ws = wb["파싱 결과"]


def find_row(chunk_id):
    for row in ws.iter_rows(min_row=6):
        if row[7].value == chunk_id:
            return row
    raise ValueError(f"chunk_id not found: {chunk_id}")


NEW_TABLE_MD = (
    "| 재원구분 | 연금수령 | 연금 외 수령 |\n"
    "|---|---|---|\n"
    "| 세액공제 받은 금액 | 연금소득 & 분리과세 5.5~3.3% 세율 "
    "(단, 연 1,500만원 초과 시 종합과세 세율 OR 16.5% 단일세율 선택) | 기타소득 & 분리과세 16.5% 세율 |\n"
    "| 운용수익 | 연금소득 & 분리과세 5.5~3.3% 세율 "
    "(단, 연 1,500만원 초과 시 종합과세 세율 OR 16.5% 단일세율 선택) | 기타소득 & 분리과세 16.5% 세율 |"
)

NEW_CHUNK_TEXT = (
    "[표] 연금계좌 운용수익 과세 구분\n"
    "연금계좌에서 발생한 수익은 금융소득이 아니다. 연금계좌에서 발생한 수익은 연금수령 시 연금소득으로 "
    "분리과세 되며 연금외 수령 시 기타소득으로 분리과세 된다. 따라서 금융소득에 해당되지 않는다.\n"
    + NEW_TABLE_MD
)

r = find_row("doc36_chunk04")
r[8].value = NEW_CHUNK_TEXT       # chunk_text
r[10].value = NEW_TABLE_MD        # 표구조화텍스트
r[11].value = (
    (r[11].value or "")
    + " ⚠️Markdown 표 형식으로 재정리 — 원문 표(병합 셀 포함)를 파이프 구분 Markdown 표로 변환, "
    "병합된 셀 값(연금소득 5.5~3.3%/기타소득 16.5%)은 두 행에 동일하게 반복 기입함."
).strip()
r[12].value = "검수완료(수정)"

wb.save(PATH)
print("doc36_chunk04 Markdown 변환 완료")
print()
print("=== chunk_text ===")
print(NEW_CHUNK_TEXT)
print()
print("=== 표구조화텍스트 ===")
print(NEW_TABLE_MD)
