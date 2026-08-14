import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"

wb = openpyxl.load_workbook(PATH, keep_vba=True)
ws = wb["파싱 결과"]

table = "| 구분 | 계산식 |\n|---|---|\n| DC 형 휴직기간 포함 시 급여 계산 | (연간임금총액 ÷ (12 - 휴업월수)) + (연 1회 임금성이 인정되는 상여금, 연차수당 ÷ 12) |"

text = ("○ 퇴직연금 DC 형 가입자입니다. 육아휴직이 있는 경우 DC 형 부담금은 어떻게 되나요?\n"
        " • 법령에 따라 일정사유에 해당하는 휴직기간의 경우 다음과 같이 계산되며, 자세한 내용은 재직하신 곳에\n"
        "  인사담당자에게 문의하시길 바랍니다.\n\n"
        "[표] DC 형 휴직기간 포함 시 급여 계산\n" + table)

for row in ws.iter_rows(min_row=6):
    if row[7].value == "doc15_chunk04":
        row[8].value = text
        row[9].value = "Y"
        row[10].value = table
        row[11].value = ((row[11].value or "") + " " +
                          "⚠️표포함 태그가 N으로 잘못 붙어 있어 기존 67건 스윕에서 누락됐던 표. "
                          "PyMuPDF find_tables()로 원문 재확인 후 표포함=Y로 정정하고 Markdown 표로 변환.").strip()
        row[12].value = "검수완료(수정)"
        break

wb.save(PATH)
print("done")
