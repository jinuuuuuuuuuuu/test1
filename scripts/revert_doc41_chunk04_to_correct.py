import openpyxl

PATH = r"C:\Users\kevin\Downloads\docs 수정.xlsm"

wb = openpyxl.load_workbook(PATH, keep_vba=True)
ws = wb["파싱 결과"]

text = ("유의할 점\n"
        "IRP에만 900만원 납입해도 세액공제 효과는 같은데 연금저축, IRP에 나눠서 입금하는 이유는 뭘까? 혹시 모를 중도인출 "
        "때문이다. 연금수령 전까지 인출하지 않는 것이 가장 절세되지만 워낙 20~30년 이상 장기로 운용하다보니 예기치 않게 "
        "일부 금액이 필요할 수 있다. 연금저축펀드는 부분 인출이 자유롭다. 원할 때 필요한 금액을 인출할 수 있다. 물론 "
        "인출금액이 과세재원이면 16.5% 기타소득세가 적용된다. 그러나 남아 있는 금액은 계속 운용하면서 절세 혜택을 누릴 수 "
        "있다. 반면 IRP는 무주택자의 주택 구입 등 법정 사유를 충족해야 부분 인출이 가능해 까다롭다. IRP로만 운용하다 일부 "
        "금액이 필요한데 법정사유에 해당되지 않으면 전체를 해지해야 되서 불이익이 크다.\n")

for row in ws.iter_rows(min_row=6):
    if row[7].value == "doc41_chunk04":
        row[8].value = text
        row[9].value = "N"
        row[10].value = None
        row[11].value = ("⚠️doc41_chunk03과 동일한 표(연금계좌 세액공제 최대 절세액)가 청크 경계에서 중복 포함돼 있던 것을 "
                          "확인. 팀원의 [파싱]renamed폴더 (1).xlsm '파싱 결과_검수본' 기준에 맞춰 중복 표를 삭제하고 "
                          "표포함=N인 순수 서술 텍스트로 정정(표는 chunk03에만 존재).")
        row[12].value = "검수완료(수정)"
        break

wb.save(PATH)
print("done")
