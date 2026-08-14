import openpyxl

PATH = r"C:\Users\kevin\Downloads\[파싱]renamed폴더.xlsm"

wb = openpyxl.load_workbook(PATH, keep_vba=True)
ws = wb["파싱 결과"]


def find_row(chunk_id):
    for row in ws.iter_rows(min_row=6):
        if row[7].value == chunk_id:
            return row
    raise ValueError(f"chunk_id not found: {chunk_id}")


def set_status(chunk_id, status):
    find_row(chunk_id)[12].value = status


# ── doc56 ─────────────────────────────────────────────────────────────
set_status("doc56_chunk01", "검수완료")

row = find_row("doc56_chunk02")
row[8].value = row[8].value.replace("재원운용형펀드", "채권혼합형펀드")
row[10].value = row[10].value.replace("재원운용형펀드", "채권혼합형펀드")
row[11].value = "원문(p.1) 대조 중 상품명 오기 정정 — '재원운용형펀드'(비표준 용어) → '채권혼합형펀드'(원문 표기, 실존 상품유형)"
row[12].value = "검수완료(수정)"

row = find_row("doc56_chunk03")
row[8].value = row[8].value.replace(
    "위험자산 | 투자금지상품 - 비상장 주식 - 전환사채, 신주인수권부 사채, 교환사채 및 후순위채권 - 투자 비적격 등급의 증권/채권 - ELS/DLS(최대 손실률 40%초과) 등 | 투자금지",
    "위험자산 | 투자금지상품 - 상장 주식 - 전환사채, 신주인수권부 사채, 교환사채 및 후순위채권 - 투자 비적격 등급의 증권/채권 - ELS/DLS(최대 손실률 40%초과) 등 | 투자금지"
)
row[10].value = row[10].value.replace(
    "위험자산 | 투자금지상품 - 비상장 주식 - 전환사채, 신주인수권부 사채, 교환사채 및 후순위채권 - 투자 비적격 등급의 증권/채권 - ELS/DLS(최대 손실률 40%초과) 등 | 투자금지",
    "위험자산 | 투자금지상품 - 상장 주식 - 전환사채, 신주인수권부 사채, 교환사채 및 후순위채권 - 투자 비적격 등급의 증권/채권 - ELS/DLS(최대 손실률 40%초과) 등 | 투자금지"
)
row[11].value = (
    "⚠️ 원문(p.2) 대조 중 중대한 오류 발견 — DC/IRP 투자금지상품이 '비상장 주식'으로 표기돼 있었으나 원문은 '상장 주식'. "
    "DC/IRP는 국내 상장주식에 직접투자할 수 없다는 핵심 규칙(DB만 가능)인데, 반대로 '비상장 주식'만 금지되는 것처럼 표기돼 있어 "
    "'DC/IRP도 상장주식 직접투자 가능'으로 오독될 위험이 있었음. 세율/한도 수치급으로 중요한 정정"
)
row[12].value = "검수완료(수정)"

# ── doc58 ─────────────────────────────────────────────────────────────
row = find_row("doc58_chunk02")
row[8].value = row[8].value.replace(
    "지분증권(주식) | 국내 상장주식",
    "지분증권(주식) | 국내 상장주식 (DB만 가능)"
)
row[10].value = row[10].value.replace(
    "지분증권(주식) | 국내 상장주식",
    "지분증권(주식) | 국내 상장주식 (DB만 가능)"
)
row[11].value = (
    "⚠️ 원문(p.2) 대조 중 중요 누락 발견 — '국내 상장주식' 항목에만 '(DB만 가능)' 표시가 빠져 있었음 (같은 표의 '사모펀드', '증권예탁증권(DR)' "
    "항목에는 정상적으로 표시돼 있어 이 항목만 선택적으로 누락된 것으로 보임). doc56에서 발견한 것과 동일한 유형의 핵심 규칙 누락이라 함께 정정"
)
row[12].value = "검수완료(수정)"

set_status("doc58_chunk03", "검수완료")
set_status("doc58_chunk04", "검수완료")

wb.save(PATH)
print("Saved.")
