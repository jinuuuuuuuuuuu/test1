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


# ── doc54 (전반적으로 정확 — 텍스트 기반 PDF) ────────────────────────────
row = find_row("doc54_chunk01")
row[8].value = row[8].value.replace(
    "타자산 플랜(타자산 안내들)도 MP 구독 서비스 신청이 가능합니다.",
    "다자산 플랜(타자산 언번들 등)도 MP 구독 서비스 신청이 가능합니다."
)
row[11].value = "원문(p.4) 대조 중 용어 왜곡 정정 — '타자산 플랜(타자산 안내들)' → '다자산 플랜(타자산 언번들 등)' (doc34의 '언번들계약'과 동일 개념)"
row[12].value = "검수완료(수정)"

for cid in ["doc54_chunk02", "doc54_chunk03", "doc54_chunk04", "doc54_chunk05",
            "doc54_chunk06", "doc54_chunk07", "doc54_chunk08", "doc54_chunk09",
            "doc54_chunk10", "doc54_chunk11"]:
    set_status(cid, "검수완료")

wb.save(PATH)
print("Saved.")
