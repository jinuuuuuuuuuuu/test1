import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\[파싱]투자설명서.xlsm"
OUT = r"C:\Users\kevin\pension-agent\prospectus_check\_verify_final.txt"

codes = [
    "KR514X450008", "KR510902511M", "KR510902773M", "KR510902777M", "KR515302022M",
    "KR516702010M", "KR518101002M", "KR518101012M", "KR518102001M",
    "KR555202013M", "KR5110501016",
    "KR5110601022", "KR5111420047", "KR5111450067",
    "KR5113420012", "KR5113420013", "KR5113420015",
]

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["투자설명서_파싱"]

with io.open(OUT, "w", encoding="utf-8") as f:
    for code in codes:
        f.write(f"\n===== {code} =====\n")
        for row in ws.iter_rows(min_row=6):
            if row[2].value == code:
                cls = row[11].value
                f.write(
                    f"class={cls} | 수익률(1/3/후)={row[16].value}/{row[17].value}/{row[18].value} | "
                    f"변동성(1/3/후)={row[19].value}/{row[20].value}/{row[21].value} | "
                    f"최초설정일={row[34].value} | 매입={row[28].value!r} | 환매={row[29].value!r} | "
                    f"지급={row[30].value!r} | 수수료={row[31].value!r} | 과세={row[32].value!r} | "
                    f"상태={row[37].value}\n"
                )
