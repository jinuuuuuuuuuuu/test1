import io
import openpyxl

PATH = r"C:\Users\kevin\Downloads\[파싱]투자설명서 (2).xlsm"
OUT = r"C:\Users\kevin\pension-agent\docs_check\_prospectus2_kr515_check.txt"

wb = openpyxl.load_workbook(PATH, data_only=True)
ws = wb["투자설명서_파싱_검수본"]

with io.open(OUT, "w", encoding="utf-8") as f:
    for r in (14, 15, 158):
        vals = [ws.cell(row=r, column=c).value for c in range(1, 39)]
        f.write(f"row{r}: code={vals[2]} class={vals[11]} 1y={vals[16]} 3y={vals[17]} since={vals[18]} 최초설정일={vals[33]} 검수상태={vals[37]}\n\n")
