import io
import os
import glob
import fitz

BASE = r"C:\Users\kevin\pension-agent\data\raw\prospectus"
SRC = glob.glob(os.path.join(BASE, "**", "*KR515302022M*.pdf"), recursive=True)[0]
OUT = r"C:\Users\kevin\pension-agent\docs_check\_find_tables_kr515.txt"

doc = fitz.open(SRC)

with io.open(OUT, "w", encoding="utf-8") as f:
    for pno in [30, 47, 48, 49, 50, 51]:  # 0-indexed: pages 31,48,49,50,51,52
        page = doc[pno]
        tabs = page.find_tables()
        f.write(f"===== page{pno+1}: {len(tabs.tables)} table(s) =====\n")
        for ti, t in enumerate(tabs.tables):
            grid = t.extract()
            f.write(f"--- table{ti+1}: {len(grid)} rows x {len(grid[0]) if grid else 0} cols ---\n")
            for row in grid:
                f.write(" | ".join((str(c) if c is not None else "None") for c in row) + "\n")
            f.write("\n")
doc.close()
print("done")
