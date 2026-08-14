import io
import os
import glob
import sys

sys.path.insert(0, r"C:\Users\kevin\pension-agent\src")
import parsing.prospectus_tables as pt

BASE = r"C:\Users\kevin\pension-agent\data\raw\prospectus"
subdir = [n for n in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, n))][0]
d = os.path.join(BASE, subdir)
SRC = glob.glob(os.path.join(d, "**", "*KR5111420047*.pdf"), recursive=True)[0]
OUT = r"C:\Users\kevin\pension-agent\docs_check\_debug_since_kr5111420047.txt"

import fitz

doc = fitz.open(SRC)
with io.open(OUT, "w", encoding="utf-8") as f:
    col_map = None
    done = False
    for pno in range(doc.page_count):
        if done:
            break
        page = doc[pno]
        tabs = page.find_tables()
        for t in tabs.tables:
            if done:
                break
            grid = t.extract()
            if not grid:
                continue
            for row in grid:
                if pt._row_is_annual_header(row):
                    f.write(f"p{pno+1}: ANNUAL HEADER -> done\n")
                    done = True
                    break
                cm = pt._detect_col_map(row)
                if cm is not None:
                    col_map = cm
                    f.write(f"p{pno+1}: HEADER row={row} -> col_map={cm}\n")
                    continue
                if col_map is not None:
                    idx = pt._find_label_index(row)
                    if idx is not None:
                        f.write(f"p{pno+1}: DATA row(len={len(row)})={row}\n")
doc.close()
print("done")
