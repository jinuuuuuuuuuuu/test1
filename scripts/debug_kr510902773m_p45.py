import io
import os
import glob
import fitz

BASE = r"C:\Users\kevin\pension-agent\data\raw\prospectus"
subdir = [n for n in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, n))][0]
d = os.path.join(BASE, subdir)
SRC = glob.glob(os.path.join(d, "**", "*KR510902773M*.pdf"), recursive=True)[0]
OUT = r"C:\Users\kevin\pension-agent\docs_check\_debug_kr510902773m_p45.txt"

doc = fitz.open(SRC)

with io.open(OUT, "w", encoding="utf-8") as f:
    for pno in [44]:  # 0-idx page 45
        f.write(f"===== page {pno+1} raw text =====\n")
        f.write(doc[pno].get_text())
        f.write("\n\n===== page {} tables =====\n".format(pno+1))
        tabs = doc[pno].find_tables()
        f.write(f"tables found: {len(tabs.tables)}\n")
        for ti, t in enumerate(tabs.tables):
            grid = t.extract()
            f.write(f"--- table{ti+1}: {len(grid)}x{len(grid[0]) if grid else 0} ---\n")
            for row in grid[:10]:
                f.write(" | ".join(str(c) for c in row) + "\n")
            f.write("\n")
doc.close()
print("done")
