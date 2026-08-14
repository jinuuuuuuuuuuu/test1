import io
import os
import glob
import fitz

BASE = r"C:\Users\kevin\pension-agent\data\raw\prospectus"
subdir = [n for n in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, n))][0]
SRC = glob.glob(os.path.join(BASE, subdir, "**", "*KR510902511M*.pdf"), recursive=True)[0]
OUT = r"C:\Users\kevin\pension-agent\docs_check\_debug_kr510902511m.txt"

doc = fitz.open(SRC)

with io.open(OUT, "w", encoding="utf-8") as f:
    for pno in range(doc.page_count):
        text = doc[pno].get_text()
        if "연평균수익률" in text or "운용실적" in text:
            f.write(f"===== page {pno+1} (text has marker) =====\n")
            tabs = doc[pno].find_tables()
            f.write(f"tables found: {len(tabs.tables)}\n")
            for ti, t in enumerate(tabs.tables):
                grid = t.extract()
                f.write(f"--- table{ti+1}: {len(grid)}x{len(grid[0]) if grid else 0} ---\n")
                for row in grid[:6]:
                    f.write(" | ".join(str(c) for c in row) + "\n")
                f.write("\n")
doc.close()
print("done")
