import io
import os
import glob
import fitz

BASE = r"C:\Users\kevin\pension-agent\data\raw\prospectus"
matches = glob.glob(os.path.join(BASE, "**", "*KR515302022M*.pdf"), recursive=True)
SRC = matches[0]
OUT = r"C:\Users\kevin\pension-agent\docs_check\_kr515302022m_part3.txt"

doc = fitz.open(SRC)
print("total pages:", doc.page_count)

with io.open(OUT, "w", encoding="utf-8") as f:
    for i in range(doc.page_count):
        text = doc[i].get_text()
        if "연평균수익률" in text or "운용실적" in text or ("C1" in text and "%" in text):
            f.write(f"\n===== PAGE {i+1} =====\n")
            f.write(text)
doc.close()
print("done")
