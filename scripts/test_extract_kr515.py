import os
import glob
import sys

sys.path.insert(0, r"C:\Users\kevin\pension-agent\src")
from parsing.prospectus_tables import extract_class_returns

BASE = r"C:\Users\kevin\pension-agent\data\raw\prospectus"
SRC = glob.glob(os.path.join(BASE, "**", "*KR515302022M*.pdf"), recursive=True)[0]

results = extract_class_returns(SRC)
for r in results:
    print(r.class_code, r.class_label, r.return_1y, r.return_3y, r.return_since_inception,
          r.volatility_1y, r.volatility_3y, r.volatility_since_inception)
print("total classes:", len(results))
