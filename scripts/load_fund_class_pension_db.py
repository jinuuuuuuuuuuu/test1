"""fund_class_pension canonical CSV를 prospectus.db에 적재한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.storage.cost_guard_loader import load_fund_class_pension_csv

DEFAULT_CSV = REPO_ROOT / "data" / "processed" / "fund_class_pension.csv"
DEFAULT_DB = REPO_ROOT / "data" / "processed" / "prospectus.db"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    stats = load_fund_class_pension_csv(args.csv, str(args.db), manifest_path=args.manifest)
    print(f"fund_class_pension 적재 완료: {stats.rows}건")
    print(f"dataset: {stats.dataset_status} ({stats.dataset_version})")
    if stats.skipped_rows:
        print(f"누락 필수값으로 스킵: {stats.skipped_rows}건")


if __name__ == "__main__":
    main()
