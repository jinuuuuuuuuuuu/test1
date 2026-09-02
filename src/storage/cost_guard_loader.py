"""Cost Guard용 연금 판매클래스 canonical CSV를 SQLite에 적재한다."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.storage.schema import connect

STANDARD_CHANNELS = {"오프라인", "온라인"}

FIELDNAMES = [
    "product_code",
    "class_code",
    "account_type",
    "channel",
    "eligibility_type",
    "total_expense_ratio",
    "synthetic_total_expense_ratio",
    "cost_3y_per_10m_krw",
    "total_expense_source_page",
    "synthetic_expense_source_page",
    "cost_3y_source_page",
    "class_label",
    "source_file",
    "parse_status",
    "validation_status",
    "validation_status_before_review",
    "review_source_page",
    "review_note",
    "dataset_version",
    "dataset_status",
]

NUMERIC_FIELDS = {
    "total_expense_ratio",
    "synthetic_total_expense_ratio",
    "cost_3y_per_10m_krw",
}


@dataclass
class CostGuardLoadStats:
    rows: int = 0
    skipped_rows: int = 0
    dataset_version: str = ""
    dataset_status: str = ""


def _clean(value: Any, *, numeric: bool = False) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if numeric:
        try:
            return float(text)
        except ValueError:
            return None
    return text


def _eligibility_type(channel: str | None) -> str:
    return "STANDARD" if channel in STANDARD_CHANNELS else "CHANNEL_CONDITIONAL"


def _default_manifest() -> dict[str, Any]:
    return {
        "dataset_version": "cost_guard_provisional",
        "dataset_status": "PROVISIONAL",
    }


def read_cost_guard_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return _default_manifest()
    manifest_path = Path(path)
    if not manifest_path.exists():
        return _default_manifest()
    return {**_default_manifest(), **json.loads(manifest_path.read_text(encoding="utf-8"))}


def _row_values(row: dict[str, Any], manifest: dict[str, Any]) -> list[Any] | None:
    product_code = _clean(row.get("product_code"))
    class_code = _clean(row.get("class_code"))
    account_type = _clean(row.get("account_type"))
    channel = _clean(row.get("channel"))
    source_file = _clean(row.get("source_file"))
    if not all((product_code, class_code, account_type, channel, source_file)):
        return None

    values = {
        field: _clean(row.get(field), numeric=field in NUMERIC_FIELDS)
        for field in FIELDNAMES
    }
    values["eligibility_type"] = values["eligibility_type"] or _eligibility_type(channel)
    values["dataset_version"] = manifest.get("dataset_version")
    values["dataset_status"] = manifest.get("dataset_status")
    return [values[field] for field in FIELDNAMES]


def _insert_manifest(conn, manifest: dict[str, Any]) -> None:
    fields = [
        "dataset_version",
        "dataset_status",
        "canonical_sha256",
        "generated_at",
        "canonical_row_count",
        "canonical_fund_count",
        "standard_pair_count",
        "channel_conditional_pair_count",
        "review_override_count",
        "p0_review_required_case_count",
        "p0_review_required_field_row_count",
        "p0_review_unresolved_field_row_count",
    ]
    conn.execute("DELETE FROM cost_guard_dataset_manifest")
    conn.execute(
        f"INSERT INTO cost_guard_dataset_manifest (id, {', '.join(fields)}) "
        f"VALUES (1, {', '.join(['?'] * len(fields))})",
        [manifest.get(field) for field in fields],
    )


def load_fund_class_pension_csv(
    csv_path: str | Path,
    db_path: str,
    manifest_path: str | Path | None = None,
) -> CostGuardLoadStats:
    """canonical CSV를 fund_class_pension 테이블에 idempotent하게 적재한다."""
    stats = CostGuardLoadStats()
    csv_path = Path(csv_path)
    manifest_path = manifest_path or csv_path.with_name("fund_class_pension_manifest.json")
    manifest = read_cost_guard_manifest(manifest_path)
    stats.dataset_version = manifest.get("dataset_version") or ""
    stats.dataset_status = manifest.get("dataset_status") or ""
    conn = connect(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM fund_class_pension")
            _insert_manifest(conn, manifest)
            with csv_path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    values = _row_values(row, manifest)
                    if values is None:
                        stats.skipped_rows += 1
                        continue
                    conn.execute(
                        f"INSERT INTO fund_class_pension ({', '.join(FIELDNAMES)}) "
                        f"VALUES ({', '.join(['?'] * len(FIELDNAMES))})",
                        values,
                    )
                    stats.rows += 1
    finally:
        conn.close()
    return stats
