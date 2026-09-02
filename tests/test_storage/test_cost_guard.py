import csv
import json
import sqlite3

import pytest

from src.storage.cost_guard_loader import load_fund_class_pension_csv
from src.storage.queries import (
    choose_pension_cost_metric,
    find_lower_cost_pension_class,
)
from src.storage.schema import connect


def _seed_master(db_path):
    conn = connect(str(db_path))
    with conn:
        conn.execute(
            "INSERT INTO fund_master (product_code, source_file, fund_name) VALUES (?, ?, ?)",
            ("KR000", "R2_KR000.pdf", "테스트펀드"),
        )
    conn.close()


def _write_csv(path, rows):
    fieldnames = [
        "product_code",
        "class_code",
        "account_type",
        "channel",
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
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path, *, status="FROZEN_V1", version="cost_guard_v1"):
    path.write_text(
        json.dumps({
            "dataset_version": version,
            "dataset_status": status,
            "canonical_sha256": "test",
            "canonical_row_count": 3,
            "canonical_fund_count": 1,
            "standard_pair_count": 1,
            "channel_conditional_pair_count": 1,
            "review_override_count": 1,
            "p0_review_required_case_count": 0,
            "p0_review_required_field_row_count": 0,
            "p0_review_unresolved_field_row_count": 0,
        }),
        encoding="utf-8",
    )


def _row(class_code, channel, total, synthetic=None, *, account_type="퇴직연금/IRP", status="MATCH"):
    return {
        "product_code": "KR000",
        "class_code": class_code,
        "account_type": account_type,
        "channel": channel,
        "total_expense_ratio": total,
        "synthetic_total_expense_ratio": synthetic,
        "cost_3y_per_10m_krw": "",
        "total_expense_source_page": "3",
        "synthetic_expense_source_page": "3" if synthetic not in ("", None) else "",
        "cost_3y_source_page": "",
        "class_label": class_code,
        "source_file": "KR000.txt",
        "parse_status": "clean",
        "validation_status": status,
    }


def test_load_fund_class_pension_csv_is_idempotent(tmp_path):
    db_path = tmp_path / "prospectus.db"
    csv_path = tmp_path / "fund_class_pension.csv"
    _seed_master(db_path)
    _write_csv(csv_path, [_row("C-P2", "오프라인", "1.0", "1.1")])

    first = load_fund_class_pension_csv(csv_path, str(db_path))
    second = load_fund_class_pension_csv(csv_path, str(db_path))

    assert first.rows == 1
    assert second.rows == 1
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM fund_class_pension").fetchone()[0]
    assert count == 1
    manifest = conn.execute("SELECT dataset_status FROM cost_guard_dataset_manifest").fetchone()
    assert manifest[0] == "PROVISIONAL"
    conn.close()


def test_choose_pension_cost_metric_is_pair_level():
    current = {"total_expense_ratio": 1.0, "synthetic_total_expense_ratio": 1.1}
    assert choose_pension_cost_metric(
        current,
        {"total_expense_ratio": 0.8, "synthetic_total_expense_ratio": 0.9},
    ) == "synthetic_total_expense_ratio"
    assert choose_pension_cost_metric(
        current,
        {"total_expense_ratio": 0.8, "synthetic_total_expense_ratio": None},
    ) == "total_expense_ratio"
    assert choose_pension_cost_metric(
        {"total_expense_ratio": None, "synthetic_total_expense_ratio": 1.1},
        {"total_expense_ratio": 0.8, "synthetic_total_expense_ratio": None},
    ) is None


def test_find_lower_cost_prefers_standard_synthetic_candidate(tmp_path):
    db_path = tmp_path / "prospectus.db"
    csv_path = tmp_path / "fund_class_pension.csv"
    _seed_master(db_path)
    _write_manifest(csv_path.with_name("fund_class_pension_manifest.json"))
    _write_csv(csv_path, [
        _row("C-P2", "오프라인", "1.00", "1.10"),
        _row("C-P2E", "온라인", "0.80", "0.90"),
        _row("S-P2", "온라인슈퍼", "0.70", "0.70"),
    ])
    load_fund_class_pension_csv(csv_path, str(db_path))

    result = find_lower_cost_pension_class("KR000", "C-P2", "IRP", db_path=str(db_path))

    assert result.found is True
    assert result.target_class_code == "C-P2E"
    assert result.comparison_metric == "synthetic_total_expense_ratio"
    assert result.eligibility == "STANDARD"
    assert result.eligibility_type == "STANDARD"
    assert result.dataset_version == "cost_guard_v1"
    assert result.dataset_status == "FROZEN_V1"
    assert result.current_value == pytest.approx(1.10)
    assert result.target_value == pytest.approx(0.90)


def test_find_lower_cost_can_fall_back_to_total_for_pair_without_mixing(tmp_path):
    db_path = tmp_path / "prospectus.db"
    csv_path = tmp_path / "fund_class_pension.csv"
    _seed_master(db_path)
    _write_manifest(csv_path.with_name("fund_class_pension_manifest.json"))
    _write_csv(csv_path, [
        _row("C-P2", "오프라인", "1.00", "1.10"),
        _row("C-P2E", "온라인", "0.80", ""),
    ])
    load_fund_class_pension_csv(csv_path, str(db_path))

    result = find_lower_cost_pension_class("KR000", "C-P2", "IRP", db_path=str(db_path))

    assert result.found is True
    assert result.target_class_code == "C-P2E"
    assert result.comparison_metric == "total_expense_ratio"


def test_find_lower_cost_excludes_source_conflict_and_ambiguous_rows(tmp_path):
    db_path = tmp_path / "prospectus.db"
    csv_path = tmp_path / "fund_class_pension.csv"
    _seed_master(db_path)
    _write_manifest(csv_path.with_name("fund_class_pension_manifest.json"))
    _write_csv(csv_path, [
        _row("C-P2", "오프라인", "1.00", "1.10"),
        _row("C-P2E", "온라인", "0.80", "0.90", status="SOURCE_CONFLICT"),
        _row("S-P2", "온라인슈퍼", "0.70", "0.70", status="AMBIGUOUS"),
    ])
    load_fund_class_pension_csv(csv_path, str(db_path))

    result = find_lower_cost_pension_class("KR000", "C-P2", "IRP", db_path=str(db_path))

    assert result.found is False
    assert result.reason == "NO_LOWER_COST_CLASS"


def test_find_lower_cost_blocks_provisional_dataset_by_default(tmp_path):
    db_path = tmp_path / "prospectus.db"
    csv_path = tmp_path / "fund_class_pension.csv"
    _seed_master(db_path)
    _write_manifest(csv_path.with_name("fund_class_pension_manifest.json"), status="PROVISIONAL", version="cost_guard_provisional")
    _write_csv(csv_path, [
        _row("C-P2", "오프라인", "1.00", "1.10"),
        _row("C-P2E", "온라인", "0.80", "0.90"),
    ])
    load_fund_class_pension_csv(csv_path, str(db_path))

    blocked = find_lower_cost_pension_class("KR000", "C-P2", "IRP", db_path=str(db_path))
    smoke = find_lower_cost_pension_class(
        "KR000",
        "C-P2",
        "IRP",
        require_frozen=False,
        db_path=str(db_path),
    )

    assert blocked.found is False
    assert blocked.reason == "DATASET_NOT_FROZEN"
    assert blocked.dataset_status == "PROVISIONAL"
    assert smoke.found is True
