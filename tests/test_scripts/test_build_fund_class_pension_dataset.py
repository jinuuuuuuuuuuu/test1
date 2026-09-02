import csv
import json
from pathlib import Path

from scripts.build_fund_class_pension_dataset import (
    GOLDEN_VIP,
    audit_summary,
    apply_review_overrides,
    build_manifest,
    build_audit_rows,
    choose_cost_metric,
    compare_parser_to_reference,
    coverage_summary,
    normalize_class_code,
    p0_review_template_rows,
    review_provenance_rows,
    review_template_rows,
    unresolved_p0_review_rows,
    validate_review_rows,
    validate_canonical_rows,
)


def test_normalize_class_code_preserves_meaningful_suffixes():
    assert normalize_class_code("C-Pe") == "C-PE"
    assert normalize_class_code(" C-P2e ") == "C-P2E"
    assert normalize_class_code("S-P2") != normalize_class_code("S-P")


def test_reference_incomplete_is_not_field_mismatch():
    parser_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "account_type": "연금저축",
            "channel": "오프라인",
            "total_expense_ratio": 1.23,
            "synthetic_total_expense_ratio": 1.24,
            "source_file": "KR000.txt",
            "class_label": "오프라인-개인연금",
        }
    ]
    reference_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "account_type": "연금저축",
            "channel": "오프라인",
            "total_expense_ratio": None,
            "source_file": "R2_KR000.pdf",
            "class_label": "C-P",
            "reference_status": "AUTO_CORRECTED",
        }
    ]

    [result] = compare_parser_to_reference(parser_rows, reference_rows)

    assert result["validation_status"] == "REFERENCE_INCOMPLETE"
    assert result["difference_note"] == "reference_total_expense_ratio_missing"


def test_coverage_counts_lower_cost_pairs_by_channel_kind():
    rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "account_type": "연금저축",
            "channel": "오프라인",
            "total_expense_ratio": 1.0,
            "synthetic_total_expense_ratio": 1.1,
            "source_file": "KR000.txt",
        },
        {
            "product_code": "KR000",
            "class_code": "C-PE",
            "account_type": "연금저축",
            "channel": "온라인",
            "total_expense_ratio": 0.8,
            "synthetic_total_expense_ratio": 0.9,
            "source_file": "KR000.txt",
        },
        {
            "product_code": "KR000",
            "class_code": "S-P",
            "account_type": "연금저축",
            "channel": "온라인슈퍼",
            "total_expense_ratio": 0.7,
            "synthetic_total_expense_ratio": 0.75,
            "source_file": "KR000.txt",
        },
    ]

    summary = coverage_summary(rows, [])

    assert summary["comparable_fund_count"] == 1
    assert summary["lower_cost_pair_count"] == 3
    assert summary["lower_cost_pair_by_kind"] == {
        "CHANNEL_CONDITIONAL": 2,
        "STANDARD": 1,
    }
    assert summary["lower_cost_pair_by_metric"] == {
        "synthetic_total_expense_ratio": 3,
    }


def test_pair_level_metric_selection_prefers_synthetic_without_blocking_total_pairs():
    current = {
        "total_expense_ratio": 1.0,
        "synthetic_total_expense_ratio": 1.1,
    }
    target_with_synthetic = {
        "total_expense_ratio": 0.9,
        "synthetic_total_expense_ratio": 1.0,
    }
    target_total_only = {
        "total_expense_ratio": 0.8,
        "synthetic_total_expense_ratio": None,
    }
    target_no_common_metric = {
        "total_expense_ratio": None,
        "synthetic_total_expense_ratio": None,
    }

    assert choose_cost_metric(current, target_with_synthetic) == "synthetic_total_expense_ratio"
    assert choose_cost_metric(current, target_total_only) == "total_expense_ratio"
    assert choose_cost_metric(current, target_no_common_metric) is None


def test_reference_value_matching_other_metric_is_semantic_not_field_mismatch():
    parser_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "account_type": "연금저축",
            "channel": "오프라인",
            "total_expense_ratio": 0.3324,
            "synthetic_total_expense_ratio": 0.3390,
            "source_file": "KR000.txt",
            "class_label": "오프라인-개인연금",
        }
    ]
    reference_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "account_type": "연금저축",
            "channel": "오프라인",
            "total_expense_ratio": 0.3324,
            "source_file": "R2_KR000.pdf",
            "class_label": "C-P",
            "reference_status": "AUTO_CORRECTED",
        }
    ]

    [result] = compare_parser_to_reference(parser_rows, reference_rows)

    assert result["validation_status"] == "METRIC_SEMANTIC_MISMATCH"
    assert result["reference_cost_metric"] == "total_expense_ratio"
    assert "reference_cost_metric=total_expense_ratio" in result["difference_note"]


def test_audit_classifies_missing_and_extra_causes():
    validation_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "validation_status": "MISSING_IN_PARSER",
            "reference_total_expense_ratio": "",
            "difference_note": "parser_missing",
        },
        {
            "product_code": "KR001",
            "class_code": "C-P",
            "validation_status": "EXTRA_IN_PARSER",
            "reference_total_expense_ratio": "",
            "difference_note": "reference_missing",
        },
    ]
    parser_rows = [{"product_code": "KR001", "class_code": "C-P"}]
    all_reference_rows = [
        {
            "product_code": "KR001",
            "class_code": "C-P",
            "cost_guard_usable": "N",
            "account_type": "연금저축",
        }
    ]

    audit_rows = build_audit_rows(validation_rows, parser_rows, all_reference_rows)
    summary = audit_summary(audit_rows)

    assert summary["audit_by_cause"]["reference_missing_expense_ratio"] == 1
    assert summary["audit_by_cause"]["reference_not_cost_guard_usable"] == 1


def test_review_template_keeps_each_mismatched_field_separate():
    [template_row_a, template_row_b] = review_template_rows([
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "validation_status": "FIELD_MISMATCH",
            "audit_cause": "same_metric_mismatch",
            "recommended_action": "원문 확인",
            "parser_account_type": "연금저축",
            "reference_account_type": "퇴직연금/IRP",
            "parser_channel": "온라인",
            "reference_channel": "온라인",
            "parser_total_expense_ratio": "0.3",
            "reference_total_expense_ratio": "",
            "difference_note": "account_type | reference_total_expense_ratio_missing",
        }
    ])

    assert {template_row_a["field"], template_row_b["field"]} == {
        "account_type",
        "total_expense_ratio",
    }


def test_p0_review_template_keeps_only_same_metric_and_channel_mismatches():
    rows = p0_review_template_rows([
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "validation_status": "FIELD_MISMATCH",
            "audit_cause": "same_metric_mismatch",
            "recommended_action": "원문 확인",
            "parser_account_type": "연금저축",
            "reference_account_type": "연금저축",
            "parser_channel": "온라인",
            "reference_channel": "온라인",
            "parser_total_expense_ratio": "0.3",
            "reference_total_expense_ratio": "0.4",
            "difference_note": "total_expense_ratio",
        },
        {
            "product_code": "KR001",
            "class_code": "C-P2E",
            "validation_status": "MISSING_IN_PARSER",
            "audit_cause": "parser_clean_fund_missing_class",
            "recommended_action": "원문 확인",
            "parser_total_expense_ratio": "",
            "reference_total_expense_ratio": "0.2",
            "difference_note": "parser_missing",
        },
    ])

    assert len(rows) == 1
    assert rows[0]["product_code"] == "KR000"


def test_unresolved_p0_review_rows_tracks_field_level_reviews():
    p0_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "field": "total_expense_ratio",
        },
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "field": "channel",
        },
    ]
    review_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "field": "total_expense_ratio",
            "review_status": "PARSER_CORRECT",
        }
    ]

    unresolved = unresolved_p0_review_rows(p0_rows, review_rows)

    assert len(unresolved) == 1
    assert unresolved[0]["field"] == "channel"


def test_manifest_stays_provisional_until_p0_is_resolved(tmp_path):
    canonical = tmp_path / "fund_class_pension.csv"
    canonical.write_text("product_code,class_code\nKR000,C-P\n", encoding="utf-8")
    coverage = {
        "canonical_row_count": 1,
        "canonical_fund_count": 1,
        "lower_cost_pair_by_kind": {"STANDARD": 1, "CHANNEL_CONDITIONAL": 0},
    }

    provisional = build_manifest(
        coverage,
        canonical_path=canonical,
        p0_case_count=1,
        p0_template=[{"product_code": "KR000", "class_code": "C-P", "field": "total_expense_ratio"}],
        unresolved_p0_rows=[{"product_code": "KR000", "class_code": "C-P", "field": "total_expense_ratio"}],
        review_rows=[],
        review_errors=[],
        validation_errors=[],
    )
    frozen = build_manifest(
        coverage,
        canonical_path=canonical,
        p0_case_count=1,
        p0_template=[{"product_code": "KR000", "class_code": "C-P", "field": "total_expense_ratio"}],
        unresolved_p0_rows=[],
        review_rows=[{"review_status": "PARSER_CORRECT"}],
        review_errors=[],
        validation_errors=[],
    )

    assert provisional["dataset_status"] == "PROVISIONAL"
    assert provisional["p0_review_unresolved_field_row_count"] == 1
    assert frozen["dataset_status"] == "FROZEN_V1"
    assert frozen["dataset_version"] == "cost_guard_v1"


def test_review_override_can_restore_field_mismatch_row():
    parser_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "account_type": "연금저축",
            "channel": "오프라인",
            "total_expense_ratio": 0.33,
            "synthetic_total_expense_ratio": None,
            "source_file": "KR000.txt",
            "class_label": "오프라인-개인연금",
            "parse_status": "clean",
        }
    ]
    validation_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "validation_status": "FIELD_MISMATCH",
            "parser_account_type": "연금저축",
            "reference_account_type": "연금저축",
            "parser_channel": "오프라인",
            "reference_channel": "오프라인",
            "parser_total_expense_ratio": "0.33",
            "reference_total_expense_ratio": "0.34",
        }
    ]
    review_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "field": "total_expense_ratio",
            "parser_value": "0.33",
            "reference_value": "0.34",
            "review_status": "REFERENCE_CORRECT",
            "reviewed_value": "",
            "source_page": "3",
            "note": "원문 총보수·비용 0.34%",
        }
    ]

    canonical = apply_review_overrides([], parser_rows, validation_rows, review_rows)

    assert canonical[0]["total_expense_ratio"] == 0.34
    assert canonical[0]["validation_status"] == "REVIEWED_REFERENCE_CORRECT"
    assert canonical[0]["validation_status_before_review"] == "FIELD_MISMATCH"


def test_review_provenance_reports_restored_rows():
    before = []
    after = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "total_expense_ratio": 0.34,
            "validation_status": "REVIEWED_REFERENCE_CORRECT",
        }
    ]
    review_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "field": "total_expense_ratio",
            "parser_value": "0.33",
            "reference_value": "0.34",
            "review_status": "REFERENCE_CORRECT",
            "reviewed_value": "",
            "source_page": "3",
            "note": "원문 총보수·비용 0.34%",
        }
    ]

    [row] = review_provenance_rows(before, after, review_rows)

    assert row["canonical_action"] == "RESTORED_TO_CANONICAL"
    assert row["value_used"] == 0.34
    assert row["validation_status_after_review"] == "REVIEWED_REFERENCE_CORRECT"


def test_review_validation_requires_both_wrong_source_and_value():
    errors = validate_review_rows([
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "field": "total_expense_ratio",
            "parser_value": "0.33",
            "reference_value": "0.34",
            "review_status": "BOTH_WRONG",
            "reviewed_value": "",
            "source_page": "",
        }
    ])

    assert any("reviewed_value" in error for error in errors)
    assert any("source_page" in error for error in errors)


def test_review_validation_requires_source_conflict_page_and_note():
    errors = validate_review_rows([
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "field": "total_expense_ratio",
            "parser_value": "0.33",
            "reference_value": "0.34",
            "review_status": "SOURCE_CONFLICT",
            "reviewed_value": "",
            "source_page": "",
            "note": "",
        }
    ])

    assert any("source_page" in error for error in errors)
    assert any("note" in error for error in errors)


def test_source_conflict_review_excludes_row_from_canonical():
    canonical_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "account_type": "연금저축",
            "channel": "오프라인",
            "total_expense_ratio": 0.33,
            "synthetic_total_expense_ratio": None,
            "source_file": "KR000.txt",
        }
    ]
    review_rows = [
        {
            "product_code": "KR000",
            "class_code": "C-P",
            "field": "total_expense_ratio",
            "parser_value": "0.33",
            "reference_value": "0.34",
            "review_status": "SOURCE_CONFLICT",
            "reviewed_value": "",
            "source_page": "3,25",
            "note": "요약표와 상세표의 총보수·비용 값이 다름",
        }
    ]

    result = apply_review_overrides(canonical_rows, [], [], review_rows)

    assert result == []


def test_generated_canonical_artifact_keeps_vip_golden_case():
    path = Path("data/processed/fund_class_pension.csv")
    assert path.exists()

    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    errors = validate_canonical_rows(rows)
    assert errors == []

    vip = {
        row["class_code"]: float(row["synthetic_total_expense_ratio"] or row["total_expense_ratio"])
        for row in rows
        if row["product_code"] == "KR514X450008"
    }
    assert vip == GOLDEN_VIP


def test_generated_coverage_artifact_matches_canonical_rows():
    csv_path = Path("data/processed/fund_class_pension.csv")
    coverage_path = Path("data/processed/fund_class_pension_coverage.json")
    audit_path = Path("data/processed/fund_class_pension_audit.csv")
    assert csv_path.exists()
    assert coverage_path.exists()
    assert audit_path.exists()

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))

    assert coverage["canonical_row_count"] == len(rows)
    assert coverage["canonical_fund_count"] == len({row["product_code"] for row in rows})
    assert coverage["lower_cost_pair_count"] > 0
    assert coverage["audit"]["audit_row_count"] > 0
    assert coverage["audit"]["audit_by_status_cause"]["FIELD_MISMATCH"]
