"""Deterministic product/class resolver shared by Product Agent and Guardians."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from src.storage.queries import DEFAULT_DB_PATH, normalize_pension_account_type


def _normalize_name(text: str | None) -> str:
    return re.sub(r"[\s·,()\[\]{}]+", "", text or "").lower()


def normalize_class_code(value: Any) -> str:
    """Normalize natural class text into the canonical Cost Guard class code."""
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", "", text).upper()
    text = re.sub(r"(클래스|CLASS)$", "", text)
    return text.split("(")[0]


def extract_class_code(text: str | None) -> str | None:
    raw = text or ""
    upper = raw.upper()

    match = re.search(r"(?<![A-Z0-9])([A-Z](?:-[A-Z0-9]{1,5})?)\s*(?:클래스|CLASS)", upper)
    if match:
        return normalize_class_code(match.group(1))

    match = re.search(r"(?<![A-Z0-9])([A-Z]-[A-Z0-9]{1,5})(?![A-Z0-9])", upper)
    if match:
        return normalize_class_code(match.group(1))

    match = re.search(r"\(([^()]*)\)\s*$", raw)
    if match:
        inner = match.group(1).strip()
        if re.fullmatch(r"[A-Za-z]-?[A-Za-z0-9]{0,5}", inner):
            return normalize_class_code(inner)
    return None


def _extract_product_code(text: str | None) -> str | None:
    match = re.search(r"KR[0-9A-Z]+", (text or "").upper())
    return match.group(0) if match else None


def _explicit_account_type(text: str | None) -> str | None:
    raw = text or ""
    upper = raw.upper()
    if "연금저축" in raw:
        return "연금저축"
    for token in ("IRP", "DC", "DB"):
        if token in upper:
            return token
    return None


def _scope_from_product_text(text: str | None) -> str | None:
    raw = text or ""
    if "퇴직플랜" in raw or "퇴직연금" in raw:
        return "퇴직연금"
    if "연금저축" in raw:
        return "연금저축"
    return None


def _scope_to_cost_account_type(scope: str | None) -> str | None:
    if scope == "퇴직연금":
        return "퇴직연금/IRP"
    if scope == "연금저축":
        return "연금저축"
    return None


def _fetch_master_rows(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT product_code, fund_name, source_file
            FROM fund_master
            ORDER BY length(fund_name) DESC, product_code DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _matching_products(question: str, db_path: str) -> list[dict]:
    explicit_code = _extract_product_code(question)
    rows = _fetch_master_rows(db_path)
    if explicit_code:
        return [row for row in rows if row["product_code"] == explicit_code]

    normalized_question = _normalize_name(question)
    matches = [row for row in rows if _normalize_name(row.get("fund_name")) in normalized_question]
    if matches:
        best_len = len(_normalize_name(matches[0].get("fund_name")))
        return [row for row in matches if len(_normalize_name(row.get("fund_name"))) == best_len]
    return []


def resolve_product(question: str, db_path: str = DEFAULT_DB_PATH) -> dict:
    """Resolve product name/code, class code, and pension account scope from a question.

    The resolver is intentionally deterministic: product/class identity is data lookup and
    normalization work, not something the LLM should infer ad hoc.
    """
    matches = _matching_products(question, db_path)
    class_code = extract_class_code(question)
    explicit_account = _explicit_account_type(question)
    product_scope = None
    if matches:
        product_scope = _scope_from_product_text(matches[0].get("fund_name"))
    product_scope = product_scope or _scope_from_product_text(question)
    cost_account_type = (
        normalize_pension_account_type(explicit_account)
        if explicit_account
        else _scope_to_cost_account_type(product_scope)
    )

    if not matches:
        return {
            "resolved": False,
            "product_name": "",
            "product_code": "",
            "product_codes": [],
            "class_code": class_code,
            "account_type": explicit_account,
            "account_scope": product_scope,
            "cost_account_type": cost_account_type,
            "account_type_source": "explicit" if explicit_account else "none",
            "confidence": "no_match",
            "candidates": [],
        }

    candidates = [
        {
            "product_code": row["product_code"],
            "product_name": row.get("fund_name") or "",
            "source_file": row.get("source_file") or "",
        }
        for row in matches
    ]
    account_source = "explicit" if explicit_account else "product_scope" if cost_account_type else "none"
    return {
        "resolved": True,
        "product_name": candidates[0]["product_name"],
        "product_code": candidates[0]["product_code"],
        "product_codes": [candidate["product_code"] for candidate in candidates],
        "class_code": class_code,
        "account_type": explicit_account,
        "account_scope": product_scope,
        "cost_account_type": cost_account_type,
        "account_type_source": account_source,
        "confidence": "exact_code" if _extract_product_code(question) else "exact_name",
        "candidates": candidates,
    }
