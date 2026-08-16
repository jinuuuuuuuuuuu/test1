"""Helpers for assembling agent drafts before verification/generation."""

from __future__ import annotations

from src.agents.state import PensionAgentState


def combined_draft(state: PensionAgentState) -> str:
    """Return all available agent drafts without dropping complex-answer parts."""
    parts = []
    info_draft = state.get("info_draft")
    product_draft = state.get("product_draft")

    if info_draft:
        parts.append(f"[정보 Agent 초안]\n{info_draft}")
    if product_draft:
        parts.append(f"[상품 Agent 초안]\n{product_draft}")
    return "\n\n".join(parts)
