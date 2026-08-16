from src.agents.drafts import combined_draft


def test_combined_draft_preserves_info_and_product_parts():
    draft = combined_draft(
        {
            "info_draft": "세액공제 한도를 설명합니다.",
            "product_draft": "상품 유형을 설명합니다.",
        }
    )

    assert "[정보 Agent 초안]" in draft
    assert "세액공제 한도" in draft
    assert "[상품 Agent 초안]" in draft
    assert "상품 유형" in draft
