from src.agents.grounding import GroundingResult, _numeric_grounding_issues


def test_grounding_result_accepts_string_list_fields():
    result = GroundingResult(
        grounded=False,
        issues="근거 부족",
        premise_issues="",
        requirements_met=False,
        missing_requirements="상품 추천",
    )

    assert result.issues == ["근거 부족"]
    assert result.premise_issues == []
    assert result.missing_requirements == ["상품 추천"]


def test_numeric_grounding_flags_numbers_without_evidence():
    issues = _numeric_grounding_issues("세액공제는 16.5%이고 한도는 900만원입니다.", "(근거 없음)")

    assert issues


def test_numeric_grounding_allows_numbers_from_user_profile_evidence():
    issues = _numeric_grounding_issues(
        "현재 조건은 월 30만원, 20년 이상으로 정리됩니다.\n1. TDF",
        "사용자: 월 30만원 가능해\nrecommendation_profile={\"investment_horizon\": \"20년 이상\"}",
    )

    assert issues == []
