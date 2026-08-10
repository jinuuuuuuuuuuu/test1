import pytest
from src.rules.in_kind_transfer import check_transfer_eligibility, get_code_info, TRANSFER_BLOCK_CODES


def test_no_flags_is_eligible_but_flags_manual_review_directional_codes():
    r = check_transfer_eligibility()
    assert r.eligible is True
    assert r.blocking_codes == []
    # directional 코드(06, 17, 20)는 항상 안내되어야 함
    assert set(r.needs_manual_review_codes) == {"06", "17", "20"}


def test_mmf_blocks():
    r = check_transfer_eligibility(is_mmf=True)
    assert r.eligible is False
    assert "04" in r.blocking_codes


def test_default_option_product_blocks():
    r = check_transfer_eligibility(is_default_option_product=True)
    assert "23" in r.blocking_codes


def test_multiple_flags_collect_all_codes():
    r = check_transfer_eligibility(is_mmf=True, private_fund=True, seized_or_pledged=True)
    assert r.eligible is False
    assert r.blocking_codes == ["03", "04", "08"]


def test_savings_bank_limit_exceeded():
    r = check_transfer_eligibility(exceeds_savings_bank_protection_limit=True)
    assert "18" in r.blocking_codes


def test_get_code_info():
    info = get_code_info("23")
    assert info["name"] == "실물이전불가(디폴트옵션)"


def test_get_code_info_unknown_raises():
    with pytest.raises(KeyError):
        get_code_info("999")


def test_all_25_codes_plus_present():
    # 01~25가 모두 등록돼 있는지 (누락 방지용 회귀 테스트)
    expected = {f"{i:02d}" for i in range(1, 26)}
    assert expected.issubset(set(TRANSFER_BLOCK_CODES.keys()))
