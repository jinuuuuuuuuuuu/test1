from src.agents.product_resolver import extract_class_code, normalize_class_code, resolve_product


def test_normalize_class_code_accepts_natural_class_text():
    assert normalize_class_code("C클래스") == "C"
    assert normalize_class_code("C 클래스") == "C"
    assert normalize_class_code("C-P2e") == "C-P2E"
    assert normalize_class_code("c-p2e 클래스") == "C-P2E"


def test_extract_class_code_from_natural_question():
    assert extract_class_code("미래에셋퇴직플랜증권자투자신탁1호(주식) C클래스 특징 알려줘") == "C"
    assert extract_class_code("KR000 C-P2e 클래스 특징 알려줘") == "C-P2E"


def test_resolve_product_name_class_and_retirement_scope():
    resolved = resolve_product("미래에셋퇴직플랜증권자투자신탁1호(주식) C클래스 특징 알려줘.")

    assert resolved["resolved"] is True
    assert resolved["product_name"] == "미래에셋퇴직플랜증권자투자신탁1호(주식)"
    assert resolved["product_codes"]
    assert resolved["class_code"] == "C"
    assert resolved["account_type"] is None
    assert resolved["account_scope"] == "퇴직연금"
    assert resolved["cost_account_type"] == "퇴직연금/IRP"
    assert resolved["account_type_source"] == "product_scope"


def test_resolver_does_not_match_unknown_product():
    resolved = resolve_product("존재하지않는퇴직플랜999 C클래스 특징 알려줘.")

    assert resolved["resolved"] is False
    assert resolved["product_codes"] == []
