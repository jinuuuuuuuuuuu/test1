from src.storage.queries import get_fund_detail, search_funds
from src.storage.schema import connect


def _seed(db_path):
    conn = connect(str(db_path))
    with conn:
        conn.execute(
            "INSERT INTO fund_master (product_code, source_file, fund_name, manager_name, risk_grade, fund_category) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("KR000000001", "a.pdf", "안전채권펀드", "가나자산운용", "5등급[낮은 위험]", "증권(채권형)"),
        )
        conn.execute(
            "INSERT INTO fund_master (product_code, source_file, fund_name, manager_name, risk_grade, fund_category) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("KR000000002", "b.pdf", "공격주식펀드", "다라자산운용", "1등급[매우 높은 위험]", "증권(주식형)"),
        )
        conn.execute(
            "INSERT INTO fund_class (product_code, class_name, sales_channel, total_expense_ratio, return_1y) "
            "VALUES (?, ?, ?, ?, ?)",
            ("KR000000001", "C1", "온라인", 0.5, 3.2),
        )
        conn.execute(
            "INSERT INTO fund_class (product_code, class_name, sales_channel, total_expense_ratio, return_1y) "
            "VALUES (?, ?, ?, ?, ?)",
            ("KR000000002", "A", "오프라인", 1.8, 25.0),
        )
    conn.close()


def test_search_funds_filters_by_risk_grade_range(tmp_path):
    db_path = tmp_path / "test.db"
    _seed(db_path)

    safe_only = search_funds(risk_grade_min=4, db_path=str(db_path))
    assert {r.product_code for r in safe_only} == {"KR000000001"}

    risky_only = search_funds(risk_grade_max=2, db_path=str(db_path))
    assert {r.product_code for r in risky_only} == {"KR000000002"}


def test_search_funds_filters_by_expense_ratio_and_return(tmp_path):
    db_path = tmp_path / "test.db"
    _seed(db_path)

    results = search_funds(max_expense_ratio=1.0, db_path=str(db_path))
    assert {r.product_code for r in results} == {"KR000000001"}

    results2 = search_funds(min_return_1y=10.0, db_path=str(db_path))
    assert {r.product_code for r in results2} == {"KR000000002"}


def test_search_funds_keyword_matches_fund_name_or_category(tmp_path):
    db_path = tmp_path / "test.db"
    _seed(db_path)

    results = search_funds(keyword="채권", db_path=str(db_path))
    assert {r.product_code for r in results} == {"KR000000001"}


def test_search_funds_keyword_ignores_spacing_differences(tmp_path):
    # 실데이터 사례: "미래에셋솔로몬단기국공채증권자투자신탁1호"처럼 펀드명에 띄어쓰기가
    # 없어도 "솔로몬 국공채" 같은 띄어 쓴 검색어(토큰 AND)로 찾을 수 있어야 한다.
    db_path = tmp_path / "test.db"
    conn = connect(str(db_path))
    with conn:
        conn.execute(
            "INSERT INTO fund_master (product_code, source_file, fund_name, manager_name, risk_grade, fund_category) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("KR000000003", "c.pdf", "미래에셋솔로몬단기국공채증권자투자신탁1호(채권)", "미래에셋자산운용", "5등급[낮은 위험]", "증권(채권형)"),
        )
        conn.execute(
            "INSERT INTO fund_class (product_code, class_name, sales_channel, total_expense_ratio, return_1y) "
            "VALUES (?, ?, ?, ?, ?)",
            ("KR000000003", "C-P2", "온라인", 0.3, 2.9),
        )
    conn.close()

    assert {r.product_code for r in search_funds(keyword="솔로몬 국공채", db_path=str(db_path))} == {"KR000000003"}
    assert {r.product_code for r in search_funds(keyword="솔로몬 단기 국공채", db_path=str(db_path))} == {"KR000000003"}
    # 운용사명으로도 검색 가능해야 한다
    assert {r.product_code for r in search_funds(keyword="미래에셋", db_path=str(db_path))} == {"KR000000003"}
    # 없는 토큰이 섞이면 다른 상품이므로 제외
    assert search_funds(keyword="솔로몬 주식형", db_path=str(db_path)) == []


def test_get_fund_detail_returns_master_and_classes(tmp_path):
    db_path = tmp_path / "test.db"
    _seed(db_path)

    detail = get_fund_detail("KR000000001", db_path=str(db_path))
    assert detail is not None
    assert detail.master["fund_name"] == "안전채권펀드"
    assert len(detail.classes) == 1
    assert detail.classes[0]["class_name"] == "C1"


def test_get_fund_detail_returns_none_for_unknown_code(tmp_path):
    db_path = tmp_path / "test.db"
    _seed(db_path)

    assert get_fund_detail("KR_NOT_EXIST", db_path=str(db_path)) is None


# ── 보유 데이터 접점 조회 (①라우터 scope 판정 보조, F-2) ──────────────────

def test_asset_overlap_finds_fund_named_in_question():
    """상품명이 언급되면 제도 어휘가 없어도 접점을 찾아야 한다 (대회 공식 질의 유형)."""
    from src.storage.queries import find_asset_overlap

    hits = find_asset_overlap("솔로몬 국공채 단기 · 중장기 · 장기, 뭐가 달라요?")

    assert hits
    assert any("솔로몬" in name for name in hits)


def test_asset_overlap_matches_without_spacing():
    """붙여 쓴 상품명도 찾아야 한다 — 사용자는 정식 띄어쓰기를 모른다."""
    from src.storage.queries import find_asset_overlap

    assert find_asset_overlap("미래에셋솔로몬장기국공채 위험등급 알려줘")


def test_asset_overlap_ignores_generic_words():
    """일반어·속성어만 있는 질문은 접점 0건이어야 게이트가 유지된다.

    "하나"가 "하나파워e단기채"에, "장기"가 "장기성장포커스"에 걸리면 범위 밖
    질문까지 통과해 scope 게이트가 무력화된다 (실측으로 확인해 좁힌 조건).
    """
    from src.storage.queries import find_asset_overlap

    assert find_asset_overlap("좋은 연금 상품 하나 추천해 주세요") == []
    assert find_asset_overlap("위험등급 낮은 채권형 펀드 뭐가 있어요?") == []
    assert find_asset_overlap("세액공제 한도가 얼마인가요") == []


def test_asset_overlap_empty_for_out_of_scope():
    from src.storage.queries import find_asset_overlap

    assert find_asset_overlap("삼성전자 주가 지금 얼마인가요?") == []
    assert find_asset_overlap("부동산 양도세 계산해주세요") == []
    assert find_asset_overlap("오늘 점심 뭐 먹지") == []
