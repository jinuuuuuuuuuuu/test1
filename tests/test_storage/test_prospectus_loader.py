import openpyxl
import pytest

from src.storage.prospectus_loader import load_prospectus_xlsm

HEADER = [
    "파일명", "집합투자기구명칭", "상품코드", "집합투자업자명칭", "작성기준일",
    "증권신고서효력발생일", "투자위험등급", "모집(매출)증권 종류/총액", "투자목적", "투자전략",
    "상품분류", "판매클래스", "판매방식", "총보수·비용(%)", "1000만원투자시_3년총비용(천원)",
    "수익률기준일", "수익률_최근1년(%)", "수익률_최근3년(%)", "수익률_설정일이후(%)",
    "변동성_최근1년(%)", "변동성_최근3년(%)", "변동성_설정일이후(%)", "비교지수(BM)",
    "책임운용역_성명", "책임운용역_생년", "책임운용역_직위", "책임운용역_운용경력년수",
    "동종집합투자기구_운용사최근1년수익률(%)", "매입기준", "환매기준", "환매대금지급",
    "환매수수료", "과세특징", "전환가능여부(전환형)", "최초설정일", "근거위치", "검수메모", "검수상태",
]


def _make_xlsm(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "투자설명서_파싱_검수본"
    for _ in range(4):
        ws.append([None])
    ws.append(HEADER)
    ws.append([
        "fundA.pdf", "테스트펀드", "KR000000001", "테스트자산운용", "2025-01-01",
        "2025-01-05", "2등급[높은 위험]", "투자신탁 수익증권", "테스트 투자목적", "테스트 투자전략",
        "증권(주식형)", "C1", "오프라인", 1.5, 300.0,
        "2025-06-01", 10.5, 20.5, 30.5,
        15.0, 16.0, 17.0, "KOSPI x 100%",
        "홍길동", "1980", "부장", "10년",
        "5.0", "D+2", "D+3", "D+4",
        "없음", "일반과세", "N", "2020-01-01", "요약정보", "확인완료", "검수완료",
    ])
    ws.append([
        "fundA.pdf", "테스트펀드", "KR000000001", "테스트자산운용", "2025-01-01",
        "2025-01-05", "2등급[높은 위험]", "투자신탁 수익증권", "테스트 투자목적", "테스트 투자전략",
        "증권(주식형)", "Ce", "온라인", 1.2, 250.0,
        "2025-06-01", "NULL", "NULL", "NULL",
        "NULL", "NULL", "NULL", "KOSPI x 100%",
        "홍길동", "1980", "부장", "10년",
        "5.0", "D+2", "D+3", "D+4",
        "없음", "일반과세", "N", "NULL", "요약정보", "NULL 처리", "추가확인",
    ])
    ws.append([
        "fundB.pdf", "테스트펀드2", "KR000000002", "테스트자산운용2", "2025-02-01",
        "2025-02-05", "4등급[보통 위험]", "투자신탁 수익증권", "테스트 투자목적2", "테스트 투자전략2",
        "증권(채권형)", "A", "오프라인", 0.8, 100.0,
        "2025-06-01", 3.1, 4.2, 5.3,
        2.0, 2.5, 3.0, "KOFR x 100%",
        "김철수", "1975", "이사", "15년",
        "2.0", "D+1", "D+2", "D+3",
        "해당사항 없음", "연금저축", "Y", "2019-05-05", "제3부", "확인완료", "검수완료",
    ])
    wb.save(path)


def test_load_prospectus_xlsm_builds_master_and_class_tables(tmp_path):
    xlsm_path = tmp_path / "prospectus.xlsm"
    db_path = tmp_path / "prospectus.db"
    _make_xlsm(xlsm_path)

    stats = load_prospectus_xlsm(str(xlsm_path), str(db_path), sheet_name="투자설명서_파싱_검수본")

    assert stats.funds == 2
    assert stats.classes == 3

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    masters = conn.execute("SELECT * FROM fund_master ORDER BY product_code").fetchall()
    assert len(masters) == 2
    assert masters[0]["product_code"] == "KR000000001"
    assert masters[0]["fund_name"] == "테스트펀드"
    assert masters[0]["risk_grade"] == "2등급[높은 위험]"

    classes = conn.execute(
        "SELECT * FROM fund_class WHERE product_code = 'KR000000001' ORDER BY class_name"
    ).fetchall()
    assert len(classes) == 2
    c1 = next(c for c in classes if c["class_name"] == "C1")
    assert c1["return_1y"] == pytest.approx(10.5)
    assert c1["total_expense_ratio"] == pytest.approx(1.5)

    ce = next(c for c in classes if c["class_name"] == "Ce")
    assert ce["return_1y"] is None  # "NULL" 문자열은 None으로 정규화되어야 함
    assert ce["inception_date"] is None

    conn.close()


def test_load_prospectus_xlsm_is_idempotent_on_rerun(tmp_path):
    xlsm_path = tmp_path / "prospectus.xlsm"
    db_path = tmp_path / "prospectus.db"
    _make_xlsm(xlsm_path)

    load_prospectus_xlsm(str(xlsm_path), str(db_path), sheet_name="투자설명서_파싱_검수본")
    stats2 = load_prospectus_xlsm(str(xlsm_path), str(db_path), sheet_name="투자설명서_파싱_검수본")

    assert stats2.funds == 2
    assert stats2.classes == 3

    import sqlite3

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM fund_class").fetchone()[0]
    assert count == 3  # 재실행해도 중복 적재되지 않아야 함
    conn.close()
