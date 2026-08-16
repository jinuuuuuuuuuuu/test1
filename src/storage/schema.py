"""투자설명서 구조화 DB 스키마 (SQLite).

정확한 숫자·범주형 답변이 필요한 필드는 여기(fund_master/fund_class)에서 조회하고,
투자전략/투자위험 등 서술형 텍스트는 별도 벡터DB(청크 단위 임베딩)에서 검색한다.

fund_master: 펀드(상품코드) 단위로 100% 동일한 필드만 (검수본 100개 상품코드 전수 대조로 확인).
fund_class: 판매클래스마다 달라질 수 있는 필드(과세특징/전환가능여부/환매수수료 포함 — 실제로
클래스별로 다른 사례가 확인됨. 예: 연금저축용 vs 퇴직연금용 클래스는 과세특징이 다름).
"""

import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fund_master (
    product_code TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    fund_name TEXT NOT NULL,
    manager_name TEXT,
    base_date TEXT,
    prospectus_effective_date TEXT,
    risk_grade TEXT,
    issue_type TEXT,
    investment_objective TEXT,
    investment_strategy TEXT,
    fund_category TEXT,
    benchmark TEXT,
    manager_person_names TEXT,
    manager_birth_years TEXT,
    manager_titles TEXT,
    manager_years_experience TEXT,
    peer_avg_return_1y TEXT,
    purchase_rule TEXT,
    redemption_rule TEXT,
    redemption_payment_rule TEXT,
    aum_krw_million REAL,
    aum_base_date TEXT,
    aum_period_label TEXT
);

CREATE TABLE IF NOT EXISTS fund_class (
    product_code TEXT NOT NULL,
    class_name TEXT NOT NULL,
    sales_channel TEXT,
    total_expense_ratio REAL,
    cost_3y_per_10m_krw REAL,
    return_asof_date TEXT,
    return_1y REAL,
    return_3y REAL,
    return_since_inception REAL,
    volatility_1y REAL,
    volatility_3y REAL,
    volatility_since_inception REAL,
    redemption_fee TEXT,
    tax_note TEXT,
    conversion_available TEXT,
    inception_date TEXT,
    extraction_source TEXT,
    extraction_note TEXT,
    review_status TEXT,
    PRIMARY KEY (product_code, class_name),
    FOREIGN KEY (product_code) REFERENCES fund_master(product_code)
);

CREATE INDEX IF NOT EXISTS idx_fund_class_product_code ON fund_class(product_code);
"""


# AUM(시장잔고) 3개 컬럼 — 대회 6축 중 마지막 축. xlsm이 아니라 투자설명서 PDF의
# "요약 재무상태표"에서 추출한다 (scripts/extract_aum.py). 단위: 백만원.
AUM_COLUMNS = ("aum_krw_million REAL", "aum_base_date TEXT", "aum_period_label TEXT")


def ensure_aum_columns(conn: sqlite3.Connection) -> list[str]:
    """기존 DB(컬럼 추가 전 생성분)에 AUM 컬럼이 없으면 추가한다. 추가한 컬럼명 반환."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(fund_master)")}
    added = []
    for column_def in AUM_COLUMNS:
        name = column_def.split()[0]
        if name not in existing:
            conn.execute(f"ALTER TABLE fund_master ADD COLUMN {column_def}")
            added.append(name)
    return added


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    ensure_aum_columns(conn)
    return conn
