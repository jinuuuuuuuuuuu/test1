"""fund_master/fund_class SQLite DB 조회 + docs 벡터DB 검색 — 순수 함수, 에이전트 툴에서 감싸서 사용."""

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_DB_PATH = "data/processed/prospectus.db"
DEFAULT_CHROMA_DIR = "data/processed/chroma_docs"
DEFAULT_COLLECTION = "pension_docs"


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _extract_risk_grade_num(risk_grade: Optional[str]) -> Optional[int]:
    """'2등급[높은 위험]' -> 2. 1등급이 가장 위험, 6등급이 가장 안전(숫자가 클수록 안전)."""
    if not risk_grade:
        return None
    m = re.match(r"\s*(\d+)", risk_grade)
    return int(m.group(1)) if m else None


@dataclass
class FundSearchResult:
    product_code: str
    fund_name: str
    manager_name: Optional[str]
    risk_grade: Optional[str]
    fund_category: Optional[str]
    class_name: str
    sales_channel: Optional[str]
    total_expense_ratio: Optional[float]
    return_1y: Optional[float]
    return_3y: Optional[float]
    return_since_inception: Optional[float]


def search_funds(
    keyword: Optional[str] = None,
    risk_grade_min: Optional[int] = None,
    risk_grade_max: Optional[int] = None,
    max_expense_ratio: Optional[float] = None,
    min_return_1y: Optional[float] = None,
    sales_channel: Optional[str] = None,
    limit: int = 10,
    db_path: str = DEFAULT_DB_PATH,
) -> list[FundSearchResult]:
    """조건에 맞는 펀드(판매클래스 단위)를 검색한다.

    risk_grade_min/max: 위험등급 숫자 범위(1=가장 위험 ~ 6=가장 안전). 예를 들어 "안전한 상품만"
    이면 risk_grade_min=4 처럼 큰 쪽으로, "적극적으로 위험 감수 가능"이면 risk_grade_max=2처럼
    작은 쪽으로 좁힌다.
    """
    conn = _connect(db_path)
    try:
        query = """
            SELECT m.product_code, m.fund_name, m.manager_name, m.risk_grade, m.fund_category,
                   c.class_name, c.sales_channel, c.total_expense_ratio,
                   c.return_1y, c.return_3y, c.return_since_inception
            FROM fund_class c
            JOIN fund_master m ON m.product_code = c.product_code
            WHERE 1=1
        """
        params: list = []
        if max_expense_ratio is not None:
            query += " AND c.total_expense_ratio <= ?"
            params.append(max_expense_ratio)
        if min_return_1y is not None:
            query += " AND c.return_1y >= ?"
            params.append(min_return_1y)
        if sales_channel:
            query += " AND c.sales_channel LIKE ?"
            params.append(f"%{sales_channel}%")
        query += " ORDER BY c.return_1y DESC"

        rows = conn.execute(query, params).fetchall()

        results: list[FundSearchResult] = []
        for row in rows:
            if risk_grade_min is not None or risk_grade_max is not None:
                grade_num = _extract_risk_grade_num(row["risk_grade"])
                if grade_num is None:
                    continue
                if risk_grade_min is not None and grade_num < risk_grade_min:
                    continue
                if risk_grade_max is not None and grade_num > risk_grade_max:
                    continue
            if keyword:
                haystack = " ".join(
                    str(row[f]) for f in ("fund_name", "fund_category") if row[f] is not None
                )
                if keyword not in haystack:
                    continue

            results.append(
                FundSearchResult(
                    product_code=row["product_code"],
                    fund_name=row["fund_name"],
                    manager_name=row["manager_name"],
                    risk_grade=row["risk_grade"],
                    fund_category=row["fund_category"],
                    class_name=row["class_name"],
                    sales_channel=row["sales_channel"],
                    total_expense_ratio=row["total_expense_ratio"],
                    return_1y=row["return_1y"],
                    return_3y=row["return_3y"],
                    return_since_inception=row["return_since_inception"],
                )
            )
            if len(results) >= limit:
                break
        return results
    finally:
        conn.close()


@dataclass
class FundDetail:
    master: dict = field(default_factory=dict)
    classes: list[dict] = field(default_factory=list)


def get_fund_detail(product_code: str, db_path: str = DEFAULT_DB_PATH) -> Optional[FundDetail]:
    """특정 상품코드의 펀드 마스터 정보 + 전체 판매클래스 상세를 반환한다. 없으면 None."""
    conn = _connect(db_path)
    try:
        master_row = conn.execute(
            "SELECT * FROM fund_master WHERE product_code = ?", (product_code,)
        ).fetchone()
        if master_row is None:
            return None
        class_rows = conn.execute(
            "SELECT * FROM fund_class WHERE product_code = ? ORDER BY class_name", (product_code,)
        ).fetchall()
        return FundDetail(master=dict(master_row), classes=[dict(r) for r in class_rows])
    finally:
        conn.close()


@dataclass
class DocSearchResult:
    chunk_id: str
    file_title: str
    section: str
    source_location: str
    category: str
    content: str


def search_pension_docs(
    query: str,
    k: int = 5,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[DocSearchResult]:
    """docs.zip 벡터DB에서 질의와 의미상 가장 관련 있는 청크 k개를 검색한다. 임베딩 API 키 필요."""
    from langchain_chroma import Chroma

    from src.agents.llm import get_embeddings

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )
    hits = vectorstore.similarity_search(query, k=k)
    return [
        DocSearchResult(
            chunk_id=doc.metadata.get("chunk_id", ""),
            file_title=doc.metadata.get("file_title", ""),
            section=doc.metadata.get("section", ""),
            source_location=doc.metadata.get("source_location", ""),
            category=doc.metadata.get("category", ""),
            content=doc.page_content,
        )
        for doc in hits
    ]
