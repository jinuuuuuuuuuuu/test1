"""fund_master/fund_class SQLite DB 조회 + docs 벡터DB 검색 — 순수 함수, 에이전트 툴에서 감싸서 사용."""

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_DB_PATH = "data/processed/prospectus.db"
DEFAULT_CHROMA_DIR = "data/processed/chroma_docs"
DEFAULT_COLLECTION = "pension_docs"
DEFAULT_PROSPECTUS_CHROMA_DIR = "data/processed/chroma_prospectus"
DEFAULT_PROSPECTUS_COLLECTION = "prospectus_text"

# 관련성 임계값(L2 거리, 낮을수록 관련) — 이 거리를 넘는 검색 결과는 버린다. 임계값이
# 없으면 데이터에 없는 질문("미국 401k 이전")에도 가장 덜 무관한 청크 5개가 "근거"로
# 포장되어 무리한 답변을 유도한다 (평가지표 "정보한계 대응"/"근거 완전성" 직결). 전부
# 초과면 빈 결과가 되고, ②정보 Agent는 이를 "보유 자료에 없음"으로 해석해 한계를 고지한다.
#
# ⚠️ LangChain의 relevance score(0~1 정규화)는 이 컬렉션(비정규화 임베딩)에서 -9~-14가
# 나와 쓸 수 없다 (실측 2026-08-16) — 반드시 raw 거리(similarity_search_with_score)를 쓴다.
# 40.0의 실측 근거 (scripts/calibrate_relevance.py, 2026-08-16):
#   범위내 질의 4종의 히트 거리 12.8~33.0 / 명백한 범위외(401k·점심메뉴·주가) 최상위 43.6~63.0
#   → 40은 범위내 전부 보존 + 명백한 무관 차단. 경계 사례(부동산 양도세 31~36)는 일부
#   통과하지만 라우터 scope 게이트가 1차로 막는다. 데이터/임베딩 재적재 시 재캘리브레이션 필수.
DEFAULT_MAX_DISTANCE = 40.0


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


def _normalize_for_match(text: str) -> str:
    """키워드 매칭용 정규화 — 공백류 전부 제거 + 소문자화.

    한국 펀드명은 띄어쓰기가 제각각이라("솔로몬국공채" vs 검색어 "솔로몬 국공채") 단순
    부분문자열 매칭은 펀드명 지목 질문에서 0건 반환 → "해당 상품 없음" 오답을 만든다.
    """
    return re.sub(r"\s+", "", text or "").lower()


def _keyword_matches(keyword: str, haystack: str) -> bool:
    """검색어를 공백 단위 토큰으로 쪼개 전부(AND) 포함되는지 정규화 비교한다.

    "솔로몬 국공채" -> ["솔로몬", "국공채"] 각각이 정규화된 haystack에 있으면 매치 —
    토큰 순서·붙여쓰기 차이에 관대하되, 하나라도 없으면 다른 상품이므로 제외한다.
    """
    normalized_haystack = _normalize_for_match(haystack)
    tokens = [_normalize_for_match(t) for t in (keyword or "").split()]
    return all(t in normalized_haystack for t in tokens if t)


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
    aum_krw_million: Optional[float] = None  # 시장잔고(백만원) — 요약 재무상태표 최신 기 자본총계
    aum_base_date: Optional[str] = None      # 그 값의 결산 기준일


def search_funds(
    keyword: Optional[str] = None,
    risk_grade_min: Optional[int] = None,
    risk_grade_max: Optional[int] = None,
    max_expense_ratio: Optional[float] = None,
    min_return_1y: Optional[float] = None,
    min_aum_krw_million: Optional[float] = None,
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
                   c.return_1y, c.return_3y, c.return_since_inception,
                   m.aum_krw_million, m.aum_base_date
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
        if min_aum_krw_million is not None:
            query += " AND m.aum_krw_million >= ?"
            params.append(min_aum_krw_million)
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
                    str(row[f])
                    for f in ("fund_name", "fund_category", "manager_name")
                    if row[f] is not None
                )
                if not _keyword_matches(keyword, haystack):
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
                    aum_krw_million=row["aum_krw_million"],
                    aum_base_date=row["aum_base_date"],
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
    distance: float = 0.0  # L2 거리(낮을수록 관련) — 임계값 캘리브레이션·디버깅용


def _filter_relevant(scored_hits: list, max_distance: float) -> list:
    """(Document, L2 거리) 쌍에서 임계값 초과를 제거한다. 전부 초과면 빈 리스트."""
    return [(doc, distance) for doc, distance in scored_hits if distance <= max_distance]


def _merge_overlapping_parts(parts: list[str], max_overlap: int = 200) -> str:
    """part 조각들을 겹침(chunk_overlap으로 생긴 중복 구간)을 제거하며 원문으로 이어붙인다.

    분할기(RecursiveCharacterTextSplitter)의 조각은 원문의 연속 부분문자열이라 인접 조각의
    앞뒤가 정확히 겹친다 — 뒤 조각의 접두사가 앞 조각의 접미사와 일치하는 최대 길이를 찾아
    한 번만 남긴다. (경계 공백 처리 등으로) 겹침을 못 찾으면 줄바꿈으로 잇는다(최선 노력).
    """
    merged = parts[0] if parts else ""
    for part in parts[1:]:
        limit = min(len(merged), len(part), max_overlap)
        for size in range(limit, 0, -1):
            if merged.endswith(part[:size]):
                merged += part[size:]
                break
        else:
            merged += "\n" + part
    return merged


def _fetch_parent_content(vectorstore, parent_chunk_id: str) -> Optional[str]:
    """parent_chunk_id의 모든 part를 모아 원본 청크 내용을 복원한다. 실패 시 None."""
    got = vectorstore.get(where={"parent_chunk_id": parent_chunk_id})
    documents = got.get("documents") or []
    metadatas = got.get("metadatas") or []
    if not documents:
        return None
    pairs = sorted(zip(metadatas, documents), key=lambda p: p[0].get("part_index", 0))
    return _merge_overlapping_parts([text for _, text in pairs])


def search_pension_docs(
    query: str,
    k: int = 5,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> list[DocSearchResult]:
    """docs.zip 벡터DB에서 질의와 관련 있는 청크를 최대 k개 검색한다. 임베딩 API 키 필요.

    거리가 max_distance를 넘는 청크는 제외한다 — 빈 리스트는 "보유 문서에 관련 내용이
    없음"을 뜻하며, 호출측은 이를 한계 고지의 신호로 쓴다. 질의 시점 임베딩 호출도
    CLOVA 플레이크/429 대상이므로 재시도로 감싼다.

    임베딩 한도 때문에 여러 part로 쪼개 적재된 긴 청크는, part 조각이 아니라 사람이 검수한
    원본 청크 단위로 복원해서 반환한다 (같은 parent의 part가 여러 개 걸리면 하나로 합침) —
    조각만 근거로 주면 규정 문장이 중간에 끊겨 근거 완전성이 떨어진다.
    """
    from langchain_chroma import Chroma

    from src.agents.llm import call_with_retry, get_embeddings

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )
    hits = call_with_retry(vectorstore.similarity_search_with_score, query, k=k)

    results: list[DocSearchResult] = []
    seen_parents: set[str] = set()
    for doc, distance in _filter_relevant(hits, max_distance):
        meta = doc.metadata
        parent_id = meta.get("parent_chunk_id")
        chunk_id = meta.get("chunk_id", "")
        content = doc.page_content
        if parent_id:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            chunk_id = parent_id
            content = _fetch_parent_content(vectorstore, parent_id) or content
        results.append(
            DocSearchResult(
                chunk_id=chunk_id,
                file_title=meta.get("file_title", ""),
                section=meta.get("section", ""),
                source_location=meta.get("source_location", ""),
                category=meta.get("category", ""),
                content=content,
                distance=round(distance, 4),
            )
        )
    return results


@dataclass
class ProspectusTextResult:
    product_code: str
    fund_name: str
    section: str   # "투자목적" | "투자전략" | "투자위험"
    content: str
    distance: float = 0.0


def search_prospectus_text(
    query: str,
    product_code: Optional[str] = None,
    k: int = 4,
    persist_dir: str = DEFAULT_PROSPECTUS_CHROMA_DIR,
    collection_name: str = DEFAULT_PROSPECTUS_COLLECTION,
) -> list[ProspectusTextResult]:
    """투자설명서 서술형 컬렉션(투자목적/투자전략/투자위험)에서 의미 검색한다. 임베딩 키 필요.

    product_code를 주면 그 펀드의 서술로 한정한다 — 특정 펀드의 전략·위험 설명 질의에서는
    반드시 한정해서 다른 펀드의 서술이 근거로 섞이지 않게 한다. part 분할 청크는
    search_pension_docs와 동일하게 원본 단위로 복원한다.
    """
    from langchain_chroma import Chroma

    from src.agents.llm import call_with_retry, get_embeddings

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )
    search_kwargs: dict = {"k": k}
    if product_code:
        search_kwargs["filter"] = {"product_code": product_code}
    hits = call_with_retry(vectorstore.similarity_search_with_score, query, **search_kwargs)

    results: list[ProspectusTextResult] = []
    seen_parents: set[str] = set()
    for doc, distance in hits:
        meta = doc.metadata
        parent_id = meta.get("parent_chunk_id")
        content = doc.page_content
        if parent_id:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            content = _fetch_parent_content(vectorstore, parent_id) or content
        results.append(
            ProspectusTextResult(
                product_code=meta.get("product_code", ""),
                fund_name=meta.get("fund_name", ""),
                section=meta.get("section", ""),
                content=content,
                distance=round(distance, 4),
            )
        )
    return results
