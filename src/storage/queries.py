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
PENSION_STANDARD_CHANNELS = {"오프라인", "온라인"}
PENSION_COST_METRICS = ("synthetic_total_expense_ratio", "total_expense_ratio")


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


@dataclass
class LowerCostPensionClass:
    found: bool
    product_code: str
    account_type: str
    current_class_code: str
    target_class_code: str = ""
    comparison_metric: str = ""
    current_value: Optional[float] = None
    target_value: Optional[float] = None
    difference_pct_point: Optional[float] = None
    eligibility: str = ""
    eligibility_type: str = ""
    current_channel: str = ""
    target_channel: str = ""
    current_source_page: str = ""
    target_source_page: str = ""
    current_validation_status: str = ""
    target_validation_status: str = ""
    dataset_version: str = ""
    dataset_status: str = ""
    fund_name: str = ""
    reason: str = ""


def get_pension_class_detail(
    product_code: str,
    class_code: str,
    account_type: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> dict | None:
    """Cost Guard canonical 기준의 특정 연금 판매클래스 상세를 반환한다.

    기존 fund_class 테이블은 일반 C/Ce 클래스 중심이라 C-P2/C-P2E 같은 연금 전용
    클래스 질문에서 다른 축의 데이터를 답할 수 있다. 명시된 product/class/account
    맥락은 frozen canonical 테이블을 우선 조회한다.
    """
    normalized_class = normalize_pension_class_code(class_code)
    normalized_account = normalize_pension_account_type(account_type)
    conn = _connect(db_path)
    try:
        try:
            rows = conn.execute(
                """
                SELECT p.*, m.fund_name, m.manager_name, m.risk_grade, m.fund_category,
                       m.investment_objective, m.investment_strategy
                FROM fund_class_pension p
                LEFT JOIN fund_master m ON m.product_code = p.product_code
                WHERE p.product_code = ? AND p.account_type = ?
                ORDER BY p.class_code
                """,
                (product_code, normalized_account),
            ).fetchall()
        except sqlite3.OperationalError:
            return None
    finally:
        conn.close()

    for row in rows:
        item = dict(row)
        if normalize_pension_class_code(item.get("class_code")) == normalized_class:
            return item
    return None


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


def normalize_pension_class_code(value: str | None) -> str:
    """Cost Guard 매칭용 판매클래스 코드 정규화."""
    text = "" if value is None else str(value)
    return re.sub(r"\s+", "", text).split("(")[0].upper()


def normalize_pension_account_type(value: str | None) -> str:
    """사용자/상품 경로의 계좌 표현을 canonical Cost Guard 계좌유형으로 맞춘다."""
    text = (value or "").upper()
    if "연금저축" in (value or ""):
        return "연금저축"
    if "IRP" in text or "DC" in text or "DB" in text or "퇴직연금" in (value or ""):
        return "퇴직연금/IRP"
    return value or ""


def _cost_value(row: dict, metric: str) -> Optional[float]:
    value = row.get(metric)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def choose_pension_cost_metric(current: dict, target: dict) -> str | None:
    """current↔target pair 단위로 공통 비용 metric을 선택한다."""
    for metric in PENSION_COST_METRICS:
        if _cost_value(current, metric) is not None and _cost_value(target, metric) is not None:
            return metric
    return None


def _pension_pair_kind(current: dict, target: dict) -> str:
    if current.get("channel") in PENSION_STANDARD_CHANNELS and target.get("channel") in PENSION_STANDARD_CHANNELS:
        return "STANDARD"
    return "CHANNEL_CONDITIONAL"


def _source_page(row: dict, metric: str) -> str:
    if metric == "synthetic_total_expense_ratio":
        return row.get("synthetic_expense_source_page") or row.get("review_source_page") or ""
    return row.get("total_expense_source_page") or row.get("review_source_page") or ""


def _blocked_validation(row: dict) -> bool:
    status = row.get("validation_status") or ""
    return "AMBIGUOUS" in status or "SOURCE_CONFLICT" in status


def get_cost_guard_dataset_manifest(db_path: str = DEFAULT_DB_PATH) -> dict:
    conn = _connect(db_path)
    try:
        try:
            row = conn.execute("SELECT * FROM cost_guard_dataset_manifest WHERE id = 1").fetchone()
        except sqlite3.OperationalError:
            return {}
        return dict(row) if row else {}
    finally:
        conn.close()


def find_lower_cost_pension_class(
    product_code: str,
    current_class_code: str,
    account_type: str,
    *,
    include_channel_conditional: bool = False,
    require_frozen: bool = True,
    db_path: str = DEFAULT_DB_PATH,
) -> LowerCostPensionClass:
    """같은 상품·계좌유형에서 더 낮은 비용의 판매클래스 1건을 찾는다.

    Cost Guard는 metric을 섞어 비교하지 않는다. current↔target pair 양쪽에
    synthetic_total_expense_ratio가 있으면 합성총보수·비용으로, 아니면 양쪽에
    total_expense_ratio가 있을 때만 총보수·비용으로 비교한다.
    """
    normalized_class = normalize_pension_class_code(current_class_code)
    normalized_account = normalize_pension_account_type(account_type)
    manifest = get_cost_guard_dataset_manifest(db_path)
    dataset_version = manifest.get("dataset_version") or ""
    dataset_status = manifest.get("dataset_status") or ""
    if require_frozen and dataset_status != "FROZEN_V1":
        return LowerCostPensionClass(
            found=False,
            product_code=product_code,
            account_type=normalized_account,
            current_class_code=normalized_class,
            dataset_version=dataset_version,
            dataset_status=dataset_status,
            reason="DATASET_NOT_FROZEN" if manifest else "DATASET_MANIFEST_NOT_LOADED",
        )
    conn = _connect(db_path)
    try:
        try:
            rows = conn.execute(
                """
                SELECT p.*, m.fund_name
                FROM fund_class_pension p
                LEFT JOIN fund_master m ON m.product_code = p.product_code
                WHERE p.product_code = ? AND p.account_type = ?
                ORDER BY p.class_code
                """,
                (product_code, normalized_account),
            ).fetchall()
        except sqlite3.OperationalError:
            return LowerCostPensionClass(
                found=False,
                product_code=product_code,
                account_type=normalized_account,
                current_class_code=normalized_class,
                dataset_version=dataset_version,
                dataset_status=dataset_status,
                reason="TABLE_NOT_LOADED",
            )
    finally:
        conn.close()

    dict_rows = [dict(row) for row in rows]
    current = next((row for row in dict_rows if normalize_pension_class_code(row.get("class_code")) == normalized_class), None)
    if current is None:
        return LowerCostPensionClass(
            found=False,
            product_code=product_code,
            account_type=normalized_account,
            current_class_code=normalized_class,
            dataset_version=dataset_version,
            dataset_status=dataset_status,
            reason="CURRENT_CLASS_NOT_FOUND",
        )
    if _blocked_validation(current):
        return LowerCostPensionClass(
            found=False,
            product_code=product_code,
            account_type=normalized_account,
            current_class_code=normalized_class,
            dataset_version=dataset_version,
            dataset_status=dataset_status,
            reason="CURRENT_CLASS_BLOCKED",
        )

    candidates: list[tuple[int, int, float, str, dict, str, float, float, str]] = []
    for target in dict_rows:
        if normalize_pension_class_code(target.get("class_code")) == normalized_class:
            continue
        if _blocked_validation(target):
            continue
        kind = _pension_pair_kind(current, target)
        if kind == "CHANNEL_CONDITIONAL" and not include_channel_conditional:
            continue
        metric = choose_pension_cost_metric(current, target)
        if metric is None:
            continue
        current_value = _cost_value(current, metric)
        target_value = _cost_value(target, metric)
        if current_value is None or target_value is None or target_value >= current_value:
            continue
        kind_priority = 0 if kind == "STANDARD" else 1
        metric_priority = 0 if metric == "synthetic_total_expense_ratio" else 1
        candidates.append((
            kind_priority,
            metric_priority,
            target_value,
            target.get("class_code") or "",
            target,
            metric,
            current_value,
            target_value,
            kind,
        ))

    if not candidates:
        return LowerCostPensionClass(
            found=False,
            product_code=product_code,
            account_type=normalized_account,
            current_class_code=normalized_class,
            dataset_version=dataset_version,
            dataset_status=dataset_status,
            reason="NO_LOWER_COST_CLASS",
        )

    _, _, _, _, target, metric, current_value, target_value, kind = sorted(candidates)[0]
    return LowerCostPensionClass(
        found=True,
        product_code=product_code,
        account_type=normalized_account,
        current_class_code=current["class_code"],
        target_class_code=target["class_code"],
        comparison_metric=metric,
        current_value=current_value,
        target_value=target_value,
        difference_pct_point=round(current_value - target_value, 6),
        eligibility=kind,
        eligibility_type=kind,
        current_channel=current.get("channel") or "",
        target_channel=target.get("channel") or "",
        current_source_page=_source_page(current, metric),
        target_source_page=_source_page(target, metric),
        current_validation_status=current.get("validation_status") or "",
        target_validation_status=target.get("validation_status") or "",
        dataset_version=dataset_version,
        dataset_status=dataset_status,
        fund_name=current.get("fund_name") or "",
    )


# ── 보유 데이터 접점 조회 (①라우터 scope 판정 보조) ──────────────────────
#
# scope 판정은 "이 질문에 답할 수 있는가"인데, 라우터 LLM은 **우리가 무엇을 보유했는지
# 모른 채** 자기 상식으로 그걸 추측한다. 실측(2026-08-27): DB에 실재하는 펀드를 두고
#   "미래에셋솔로몬장기국공채 위험등급 알려줘" → 범위외
#   scope_note: "특정 펀드의 위험등급 문의 — 개별 상품 정보는 제공 불가능"
# 라고 판정했다. fund_master에 risk_grade 컬럼이 있는데도 "없다"고 단정한 것이다.
# 반대로 "IRP에서 ~ 살 수 있나요"처럼 제도 어휘가 붙으면 정상 판정한다 — 즉 상품명만
# 아는 고객일수록 거부당하는 역진적 실패다.
#
# 이는 "세액공제+얼마 = 한도질문"으로 단정하던 사고(7cddb1f)와 같은 클래스다: 판정에
# 필요한 사실을 조회하지 않고 표면 신호로 단정한다. 그때의 해법(규칙이 사실을 조회해
# LLM에 힌트를 주고, 판단은 LLM이 한다)을 scope 축에도 동일하게 적용한다.

# 펀드명에 흔히 등장하지만 그 자체로는 특정 상품을 지목하지 못하는 어휘. 이걸 매칭에
# 쓰면 "좋은 연금 상품 하나 추천해줘"의 "하나"가 "하나파워e단기채"에 걸리는 식으로
# 게이트가 무력화된다 — 접점 판정은 "고유명사성"이 있는 토큰으로만 해야 한다.
_ASSET_GENERIC_TOKENS = frozenset({
    # 도메인 일반어
    "펀드", "상품", "투자", "증권", "신탁", "자산", "수익", "연금", "퇴직", "계좌",
    "자투자신탁", "증권자투자신탁", "투자신탁",
    # 제도·계좌 유형 명칭 — "연금저축"·"퇴직연금"은 실제 펀드명(운용사 브랜드)에도 흔히
    # 들어있어("미래에셋고배당포커스연금저축증권전환형...", "한국투자 퇴직연금 증권...")
    # 제도 질문을 상품 질문으로 오판하게 만든다. 실측: "연금저축 600만원 납입하고...
    # 세액공제 얼마?"가 상품 2건과 매칭돼 라우터의 세액공제 분류를 흔들었다.
    # "DB"는 확정급여형 제도 약칭이면서 DB자산운용 브랜드와도 겹쳐 더 위험하다.
    "연금저축", "퇴직연금", "개인연금", "확정급여형", "확정기여형",
    "DB", "DC", "IRP", "db", "dc", "irp",
    # 상품 속성어 (질문의 조건이지 상품 지목이 아니다)
    "단기", "중기", "장기", "초단기", "중장기", "채권", "주식", "혼합", "국내", "해외",
    "배당", "성장", "가치", "안정", "위험", "등급", "보수", "수익률", "잔고",
    # 수량·지시어
    "하나", "둘", "셋", "두개", "세개", "여러", "모든", "각각",
    # 질문 상용어
    "어때요", "알려줘", "얼마", "뭐가", "달라요", "어떤", "추천", "비교", "설명",
    "좋은", "괜찮은", "적당한", "무엇", "어디", "언제",
})


def find_asset_overlap(question: str, db_path: str = DEFAULT_DB_PATH, limit: int = 5) -> list[str]:
    """질문이 보유 펀드 데이터와 겹치는지 조회해, 매칭된 펀드명을 반환한다.

    scope 판정의 **사실 확인용**이다 — 답을 만들지 않고 "답할 재료가 있는가"만 본다.
    특정 상품명을 하드코딩하지 않고 fund_master 전체와 대조하므로, 데이터가 바뀌면
    판정도 자동으로 따라간다.

    판정 기준은 "질문에 우리 펀드를 **지목하는 고유명사가 있는가**"다:
      - 일반어·속성어(_ASSET_GENERIC_TOKENS)는 제외 — 상품을 지목하지 못한다
      - 3글자 이상만 본다 — 2글자는 우연 일치가 잦다("장기" 등은 위에서도 걸러진다)
      - 한 펀드명 안에서 고유 토큰이 겹칠수록 확실하므로, 매칭 수가 많은 순으로 준다

    "좋은 연금 상품 추천해줘"처럼 속성만 있는 질문은 0건이 정상이다 — 이런 질문은
    이미 라우터가 범위내로 정상 판정하므로 이 보조 신호가 필요 없다.
    """
    tokens = [
        _normalize_for_match(t)
        for t in re.split(r"[\s·,()\[\]]+", question or "")
    ]
    tokens = [t for t in tokens if len(t) >= 3 and t not in _ASSET_GENERIC_TOKENS]
    if not tokens:
        return []

    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT fund_name FROM fund_master").fetchall()
    finally:
        conn.close()

    scored: list[tuple[int, str]] = []
    for row in rows:
        name = row["fund_name"] or ""
        normalized = _normalize_for_match(name)
        hits = sum(1 for t in tokens if t in normalized)
        if hits:
            scored.append((hits, name))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in scored[:limit]]


@dataclass
class DocSearchResult:
    chunk_id: str
    file_title: str
    section: str
    source_location: str
    category: str
    content: str
    distance: float = 0.0  # L2 거리(낮을수록 관련) — 임계값 캘리브레이션·디버깅용


# 이름이 비슷하지만 지금과 다른 폐지 제도의 문서를 가리키는 표시.
# "(구)개인연금저축"은 현행 "연금저축"과 한도·과세·요건이 전부 다르다
# (중도해지 과세: 옛 제도 이자소득세 15.4% vs 현행 기타소득세 16.5%).
_OBSOLETE_REGIME_MARKERS = ("(구)개인연금", "(구) 개인연금", "구 개인연금저축")

# 사용자가 그 옛 제도를 **명시적으로** 물은 경우엔 걸러내면 안 된다.
_OBSOLETE_REGIME_QUERY_MARKERS = ("(구)", "구 개인연금", "옛 개인연금", "예전 개인연금", "종전 개인연금")


def _asks_obsolete_regime(query: str) -> bool:
    """질문이 폐지된 (구)개인연금저축을 명시적으로 묻는지 판정한다."""
    return any(marker in query for marker in _OBSOLETE_REGIME_QUERY_MARKERS)


def _is_obsolete_regime_doc(meta: dict, content: str) -> bool:
    """검색된 청크가 폐지 제도 문서인지 판정한다.

    ⚠️ 실측(501문항): "연금저축 중도해지하면 세액공제는 어떻게 되나요?"에
    (구)개인연금저축 문서가 근거로 잡혀 "이자소득세 15.4%"라고 답했다(정답은
    기타소득세 16.5%). grounded=True로 통과까지 됐다 — 근거에 실재하는 숫자였기
    때문이다. 22건에서 이 문서가 근거로 잡혔고 3건은 답변까지 오염됐다.
    프롬프트로 "제도를 구분하라"고 지시해도 2/3이 여전히 15.4%를 답해, 검색
    단계에서 걸러낸다.
    """
    haystack = f"{meta.get('file_title', '')} {meta.get('section', '')} {content[:200]}"
    return any(marker in haystack for marker in _OBSOLETE_REGIME_MARKERS)


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

    asks_obsolete = _asks_obsolete_regime(query)

    results: list[DocSearchResult] = []
    seen_parents: set[str] = set()
    for doc, distance in _filter_relevant(hits, max_distance):
        meta = doc.metadata
        if not asks_obsolete and _is_obsolete_regime_doc(meta, doc.page_content):
            continue
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
