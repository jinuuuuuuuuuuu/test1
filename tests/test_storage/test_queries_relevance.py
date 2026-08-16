"""search_pension_docs 관련성 임계값 필터링(_filter_relevant) 검증 — 벡터DB/API 키 불필요.

임계값은 L2 거리 기준(낮을수록 관련)이다. LangChain relevance score는 이 컬렉션에서
음수가 나와 쓰지 않는다 (실측 2026-08-16, queries.py 주석 참고).
"""

from src.storage.queries import DEFAULT_MAX_DISTANCE, _filter_relevant


def test_drops_hits_beyond_max_distance():
    hits = [("docA", 13.0), ("docB", 33.0), ("docC", 47.0)]
    assert _filter_relevant(hits, 40.0) == [("docA", 13.0), ("docB", 33.0)]


def test_threshold_is_inclusive():
    assert _filter_relevant([("docA", 40.0)], 40.0) == [("docA", 40.0)]


def test_all_irrelevant_returns_empty():
    # 범위외 질의(실측: 401k 43.6, 점심메뉴 63.0): 전부 초과면 빈 결과 —
    # ②는 이를 "보유 자료에 없음"으로 해석해 한계를 고지한다.
    assert _filter_relevant([("docA", 43.6), ("docB", 63.0)], 40.0) == []


# ── parent 청크 복원 (_merge_overlapping_parts) ──────────────────────────


def test_merge_removes_splitter_overlap():
    # 분할기 조각은 원문의 연속 부분문자열이라 인접 조각 경계가 정확히 겹친다.
    # (반복 없는 문자열로 겹침 판정이 유일하게 떨어지게 한다)
    original = "".join(chr(0xAC00 + i) for i in range(300))
    part1, part2, part3 = original[:120], original[100:220], original[200:]
    from src.storage.queries import _merge_overlapping_parts

    assert _merge_overlapping_parts([part1, part2, part3]) == original


def test_merge_falls_back_to_newline_join_without_overlap():
    from src.storage.queries import _merge_overlapping_parts

    assert _merge_overlapping_parts(["앞부분", "완전히다른뒷부분"]) == "앞부분\n완전히다른뒷부분"


def test_merge_single_and_empty():
    from src.storage.queries import _merge_overlapping_parts

    assert _merge_overlapping_parts(["하나"]) == "하나"
    assert _merge_overlapping_parts([]) == ""


def test_default_threshold_matches_calibration():
    # 실측 분포(범위내 12.8~33.0 / 명백한 범위외 43.6~) 사이에 있어야 한다.
    # 재적재 후 분포가 바뀌면 scripts/calibrate_relevance.py로 재캘리브레이션할 것.
    assert 33.0 < DEFAULT_MAX_DISTANCE < 43.6
