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


# ── 폐지 제도 문서 필터 (_is_obsolete_regime_doc / _asks_obsolete_regime) ──
#
# 실측(501문항 2차): "연금저축 중도해지하면 세액공제는 어떻게 되나요?"에
# (구)개인연금저축 문서가 근거로 잡혀 "이자소득세 15.4%"라고 답했다(정답은 현행
# 연금저축의 기타소득세 16.5%). grounded=True로 통과까지 됐다 — 근거에 실재하는
# 숫자였기 때문이다. 22건에서 이 문서가 근거로 잡혔고 3건은 답변까지 오염됐다.
# ②정보 Agent 프롬프트에 "제도를 구분하라"고 지시해도 3건 중 2건이 여전히 15.4%를
# 답해, 검색 단계에서 코드로 걸러낸다.


def test_obsolete_regime_doc_is_detected_by_title():
    from src.storage.queries import _is_obsolete_regime_doc

    assert _is_obsolete_regime_doc(
        {"file_title": "(구)개인연금저축 절세 혜택 안내", "section": "개인연금저축 주요 특징"},
        "중도해지 과세 | 이자소득세 15.4%",
    )


def test_current_regime_doc_is_not_filtered():
    from src.storage.queries import _is_obsolete_regime_doc

    assert not _is_obsolete_regime_doc(
        {"file_title": "중도인출 및 계좌해지", "section": "계좌 해지: 일반·특별·부득이한 사유"},
        "연금저축계좌를 일반 해지하면 기타소득세 16.5%가 부과됩니다.",
    )


def test_explicit_obsolete_regime_question_is_recognized():
    """사용자가 옛 제도를 명시적으로 물으면 걸러내지 않는다."""
    from src.storage.queries import _asks_obsolete_regime

    assert _asks_obsolete_regime("(구)개인연금저축 중도해지 과세는 어떻게 되나요?")
    assert _asks_obsolete_regime("예전 개인연금저축은 한도가 얼마였나요?")
    assert not _asks_obsolete_regime("연금저축 중도해지하면 세액공제는 어떻게 되나요?")
