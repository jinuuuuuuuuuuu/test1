"""L0 결정론적 검증 — LLM 없이 코드로 강제하는 근거 검사.

④grounding의 LLM 판정(L1)은 프롬프트 순종에 의존하므로 확률적으로 실패한다(실측:
근거 0건인데 학습 지식 속 숫자로 답한 초안을 grounded=True로 통과시킨 사례). 여기의
검사는 그 위에 있는 결정론적 방어선이다:

  1. extract_number_tokens / find_unsupported_numbers — 초안에서 "숫자+단위" 토큰을
     추출해 근거 원문과 기계적으로 대조하고, 근거에 없는 수치 목록(의심 목록)을 만든다.
     이 목록은 ④ LLM의 입력으로 넘어가 "각 수치를 근거와 대조해 확인하라"는 기계적
     과제로 바뀐다 (막연한 "근거에 부합하는가"보다 훨씬 실패하기 어렵다).
  2. apply_l0_overrides — ④ LLM의 판정 결과에 코드가 최종 오버라이드를 건다:
     근거가 0건인데 초안에 구체적 수치가 있으면 LLM이 뭐라 하든 grounded=False다.

단위 없는 맨 숫자("1.", "2)")는 목록 번호·지시어일 가능성이 높아 L0에서는 잡지 않는다 —
여기서의 오탐(false positive)은 맞는 답의 숫자를 지우게 만들므로, 애매한 것은 L1(LLM)에
맡기고 L0는 확실한 것만 잡는다.
"""

import re

# "숫자+단위" 토큰만 추출한다. 단위 후보는 연금 도메인에서 사실 주장에 쓰이는 것들로 한정:
# 금액(억원/만원/천원/원, 맨 억/만), 비율(%/퍼센트/프로), 나이(세), 기간(년/개월), 위험등급(등급).
# '세'는 세금/세율/세액/세대 같은 복합어 오탐을 막기 위해 뒤 글자를 제한한다.
_NUMBER_TOKEN_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s?(?:"
    r"억\s?원|만\s?원|천\s?원"
    r"|억(?![가-힣])|만(?![가-힣])"
    r"|원|%|퍼센트|프로"
    r"|세(?![금율액대])|년|개월|등급"
    r")"
    # "1/12", "12분의 1"처럼 분수로 사실을 주장하는 경우 — DC형 회사 부담금
    # "연간 임금총액의 1/12 이상"이 대표적이다. 이 형태를 단위 목록에서 빠뜨리면
    # extract_number_tokens/find_unsupported_numbers 양쪽에서 통째로 안 잡혀, L0
    # 근거대조는 그 수치를 검사하지 못하고 enforce_missing_requirements는 답변에
    # 이미 있는 값을 "빠졌다"고 오판해 자기모순 문장을 덧붙인다(실측 no.6).
    r"|\d[\d,]*\s?/\s?\d[\d,]*"
    r"|\d[\d,]*분의\s?\d[\d,]*"
)

_NUMERIC_CORE_RE = re.compile(r"[\d,\.]+(?:\s?/\s?[\d,\.]+)?")

# 인라인 마크다운 서식 문자. LLM이 수치를 강조할 때 숫자만 감싸고 단위를 밖에 두는 일이
# 잦은데(**16.5**%), 그러면 "숫자+단위" 패턴이 서식 문자로 갈라져 L0가 토큰을 아예
# 추출하지 못한다 — 지어낸 수치도 강조로 감싸기만 하면 검증을 통과하는 구멍이 된다.
#
# 실측(2026-08-27, API 응답):
#   "**16.5**%"      -> 추출 []       (강조 없으면 ['16.5%'])
#   "**99.9**%"      -> 미지원 판정 [] (근거에 없는 값인데도 통과)
#   "**74**세", "**10**년", "1,**800**만원" 도 동일
#   `*`, `__`, `` ` `` 등 다른 서식도 같은 결과
#
# 정규식에 서식 문자를 하나씩 끼워 넣는 대신(새 서식마다 땜질이 필요하다) 검사 전에
# 서식을 제거한다 — L0가 봐야 하는 것은 표기가 아니라 수치 그 자체다.
#
# ⚠️ '~'는 취소선(~~) 서식이기도 하지만 한국어에서 **범위 표기**로 훨씬 자주 쓰인다
# ("5~6등급", "55~70세", "3.3~5.5%"). 무조건 지우면 범위의 양끝이 붙어 존재하지 않는
# 수치가 만들어진다 — 실측(S02): 답변 초안의 "위험등급 5~6등급"이 "56등급"으로 뭉개져
# L0가 "근거에 없는 수치 56등급"이라고 확정했다(근거에 있을 수가 없는 유령 값이다).
# 그래서 숫자 사이에 낀 단일 '~'는 범위 구분자로 보고 보존하며, 취소선으로 쓰인
# '~~'만 제거한다.
_STRIKETHROUGH_RE = re.compile(r"~~+")
_INLINE_MARKUP_RE = re.compile(r"[*_`]+")


def strip_inline_markup(text: str) -> str:
    """수치 대조를 방해하는 인라인 마크다운 서식 문자를 제거한다.

    서식은 의미가 아니라 표현이므로, 근거 대조 전에 걷어내야 "**16.5**%"와 "16.5%"가
    같은 사실로 취급된다. 원문을 바꾸지 않고 검사용 사본에만 적용한다.

    단, 범위 표기의 '~'("5~6등급")는 서식이 아니라 의미라서 보존한다 — 지우면
    양끝이 붙어 "56등급" 같은 유령 수치가 생긴다.
    """
    return _INLINE_MARKUP_RE.sub("", _STRIKETHROUGH_RE.sub("", text or ""))


# 범위 표기에서 **앞쪽 수치는 단위가 없어** 토큰 정규식에 잡히지 않는다
# ("5~6등급"의 5, "3.3~5.5%"의 3.3, "55~70세"의 55). 뒤쪽 단위를 앞쪽에도 나눠 붙여
# 양끝을 모두 검사 대상으로 만든다 — 안 그러면 범위의 앞 숫자를 지어내도 L0가 놓친다.
_NUMBER_RANGE_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*[~∼－-]\s*(\d[\d,]*(?:\.\d+)?\s?"
    r"(?:억\s?원|만\s?원|천\s?원|억(?![가-힣])|만(?![가-힣])|원|%|퍼센트|프로"
    r"|세(?![금율액대])|년|개월|등급))"
)


def _expand_number_ranges(text: str) -> str:
    """'5~6등급'을 '5등급~6등급'처럼 펼쳐 양끝 모두 토큰으로 잡히게 한다."""

    def _sub(match: re.Match) -> str:
        head, tail = match.group(1), match.group(2)
        unit = re.sub(r"^\d[\d,]*(?:\.\d+)?\s?", "", tail)
        return f"{head}{unit}~{tail}"

    return _NUMBER_RANGE_RE.sub(_sub, text)


def extract_number_tokens(text: str) -> list[str]:
    """텍스트에서 '숫자+단위' 토큰을 등장 순서대로 중복 없이 추출한다. 예: ['900만원', '16.5%']

    인라인 서식은 먼저 제거한다 — "**16.5**%"처럼 강조가 숫자와 단위를 갈라놓으면
    수치가 통째로 검사에서 빠져나간다.
    """
    seen: list[str] = []
    for m in _NUMBER_TOKEN_RE.finditer(_expand_number_ranges(strip_inline_markup(text))):
        token = m.group(0).strip()
        if token not in seen:
            seen.append(token)
    return seen


_WITHDRAWAL_TOPIC_ANSWER_MARKERS = {
    "DOCUMENTS": ("서류", "결정문", "확인서", "증명서", "진단서", "소견서"),
    "DEADLINE": ("신청기한", "이내", "언제까지", "피해발생일", "결정일"),
    "TAX": ("세금", "세율", "과세", "퇴직소득세", "기타소득세", "연금소득세"),
    "ELIGIBILITY": ("가능", "허용", "사유에 해당", "신청할 수", "신청할수"),
}
_WITHDRAWAL_TOPIC_LABELS = {
    "DOCUMENTS": "중도인출 필요서류",
    "DEADLINE": "중도인출 신청기한",
    "TAX": "중도인출 세금",
    "ELIGIBILITY": "중도인출 가능 여부",
    "ELIGIBILITY_PRECONDITION": "법정 중도인출 요건 충족 전제",
}


def apply_withdrawal_context_override(
    verification: dict,
    withdrawal_context: dict | None,
    draft: str,
) -> dict:
    """중도인출 Verifier가 질문 밖 요구사항을 새로 만들지 못하게 범위를 고정한다.

    explicit_topics와 사전 정의된 task_required_topics만 요구사항으로 인정한다. 사유·재원·
    수령방식은 여기서 다시 추정하지 않으며, grounding/issue 판정은 그대로 유지한다.
    """
    if not withdrawal_context:
        return verification

    explicit_topics = set(withdrawal_context.get("explicit_topics") or [])
    required_topics = set(withdrawal_context.get("task_required_topics") or [])
    if not explicit_topics:
        return verification

    compact_draft = re.sub(r"\s+", "", draft or "")
    missing: list[str] = []
    for topic in sorted(explicit_topics):
        markers = _WITHDRAWAL_TOPIC_ANSWER_MARKERS.get(topic, ())
        if markers and not any(marker in compact_draft for marker in markers):
            missing.append(_WITHDRAWAL_TOPIC_LABELS.get(topic, topic))

    if "ELIGIBILITY_PRECONDITION" in required_topics and not any(
        marker in compact_draft for marker in ("법정사유", "법정중도인출요건", "요건을충족")
    ):
        missing.append(_WITHDRAWAL_TOPIC_LABELS["ELIGIBILITY_PRECONDITION"])

    result = dict(verification)
    result["missing_requirements"] = missing
    result["requirements_met"] = not missing
    return result


def _numeric_core(token: str) -> str:
    """토큰의 숫자 부분만 콤마 제거 형태로 반환한다. '1,200만원' -> '1200'"""
    m = _NUMERIC_CORE_RE.match(token)
    return (m.group(0) if m else token).replace(",", "")


# "5천만원"처럼 아라비아 숫자와 단위 사이에 한글 보조단위가 낀 표기.
# deterministic_info._parse_korean_amount와 같은 대상을 다루지만, 여기서는 금액을
# 계산하는 게 아니라 "같은 값의 다른 표기"를 만들어 비교에 보태는 용도다.
_KOREAN_SUBUNIT_RE = re.compile(r"(\d[\d,]*)\s*(천|백|십)\s*(만|억)?")
_KOREAN_SUBUNIT_FACTOR = {"천": 1_000, "백": 100, "십": 10}


def _expand_korean_numerals(text: str) -> str:
    """한글 보조단위 표기를 아라비아 표기로 펼친 문자열을 만든다.

    "5천만원" -> "5000만 50000000" 처럼 초안이 쓸 법한 표기를 모두 만들어 두면,
    부분문자열 비교만으로도 같은 값을 알아본다.
    """
    expanded: list[str] = []
    for raw, subunit, big_unit in _KOREAN_SUBUNIT_RE.findall(text):
        base = int(raw.replace(",", "")) * _KOREAN_SUBUNIT_FACTOR[subunit]
        expanded.append(str(base))
        if big_unit == "만":
            expanded.append(str(base * 10_000))
        elif big_unit == "억":
            expanded.append(str(base * 100_000_000))
    return " ".join(expanded)


def find_unsupported_numbers(
    draft: str,
    evidence_texts: list[str],
    user_texts: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """초안의 수치 토큰 중 근거 원문 어디에도 숫자 부분이 등장하지 않는 것들을 반환한다.

    user_texts(현재 질문 + 이전 턴의 사용자 발화)에 등장한 수치는 지원된 것으로 본다 —
    "월 30만원 정도 투자 가능해" 같은 사용자 조건을 초안이 되받아 정리하는 것은
    할루시네이션이 아니다. 단 이전 턴의 '답변'은 절대 포함하지 않는다 — 과거 답변에 섞인
    오류 수치가 재인용을 정당화하면 안 된다 (⑤ 프롬프트의 재검증 원칙과 동일).

    콤마 표기 차이(1,200 vs 1200)와 인라인 마크다운 서식(1,**800**만원)은 정규화해서
    비교한다. 숫자 부분의 부분문자열 일치만 보므로 "지원됨" 쪽으로 관대하다 — 여기서
    잡히지 않은 표기 차이(9백만 원 등)는 ④ LLM이 의심 목록을 근거와 대조할 때 걸러진다.

    한글 수사("5천만원")는 아라비아 표기("5,000만원")로도 함께 펼쳐 비교한다 —
    사용자가 "연봉 5천만원"이라 쓰고 답변이 "총급여 5,000만원"으로 되받는 것은
    할루시네이션이 아닌데, 표기가 달라 의심 목록에 올라가면 ④ LLM이 그 목록에서
    "진짜 지어낸 값"을 골라내는 일이 그만큼 어려워진다(실측 no.486: 의심 4개 중
    3개가 이런 표기 차이였고, 정작 진짜 할루시네이션인 "700만원"은 확정에서
    누락돼 최종 답변에 그대로 나갔다).
    """
    support_text = strip_inline_markup(" ".join([*evidence_texts, *user_texts]))
    normalized_support = support_text.replace(",", "") + " " + _expand_korean_numerals(support_text)
    return [
        token
        for token in extract_number_tokens(draft)
        if not _numeric_core_supported(_numeric_core(token), normalized_support)
    ]


def _numeric_core_supported(core: str, normalized_support: str) -> bool:
    """core(예: "60")가 근거 텍스트에 그 자체 숫자로 등장하는지, 다른 숫자에 우연히
    포함된 것인지 구분한다.

    ⚠️ 예전엔 `core in normalized_support`(부분문자열 포함)로만 판정해서, "60%"가
    근거의 "6,000,000원"(콤마 제거 후 "6000000")에 "60"이 부분문자열로 들어있다는
    이유만으로 "지원됨"으로 오판됐다. 실측 no.26/123: DB형 급여 계산식이 근거에
    없는데도 "평균임금의 60% × 근속연수"라는 지어낸 공식이 검증을 그대로 통과해
    L0 오버라이드도, ⑤의 디스클레이머도 발동하지 못했다 — 부분문자열 일치가 검증
    전체를 무력화하는 구멍이었다.

    앞뒤가 숫자가 아닌 위치(단어 경계)에서 core가 등장할 때만 "지원됨"으로 본다.
    """
    if not core:
        return True
    pattern = re.compile(rf"(?<!\d){re.escape(core)}(?!\d)")
    return pattern.search(normalized_support) is not None


def apply_l0_overrides(verification: dict, suspects: list[str], has_evidence: bool) -> dict:
    """④ LLM의 판정 위에 결정론적 오버라이드를 적용한 verification dict를 반환한다.

    - 근거 0건 + 초안에 수치 존재 → 무조건 grounded=False (LLM 판정 무시)
    - LLM이 의심 목록 중 실제 근거 부재로 확인(unsupported_numbers_confirmed)한 수치가
      있으면 grounded=False (확인 목록은 의심 목록의 부분집합으로 강제 — LLM이 목록에
      없던 수치를 지어내 넣어도 무시)
    - LLM 스스로 grounded=False로 판정한 것은 그대로 유지
    """
    result = dict(verification)
    issues = list(result.get("issues") or [])
    confirmed = [n for n in (result.get("unsupported_numbers_confirmed") or []) if n in suspects]

    if not has_evidence and suspects:
        result["grounded"] = False
        issues.append(f"근거가 0건인데 초안에 구체적 수치가 있습니다: {', '.join(suspects)}")
    elif confirmed:
        result["grounded"] = False
        issues.append(f"근거 원문에 없는 수치입니다: {', '.join(confirmed)}")

    # ④ LLM이 issues에 위반 사항을 적어놓고도 grounded=True로 통과시키는 모순을 코드가
    # 바로잡는다 — 실측: 개별 펀드 2건을 근거로 "연금저축 펀드는 일반적으로 환매 제한이
    # 없다"고 단정한 초안에 대해 LLM이 issues에는 "일부 펀드 정보를 일반 규칙으로 단정했다"고
    # 정확히 지적하면서도 grounded=True를 반환했다. 판정과 근거가 어긋나면 판정을 신뢰할 수
    # 없으므로 위반 쪽으로 확정한다 (L0가 프롬프트 순종에 의존하지 않는다는 원칙과 동일).
    if issues and result.get("grounded"):
        result["grounded"] = False

    result["issues"] = issues
    result["unsupported_numbers_confirmed"] = confirmed
    result["l0_suspect_numbers"] = suspects
    return result


def apply_clarification_override(verification: dict) -> dict:
    """역질문(needs_clarification) 초안에 대한 요구사항 검증 면제를 강제한다.

    ②③이 조건 불충분으로 의도적으로 답을 유보하고 역질문했을 때, ④가 이를 "요구사항
    미충족"으로 판정하면 ⑤가 근거를 긁어 추천/답변을 되살리려 든다 — "단정적 추천 금지"
    원칙을 파이프라인 스스로 무너뜨리는 경로라 코드로 차단한다. grounded/premise 검증은
    그대로 유지한다 (역질문 문장에도 근거 없는 수치가 섞이면 안 되므로).
    """
    result = dict(verification)
    result["requirements_met"] = True
    result["missing_requirements"] = []
    result["clarification_mode"] = True
    return result


# ── ⑤ 생성기 출력 강제 (L0와 같은 사상: 반드시 지켜야 하는 것은 코드로 강제) ──────
#
# ④는 L0 오버라이드로 코드 방어선을 깔아뒀지만, ⑤는 검증 결과를 프롬프트 텍스트로
# 넘기고 "지켜달라"고 부탁만 했다 — verification.py 자신이 경고한 "프롬프트 순종은
# 확률적으로 실패한다"가 ⑤에서 그대로 재현됐다(실측 4/4 위반).
#
# ⚠️ 설계 근거 (실측으로 확정한 것):
# "⑤ 출력에 L0(find_unsupported_numbers)를 재적용한다"는 처방은 이 실패들을 못 잡는다.
#   - S1(2027년 개편안): 답변의 수치 13개가 전부 근거(doc38~41)에 실재해 L0 통과.
#     실패는 "지어낸 숫자"가 아니라 "묻지 않은 걸 답하고 한계를 고지하지 않은 것".
#   - M2(DC 상품 미선택): 답변에 수치 토큰이 0개라 L0가 검사할 대상 자체가 없다.
# 따라서 여기서 강제하는 것은 수치 대조가 아니라 **"④가 지적한 사항이 답변에 실제로
# 반영됐는가"**이다. 반영되지 않았으면 코드가 문장을 덧붙여 확정한다.

_LIMIT_DISCLOSURE_MARKERS = (
    "확인이 어렵", "확인하기 어렵", "확인되지 않", "확인할 수 없",
    "제공된 자료", "보유한 자료", "자료에 없", "자료에는 없",
    "포함되어 있지 않", "안내드리기 어렵", "답변드리기 어렵", "알 수 없",
)

_PREMISE_CORRECTION_MARKERS = (
    "말씀하신", "알려진 것과", "정확히는", "사실과 다", "오해", "그렇지 않",
    "완전히 자유로운 것은 아", "만큼 크지는",
)


def has_limit_disclosure(answer: str) -> bool:
    """답변이 '이건 확인이 어렵다'는 한계 고지를 실제로 담고 있는지 판정한다."""
    return any(marker in (answer or "") for marker in _LIMIT_DISCLOSURE_MARKERS)


def has_premise_correction(answer: str) -> bool:
    """답변이 질문의 잘못된 전제를 바로잡는 문장을 담고 있는지 판정한다."""
    return any(marker in (answer or "") for marker in _PREMISE_CORRECTION_MARKERS)


def _requirement_already_covered(answer: str, requirement: str) -> bool:
    """missing_requirements 항목이 실제로는 답변 본문에 이미 서술돼 있는지 확인한다.

    ④(LLM)는 같은 결함을 requirements_met과 grounded 양쪽에 겹쳐서 반영하는 경우가
    있다 — 완결성 부족을 grounded=False의 issues에도 함께 적는 식이다. 그런데 더 심각한
    건 이 방향의 자기모순이다: missing_requirements에 적은 항목의 숫자가 답변 본문에
    이미 정확히 들어있는 경우(실측 no.6 "DC형은 회사가 매년 얼마를 넣어주는지 알 수
    있나요?" — 답변 본문에 "매년 연간 임금총액의 **1/12** 이상"이 이미 있는데도
    missing_requirements에 같은 내용이 올라와, 무조건 붙이면 "1/12 이상이라는 점은
    확인이 어려워 포함하지 못했습니다"라는 자기모순 문장이 뒤에 따라붙는다).

    완전한 의미 판정은 불가능하므로, 항목에 등장하는 숫자 토큰이 답변 본문에 그대로
    있으면 "이미 다뤘다"고 본다 — 애매하면 그냥 붙이는 쪽(기존 동작)을 유지해, 실제로
    빠진 항목을 숨기는 반대 방향 실패는 만들지 않는다.

    ⚠️ 단, 항목 텍스트가 "④의 판정 서술문"(예: "질문은 '74세'라는 특정 나이에서의
    세율을 묻고 있으나, 초안은 이를 다루지 않고...")이면 이 필터를 적용하지 않는다.
    이런 서술문은 질문에 나온 숫자를 그대로 인용하는 경우가 많아, 그 숫자가 답변
    본문에도 우연히 있으면 "이미 다뤘다"고 오판한다(실측 no.85: missing 항목이
    "질문은 '74세'..."였는데 답변 본문에도 "만 74세"가 있어 오판 — 실제로는 ④가
    지적한 결함이 그대로 남아있는 채였다). 판정 서술문은 "~하지만/하나", "묻고
    있으나", "질문은" 같은 대조·인용 표현으로 사실 항목 나열과 구분된다.
    """
    if any(marker in requirement for marker in ("질문은", "하지만", "하나,", "있으나", "묻고 있")):
        return False
    requirement_numbers = extract_number_tokens(requirement)
    if not requirement_numbers:
        return False
    answer_core = strip_inline_markup(answer).replace(",", "")
    return all(_numeric_core(token) in answer_core for token in requirement_numbers)


def enforce_missing_requirements(answer: str, missing: list[str]) -> str:
    """④가 '질문이 요구했는데 빠졌다'고 지적한 항목을 답변이 다루지 않았으면 한계를 명시한다.

    빠진 항목을 지어내 채우는 게 아니라, **답하지 못했다는 사실 자체를 드러내는 것**이
    목적이다 (대회 평가지표 "정보한계 대응": 무리한 답변 대신 한계 고지 또는 역질문).

    이미 한계를 고지한 답변에는 덧붙이지 않는다 — 중복 고지는 답변 품질을 떨어뜨린다.
    항목별로도 답변 본문에 이미 서술된 것은 걸러낸다(자기모순 방지, _requirement_already_covered).
    """
    if not missing or has_limit_disclosure(answer):
        return answer
    missing = [item for item in missing if not _requirement_already_covered(answer, item)]
    if not missing:
        return answer
    items = "".join(f"\n- {m}" for m in missing)
    return (
        f"{answer}\n\n"
        f"다만 다음 항목은 제공된 자료만으로는 확인이 어려워 답변에 포함하지 못했습니다:{items}\n"
        "해당 부분은 가입하신 금융기관이나 관련 기관에 확인해 주시기 바랍니다."
    )


# 수치 뒤에 이 표현이 이어지면 그 수치를 사실로 주장하는 게 아니라 "틀렸다"고 바로잡는
# 문맥이다. 실측(500문항 평가 스크리닝 중 발견): "평균 임금의 60%가 아니라 30일분에"처럼
# 틀린 수치를 부정하며 교정하는 답변까지 위반으로 잡으면 오히려 올바른 답변에 경고가
# 붙는다. eval/screen_results.py의 _is_asserted와 같은 판정을 코드 강제에도 적용한다.
_NUMBER_NEGATION_MARKERS = (
    "가 아니", "이 아니", "은 아니", "는 아니", "아닙니다", "아니라",
    "가 아닌", "이 아닌", "잘못", "오해", "사실과 다",
)


def _number_is_asserted(answer: str, number_token: str) -> bool:
    """답변이 그 수치를 사실로 주장하는지 판정한다 (부정 문맥에서만 등장하면 False)."""
    core = _numeric_core(number_token)
    normalized = strip_inline_markup(answer).replace(",", "")
    start = 0
    while True:
        idx = normalized.find(core, start)
        if idx == -1:
            return False
        tail = normalized[idx + len(core): idx + len(core) + 25]
        if not any(marker in tail for marker in _NUMBER_NEGATION_MARKERS):
            return True
        start = idx + len(core)


# 문장 분리 — 마침표·물음표·느낌표 뒤 공백, 또는 줄바꿈을 경계로 본다.
# 목록 항목("- 2013년 3월 1일 이후...")도 줄 단위로 하나의 문장처럼 다룬다.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# 번호 목록 항목("1. ", "  2. "). 이런 줄은 통째로 보존한다 — 한 항목만 지우면
# 번호가 어긋나고, 다시 매기면 원문 서식을 더 훼손한다.
_NUMBERED_ITEM_RE = re.compile(r"^[ \t]*\d+\.[ \t]+\S")

# 이 표현이 문장에 있으면 삭제하지 않는다 — 수치를 부정·교정하거나 한계를 고지하는
# 문장은 지우면 오히려 정확한 정보가 사라진다.
_SENTENCE_KEEP_MARKERS = _NUMBER_NEGATION_MARKERS + _LIMIT_DISCLOSURE_MARKERS


def _drop_sentences_with_numbers(answer: str, leaked: list[str]) -> tuple[str, list[str]]:
    """확정된 미지원 수치를 '사실로 주장하는 문장'만 통째로 제거한다.

    반환값은 (정리된 답변, 실제로 제거하지 못해 남은 수치들)이다.

    ⚠️ 왜 문장 단위인가: 수치만 지우면 문장이 깨진다("연금소득세율은 %입니다").
    문장을 통째로 들어내면 나머지 문장은 온전하다. 실측으로 확인한 결과 지어낸
    주장은 한 문장에 고립돼 있었다 —
      no.27  "...평균 임금의 60% 이상으로 계산된다는 규정이 있으나..."  (10문장 중 1개)
      no.56  "- 2013년 3월 1일 이후 가입한 연금 계좌의 자금을..."       (19문장 중 1개)
    둘 다 근거에 없는 규정을 사실처럼 서술한 문장이고, 지워도 답변의 나머지 설명은
    그대로 성립한다.

    ⚠️ 보수적으로 동작한다 — 다음 경우에는 지우지 않고 기존처럼 경고만 붙인다:
      - 부정·교정 문맥("60%가 아니라")이나 한계 고지 문맥의 문장
      - 그 문장을 지우면 본문이 거의 남지 않는 경우(문장이 1~2개뿐인 짧은 답변)
    답변을 과하게 훼손하는 것은 할루시네이션 못지않게 나쁘기 때문이다.
    """
    body, separator, tail = answer.partition("참고 근거:")
    kept_lines: list[str] = []
    removed_numbers: set[str] = set()
    for line in body.split("\n"):
        # 번호 목록 항목은 건드리지 않는다 — 한 항목만 지우면 번호가 어긋나고
        # (1. 다음에 3.) 재부여는 원문 서식을 더 망가뜨린다. 목록 안의 지어낸
        # 값은 경고로 처리한다(실측 no.140).
        if _NUMBERED_ITEM_RE.match(line):
            kept_lines.append(line)
            continue

        # 한 줄 안에서 문장 단위로만 들어낸다 — 줄바꿈은 서식이므로 보존한다.
        pieces = _SENTENCE_SPLIT_RE.split(line)
        kept_pieces: list[str] = []
        for piece in pieces:
            target = next(
                (
                    n
                    for n in leaked
                    if _numeric_core(n) in strip_inline_markup(piece).replace(",", "")
                ),
                None,
            )
            if target is not None and not any(m in piece for m in _SENTENCE_KEEP_MARKERS):
                removed_numbers.add(target)
                continue
            kept_pieces.append(piece)
        if pieces and not kept_pieces:
            continue  # 줄 전체가 지어낸 내용이면 줄째로 뺀다
        kept_lines.append(" ".join(p for p in kept_pieces if p).strip())

    if not removed_numbers:
        return answer, leaked

    # 남은 분량은 **문장 수**로 센다. 줄 수로 세면 여러 문장이 한 줄에 있는 답변에서
    # "1줄이니 너무 짧다"고 오판해 삭제가 통째로 무산된다.
    def _sentence_count(text: str) -> int:
        return len([s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()])

    kept_sentences = _sentence_count("\n".join(kept_lines))
    original_sentences = _sentence_count(body)
    # 본문이 절반 넘게 사라지거나 거의 남지 않으면 삭제하지 않는다 — 지나친 훼손은
    # 할루시네이션 못지않게 나쁘므로 경고로 물러선다.
    if kept_sentences < 2 or kept_sentences * 2 < original_sentences:
        return answer, leaked

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    still_leaked = [n for n in leaked if n not in removed_numbers]
    return f"{cleaned}\n\n{separator}{tail}" if separator else cleaned, still_leaked


def enforce_unsupported_numbers(answer: str, confirmed: list[str]) -> str:
    """④가 '근거에 없다'고 확정한 수치를 최종 답변에서 제거하거나, 못 지우면 경고를 붙인다.

    ⑤ 프롬프트에 이 목록을 넘기고 "쓰지 말라"고 부탁하지만(generator.py), 그 실측
    4/4 위반이 이 프로젝트가 "프롬프트 순종은 확률적으로 실패한다"는 원칙을 세운
    근거였다 — missing_requirements/premise_issues는 이미 코드로 강제하면서 정작
    grounded=False의 핵심 증거인 unsupported_numbers_confirmed는 강제가 없었다.

    ⚠️ 예전에는 경고만 붙이고 본문은 그대로 뒀다("수치를 지우면 문장이 깨진다"는
    이유). 그러나 사용자는 본문을 먼저 읽고 경고는 맨 아래에 있어, **처음 읽을 때는
    지어낸 값을 사실로 받아들인다.** 실측:
      no.27 "평균 임금의 60% 이상으로 계산된다는 규정"  (근거에 없는 계산식)
      no.56 "2013년 3월 1일 이후 가입한 계좌는 이전 불가"  (근거에 없는 규정)
      실사용 "연간 최대 700만원까지 세액공제"            (2023년 폐지된 한도)
    수치만 지우면 문장이 깨지는 것은 맞지만, **문장을 통째로 들어내면** 나머지는
    온전하다. 그래서 문장 단위 제거를 먼저 시도하고, 안전하지 않으면 경고로 물러선다.

    이미 부정 문맥으로 쓰였다면(수치를 틀렸다고 바로잡는 중) 손대지 않는다 — 그건
    할루시네이션이 아니라 올바른 답변이다.
    """
    leaked = [n for n in confirmed if _number_is_asserted(answer, n)]
    if not leaked:
        return answer

    answer, leaked = _drop_sentences_with_numbers(answer, leaked)
    if not leaked:
        return answer

    items = ", ".join(dict.fromkeys(leaked))
    return (
        f"{answer}\n\n"
        f"※ 위 답변에 포함된 다음 수치는 제공된 자료에서 확인되지 않아 참고용입니다: "
        f"{items}. 정확한 수치는 가입하신 금융기관에 확인해 주시기 바랍니다."
    )


# ④가 premise_issues에 "답변의 결함"을 적어 넣는 경우가 있다. 실측 사례:
#   - "2027년 연금 세제 개편안 확정 내용"      (자료에 없어 못 답한 것)
#   - "초안이 날짜에 직접 답하지 않음"          (답변 누락)
# 이건 "질문의 잘못된 전제"가 아니라 "요구사항 미충족"이다. 그대로 두면 최종 답변이
# "먼저 질문에 담긴 전제를 짚고 넘어가겠습니다: 초안이 날짜에 직접 답하지 않음"처럼
# 사용자가 하지도 않은 말을 전제라고 지적하는 이상한 문장으로 시작한다.
#
# 개별 문구를 블랙리스트로 거르면 표현이 바뀔 때마다 다시 뚫리므로, **서술 대상**으로
# 구분한다: 진짜 전제는 사용자 발화를 인용하고(~다던데/~라는데/~맞죠), 오분류된 항목은
# 답변·초안의 상태를 서술한다(~답하지 않음/~누락/~부족).
_ANSWER_DEFECT_MARKERS = (
    "답하지 않", "답변하지 않", "언급하지 않", "다루지 않", "포함하지 않",
    "제공하지 않", "제시하지 않", "계산하지 않", "설명하지 않", "반영하지 않",
    "누락", "빠졌", "빠져", "부족", "직접 답",
    "확인되지 않", "확인할 수 없", "정보가 없", "자료에 없",
)
# 사용자가 실제로 한 말을 인용하는 표현. 이게 있으면 진짜 전제로 본다.
# ⚠️ "라고 하여"처럼 인용 어미가 결함 서술 안에 섞이는 경우가 있어("질문은 '...'라고 하여
# 정보를 제공하지 않고 있으며"), 인용 표현만으로 단정하지 않고 결함 표현과 함께 본다.
_USER_PREMISE_MARKERS = (
    "다던데", "라던데", "다는데", "라는데", "맞죠", "맞나요", "아닌가요",
    "들었", "알고 있", "다고 하",
)
# ④가 "초안/질문이 ~하다"처럼 답변 과정을 서술하는 주어. 이게 등장하면 사용자 발화가
# 아니라 시스템 내부 상태를 말하는 것이다.
_META_SUBJECT_MARKERS = ("초안", "답변이", "답변은", "질문은", "질문이")
# ⚠️ "이라는 전제"·"라는 전제"는 여기 두면 안 된다(제거 이력). "전제"는 ④가 **모든**
# premise 항목에 붙이는 일반 용어라, 조건 되뇜과 진짜 오류를 전혀 구분하지 못한다.
# 게다가 "이라는/라는"은 앞 명사의 받침 유무로 갈리는 조사일 뿐이라, 판정이 의미가
# 아니라 철자에 좌우됐다:
#   "IRP는 원금보장 상품이라는 전제"   -> 필터링(진짜 오류인데 사라짐)
#   "중도인출이 가능하다는 전제"        -> 유지
# 실측: 이 마커 때문에 진짜 전제 오류 5개 중 4개가 조용히 걸러졌다(IRP 원금보장,
# 위험등급 6등급, DB형 중도인출, 폐지된 700만원 한도). 원래 의도했던 "사용자가 준
# 조건을 되뇐 항목"(잔금지급일 2026-01-31 등)은 아래 숫자 분기가 이미 전부 커버한다.
_BENIGN_CONDITION_MARKERS = (
    "이라고 가정",
    "라고 가정",
    "나이가",
    "연금으로 받을",
    "연금으로 받을 때",
    "잔금지급일이",
    "피해발생일이라는",
    "결정일이",
    "DB형",
    "개인워크아웃",
)
_FALSE_PREMISE_MARKERS = ("사실과 다", "과장", "잘못", "오해", "자유롭", "무조건", "반드시")


# 사용자의 요청·목표·선호·희망을 서술하는 종결 표현. premise_issues는 "참·거짓을 따질 수
# 있는 사실 주장"만 담아야 하는데, ④가 사용자의 **바람**을 여기에 넣는 사고가 반복됐다.
#
# 실측 2건(같은 클래스, 표현만 다름):
#   "안정적인 것을 원한다"        (Q-4 "솔로몬 국공채... 안정적인 걸 원해요")
#   "노후를 위한 절세 방법이 필요함" ("65세 정년퇴직... 절세를 많이 하고 싶어")
# 둘 다 최종 답변이 "다음 내용은 사실과 다르거나 과장된 부분이 있어 그대로 전제하기
# 어렵습니다: 노후를 위한 절세 방법이 필요함"으로 시작해 사용자의 요청을 반박했다.
#
# 개별 문구를 _BENIGN_CONDITION_MARKERS에 추가하는 방식으로는 못 막는다 — 1차 사고 뒤에도
# 표현만 바뀐 2차 사고가 났다. 어휘가 아니라 **문장의 종류**로 판정한다: 욕구·필요·의향
# 서술어는 유한한 문법 범주라 어휘 목록보다 표현 변형에 견고하다.
_WANT_STATEMENT_ENDINGS = (
    "원한다", "원함", "원해", "원하심", "원하십니다",
    "하고싶다", "하고싶음", "하고싶어", "싶다", "싶음", "싶어", "싶어함", "싶어한다",
    "필요하다", "필요함", "필요해", "필요로한다", "필요성",
    "바란다", "바람", "희망한다", "희망함", "희망",
    "요청", "요청함", "문의", "문의함", "알고싶다", "알고싶음",
)


def is_want_statement(text: str) -> bool:
    """사용자의 요청·목표·선호를 담은 문장인지(= 참·거짓이 없는 진술인지) 판정한다.

    사용자가 무언가를 원한다는 사실 자체는 틀릴 수가 없으므로, 이런 항목은 "잘못된
    전제"가 될 수 없다. 종결부로 판정한다 — "노후를 위한 절세 방법이 필요함"처럼
    명사구로 끝나도 마지막 서술어가 욕구·필요를 나타내면 요청 진술이다.

    ⚠️ 호출부는 _FALSE_PREMISE_MARKERS("사실과 다"·"과장"·"잘못" 등)를 먼저 확인해야
    한다 — "안전하다고 잘못 알고 원함"처럼 사실 주장이 섞인 항목까지 걷어내면 진짜
    전제 오류를 놓친다.
    """
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False
    return any(compact.endswith(ending) for ending in _WANT_STATEMENT_ENDINGS)


# "제도가 이렇게 작동한다"는 주장에 쓰이는 서술어. 사용자가 준 **자기 조건**(잔금지급일,
# 본인 나이, 수령연차)과 **제도에 대한 주장**(원금보장이다, 중도인출이 된다, 한도가 얼마다)을
# 가르는 신호다. 전자는 되뇜이라 교정 대상이 아니고, 후자는 틀렸으면 반드시 바로잡아야 한다.
_INSTITUTIONAL_CLAIM_MARKERS = (
    "위험하", "안전하", "가능하", "불가능", "된다", "안된다", "안 된다",
    "보장", "유리", "불리", "면제", "비과세", "과세되", "적용되", "허용",
    "높다", "낮다", "같다", "다르다", "이다", "입니다",
)
# 제도의 기준값을 주장하는 표현. 사용자 조건은 "내 상황이 얼마"이고, 제도 주장은
# "규정상 얼마"다 — 후자는 숫자가 들어 있어도 검증 대상이다.
_INSTITUTIONAL_VALUE_MARKERS = ("한도", "기준", "요건", "세율", "공제율", "등급")


def asserts_institutional_rule(text: str) -> bool:
    """항목이 "제도가 이렇게 작동한다"는 주장인지(= 참·거짓 검증 대상인지) 판정한다.

    사용자가 제시한 자기 조건("잔금지급일이 2026년 1월 31일")과 구분하기 위한 것이다.
    조건 되뇜은 값을 그대로 옮길 뿐이지만, 제도 주장은 그 값이나 성질이 **맞는지 틀리는지**를
    말한다 — 틀렸다면 답변 앞머리에서 바로잡아야 하는 바로 그 대상이다.
    """
    return any(marker in text for marker in _INSTITUTIONAL_CLAIM_MARKERS) or any(
        marker in text for marker in _INSTITUTIONAL_VALUE_MARKERS
    )


def is_answer_defect_statement(text: str) -> bool:
    """premise_issues 항목이 '질문의 전제'가 아니라 '답변의 결함'을 서술하는지 판정한다."""
    has_defect = any(marker in text for marker in _ANSWER_DEFECT_MARKERS)
    if not has_defect:
        return False
    # 결함 표현이 있고 초안·답변·질문을 주어로 서술하면 메타 서술로 확정한다.
    if any(marker in text for marker in _META_SUBJECT_MARKERS):
        return True
    # 결함 표현은 있으나 사용자 발화 인용이면 진짜 전제로 남긴다.
    return not any(marker in text for marker in _USER_PREMISE_MARKERS)


def is_benign_condition_statement(text: str) -> bool:
    """사용자가 제공한 조건을 단순히 '전제/가정'이라고 반복한 항목인지 판정한다.

    ④가 "잔금지급일이 2026년 1월 31일이라고 가정"처럼 사용자가 준 조건을
    premise_issues에 넣는 경우가 있다. 이는 잘못된 전제가 아니므로 최종 답변 앞머리에
    교정문으로 붙이면 안 된다. 단 "중도인출이 자유롭다"처럼 실제로 틀릴 수 있는
    제도 전제는 유지한다.
    """
    if any(marker in text for marker in _FALSE_PREMISE_MARKERS):
        return False
    # 사용자의 요청·목표·선호는 참·거짓이 없으므로 교정 대상이 될 수 없다.
    # (_FALSE_PREMISE_MARKERS 확인 뒤에 둔다 — 사실 주장이 섞였으면 그쪽이 우선)
    if is_want_statement(text):
        return True
    # "제도가 이렇게 작동한다"는 주장은 사용자가 준 조건이 아니라 검증 대상이다.
    # 아래 무해 분기들(숫자 포함, 중도인출 문맥 등)보다 먼저 확인해야 한다 — 그 분기들은
    # 주제어만 보므로 제도 주장까지 함께 삼킨다(실측: "연금저축 한도가 700만원이라는
    # 전제"가 숫자+"연금" 조합으로, "DB형도 중도인출이 된다는 전제"가 "DB형" 마커로
    # 무해 처리돼 폐지된 한도·틀린 제도 이해를 교정하지 못했다).
    if asserts_institutional_rule(text):
        return False
    if re.search(r"\d", text) and any(
        marker in text
        for marker in (
            "조건",
            "전제",
            "가정",
            "세금",
            "세액공제",
            "연금",
            "중도인출",
            "실물이전",
        )
    ):
        return True
    if "중도인출" in text and any(
        marker in text
        for marker in (
            "때문에",
            "하려고",
            "신청",
            "전세보증금",
            "전월세",
            "주택구입",
            "집을",
            "집",
            "사려고",
            "할래",
            "하려",
            "DB형",
            "IRP",
        )
    ):
        return True
    return any(marker in text for marker in _BENIGN_CONDITION_MARKERS)


def _compact_for_duplicate(text: str) -> str:
    """중복 판정을 위해 공백·서식·구두점 차이를 제거한다."""
    return re.sub(r"[\s·,.\-—~()\[\]]", "", strip_inline_markup(text or "")).lower()


def split_premise_issues(
    premise_issues: list[str], missing_requirements: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """premise_issues를 (진짜 전제, 실제로는 요구사항 미충족인 항목)으로 나눈다.

    ⚠️ missing_requirements를 함께 넘기면 **같은 항목이 양쪽에 중복 기재된 경우**를
    무조건 misfiled로 분류한다. 이건 마커 판정보다 강한 신호다 — ④가 "질문이 요구했는데
    초안이 빠뜨린 항목"이라고 적은 바로 그 문자열을 동시에 "질문의 잘못된 전제"라고
    부르는 것은 정의상 성립하지 않는다(사용자가 요구한 것이 곧 사용자의 틀린 전제일 수는
    없다). 둘 중 하나는 오분류이고, 성격상 요구사항 미충족 쪽이 맞다.

    실측(eval no.199 "솔로몬 국공채 단기·중장기·장기, 뭐가 달라요?"):
      missing_requirements = ["솔로몬 국공채 단기·중장기·장기 펀드의 구체적인 차이점과 특징"]
      premise_issues       = ["솔로몬 국공채 단기·중장기·장기 펀드의 구체적인 차이점과 특징"]
    이 항목은 명사구라 _ANSWER_DEFECT_MARKERS("~다루지 않", "누락" 등)에 하나도 걸리지
    않아 is_answer_defect_statement가 False를 냈고, 진짜 전제로 통과해 ⑤ 프롬프트의
    "premise_issues가 있으면 답변 시작 부분에서 그 전제를 바로잡아라" 지시를 발동시켰다.
    그 결과 ③이 8개 펀드를 정확히 조회해 놓고도 최종 답변 첫 문장이 "제공된 자료에서
    찾을 수 없으므로 정확히 알려드릴 수 없습니다"가 됐다(grounded=True였는데도).

    마커 목록에 명사구 패턴을 더 넣는 방식으로는 막을 수 없다 — 결함 항목은 어떤
    명사구로도 표현될 수 있어 블랙리스트가 계속 뚫린다. 중복이라는 구조적 사실로 잡는다.
    """
    duplicated = {_compact_for_duplicate(m) for m in (missing_requirements or [])}
    duplicated.discard("")
    real_premises: list[str] = []
    misfiled_defects: list[str] = []
    for issue in premise_issues:
        if is_benign_condition_statement(issue):
            continue
        is_duplicate = _compact_for_duplicate(issue) in duplicated
        if is_duplicate or is_answer_defect_statement(issue):
            misfiled_defects.append(issue)
        else:
            real_premises.append(issue)
    return real_premises, misfiled_defects


def apply_premise_issue_normalization(verification: dict) -> dict:
    """premise_issues 오분류를 verification 단계에서 정리한다.

    generator에서만 정리하면 최종 답변은 괜찮아도 think_trace에는 여전히
    "전제 교정: 잔금지급일이 ...라고 가정"처럼 남는다. 검증 dict 자체를 정규화해
    이후 모든 계층이 같은 해석을 보게 한다.
    """
    result = dict(verification)
    missing = list(result.get("missing_requirements") or [])
    real, misfiled = split_premise_issues(list(result.get("premise_issues") or []), missing)
    missing.extend(item for item in misfiled if item not in missing)
    result["premise_issues"] = real
    result["missing_requirements"] = missing
    if misfiled and result.get("requirements_met") is not False:
        result["requirements_met"] = False
    return result


def apply_source_limited_override(
    verification: dict,
    draft: str,
    evidence_texts: list[str],
) -> dict:
    """DB가 exact-date 계산 방식을 정의하지 않는다고 밝힌 응답을 검증 실패로 보지 않는다.

    중도인출 문서에는 "1개월/3개월 이내"는 있지만 exact date 환산 방식이 없다.
    초안이 이를 근거로 "DB 근거만으로 특정 날짜를 단정하지 않겠다"고 답한 경우,
    ④ LLM이 "회피"라고 판단해 grounded=False를 낼 수 있다. 이 케이스는 hallucination이
    아니라 근거 한계 대응이므로, 해당 이슈를 제거하고 요구사항을 충족한 것으로 본다.
    """
    evidence = "\n".join(evidence_texts)
    if "calculation_basis=not_defined_in_source" not in evidence:
        return verification
    if "DB 근거만으로" not in draft and "정확한 날짜로 계산하는 방식" not in draft:
        return verification

    result = dict(verification)
    source_limit_markers = ("회피", "직접적인 답변", "직접 답변", "정확히 판정하지")
    issues = [
        issue
        for issue in (result.get("issues") or [])
        if not any(marker in issue for marker in source_limit_markers)
    ]
    result["issues"] = issues
    if not issues and not result.get("unsupported_numbers_confirmed"):
        result["grounded"] = True
    result["requirements_met"] = True
    result["missing_requirements"] = []
    result["source_limited_mode"] = True
    return result


def _compact_for_requirement(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _question_explicitly_requests_alternative_withdrawal_plans(question: str) -> bool:
    text = _compact_for_requirement(question)
    if "중도인출" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "다른제도",
            "다른퇴직연금",
            "다른계좌",
            "가능한제도",
            "가능한종류",
            "어떤제도",
            "어떤종류",
            "무슨제도",
            "종류도",
            "뭐가가능",
            "무엇이가능",
        )
    )


def _question_has_housing_deposit_withdrawal_anchor(question: str) -> bool:
    """전세/보증금 단어를 일반 전세대출 문맥으로 확장하지 않기 위한 도메인 앵커."""
    text = _compact_for_requirement(question)
    if "중도인출" not in text:
        return False
    return any(marker in text for marker in ("전세", "전월세", "보증금", "임대차"))


def _question_explicitly_requests_lease_loan_info(question: str) -> bool:
    text = _compact_for_requirement(question)
    return any(marker in text for marker in ("전세대출", "대출", "중도상환", "수수료"))


def _missing_is_lease_loan_expansion(item: str) -> bool:
    text = _compact_for_requirement(item)
    has_lease_context = any(marker in text for marker in ("전세", "전월세", "보증금", "임대차"))
    has_loan_context = any(marker in text for marker in ("대출", "중도상환", "수수료"))
    return has_lease_context and has_loan_context


def _premise_is_pension_context_false_negative(issue: str, question: str, draft: str) -> bool:
    """④가 전세 중도인출을 전세대출 문맥으로 잘못 재분류한 전제 오판을 제거한다."""
    if not _question_has_housing_deposit_withdrawal_anchor(question):
        return False
    issue_text = _compact_for_requirement(issue)
    draft_text = _compact_for_requirement(draft)
    says_wrong_pension_context = (
        "퇴직연금" in issue_text
        and any(marker in issue_text for marker in ("맞지않", "관련내용을다루고", "질문에맞지"))
    )
    draft_has_pension_withdrawal = "중도인출" in draft_text and any(
        marker in draft_text for marker in ("퇴직연금", "irp", "dc", "전월세보증금")
    )
    return says_wrong_pension_context and draft_has_pension_withdrawal


def _premise_is_inferred_alternative_plan_claim(issue: str, question: str) -> bool:
    """다른 제도 질문에서 사용자가 하지 않은 'DB형 아니면 모두 가능'류 전제를 제거한다."""
    if not _question_explicitly_requests_alternative_withdrawal_plans(question):
        return False
    text = _compact_for_requirement(issue)
    return "db형" in text and any(
        marker in text
        for marker in (
            "db형이아니면",
            "db형아니면",
            "아니면중도인출가능",
            "모두가능",
            "가능하다라는잘못된전제",
        )
    )


def _missing_is_withdrawal_plan_expansion(item: str) -> bool:
    text = _compact_for_requirement(item)
    has_withdrawal_context = any(marker in text for marker in ("중도인출", "퇴직연금", "제도", "종류", "계좌"))
    has_plan_context = any(marker in text for marker in ("dc", "irp", "제도", "종류", "퇴직연금"))
    has_optional_expansion = any(marker in text for marker in ("다른", "추가", "기타", "종류"))
    return "가능" in text and has_withdrawal_context and has_plan_context and has_optional_expansion


def _missing_is_answered_by_withdrawal_plan_text(item: str, draft: str) -> bool:
    item_text = _compact_for_requirement(item)
    draft_text = _compact_for_requirement(draft)
    if "중도인출" not in item_text or "가능" not in item_text:
        return False
    asks_plan = any(marker in item_text for marker in ("제도", "종류", "퇴직연금", "dc", "irp"))
    answers_plan = "중도인출" in draft_text and "가능" in draft_text and "dc" in draft_text and "irp" in draft_text
    return asks_plan and answers_plan


# 질문에서 상품 고유명사를 뽑아낼 때 무시할 일반 단어. 이 단어들만으로는 특정 상품을
# 지목했다고 볼 수 없다(예: "펀드가 뭐예요?"는 특정 상품 질문이 아니다).
_PRODUCT_NAME_STOPWORDS = {
    "펀드", "상품", "국공채", "채권", "주식", "연금", "퇴직연금", "연금저축", "증권",
    "투자", "신탁", "단기", "장기", "중장기", "초단기", "안정", "수익", "위험", "등급",
    "차이", "특징", "비교", "추천", "가입", "운용", "계좌", "무엇", "뭐", "어떤",
}
# 상품 고유명사로 볼 수 있는 한글 토큰(2자 이상). 조사·구두점을 떼고 뽑는다.
_PRODUCT_TOKEN_RE = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9]{1,}")


def _question_names_specific_products(question: str, evidence_sources: list[str]) -> list[str]:
    """질문이 지목한 상품 고유명사 중 근거 출처명에 실재하는 것을 돌려준다.

    "솔로몬 국공채 단기·중장기·장기, 뭐가 달라요?" → ["솔로몬"]
    (근거 출처명 "미래에셋솔로몬단기국공채증권자투자신탁1호(채권) (C-P)"에 실재)

    출처명과 대조하는 것이 핵심이다 — 질문에 단어가 있다는 것만으로는 부족하고, ③이
    실제로 그 상품을 조회해 왔을 때만 "요구사항을 충족했다"고 말할 수 있다.
    """
    if not evidence_sources:
        return []
    haystack = _compact_for_duplicate(" ".join(evidence_sources))
    if not haystack:
        return []
    named: list[str] = []
    for token in _PRODUCT_TOKEN_RE.findall(question or ""):
        if token in _PRODUCT_NAME_STOPWORDS or len(token) < 2:
            continue
        compact = _compact_for_duplicate(token)
        if compact and compact in haystack:
            named.append(token)
    return named


def _missing_is_named_product_explanation(item: str, named_products: list[str]) -> bool:
    """지목된 상품의 '차이/특징 설명'을 누락으로 본 항목인지 판정한다.

    ④가 특정 상품 질문을 "일반적 설명 요구"로 잘못 읽어 내는 판정이다. 항목이 그
    상품명을 그대로 담고 있으면서 설명·비교를 요구하는 형태이면, 근거에 그 상품이
    실재하는 이상(named_products가 그것을 보증한다) 초안은 답할 재료를 갖고 있었다.
    """
    if not named_products:
        return False
    text = _compact_for_duplicate(item)
    if not any(_compact_for_duplicate(name) in text for name in named_products):
        return False
    return any(
        marker in text
        for marker in ("차이", "특징", "설명", "비교", "구분", "다른점", "무엇이다른")
    )


def apply_requirement_scope_override(
    verification: dict,
    question: str,
    draft: str,
    evidence_sources: list[str] | None = None,
) -> dict:
    """④가 선택적 확장정보를 필수 요구사항으로 만든 경우를 정규화한다.

    요구사항 검증은 사용자가 직접 요구한 것(EXPLICIT)과 답변에 필수적인 것(NECESSARY)을
    봐야 한다. "DB형인데 집 사려고 중도인출할래"의 핵심 요구는 DB형에서 가능한지 여부다.
    ④가 "다른 퇴직연금 종류"처럼 유용하지만 선택적인 정보를 필수 missing으로 확장하면,
    답변은 맞는데 repair loop가 돌고 최종 답변에 잘못된 한계고지가 붙는다.

    또한 초안이 이미 DC/IRP를 답했는데도 같은 항목을 누락으로 보는 coverage false
    negative를 함께 제거한다.
    """
    original_missing = list(verification.get("missing_requirements") or [])
    original_premises = list(verification.get("premise_issues") or [])
    if not original_missing and not original_premises:
        return verification

    explicit_alt_plan_request = _question_explicitly_requests_alternative_withdrawal_plans(question)
    anchored_housing_withdrawal = _question_has_housing_deposit_withdrawal_anchor(question)
    explicit_lease_loan_request = _question_explicitly_requests_lease_loan_info(question)
    named_products = _question_names_specific_products(question, evidence_sources or [])
    missing: list[str] = []
    for item in original_missing:
        if _missing_is_answered_by_withdrawal_plan_text(item, draft):
            continue
        if _missing_is_named_product_explanation(item, named_products):
            continue
        if (
            anchored_housing_withdrawal
            and not explicit_lease_loan_request
            and _missing_is_lease_loan_expansion(item)
        ):
            continue
        if (
            not explicit_alt_plan_request
            and _missing_is_withdrawal_plan_expansion(item)
        ):
            continue
        missing.append(item)

    premises = [
        issue for issue in original_premises
        if not _premise_is_pension_context_false_negative(issue, question, draft)
        and not _premise_is_inferred_alternative_plan_claim(issue, question)
        and not _missing_is_named_product_explanation(issue, named_products)
    ]

    if missing == original_missing and premises == original_premises:
        return verification

    result = dict(verification)
    result["missing_requirements"] = missing
    result["premise_issues"] = premises
    if not missing and not result.get("issues") and not premises:
        result["requirements_met"] = True
    return result


# 세제 효과를 과장하는 유도성 표현 + 그 표현이 인용임을 드러내는 어미.
# 대회 요강 평가지표 "정확성"이 명시적으로 요구하는 항목이다 — "고객의 잘못된 전제나
# 유도성 질문을 그대로 수용하지 않고 바로잡는가". 요강의 참고 질의에도
# "명퇴수당을 연금계좌에 넣으면 세금 감면이 어마어마하다던데"가 예시로 들어 있다.
_EXAGGERATED_TAX_CLAIM_MARKERS = (
    "어마어마", "엄청", "무제한", "완전히 안", "하나도 안", "거의 안", "안 낸다", "안낸다",
    "전혀 안", "공짜", "면제된다",
)
# "세금이 없다"/"세금도 없다"/"세금 없다"처럼 조사만 달라지는 형태를 하나로 잡는다 —
# 조사를 리터럴로 하나씩 늘리면 표현이 조금만 달라져도 계속 뚫린다(실측: "세금도 없다던데"가
# "세금이 없"·"세금 없" 두 마커를 모두 비껴갔다).
_NO_TAX_CLAIM_RE = re.compile(r"세금[이가도는을]?없")
# 사용자가 남의 말을 옮기는 어미 — 이게 있으면 본인 주장이 아니라 "들은 이야기"라
# 확인을 구하는 것이므로, 바로잡아 주는 것이 정확히 요강이 요구하는 대응이다.
_HEARSAY_ENDINGS = ("다던데", "라던데", "다는데", "라는데", "다고 하", "라고 하", "들었", "맞나요", "맞죠", "사실인가요")
_TAX_TOPIC_MARKERS = ("세금", "세액공제", "절세", "감면", "과세", "세율", "혜택")


def detect_exaggerated_tax_premise(question: str) -> list[str]:
    """질문에 담긴 "세금이 거의 없다"류 과장 전제를 결정론적으로 찾아낸다.

    ⚠️ 결정론 답변 경로는 ④grounding을 건너뛰므로(불필요한 repair 47/184건을 막기
    위한 의도된 우회), premise_issues가 항상 빈 리스트로 고정된다. 그 부작용으로
    **전제 교정이 결정론 경로에서 아예 작동하지 않았다** — 실측 T18("퇴직금 받아서
    연금으로 굴리면 세금 거의 안 낸다던데 맞나요?")과 요강 참고질의("세금 감면이
    어마어마하다던데")가 모두 이 경로라 과장을 그대로 통과시켰다.

    LLM을 다시 부르지 않고 코드로 잡는다 — 우회의 이점(속도·비용·불필요한 repair 제거)을
    유지하면서 교정만 되살리는 방법이다. 과장 표현 + 세금 주제 + 인용 어미가 모두
    있을 때만 잡아 오탐을 억제한다(본인이 단정하는 게 아니라 "들었다"고 확인을 구하는
    형태여야 한다).
    """
    compact = re.sub(r"\s+", "", question or "")
    if not compact:
        return []
    has_exaggeration = any(
        re.sub(r"\s+", "", m) in compact for m in _EXAGGERATED_TAX_CLAIM_MARKERS
    ) or _NO_TAX_CLAIM_RE.search(compact) is not None
    has_tax_topic = any(m in compact for m in _TAX_TOPIC_MARKERS)
    has_hearsay = any(re.sub(r"\s+", "", m) in compact for m in _HEARSAY_ENDINGS)
    if has_exaggeration and has_tax_topic and has_hearsay:
        return [
            "연금계좌의 세제 혜택이 세금을 거의 내지 않아도 될 만큼 크다는 전제"
        ]
    return []


# ④의 issues 항목이 "근거 없이 단정했다"는 지적인지 판정하는 표현.
_UNSUPPORTED_CLAIM_MARKERS = (
    "근거 없", "근거가 없", "근거 부족", "근거가 부족", "뒷받침되지",
    "근거 없이", "확인되지 않", "제공된 근거 없",
)
# 위 표현이 있어도 이 표현이 함께 있으면 "문제 없다"는 서술이다 — ④가 issues 칸에
# 검토 결과를 그대로 적는 경우가 있다(실측 no.446 "구체적인 수치나 단정적인 주장을
# 포함하지 않으므로, 이 부분은 문제가 없습니다").
_ISSUE_BENIGN_MARKERS = (
    "문제가 없", "문제없", "일치하므로", "위반이 아니", "해당하지 않습니다",
    "포함하지 않으므로", "적절합니다", "타당합니다",
)
# ④가 issues에 "답변이 빠뜨렸다"를 적는 경우 — 성격상 요구사항 미충족이라
# enforce_missing_requirements가 담당한다. 여기서 중복 고지하면 안 된다.
_ISSUE_OMISSION_MARKERS = (
    "답변하지 않", "다루지 않", "제공하지 않", "언급하지 않", "직접적으로 답",
)


def _is_unsupported_claim_issue(issue: str) -> bool:
    """issues 항목이 '근거 없는 단정'을 지적한 것인지 판정한다."""
    if any(marker in issue for marker in _ISSUE_BENIGN_MARKERS):
        return False
    if any(marker in issue for marker in _ISSUE_OMISSION_MARKERS):
        return False
    return any(marker in issue for marker in _UNSUPPORTED_CLAIM_MARKERS)


def enforce_unsupported_claims(answer: str, issues: list[str]) -> str:
    """④가 '근거 없이 단정했다'고 지적한 서술에 대해 한계를 고지한다.

    ## 왜 필요한가

    ④의 출력 중 issues만 **코드 강제가 전혀 없었다**. unsupported_numbers_confirmed·
    missing_requirements·premise_issues에는 각각 enforce_* 가 있는데, issues는 ⑤
    프롬프트에 넘기고 "반영해달라"고 부탁만 했다 — 이 프로젝트가 반복 확인한
    "프롬프트 순종은 확률적으로 실패한다"가 그대로 적용되는 자리다.

    실측(501문항): grounded=False 46건 중 32건이 "확정 수치는 없고 issues만 있는"
    경우였고, 그중 근거 없는 단정을 지적했는데 답변에 한계 고지가 없는 사례가 6건이었다.
      no.9   "퇴직연금 규약 변경 시 고용노동부 승인 필요성에 대한 근거 부족"
      no.48  "포트폴리오형으로 간주된다는 내용은 근거 없이 단정적으로 서술됨"
      no.324 "배우자 명의 납입 시 세액공제가 안 된다는 정보는 제공된 근거 없이 작성됨"
    수치가 아니라 **서술**이라 L0 수치 대조로는 잡히지 않는 유형이다.

    ## 왜 문장을 지우지 않고 고지만 하는가

    수치는 토큰 대조로 위치를 특정할 수 있지만, "근거 없는 단정"은 ④가 자연어로
    서술할 뿐 답변의 **어느 문장인지 알 수 없다**. 위치를 모른 채 지우면 맞는 내용을
    지울 위험이 크므로, 삭제 대신 한계를 고지해 사용자가 판단할 수 있게 한다.

    ⚠️ issues는 ④가 자유 서술로 적는 칸이라 성격이 섞여 있다(실측 32건 중 2건은
    "문제가 없다"는 서술, 10건은 답변 누락 지적). 그래서 세 겹으로 거른다:
    무해 서술 제외, 누락 지적 제외(enforce_missing_requirements 담당), 그리고
    이미 한계를 고지한 답변에는 덧붙이지 않는다.
    """
    if not issues or has_limit_disclosure(answer):
        return answer
    flagged = [issue for issue in issues if _is_unsupported_claim_issue(issue)]
    if not flagged:
        return answer
    items = "".join(f"\n- {issue}" for issue in flagged)
    return (
        f"{answer}\n\n"
        f"※ 위 답변 중 다음 내용은 제공된 자료에서 확인되지 않아 참고용으로만 봐주세요:{items}\n"
        "정확한 내용은 가입하신 금융기관에 확인해 주시기 바랍니다."
    )


def enforce_premise_issues(answer: str, premise_issues: list[str]) -> str:
    """④가 짚은 '질문의 잘못된 전제'를 답변이 바로잡지 않았으면 앞머리에 교정문을 붙인다.

    대회 평가지표 "정확성"이 요구하는 항목이다 — 고객의 잘못된 전제나 유도성 질문을
    그대로 수용하지 않고 바로잡는가.
    """
    if not premise_issues or has_premise_correction(answer):
        return answer
    items = "".join(f"\n- {p}" for p in premise_issues)
    return (
        "먼저 질문에 담긴 전제를 짚고 넘어가겠습니다. 다음 내용은 사실과 다르거나 "
        f"과장된 부분이 있어 그대로 전제하기 어렵습니다:{items}\n\n"
        f"{answer}"
    )


# ⑤ 프롬프트는 "[근거 N] 대괄호 안의 출처를 그대로 쓰라"고 지시하지만, 실측상 LLM이
# 생성한 답변 7건 중 5건이 출처명 대신 내부 인덱스 "[근거 1]"을 그대로 노출했다
# (정형 응답 경로는 코드가 문자열을 붙여 100% 정확 — 프롬프트 순종 실패의 또 다른 사례).
# 대회 요강은 "모든 답변에는 근거 문서 표시할 것"을 명시하므로 코드로 치환한다.
_EVIDENCE_PLACEHOLDER_RE = re.compile(r"\[\s*근거\s*(\d+)\s*\]")


def replace_evidence_placeholders(answer: str, context: list[dict]) -> str:
    """답변에 남은 '[근거 N]' 표기를 N번 근거의 실제 출처명으로 치환한다.

    context는 ⑤가 프롬프트에 넣은 것과 같은 순서(1-based)여야 한다.
    범위를 벗어난 번호는 LLM이 지어낸 것이므로 표기를 지운다 — 존재하지 않는 근거를
    가리키는 인용은 없느니만 못하다.

    출처명만 넣고 "출처:" 접두사는 붙이지 않는다. 실측상 이 표기는 세 문맥에서 쓰이는데
    (`(출처: [근거 3])`, `참고 근거: [근거 1]; [근거 2]`, `[근거 1]과 [근거 2]`),
    앞 두 경우는 이미 라벨이 있어 접두사를 붙이면 "출처: 출처: ..."가 된다.
    """
    if not answer:
        return answer

    def _sub(match: re.Match) -> str:
        index = int(match.group(1))
        if 1 <= index <= len(context):
            return context[index - 1]["source"]
        return ""

    return _EVIDENCE_PLACEHOLDER_RE.sub(_sub, answer)


# ── ⑤ 출력에서 도구 호출·코드 텍스트 제거 ────────────────────────────────────
#
# 실측(정도부사 25문항 V06 "연금저축이랑 IRP에 최대한 많이 넣고 싶어요. 얼마까지 되나요?"):
# 최종 답변에 아래가 그대로 실려 사용자에게 노출됐다.
#
#     연금저축과 IRP의 최대 납입 한도에 대한 정보는 검색을 통해 확인해 보겠습니다.
#     ```python
#     search_result = search_pension_docs("연금저축 IRP 연간 납입 한도 및 세액공제 비율")
#     ```
#
# 원인은 프롬프트 누락이 아니라 **구조적 공백**이다. 이 파이프라인의 도구 호출은
# LangChain tool-calling으로 이뤄지므로 정상 경로에서는 코드가 텍스트로 나올 일이 없다.
# 그런데 LLM이 "도구를 쓰는 시늉"을 텍스트로 흉내내면(HCX 계열에서 실측됨) 그 텍스트는
# 그냥 평범한 답변 문자열이라 어떤 계층도 걸러내지 않는다 — grounded 검증은 수치만 보고,
# enforce_* 계열은 덧붙이기만 한다.
#
# 프롬프트에 "코드를 쓰지 마세요"를 넣는 방식은 이 프로젝트가 반복 확인한 대로 확률적으로
# 실패한다. 사용자에게 내부 구현이 노출되는 것은 확률적으로도 허용할 수 없으므로 코드로
# 제거한다 — 모든 최종 답변이 통과하는 _finalize_answer 한 곳에서 강제한다.
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
# 우리 도구 이름을 그대로 호출문 형태로 쓴 줄. 펜스 없이 맨줄로 나오는 경우가 있다.
_TOOL_CALL_LINE_RE = re.compile(
    r"^[^\n]*\b(?:search_pension_docs|search_funds|get_fund_detail|search_prospectus)\s*\([^\n]*$",
    re.MULTILINE,
)
# "~해 보겠습니다/확인해 보겠습니다"처럼 도구를 쓰겠다고 예고하는 문장만 남으면 어색하므로
# 코드 제거 후 함께 지운다. 답변 본문의 정보성 문장은 건드리지 않는다.
_TOOL_PREAMBLE_RE = re.compile(
    r"^[^\n]*(?:검색을 통해|툴을 사용해|도구를 사용해|아래 코드|다음 코드)[^\n]*"
    r"(?:확인해 보겠습니다|알아보겠습니다|조회하겠습니다|실행하겠습니다)[^\n]*$",
    re.MULTILINE,
)


def strip_tool_call_artifacts(answer: str) -> str:
    """최종 답변에서 코드블록·도구 호출 텍스트를 제거한다.

    내부 구현(도구 이름·파라미터)이 사용자에게 노출되는 것을 코드로 차단한다.
    제거 후 공백 줄이 연달아 남지 않도록 정리한다.
    """
    if not answer:
        return answer
    cleaned = _FENCED_CODE_RE.sub("", answer)
    cleaned = _TOOL_CALL_LINE_RE.sub("", cleaned)
    cleaned = _TOOL_PREAMBLE_RE.sub("", cleaned)
    if cleaned == answer:
        return answer
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── 제도 용어 오표기 교정 ────────────────────────────────────────────────────
#
# 연금 제도명의 영문 표기는 법령으로 고정돼 있다(DB=Defined Benefit, DC=Defined
# Contribution). 그런데 LLM이 약어를 풀어 쓰면서 **그럴듯하지만 틀린 단어**를
# 끼워 넣는 일이 있다.
#
# 실측(501문항 no.1 "DC와 DB, 퇴직금이 정해지는 방식이랑 운용 주체가 어떻게
# 다른가요?"): "DC(Dividend Contribution)형"이라고 썼다. Dividend는 '배당'이라
# 확정기여와 아무 관련이 없고, 근거 문서에는 정확한 표기가 있었는데도 창작했다.
# 전수 확인 결과 Defined Contribution 8회(정상) 대 Dividend Contribution 2회(오기)로,
# 체계적 오류가 아니라 확률적으로 튀는 유형이다 — 즉 프롬프트로는 막기 어렵다.
#
# 수치 검증(L0)은 숫자만 보므로 이 오류를 구조적으로 못 잡는다. 제도명은 근거와
# 무관하게 **정답이 하나로 고정**돼 있어 코드로 교정해도 안전하다.
#
# ⚠️ 여기 담는 것은 "틀린 표기 -> 옳은 표기"가 1:1로 확정되는 것만이다. 문맥에
# 따라 달라질 수 있는 표현은 넣지 않는다 — 잘못 고치면 오히려 정확한 답변을
# 훼손한다.
_TERM_CORRECTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # DC = 확정기여(Defined Contribution). Dividend/Definite 등으로 잘못 쓰는 사례.
    (re.compile(r"Dividend\s+Contribution", re.IGNORECASE), "Defined Contribution"),
    (re.compile(r"Definite\s+Contribution", re.IGNORECASE), "Defined Contribution"),
    # DB = 확정급여(Defined Benefit).
    (re.compile(r"Definite\s+Benefit", re.IGNORECASE), "Defined Benefit"),
    (re.compile(r"Defined\s+Benefits\b"), "Defined Benefit"),
    # IRP = 개인형 퇴직연금(Individual Retirement Pension).
    (re.compile(r"Individual\s+Retirement\s+Plan\b", re.IGNORECASE), "Individual Retirement Pension"),
)


def correct_institution_terms(answer: str) -> str:
    """제도명 영문 표기 오류를 교정한다 (DC=Defined Contribution 등).

    법령상 표기가 하나로 고정된 용어만 다루므로, 근거를 확인하지 않고 치환해도
    안전하다. 답변의 다른 내용은 건드리지 않는다.
    """
    if not answer:
        return answer
    for pattern, correct in _TERM_CORRECTIONS:
        answer = pattern.sub(correct, answer)
    return answer
