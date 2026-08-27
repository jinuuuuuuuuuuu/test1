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
)

_NUMERIC_CORE_RE = re.compile(r"[\d,\.]+")

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
_INLINE_MARKUP_RE = re.compile(r"[*_`~]+")


def strip_inline_markup(text: str) -> str:
    """수치 대조를 방해하는 인라인 마크다운 서식 문자를 제거한다.

    서식은 의미가 아니라 표현이므로, 근거 대조 전에 걷어내야 "**16.5**%"와 "16.5%"가
    같은 사실로 취급된다. 원문을 바꾸지 않고 검사용 사본에만 적용한다.
    """
    return _INLINE_MARKUP_RE.sub("", text or "")


def extract_number_tokens(text: str) -> list[str]:
    """텍스트에서 '숫자+단위' 토큰을 등장 순서대로 중복 없이 추출한다. 예: ['900만원', '16.5%']

    인라인 서식은 먼저 제거한다 — "**16.5**%"처럼 강조가 숫자와 단위를 갈라놓으면
    수치가 통째로 검사에서 빠져나간다.
    """
    seen: list[str] = []
    for m in _NUMBER_TOKEN_RE.finditer(strip_inline_markup(text)):
        token = m.group(0).strip()
        if token not in seen:
            seen.append(token)
    return seen


def _numeric_core(token: str) -> str:
    """토큰의 숫자 부분만 콤마 제거 형태로 반환한다. '1,200만원' -> '1200'"""
    m = _NUMERIC_CORE_RE.match(token)
    return (m.group(0) if m else token).replace(",", "")


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
    """
    normalized_support = strip_inline_markup(
        " ".join([*evidence_texts, *user_texts])
    ).replace(",", "")
    return [
        token
        for token in extract_number_tokens(draft)
        if _numeric_core(token) not in normalized_support
    ]


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


def enforce_missing_requirements(answer: str, missing: list[str]) -> str:
    """④가 '질문이 요구했는데 빠졌다'고 지적한 항목을 답변이 다루지 않았으면 한계를 명시한다.

    빠진 항목을 지어내 채우는 게 아니라, **답하지 못했다는 사실 자체를 드러내는 것**이
    목적이다 (대회 평가지표 "정보한계 대응": 무리한 답변 대신 한계 고지 또는 역질문).

    이미 한계를 고지한 답변에는 덧붙이지 않는다 — 중복 고지는 답변 품질을 떨어뜨린다.
    """
    if not missing or has_limit_disclosure(answer):
        return answer
    items = "".join(f"\n- {m}" for m in missing)
    return (
        f"{answer}\n\n"
        f"다만 다음 항목은 제공된 자료만으로는 확인이 어려워 답변에 포함하지 못했습니다:{items}\n"
        "해당 부분은 가입하신 금융기관이나 관련 기관에 확인해 주시기 바랍니다."
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
_BENIGN_CONDITION_MARKERS = (
    "이라는 전제",
    "라는 전제",
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


def split_premise_issues(premise_issues: list[str]) -> tuple[list[str], list[str]]:
    """premise_issues를 (진짜 전제, 실제로는 요구사항 미충족인 항목)으로 나눈다."""
    real_premises: list[str] = []
    misfiled_defects: list[str] = []
    for issue in premise_issues:
        if is_benign_condition_statement(issue):
            continue
        (misfiled_defects if is_answer_defect_statement(issue) else real_premises).append(issue)
    return real_premises, misfiled_defects


def apply_premise_issue_normalization(verification: dict) -> dict:
    """premise_issues 오분류를 verification 단계에서 정리한다.

    generator에서만 정리하면 최종 답변은 괜찮아도 think_trace에는 여전히
    "전제 교정: 잔금지급일이 ...라고 가정"처럼 남는다. 검증 dict 자체를 정규화해
    이후 모든 계층이 같은 해석을 보게 한다.
    """
    result = dict(verification)
    real, misfiled = split_premise_issues(list(result.get("premise_issues") or []))
    missing = list(result.get("missing_requirements") or [])
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


def apply_requirement_scope_override(verification: dict, question: str, draft: str) -> dict:
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
    missing: list[str] = []
    for item in original_missing:
        if _missing_is_answered_by_withdrawal_plan_text(item, draft):
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
    ]

    if missing == original_missing and premises == original_premises:
        return verification

    result = dict(verification)
    result["missing_requirements"] = missing
    result["premise_issues"] = premises
    if not missing and not result.get("issues") and not premises:
        result["requirements_met"] = True
    return result


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
