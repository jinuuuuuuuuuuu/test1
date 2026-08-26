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


def extract_number_tokens(text: str) -> list[str]:
    """텍스트에서 '숫자+단위' 토큰을 등장 순서대로 중복 없이 추출한다. 예: ['900만원', '16.5%']"""
    seen: list[str] = []
    for m in _NUMBER_TOKEN_RE.finditer(text or ""):
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

    콤마 표기 차이(1,200 vs 1200)는 정규화해서 비교한다. 숫자 부분의 부분문자열 일치만
    보므로 "지원됨" 쪽으로 관대하다 — 여기서 잡히지 않은 표기 차이(9백만 원 등)는 ④ LLM이
    의심 목록을 근거와 대조할 때 걸러진다.
    """
    normalized_support = " ".join([*evidence_texts, *user_texts]).replace(",", "")
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
