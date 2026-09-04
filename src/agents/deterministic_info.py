"""Deterministic answer generation for high-risk pension information questions.

분류(어느 카테고리인지)와 답변 생성(그 카테고리에서 실제 숫자·조건을 인용하는 것)의 역할을
분리한다 — 근거: "세액공제"+"얼마"만 있으면 무조건 한도질문으로 오분류하던 사고, 그리고
같은 클래스(주제어+느슨한 동반어)의 트리거가 "언제"/"시점"/"상품"/"세율" 등에서도 반복
재발할 위험이 있다는 판단(2026-08-25 재점검).

- candidate_category(): 1단계, 순수 키워드 규칙. 주제어(예: "세액공제")만 보고 "이 카테고리일
  수도 있다"는 느슨한 후보만 낸다. 결정적(같은 입력엔 항상 같은 결과)이고 빠르다.
- DETERMINISTIC_CATEGORIES: 후보로 나올 수 있는 전체 카테고리 목록 — router.py가
  RouterDecision.deterministic_category의 허용값을 정의할 때 이 목록을 그대로 쓴다.
- deterministic_response_for(category, question): 2단계, router의 LLM이 후보를 검증/확정한
  카테고리를 받아 실제 정형 답변(숫자·조건)을 만든다. 여기서부터는 결정론 그대로 유지한다 —
  숫자를 LLM에게 맡기면 학습 지식으로 틀린 값을 지어내는 사고가 실측된 바 있다.

왜 분류까지 전부 규칙으로 하지 않는가: 키워드 매칭은 "얼마"/"언제"/"시점"/"상품"/"세율"처럼
그 자체로 의미가 넓은 동반어를 요구하면 반드시 오탐(다른 의도의 질문을 잘못 걸러냄)을
만든다 — 이게 원래 사고의 근본 원인이었다. 반대로 분류를 통째로 LLM에게 맡기면(동반어 없이
전체 판단) 오탐은 줄지만 판단이 확률적이라 같은 입력에도 흔들릴 수 있고 실패 원인을 코드로
특정하기 어렵다. candidate_category()가 "주제어 존재 여부"라는 가장 안정적인 신호만
규칙으로 담당하고, 그 후보가 맞는지 최종 확정은 router의 LLM(이미 있는 호출에 필드만
추가돼 비용 증가 없음)에게 맡기는 하이브리드로 양쪽의 실패 모드를 서로 보완한다.

## 카테고리는 (도메인 × 작업종류)로 설계한다 — 2026-08-27 재설계

카테고리를 도메인(중도인출·실물이전·세제)으로만 나누면, 같은 도메인 안의 서로 다른
작업이 갈 곳을 잃는다. 실측된 증상:
  - 중도인출 날짜계산 4건(요양/전월세/재난/주택구입)이 전부 후보 ['중도인출_일반']을
    받았다. 목록답변용 카테고리라 라우터가 기각했고, 답변은 사유 목록만 나열했다.
  - "MMF인데 실물이전 되나요"(개별판정)가 ['실물이전_불가사유'](목록답변)만 후보로
    받아 기각됐고, LLM이 자유롭게 툴을 고르다 투자 가능 여부 툴을 불러
    "네, 실물이전 문제없습니다"라는 정반대 답을 냈다.

작업종류는 최소 다음 다섯이며, 새 카테고리를 만들 때 "이 도메인의 이 작업이 이미
있는가"를 먼저 확인한다:
  일반설명   — 제도 자체를 설명 (예: 세액공제_한도)
  목록답변   — 해당 사유·항목을 나열 (예: 중도인출_일반, 실물이전_불가사유)
  기한규칙   — 기준일·기간 규칙을 안내 (exact date는 DB-grounded 답변에서 단정하지 않음)
  요건판정   — 주어진 조건이 요건을 충족하는지 (예: 실물이전_개별판정)
  개인계산   — 사용자 수치를 대입해 산출 (예: 세액공제_계산_입력부족)

⚠️ 사유가 늘 때마다 "요양_신청기한", "주택구입_신청기한"처럼 (도메인×사유×작업)으로
카테고리를 늘리지 말 것. 같은 작업은 하나의 공통 경로가 처리하고, 사유별 차이는
데이터(WITHDRAWAL_DEADLINE_RULES, TRANSFER_BLOCK_CODES)로 표현한다.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal, Optional

from src.agents.state import RetrievedItem
from src.rules.comprehensive_tax import (
    ANNUAL_THRESHOLD,
    SEPARATE_TAXATION_RATE_OVER_THRESHOLD,
    get_pension_income_tax_rate,
)
from src.rules.default_option import (
    NOTICE_DELAY_DAYS_EXISTING,
    WAIT_DAYS_AFTER_NOTICE,
    check_optin_eligibility,
)
from src.rules.early_withdrawal import (
    MEDICAL_EXPENSE_RATIO_THRESHOLD,
    MEDICAL_TREATMENT_ELIGIBLE_PERSONS,
    MEDICAL_TREATMENT_MIN_MONTHS,
    WITHDRAWAL_DEADLINE_RULES,
    PlanType,
)
from src.rules.in_kind_transfer import TRANSFER_BLOCK_CODES
from src.rules.irp_mandatory_transfer import (
    IRP_MANDATORY_TRANSFER_EXCEPTIONS,
    IRP_POST_RECEIPT_DEPOSIT_DAYS,
)
from src.rules.investment_limit import (
    PRODUCT_RISK_TIER,
    RISKY_ASSET_LIMIT,
    TDF_QUALIFIED_LIMIT,
    RiskTier,
)
from src.rules.retirement_tax_reduction import get_deferred_retirement_tax_rate
from src.rules.tax_credit import (
    COMBINED_CREDIT_LIMIT,
    CREDIT_RATE_HIGH,
    CREDIT_RATE_LOW,
    INCOME_THRESHOLD_COMPREHENSIVE,
    INCOME_THRESHOLD_SALARY,
    PENSION_SAVINGS_ONLY_LIMIT,
    TOTAL_CONTRIBUTION_LIMIT,
    calculate_tax_credit,
)
from src.rules.withdrawal_limit import (
    SIX_YEAR_EXCEPTION_CUTOFF,
    SIX_YEAR_EXCEPTION_START,
    UNLIMITED_FROM_YEAR,
    calculate_withdrawal_limit,
)
from src.agents.tax_context import extract_tax_context, personal_tax_response
from src.agents.withdrawal_context import extract_withdrawal_context

DeterministicCategory = Literal[
    "복합정보_태스크플랜",
    "세액공제_계산_입력부족",
    "세액공제_한도",
    "세금혜택_개요",
    "개인세금_입력충분성",
    "중도인출_기한판정",
    "중도인출_요건판정",
    "중도인출_일반",
    "디폴트옵션_자동매수",
    "디폴트옵션_옵트인판정",
    "실물이전_불가사유",
    "실물이전_개별판정",
    "투자한도_위험자산",
    "투자가능여부_상품유형",
    "퇴직시_IRP의무이전",
    "연금수령한도",
    "퇴직소득세감면",
    "연금소득세_종합과세",
    "연금소득세율_연령별",
    "해당없음",
]

# router.py가 프롬프트/RouterDecision의 Literal 정의에 그대로 재사용한다.
DETERMINISTIC_CATEGORIES: tuple[str, ...] = (
    "복합정보_태스크플랜",
    "세액공제_계산_입력부족",
    "세액공제_한도",
    "세금혜택_개요",
    "개인세금_입력충분성",
    "중도인출_기한판정",
    "중도인출_요건판정",
    "중도인출_일반",
    "디폴트옵션_자동매수",
    "디폴트옵션_옵트인판정",
    "실물이전_불가사유",
    "실물이전_개별판정",
    "투자한도_위험자산",
    "투자가능여부_상품유형",
    "퇴직시_IRP의무이전",
    "연금수령한도",
    "퇴직소득세감면",
    "연금소득세_종합과세",
    "연금소득세율_연령별",
    "해당없음",
)


# 라우터가 "해당없음"으로 기각했을 때 코드가 되살려도 되는 카테고리.
#
# 배경: "개인세금_입력충분성"은 후보로 정확히 주어져도 라우터가 3/3 기각했다. 프롬프트
# 문구를 여러 형태로 조정해도 재현됐고(상위 판정 원칙 블록을 완전히 제거한 최소
# 프롬프트에서만 성공), 프롬프트 순종에 기댈 수 없는 실패라 코드로 되살린다.
#
# ## 여기 넣어도 되는 기준: "라우터가 판단할 게 남아있지 않은가"
#
# 후보 목록(candidate_categories)은 "이 카테고리를 검토할 신호가 질문에 있다"까지만
# 말한다. 카테고리에 따라 그 다음 판단이 더 필요할 수 있고, 그 판단은 라우터의 몫이다.
#   - 세액공제_계산_입력부족: "세액공제" 신호만으로는 부족하고, **입력값이 충분한지**를
#     더 봐야 한다. 충분하면 계산 툴로 넘겨야 하는데 이 핸들러는 입력이 충분해도 항상
#     "입력값이 부족합니다"를 반환한다 — 라우터가 정확히 기각한 케이스("연금저축
#     600만원, 총급여 5000만원인데 세액공제 얼마?")를 코드가 되살리면 오답이 나간다.
#   - 여기 담긴 셋은 후보에 오른 시점에 판정이 사실상 끝난다. 개인세금_입력충분성은
#     충분/부족을 핸들러가 스스로 분기하고, 중도인출_기한판정·실물이전_개별판정은
#     사유·상품이 인식되지 않으면 핸들러가 None을 낸다.
#
# 참고: "무관한 질문에도 답하는가"는 더 이상 기준이 아니다. 그 문제는
# deterministic_response_for가 후보 목록을 재확인하도록 고쳐 13개 전부 해결됐다.
CODE_OVERRIDABLE_CATEGORIES: frozenset[str] = frozenset({
    "복합정보_태스크플랜",
    "개인세금_입력충분성",
    # 개인 상황 신호가 있으면 스스로 None을 낸다(_DB_DC_PERSONAL_SIGNAL_MARKERS).
    "제도비교_DB_DC",
    # 이전/이체 요구가 없거나 개인 판정 신호가 있으면 스스로 None을 낸다.
    "계좌이전_절차",
    "중도인출_기한판정",
    "중도인출_요건판정",
    "실물이전_개별판정",
    "디폴트옵션_옵트인판정",
    # 둘 다 핸들러가 상품유형/질문유형을 못 찾으면 None을 내므로 위 기준을 충족한다.
    # 특히 투자가능여부_상품유형은 "추천해주세요" 형태로 오는 일이 많아 라우터가
    # 상품형으로 볼 여지가 크다 — 금지 상품인데 조건만 되묻는 실측 오답(no.459)이
    # 재현되지 않도록 코드가 되살릴 수 있어야 한다.
    "투자가능여부_상품유형",
    "퇴직시_IRP의무이전",
    # ⚠️ 세액공제 한도는 **틀린 답이 가장 잦은 주제**다. 2023년 개정 전 수치(400만원/
    # 700만원/13.2%)가 학습 데이터에 대량으로 남아 있어, 정형 경로를 못 타면 LLM이
    # 옛 수치를 자신 있게 답한다.
    # 실측(정도부사 25문항):
    #   V03 "세액공제 대박으로 받을 수 있는 방법 있나요?"    -> 700만원, 50세 (오답)
    #   V05 "세액공제를 많이 받고 싶은데 얼마나 넣어야 하나요?" -> 400만원, 14.6%, 700만원 (오답)
    # 정답(600만원/900만원/16.5%·13.2%)은 _tax_credit_limit_response가 이미 갖고 있었다 —
    # 라우터가 "얼마/한도" 형태가 아닌 질문("~받는 방법", "~많이 받고 싶은데")을
    # 기각하면서 답을 못 쓴 것이다. 핸들러가 스스로 None을 내므로 되살려도 안전하다.
    "세액공제_한도",
    # ⚠️ 아래 세제 카테고리들은 TAX_FALLBACK_CATEGORIES와 한 쌍이다 — 후보가 0건이어도
    # 되살릴 수 있게 열어둔다. 세제 질문은 표현이 무한한데 candidate_categories는 유한한
    # 키워드 목록이라 누락이 반복됐고(퇴직금/1600만원/절세방법/납입한도…), 누락되면
    # LLM 자유응답으로 새면서 2023년 개정 전 폐지 수치를 자신 있게 답한다.
    # 오탐보다 누락이 훨씬 비싼 영역이라 판정을 넓게 열고, 안전은 "핸들러가 스스로
    # None을 낸다"로 담보한다(세제 무관·인접 주제 질문 8종에 전부 None 확인).
    # ⚠️ 세액공제_계산_입력부족은 일부러 넣지 않는다. 이 집합은 **후보가 있을 때의
    # 첫 루프**에도 쓰이는데, 후보 순서상 이 카테고리가 세액공제_한도보다 앞이라
    # 넣는 순간 "얼마나 넣어야 하나요?"류 질문이 한도 안내 대신 "입력값이 부족하다"는
    # 계산 보류 답변으로 바뀐다(실측 회귀). 후보 0건일 때만 필요하므로
    # TAX_FALLBACK_CATEGORIES에만 둔다.
    "세금혜택_개요",
    "연금수령한도",
    "퇴직소득세감면",
    "연금소득세_종합과세",
    "연금소득세율_연령별",
})


# 후보가 0건이어도 라우터의 "해당없음"을 되살릴 수 있는 카테고리.
#
# candidate_categories는 "이 카테고리를 검토할 근거가 있다"를 키워드로 판정하는데,
# 세제 영역에서 이 방식이 반복적으로 무너졌다 — 사용자는 제도 용어를 모른 채 일상어로
# 묻기 때문이다("퇴직소득세" 대신 "퇴직금", "1,500만원 기준" 대신 본인 금액 "1,600만원",
# "한도" 대신 "얼마나 넣어야"). 표현은 무한하고 키워드 목록은 유한해서 구조적으로 진다.
#
# 그래서 세제에 한해 순서를 뒤집는다: 키워드로 **차단**하지 않고, 라우터가 "해당없음"을
# 냈을 때 세제 핸들러들에게 직접 물어본다. 핸들러가 답을 내면(= 자기 소관이면) 그걸
# 쓰고, 전부 None이면 원래대로 LLM 경로로 간다. 잘 작동하는 다른 영역(중도인출·실물이전·
# 디폴트옵션 등)은 건드리지 않는다 — 결함이 실측된 곳만 연다.
#
# ⚠️ 알려진 한계 (2026-09-03 실측으로 확인, 시도했다가 되돌림):
# 이 폴백 루프는 **실제로는 발동하지 않는다**. deterministic_response_for가 내부에서
# candidate_categories를 다시 확인하기 때문에(후보 게이트), 후보가 0건이면 무조건
# None이 나온다.
#
# 그렇다고 게이트를 건너뛰는 우회로를 만들면 안 된다. 핸들러 대부분이 자기 소관을
# 판단하지 못하기 때문이다 — 실측: "안정적인 연금 상품 추천해줘", "DB형과 DC형
# 차이가 뭔가요"에 세액공제_한도·연금소득세율_연령별이 각각 한도표·세율표를 반환했다.
# (기존 회귀 테스트 test_tax_fallback_does_not_hijack_non_tax_questions가 이를 잡는다.)
# 즉 이 폴백이 지금까지 사고를 내지 않은 이유는 설계가 안전해서가 아니라 발동한 적이
# 없어서다.
#
# 자기방어가 확인된 핸들러(개인세금·복합정보·중도인출 기한/요건·투자가능여부·
# 퇴직시IRP)만 게이트 밖에서 부르는 안을 구현해 측정했으나, **501문항의 후보 0건
# 177건 중 0건을 되살렸다.** 그 핸들러들이 내부적으로 candidate_categories와 같은
# 키워드 판정을 쓰기 때문에, 키워드가 놓친 질문은 핸들러도 똑같이 놓친다.
#
# 결론: 후보 0건 문제는 이 계층에서 못 푼다. 키워드 판정 자체를 의미 기반으로
# 바꾸거나(라우팅 구조 개편), 검색·검증 계층에서 방어해야 한다.
TAX_FALLBACK_CATEGORIES: tuple[str, ...] = (
    # 순서가 우선순위다. 좁고 구체적인 판정을 먼저 시도하고, 넓은 개요를 마지막에 둔다 —
    # 개요가 먼저 오면 구체적 질문까지 일반론으로 덮어버린다.
    "개인세금_입력충분성",
    "세액공제_계산_입력부족",
    "세액공제_한도",
    "퇴직소득세감면",
    "연금수령한도",
    "연금소득세_종합과세",
    "연금소득세율_연령별",
    "세금혜택_개요",
)


# 한글로 나이를 말하는 표현 — "일흔 넘었는데", "칠순인데"처럼 숫자·'세' 없이 묻는 질문이
# 실제로 들어온다(실측 no.279 "일흔 넘었는데 연금소득세율이 어떻게 되나요?": 후보
# 카테고리가 0건이라 결정론 경로를 아예 못 타고 LLM 답변에 맡겨졌다).
# 값은 그 나이대의 하한이다 — "일흔 넘었다"는 70세 이상이므로 70으로 잡으면
# 70~80세 구간(4.4%)이 정확히 걸린다.
_KOREAN_AGE_WORDS = {
    "쉰": 50, "예순": 60, "일흔": 70, "여든": 80, "아흔": 90,
    "환갑": 60, "회갑": 60, "칠순": 70, "팔순": 80, "구순": 90,
}
_KOREAN_AGE_RE = re.compile("|".join(sorted(_KOREAN_AGE_WORDS, key=len, reverse=True)))

# 나이 표현 — "74세", "만 74세", "제 나이가 74", "일흔" 등. 연령별 세율 판정의 신호다.
_AGE_MENTION_RE = re.compile(
    rf"(?:만)?\d{{1,3}}세|나이가?\d{{1,3}}|{_KOREAN_AGE_RE.pattern}"
)

# 세금 관련 표현 — "세율/세금/얼마 떼나" 류를 폭넓게 잡는다. 정확한 제도명("연금소득세")을
# 쓰는 사용자는 소수라, 어휘 목록을 좁게 잡으면 같은 의도의 질문을 놓친다.
_TAX_AMOUNT_WORDS = (
    "세율", "세금", "몇%", "몇퍼센트", "몇프로", "얼마", "떼나", "떼가", "부과", "과세",
)

# "연금을 받는" 문맥 — 납입·가입 얘기("연금저축에 납입")와 구분한다. 연령별 세율은
# 수령 시점의 나이로 결정되므로 수령 문맥이 있어야 이 카테고리가 맞다.
_PENSION_RECEIPT_WORDS = (
    "연금받", "연금수령", "연금으로받", "수령할때", "수령시", "연금개시", "연금탈때",
    "연금을받",
)


def candidate_categories(question: str) -> list[str]:
    """1단계: 주제어만 보고 **느슨한** 후보 목록을 낸다 (동반어 요구 없음).

    router의 LLM이 이 후보들 중 실제로 맞는 카테고리를 확정한다(또는 전부 기각하고
    "해당없음"). 여러 개가 동시에 후보로 나올 수 있다(예: "세액공제" 관련 두 카테고리).

    ⚠️ 이 함수는 **넓게 잡아야 한다**. 정확한 판정은 라우터의 역할이고, 여기서 후보를
    놓치면 라우터는 그 카테고리를 아예 고려조차 못 한다. 실측(2026-08-27): "연금 + 세율"
    AND 조건과 정확한 어휘 목록을 요구하던 시절, 같은 의도의 7개 표현 중 3개가 후보를
    만들지 못했다 —
        "나 74세인데 세금 어떻게 내?"      -> [] (연금이라는 단어가 없음)
        "74세인데 얼마나 떼나요?"           -> [] (세율/세금 어휘가 목록에 없음)
        "제 나이가 74인데 세율 알려주세요"    -> [] (연금이라는 단어가 없음)
    이는 "세액공제+얼마=한도질문"으로 단정하던 사고(7cddb1f)와 같은 클래스다: 표면
    어휘 조합으로 판정하려다 표현이 조금만 달라지면 무너진다. 후보는 넓히고, 대신
    라우터가 후보 밖 카테고리를 고르지 못하도록 코드로 막는다(router._enforce_candidate_scope).
    """
    text = _compact(question)
    candidates: list[str] = []

    if _build_composite_info_tasks(question):
        candidates.append("복합정보_태스크플랜")
    if personal_tax_response(question) is not None:
        candidates.append("개인세금_입력충분성")
    # DB/DC 제도 비교 — 실측(no.1/no.27): 근거를 7건씩 갖고도 DB 급여 계산식을
    # "평균 임금의 60% x 근속연수"로 창작했다. 여기서는 후보만 넓게 낸다 — 정밀
    # 판정(개인 상황 배제 등)은 핸들러(_db_dc_comparison_response)가 담당한다.
    # DB 단독 질문("DB형은 확정돼 있는 게 맞나요?")도 후보로 낸다 — DC 언급이
    # 없어도 핸들러가 답할 수 있는 실측 사례(no.27)다.
    mentions_db = "DB" in question or "확정급여" in text
    mentions_dc = "DC" in question or "확정기여" in text
    if mentions_db and mentions_dc:
        candidates.append("제도비교_DB_DC")
    elif mentions_db and any(word in text for word in ("확정돼있는", "확정되어있는", "확정된게")):
        candidates.append("제도비교_DB_DC")
    # 계좌 이전/이체 — 실측(no.56 "IRP 계좌를 다른 증권사로 옮기려면"): 근거 10건을
    # 갖고도 무관한 규정("2013년 이전 가입분은 이전 불가")을 창작했다.
    if any(word in text for word in ("IRP", "irp", "연금저축", "퇴직연금", "연금계좌")) and any(
        word in text for word in ("이전", "옮기", "옮겨", "이체")
    ):
        candidates.append("계좌이전_절차")
    # ⚠️ "세액공제"라는 단어가 없어도 **납입 한도**를 묻는 질문은 같은 정형 답변이
    # 정답이다(_tax_credit_limit_response가 연금저축+IRP 합산 납입한도와 세액공제
    # 대상 한도를 함께 제시한다). 이 어휘를 빠뜨려서 생긴 구멍이 실측으로 확인됐다 —
    # V06("연금저축이랑 IRP에 최대한 많이 넣고 싶어요. 얼마까지 되나요?")은 후보가
    # 0건이라 결정론 경로를 못 타고 LLM 자유응답으로 새서, 13.2%·5,500만원을
    # 근거 없이 지어냈다(grounded=False로 디스클레이머까지 붙었다).
    #
    # 이 함수의 docstring이 경고하는 실패 유형("표면 어휘 조합으로 판정하려다 표현이
    # 조금만 달라지면 무너진다")과 같은 클래스다. 정도부사("최대한 많이")가 원인이
    # 아니라는 점이 중요하다 — 부사를 빼도("연금저축이랑 IRP에 넣고 싶어요. 얼마까지
    # 되나요?") 여전히 후보가 0건이었다. 원인은 "넣다/납입"이라는 어휘 자체의 누락이다.
    # ⚠️ _compact는 소문자화를 하지 않는다 — "IRP"(대문자)가 실제 표기라
    # "irp"만 검사하면 영원히 안 걸린다. 두 표기를 함께 본다.
    mentions_pension_account = any(
        word in text for word in ("연금저축", "IRP", "irp", "연금계좌")
    )
    asks_contribution_limit = (
        any(word in text for word in ("납입한도", "납입limit", "불입한도"))
        or (
            mentions_pension_account
            and any(word in text for word in ("넣", "납입", "불입", "저축", "가입", "얼마까지"))
            and any(word in text for word in ("얼마", "한도", "최대", "까지"))
        )
        # 한도를 **묻지 않고 단정**하는 형태도 같은 정형 답변이 정답이다.
        # 실측 no.383("연금저축을 먼저 600만원 채우고 IRP로 300만원 추가하는 순서가
        # 맞나요?")은 "얼마/한도" 어휘가 없어 후보 0건이 됐고, LLM이 "합산 한도는
        # 연간 700만원"(폐지된 값)이라 답했다. 사용자가 한도를 이미 안다고 전제하고
        # 확인만 구하는 질문일수록 틀린 전제를 바로잡아 줘야 하는데, 정작 그런
        # 질문이 정형 경로에서 빠지고 있었다.
        or (
            mentions_pension_account
            and re.search(r"\d[\d,]*\s*만\s*원", text) is not None
            and any(word in text for word in ("맞나요", "맞는", "맞죠", "순서", "채우", "먼저", "나눠", "배분"))
        )
    )
    if "세액공제" in text or asks_contribution_limit:
        candidates.append("세액공제_계산_입력부족")
        candidates.append("세액공제_한도")
    # ⚠️ "절세"가 연금 문맥 단어와 **동시에** 나오기를 요구하면, 사용자가 연금 상황을
    # 다른 말로 표현한 순간 후보가 0건이 된다. 이 서비스는 연금 상담 전용이므로
    # (범위 판정은 라우터의 scope가 따로 한다) "절세"라는 단어 자체가 이미 충분한
    # 신호다. 동시출현 요구는 커버리지만 좁히고 얻는 것이 없다.
    #
    # 실측(2026-09-02, 실사용): "나는 올해 나이가 65세로 정년 은퇴를 앞두고 있어.
    # 이런 내가 절세를 하고자하는데 방법 알려줘" -> 후보 0건.
    # "은퇴/정년/65세"는 전부 연금 문맥인데 "연금/IRP"라는 단어가 없어서 걸러졌다.
    # 그 결과 LLM 자유응답으로 흘러 "연간 최대 700만원까지 세액공제"(폐지된 한도)를
    # 지어냈다. 정작 _tax_benefit_overview_response는 정답(600/900만원)뿐 아니라
    # 사용자가 물은 이연퇴직소득세 감면(수령연차별 30/40/50%)까지 갖고 있었다.
    # 같은 원인의 실측이 501문항에도 있었다 — no.383(700만원), no.321(700만원).
    if any(word in text for word in ("세금혜택", "세제혜택", "절세혜택", "세금상혜택", "절세")) or (
        "세금" in text and any(word in text for word in ("줄이", "아끼", "덜내", "덜 내", "혜택"))
    ):
        candidates.append("세금혜택_개요")
    # 사유별 기준일 용어("요양종료일", "잔금지급일" 등)는 그 자체로 중도인출 문맥을
    # 특정하므로, "중도인출"이라는 단어가 없어도 후보로 낸다. 실측 no.426
    # ("요양종료일이 2026년 12월 15일이면 신청기한이 다음해로 넘어가나요?")은 후보가
    # 0건이라 결정론 경로를 못 타고 LLM이 임의로 plan_type="DB"를 찍어 호출했다.
    # 제도 용어("중도인출")를 모르는 사용자는 "중간에 빼서 쓴다"처럼 풀어 쓴다.
    asks_early_withdrawal_plainly = any(
        word in text for word in ("중간에빼", "중간에찾", "미리빼", "미리찾", "중간인출")
    ) or (
        any(word in text for word in ("퇴직연금", "연금계좌", "IRP", "irp", "연금저축"))
        and any(word in text for word in ("빼서쓸", "빼서쓰", "빼쓸", "꺼내쓸", "꺼내쓰"))
    )
    if "중도인출" in text or _mentions_withdrawal_basis_event(text) or asks_early_withdrawal_plainly:
        # 같은 도메인의 두 작업을 모두 후보로 낸다: 사유 목록 나열(중도인출_일반)과
        # 기한 계산·판정(중도인출_기한판정). 사유별로 후보 조건을 따로 쓰면
        # ("요양이고 요양종료일이 있으면...") 사유가 늘 때마다 조건이 늘고, 실제로
        # 요양·주택구입만 잡히고 전월세·재난·개인회생은 빠지는 비대칭이 생겼다.
        candidates.append("중도인출_일반")
        candidates.append("중도인출_기한판정")
        candidates.append("중도인출_요건판정")
    # 제도명을 그대로 쓰지 않는 표현도 잡는다 — "디폴트옵션"이라는 단어 없이
    # "기존가입자인데 언제 자동매수되나요?"처럼 묻는 경우가 많다.
    #
    # ⚠️ "자동매수 일정"(언제 되는가)과 "옵트인 가능 여부"(지금 살 수 있는가)는
    # 규칙 엔진에 서로 다른 판정 함수(get_auto_purchase_schedule / check_optin_
    # eligibility)로 이미 분리돼 있는데, 카테고리는 자동매수 하나뿐이었다(실측:
    # "1개 보유 중 같은 상품 추가매수 가능한가요?" 같은 옵트인 질문이 트리거 자체가
    # 없어 얼버무리거나 포기하는 답변만 나갔다). 두 작업을 모두 후보로 내고
    # 라우터가 질문 의도로 고르게 한다.
    # "자동매수"를 붙여 쓰지 않는 표현("자동으로 매수", "자동으로 사")도 같은 질문이다.
    if any(word in text for word in ("디폴트옵션", "사전지정운용", "자동매수")) or (
        "자동" in text and any(word in text for word in ("매수", "매입", "사지", "사는", "삽니"))
    ):
        candidates.append("디폴트옵션_자동매수")
    if "옵트인" in text or (
        any(word in text for word in ("디폴트옵션", "사전지정운용"))
        and any(word in text for word in ("보유", "가지고", "가진", "매수", "추가로"))
    ):
        candidates.append("디폴트옵션_옵트인판정")
    # 마찬가지로 "실물이전"을 "옮기다/이관/이체"로 말하는 경우를 포함한다.
    # ⚠️ 단, 계좌 단위 이전(계좌이전 제도)은 상품 단위 실물이전과 다른 제도다.
    # 실측 no.367("연금저축 계좌를 해지하지 않고 다른 금융사로 옮기는 방법이 있나요?")은
    # "옮기" 하나로 실물이전 카테고리가 붙어, 계좌이전 방법 대신 상품 실물이전
    # **불가사유 목록**을 나열하는 동문서답이 나갔다. 보유 DB에는 계좌이전 제도 문서가
    # 없으므로, 이런 질문은 결정론 경로로 답하지 말고 일반 경로에서 한계를 고지하게 둔다.
    if any(word in text for word in ("실물이전", "이전되", "이전가능", "옮길", "옮기", "이관", "이체")) and not (
        _asks_account_level_transfer(text)
    ):
        # 같은 도메인이라도 "목록 나열"과 "내 상품 판정"은 다른 작업이다. 둘 다 후보로
        # 넣고 라우터가 고르게 한다 — 개별판정 경로가 없던 동안 라우터가 이런 질문을
        # 기각했고, LLM이 자유롭게 툴을 고르다 엉뚱한 툴(투자 가능 여부)을 불러
        # "네, 실물이전 문제없습니다"라는 정반대 답을 낸 실측이 있다.
        candidates.append("실물이전_불가사유")
        candidates.append("실물이전_개별판정")
    # 투자한도(위험자산 70%·TDF 특례)는 investment_limit.py에 원문 대조 완료된 규칙이
    # 있는데도 카테고리가 없어 전부 해당없음으로 빠졌다(실측: 14건, 그중 "DB형도
    # 위험자산 한도가 70%인가요?"에 "아니다"라고 정반대로 답한 오답 1건 확인). "위험자산"·
    # "TDF"라는 정확한 용어를 쓰지 않는 표현("주식형 펀드로 100% 채울 수 있나요")도
    # 잡도록 "비중"+"%" 조합까지 포함한다.
    if any(word in text for word in ("위험자산", "TDF")) or (
        "비중" in text and any(word in text for word in ("%", "퍼센트", "프로"))
    ):
        candidates.append("투자한도_위험자산")
    # 같은 도메인(투자한도)이라도 "비중 한도가 몇 %인가"와 "이 상품을 담을 수 있는가"는
    # 다른 작업이다. investment_limit.py의 PRODUCT_RISK_TIER에는 제도별 투자금지
    # 판정(국내상장주식은 DC/IRP 금지 등)이 이미 원문 대조까지 끝나 있는데, 후보
    # 카테고리가 한도 질문만 겨냥해 가부 질문은 전부 해당없음으로 빠졌다.
    # 실측 no.459("IRP로 국내 상장 개별주식 몇 개 담고 싶은데 추천해주세요"):
    # 근거 0건으로 투자금지 사실을 말하지 못하고 위험성향만 되물었다.
    if _detect_investment_product_type(text) is not None:
        candidates.append("투자가능여부_상품유형")
    # 퇴직·이직 시 퇴직급여가 어디로 가는지(IRP 의무이전)는 원칙과 예외를 함께 말해야
    # 하는데 규칙이 없어 LLM 판단에 맡겨졌다. 실측 no.361은 "이직할 때마다 DC 계좌가
    # 새로 생긴다"고 이전 구조 자체를 뒤집었고, no.17은 "DC 퇴직금은 나이와 상관없이
    # 반드시 IRP로"라며 55세 이후 퇴직 예외를 빠뜨렸다.
    if _asks_irp_mandatory_transfer(text):
        candidates.append("퇴직시_IRP의무이전")
    # ⚠️ "한도"라는 단어를 요구하면 이 카테고리가 실제로 답할 수 있는 질문의 상당수가
    # 후보에조차 오르지 못한다. _withdrawal_limit_response는 한도 공식뿐 아니라
    # **연금수령연차 기산**(2013.3.1 이전 가입 계좌의 6년차 특례), 11년차 한도 소멸,
    # 연금수령 요건(가입 5년·만 55세)까지 근거와 함께 답하는데, 이것들은 "한도"라는
    # 단어 없이 물어보는 게 오히려 자연스럽다.
    #
    # 실측(501문항): 후보가 빈 리스트로 나와 라우터가 고를 선택지조차 없었던 문항들 —
    #   no.104 "2013년 3월 1일 이전 가입인데 연금수령연차를 어떻게 계산하나요"
    #          (정답은 6년차 특례. 근거 0건으로 연령별 수령시기표를 창작했다)
    #   no.99  "연금 실제수령연차랑 연금수령연차가 같은 말 아닌가요"
    #   no.86  "55세 미만인데 연금을 받을 수 있나요"
    #          (사적연금 서비스인데 국민연금 조기노령연금 수치를 지어냈다)
    # 후보에 올린다고 이 카테고리로 확정되는 건 아니다 — 라우터가 최종 판단하므로,
    # 답할 수 있는 범위를 빠짐없이 후보로 올리는 쪽이 안전하다.
    #
    # ⚠️ "실제수령연차"는 이 카테고리가 아니라 퇴직소득세감면 소관이다 — 두 연차는
    # 이름만 비슷할 뿐 서로 다른 값이고(수령연차=한도 산정, 실제수령연차=감면율 산정),
    # _extract_withdrawal_limit_inputs도 "실제수령연차"가 보이면 명시적으로 손을 뗀다.
    # 이 구분을 안 하면 no.92~97·345·346 같은 감면율 질문이 한도 후보로 잘못 올라온다.
    asks_actual_receipt_year = "실제수령연차" in text
    # ⚠️ "연금"+"한도"만으로 잡으면 **납입**한도 질문까지 수령한도로 끌려온다.
    # 납입(넣는 것)과 수령(받는 것)은 정반대 개념이라 정형 답변도 완전히 다르다 —
    # 실측: "연금저축이랑 IRP 납입한도가 얼마인가요?"가 연금수령한도 후보로만 올라왔다.
    asks_contribution_not_receipt = any(
        word in text for word in ("납입한도", "불입한도")
    ) or (
        any(word in text for word in ("넣", "납입", "불입"))
        and not any(word in text for word in ("수령", "받", "인출", "연차"))
    )
    if not asks_contribution_not_receipt and (
        "연금수령한도" in text
        or ("연금" in text and "한도" in text)
        or (not asks_actual_receipt_year and "수령연차" in text)
        or (not asks_actual_receipt_year and "연금" in text and "연차" in text)
        or ("연금" in text and "55세" in text)
    ):
        candidates.append("연금수령한도")
    # ⚠️ 제도 용어("퇴직소득세"/"이연퇴직소득세")만 요구하면, 일상어로 묻는 질문이
    # 후보에조차 못 오른다 — 사용자는 "퇴직금"이라고 말하지 "이연퇴직소득"이라고 하지
    # 않는다. 실측 "퇴직금 1억원을 연금으로 받으려고 해. 세금은?"은 후보가
    # ['연금소득세율_연령별']뿐이라, **퇴직금 재원인데 사적연금소득 세율표(5.5/4.4/3.3%)**로
    # 답하는 재원 혼동이 났다. 퇴직금 재원은 이연퇴직소득세 감면 체계라 세율 체계 자체가
    # 다르므로, 재원을 잘못 잡으면 숫자가 통째로 틀린다.
    #
    # "퇴직금 + 연금수령 + 세금" 세 신호가 함께 있으면 이 카테고리가 맞다. 핸들러는
    # 연차 정보가 없으면 스스로 None을 내므로(실측 확인) 과잉 확정 위험도 없다.
    asks_retirement_pay_pension_tax = (
        any(word in text for word in ("퇴직금", "퇴직급여", "명예퇴직금", "명퇴금"))
        and any(word in text for word in ("연금으로", "연금수령", "연금으로받", "연금개시"))
        and any(word in text for word in ("세금", "세율", "과세", "감면"))
    )
    # "연금실제수령연차"는 이 카테고리 고유의 개념어다(감면율 산정 전용 — 한도 산정용
    # "연금수령연차"와 다른 값). 이 단어를 쓴 질문은 감면율을 묻는 것이 거의 확실한데
    # 조건에 없어 후보에조차 못 올랐다(실측: "연금실제수령연차 5년차인데 퇴직금 세금
    # 얼마나 감면돼?" -> 후보 ['개인세금_입력충분성']). 퇴직금 + 감면 조합도 같은 이유로
    # 넣는다 — "연금으로"라는 표현 없이도 감면을 물으면 이 카테고리다.
    asks_actual_receipt_year_reduction = "실제수령연차" in text or (
        any(word in text for word in ("퇴직금", "퇴직급여", "명예퇴직금", "명퇴금"))
        and "감면" in text
    )
    if (
        any(word in text for word in ("퇴직소득세", "이연퇴직소득세"))
        or asks_retirement_pay_pension_tax
        or asks_actual_receipt_year_reduction
    ):
        candidates.append("퇴직소득세감면")
    # ⚠️ "연금소득세"는 부분문자열 매칭이라 "연금소득세율"("세율" 부분)에도 걸린다.
    # 그러면 "연령별 연금소득세율 표 알려줘"가 종합과세로만 후보를 잡고 정작 아래
    # 연령별 세율 카테고리는 후보에서 빠지는 오발화가 생긴다(실측). "종합과세"·
    # "분리과세"는 그 자체로 명확한 단어라 문제없지만, "연금소득세"만 뒤에 "율"이
    # 붙지 않았는지 확인해 순수 종합과세 표현만 잡는다.
    # 제도명 대신 **기준 금액**으로 묻는 표현이 흔하다("1500만원 넘으면 어떻게 되나요").
    # 1,500만원은 사적연금소득 종합과세 판단 기준이라 이 금액 자체가 강한 신호다.
    mentions_annual_threshold = any(
        word in text for word in ("1500만", "1,500만", "1500만원", "천5백만", "1천5백만")
    )
    # ⚠️ 기준값(1,500만원) 자체를 말한 경우만 잡으면, 정작 **본인 연금소득 금액**을
    # 말한 질문이 후보 0건이 된다 — 규칙이 발동하는 바로 그 상황인데도. 실측 CASE 8
    # "연금소득이 1600만원이야"는 후보가 없어 LLM 자유응답으로 샜다(그 경로는 폐지된
    # 수치를 지어내는 곳이라 통제 밖으로 나가는 것과 같다).
    #
    # 연금소득을 만원 단위 금액으로 말했으면 이 카테고리의 안내 대상이다. 핸들러는
    # 기준과 판정 대상 재원만 설명하고 **사용자 금액이 과세대상인지는 단정하지 않으므로**
    # (실측 확인) 금액을 잘못 확정할 위험이 없다.
    states_pension_income_amount = ("연금소득" in text or "연금으로" in text) and re.search(
        r"\d[\d,]*\s*만\s*원", text
    ) is not None
    if any(word in text for word in ("종합과세", "분리과세")) or (
        "연금소득세" in text and "연금소득세율" not in text
    ) or (mentions_annual_threshold and any(w in text for w in ("연금", "초과", "넘으면", "넘으"))) or (
        states_pension_income_amount
    ):
        candidates.append("연금소득세_종합과세")

    # 연령별 연금소득세율 — 세금 얘기 + (나이 언급 OR 연금 수령 문맥)일 때 후보에 넣는다.
    # 사용자는 "나 74세인데 세금 어떻게 내?"처럼 제도명을 생략하므로 "연금소득세"라는
    # 정확한 단어를 요구하면 같은 의도의 질문을 놓친다.
    #
    # ⚠️ "연금"이라는 단어만으로는 부족하다. 그러면 "연금저축 600만원 납입하고 총급여
    # 5000만원인데 세액공제 얼마?"가 "연금"(연금저축) + "얼마"만으로 후보가 되어,
    # 나이가 전혀 없는데도 연령별 세율표를 답하는 경로가 열린다(실측). 이 카테고리는
    # 이름 그대로 **연령별**이므로, 나이가 없다면 "연금을 받는" 문맥이라도 있어야 한다.
    #
    # ⚠️ "연령별"이라는 단어 자체는 이 카테고리를 직접 지목하는 명시적 신호라 나이 숫자·
    # 수령 문맥 없이도 후보에 넣는다 — 실측: "연령별 연금소득세율 표 알려줘"에 만 나이가
    # 없어 후보 0건이 되고, 라우터가 맞게 판정해도 _enforce_candidate_scope가 되돌렸다.
    has_tax_context = any(word in text for word in _TAX_AMOUNT_WORDS)
    has_receipt_context = any(word in text for word in _PENSION_RECEIPT_WORDS)
    # "연금소득세율"은 이 카테고리를 글자 그대로 지목하는 이름이다. 나이·수령 문맥을
    # 추가로 요구하면 정작 가장 직접적인 질문("연금소득세율 알려줘")이 후보 0건이 된다
    # — 이 함수가 반복해서 겪은 "표면 어휘 조합" 실패와 같은 형태다.
    # 핸들러는 나이를 못 찾으면 연령별 세율표 전체를 안내하므로 되살려도 안전하다.
    if "연금소득세율" in text or "연령별" in text or (
        has_tax_context and (_AGE_MENTION_RE.search(text) or has_receipt_context)
    ):
        candidates.append("연금소득세율_연령별")

    return candidates


# ── 정형 경로 누락 관측 (silent miss) ────────────────────────────────────────
#
# candidate_categories의 실패는 **조용하다**. 후보가 0건이 되면 아무 신호 없이 LLM
# 자유응답으로 넘어가고, 틀린 답이 나가야만 비로소 발견된다. 실제로 같은 유형이
# 세 번 재발했다:
#   - no.383/no.321  "연금저축 600만원 채우고 IRP 300만원" → 700만원(폐지된 한도)
#   - V03/V05        "세액공제 대박으로 받는 방법"          → 700만원/400만원/14.6%
#   - 실사용(65세)    "정년 은퇴 앞두고 절세 방법"           → 700만원
# 셋 다 정답을 가진 핸들러가 있었는데 어휘 조건이 좁아 도달하지 못한 것이다.
#
# 오탐(넓게 잡음)은 라우터와 deterministic_response_for 게이트가 걸러내지만,
# **누락은 아무도 막지 못한다** — 이 비대칭이 문제의 핵심이다. 그래서 누락을
# 최소한 "보이게" 만든다: 정형 주제어가 있는데 후보가 0건이면 신호를 남긴다.
#
# 이 함수는 판정을 바꾸지 않는다(후보를 추가하지도, 빼지도 않는다). 오직 관측용이다 —
# think_trace와 평가 로그에 남겨, 다음 누락을 "터진 뒤"가 아니라 집계로 발견한다.
_DETERMINISTIC_TOPIC_MARKERS = (
    "세액공제", "절세", "세금혜택", "세제혜택", "연금소득세", "퇴직소득세", "기타소득세",
    "중도인출", "실물이전", "디폴트옵션", "연금수령한도", "위험자산", "종합과세",
    "납입한도", "세율", "과세",
)


def deterministic_miss_signal(question: str) -> Optional[str]:
    """정형 주제어가 있는데 후보가 0건이면 그 주제어를 돌려준다 (없으면 None).

    "정형 답변이 있어야 할 것 같은데 경로가 없다"는 의심 신호다. 확정된 결함이
    아니라 **점검 대상**이라는 뜻이다 — 정형 카테고리가 아직 없는 주제(DB/DC 운용주체
    차이 등)도 여기 걸리므로, 이 신호가 곧 버그를 의미하지는 않는다.
    """
    if candidate_categories(question):
        return None
    text = _compact(question)
    hit = [marker for marker in _DETERMINISTIC_TOPIC_MARKERS if marker in text]
    return ", ".join(hit) if hit else None


def deterministic_response_for(
    category: str, question: str
) -> Optional[tuple[str, list[RetrievedItem]]]:
    """2단계: router가 확정한 카테고리로 실제 정형 답변을 만든다.

    category가 "해당없음"이거나, 핸들러가 없거나, **그 카테고리가 이 질문의 후보가 아니면**
    None — 호출측(info_agent)은 LLM+툴 경로로 진행해야 한다.

    ## 왜 여기서 후보를 다시 확인하는가

    원래 설계는 "후보 생성(넓게) → 라우터 확정(정확히) → 핸들러 실행(신뢰)"이었고,
    라우터가 유일한 관문이므로 핸들러는 질문 적합성을 확인하지 않는다. 실제로 13개
    핸들러 중 10개는 반환 타입에 None이 없어 **물러날 방법 자체가 없다**("오늘 점심
    뭐 먹지"에도 세액공제 한도표를 반환한다).

    그 전제는 라우터를 우회하는 경로가 없을 때만 성립한다. router._restore_rejected_category
    (라우터가 잘못 기각한 카테고리를 코드가 되살리는 장치)가 관문을 우회하면서, 검증되지
    않은 카테고리로 정형 답변이 나갈 수 있는 구멍이 생겼다. 실측: 되살릴 대상을 넓게
    잡았더니 "연금저축 600만원 납입하고 총급여 5000만원인데 세액공제 얼마?"가
    연금소득세율_연령별로 되살아나 묻지도 않은 세율표를 답했다.

    핸들러 13개에 각각 가드를 붙이는 방법도 있으나, 새 핸들러가 추가될 때 빠뜨리면
    조용히 다시 뚫린다. candidate_categories가 이미 카테고리별 적합성을 판정하고 있으므로
    (무관한 질문에는 빈 목록을 낸다) 그 판정을 dispatch 한 곳에서 재사용한다 —
    핸들러가 몇 개든, 앞으로 몇 개가 더 생기든 자동으로 같은 보호를 받는다.
    """
    handler = _CATEGORY_HANDLERS.get(category)
    if handler is None:
        return None
    if category not in candidate_categories(question):
        return None
    return handler(question)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


# "종신연금 아니고", "TDF 아닌"처럼 어떤 조건을 명시적으로 **배제**하는 표현.
# 한글은 완성형 음절이라 "아니"가 "아닌"의 부분문자열이 아니므로(받침 ㄴ이 붙으면
# 통째로 다른 음절 "닌"이 된다) 활용형을 나열해야 한다.
_NEGATION_SUFFIXES = ("아니라", "아니고", "아닌", "아니에요", "아니예요", "아니야", "말고", "빼고")


def _is_negated(compact_text: str, keyword: str) -> bool:
    """질문이 그 키워드를 명시적으로 배제하는지 판정한다.

    ⚠️ 키워드가 있다는 이유만으로 조건을 적용하면 정반대 답이 나간다. 실측:
      "76세인데 **종신연금 아니고** 그냥 확정기간형으로 받으면 세율이 얼마예요?"
        -> 종신연금 3.3%로 답변(정답은 70~80세 일반수령 4.4%)
      "IRP에서 **TDF 아닌** 일반 주식형 펀드로 100% 채울 수 있나요?"
        -> TDF 특례 100%로 답변(정답은 일반 한도 70%)
    """
    escaped = re.escape(keyword)
    pattern = rf"{escaped}(?:은|는|이|가)?(?:{'|'.join(_NEGATION_SUFFIXES)})"
    return re.search(pattern, compact_text, re.IGNORECASE) is not None


def _asks_alternative_withdrawal_plan_types(compact_text: str) -> bool:
    if "중도인출" not in compact_text:
        return False
    return any(
        marker in compact_text
        for marker in (
            "다른제도",
            "다른퇴직연금",
            "다른계좌",
            "가능한제도",
            "가능한종류",
            "어떤제도",
            "어떤종류",
            "무슨제도",
            "뭐가있",
            "무엇이있",
        )
    )


def _won(amount: int) -> str:
    if amount % 10_000 == 0:
        return f"{amount // 10_000:,}만원"
    return f"{amount:,}원"


def _won_readable(amount: int) -> str:
    if amount % 10_000 == 0:
        return _won(amount)
    if amount >= 10_000:
        man, rest = divmod(amount, 10_000)
        if rest and rest % 1_000 == 0:
            return f"{man:,}만 {rest // 1_000}천원"
    return f"{amount:,}원"


def _pct(rate: float) -> str:
    return f"{rate * 100:g}%"


def _context(source: str, content: str) -> list[RetrievedItem]:
    return [{"source": source, "content": content, "node": "info_agent"}]


# 한글 숫자 단위. "6천만원"처럼 아라비아 숫자와 만원 사이에 "천"이 끼는 표기를
# 처리하기 위해 "천/백/십" 보조단위까지 인식한다. "만"·"억"은 필수 뒤 단위이고
# "천/백/십"은 그 앞에 선택적으로 붙는 보조단위다 (예: 6천만 = 6*1000*10000).
_AMOUNT_NUMBER_RE = (
    r"(?:[0-9,]+(?:\.[0-9]+)?)"  # 아라비아 숫자, 콤마/소수점 허용
    r"(?:\s*(?:천|백|십)\s*)?"    # 보조단위 (선택)
)
_AMOUNT_UNIT_RE = r"(?:만\s*원|억\s*원|만|억|원)"

# 숫자와 단위 사이에 이 토큰들이 끼면 "라벨 값" 매칭을 무효화한다 — 다른 금액이나
# 다른 라벨이 사이에 끼어 있으면 그 금액을 엉뚱한 라벨로 잘못 읽는 사고가 난다
# (실측: "IRP 200만원 넣었고 총급여 6천만원"에서 총급여를 200만원으로 오독).
_AMOUNT_BOUNDARY_BREAK_RE = re.compile(
    r"[0-9](?:만원|억원|만|억|원)|연금저축|irp|총급여|급여|연봉|종합소득|사업소득",
    re.IGNORECASE,
)


def _parse_korean_amount(number_text: str, unit_text: str) -> int:
    """'6천만', '1,200만', '3억' 같은 숫자+한글단위 조합을 원 단위 정수로 바꾼다."""
    sub_unit = 1
    stripped = number_text.strip()
    for word, mul in (("천", 1_000), ("백", 100), ("십", 10)):
        if stripped.endswith(word):
            sub_unit = mul
            stripped = stripped[: -len(word)].strip()
            break
    value = float(stripped.replace(",", "")) * sub_unit
    if "억" in unit_text:
        value *= 100_000_000
    elif "만" in unit_text:
        value *= 10_000
    return int(value)


# 금액 바로 앞에 붙는 "월 단위" 표현. 세액공제·납입한도는 모두 **연간** 기준이라,
# 월 납입액을 연액으로 그대로 쓰면 12배 틀린 답이 나간다(실측 재현: "연금저축에 매달
# 50만원씩 넣는데 세액공제 얼마?"에 연 50만원으로 계산해 82,500원을 답했다 —
# 연 600만원 기준 99만원이 정답).
_MONTHLY_PREFIX_RE = re.compile(r"(?:매\s*달|매\s*월|월\s*납|달\s*마다|한\s*달\s*에|월)$")


def _is_monthly_amount(compact: str, amount_start: int) -> bool:
    """금액 바로 앞 구간에 월 단위 표현이 있는지 확인한다."""
    head = compact[max(0, amount_start - 8): amount_start]
    return _MONTHLY_PREFIX_RE.search(head) is not None


def _extract_labeled_amount(question: str, labels: tuple[str, ...]) -> int | None:
    """질문에서 '라벨 값' 또는 '값 라벨' 형태로 언급된 금액을 **연 단위**로 추출한다.

    라벨과 숫자 사이에 다른 금액이나 다른 라벨 단어가 끼면 매칭을 버린다 — 그런
    경우는 대개 그 금액이 이 라벨의 값이 아니라 옆에 있던 다른 항목의 값이다.

    "매달 50만원"처럼 월 단위로 말한 금액은 12를 곱해 연 환산한다 — 이 함수의
    결과는 전부 연간 기준(세액공제 한도·납입한도)으로 쓰이기 때문이다.
    """
    compact = _compact(question)
    number_unit = _AMOUNT_NUMBER_RE + r"\s*" + _AMOUNT_UNIT_RE
    for label in labels:
        escaped = re.escape(label)
        after = re.search(rf"{escaped}([^\d]{{0,12}})({number_unit})", compact, re.IGNORECASE)
        if after and not _AMOUNT_BOUNDARY_BREAK_RE.search(after.group(1)):
            amount = _amount_from_match(after.group(2))
            return amount * 12 if _is_monthly_amount(compact, after.start(2)) else amount

        before = re.search(rf"({number_unit})([^\n,.;]{{0,12}}){escaped}", compact, re.IGNORECASE)
        if before and not _AMOUNT_BOUNDARY_BREAK_RE.search(before.group(2)):
            amount = _amount_from_match(before.group(1))
            return amount * 12 if _is_monthly_amount(compact, before.start(1)) else amount
    return None


def _amount_from_match(amount_text: str) -> int:
    m = re.match(rf"({_AMOUNT_NUMBER_RE})\s*({_AMOUNT_UNIT_RE})", amount_text)
    return _parse_korean_amount(m.group(1), m.group(2))


def extract_tax_credit_inputs(question: str) -> dict[str, int | None]:
    return {
        "pension_savings_paid": _extract_labeled_amount(question, ("연금저축", "연저")),
        "irp_paid": _extract_labeled_amount(question, ("IRP", "irp", "개인형IRP", "개인형퇴직연금")),
        "total_salary": _extract_labeled_amount(question, ("총급여", "급여", "연봉")),
        "comprehensive_income": _extract_labeled_amount(question, ("종합소득금액", "종합소득", "사업소득")),
    }


def _has_sufficient_tax_credit_inputs(values: dict[str, int | None]) -> bool:
    has_contribution = values["pension_savings_paid"] is not None or values["irp_paid"] is not None
    has_income = values["total_salary"] is not None or values["comprehensive_income"] is not None
    return has_contribution and has_income


def _tax_credit_calculation_missing_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc41 세액공제 계산 입력값 규칙"
    values = extract_tax_credit_inputs(question)
    if _has_sufficient_tax_credit_inputs(values):
        pension_savings_paid = values["pension_savings_paid"] or 0
        irp_paid = values["irp_paid"] or 0
        result = calculate_tax_credit(
            pension_savings_paid=pension_savings_paid,
            irp_paid=irp_paid,
            total_salary=values["total_salary"],
            comprehensive_income=values["comprehensive_income"],
        )
        income_label = (
            f"총급여 {_won(values['total_salary'])}"
            if values["total_salary"] is not None
            else f"종합소득금액 {_won(values['comprehensive_income'])}"
        )
        income_basis = (
            f"총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하"
            if values["total_salary"] is not None and values["total_salary"] <= INCOME_THRESHOLD_SALARY
            else f"총급여 {_won(INCOME_THRESHOLD_SALARY)} 초과"
            if values["total_salary"] is not None
            else f"종합소득금액 {_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하"
            if values["comprehensive_income"] <= INCOME_THRESHOLD_COMPREHENSIVE
            else f"종합소득금액 {_won(INCOME_THRESHOLD_COMPREHENSIVE)} 초과"
        )
        content = (
            f"세액공제액 계산에는 연금저축 납입액, IRP 납입액, 총급여 또는 종합소득금액이 필요합니다. "
            f"연금저축 단독 세액공제 대상 한도는 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
            f"연금저축+IRP 합산 세액공제 대상 한도는 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
            f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
            f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
            f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다. 입력 조건에서는 세액공제 대상 납입액 "
            f"{_won(result.credited_total)} x {_pct(result.credit_rate)} = {_won_readable(result.tax_credit_amount)}입니다."
        )
        # ⚠️ 아래 초과분 문구를 lines(답변)에만 넣고 content(근거)에는 안 넣으면, ④검증이
        # "답변에 있는데 근거에는 없는 수치"로 오판해 grounded=False가 나고 그 지적이
        # unsupported_numbers_confirmed에 실제 계산값(예: "100만원")으로 확정되는 사고가
        # 난다(실측). 답변에 쓸 계산 결과는 반드시 근거에도 먼저 반영한다.
        if result.excess_beyond_credit_limit:
            content += (
                f" 세액공제 대상 한도를 초과한 납입액 {_won(result.excess_beyond_credit_limit)}은 "
                "세액공제액 계산에는 포함되지 않습니다."
            )
        if result.over_contribution_limit:
            content += f" 연금저축+IRP 합산 납입한도 {_won(TOTAL_CONTRIBUTION_LIMIT)}도 초과합니다."
        lines = [
            "입력해주신 조건으로 세액공제액을 계산하면 다음과 같습니다.",
            "",
            f"- 연금저축 납입액: {_won(pension_savings_paid)}",
            f"- IRP 납입액: {_won(irp_paid)}",
            f"- 소득 기준: {income_label}",
            f"- 적용 공제율: {_pct(result.credit_rate)} ({income_basis})",
            f"- 세액공제 대상 납입액: {_won(result.credited_total)}",
            "",
            f"예상 세액공제액은 {_won(result.credited_total)} x {_pct(result.credit_rate)} = {_won_readable(result.tax_credit_amount)}입니다.",
        ]
        if result.excess_beyond_credit_limit:
            lines.append(
                f"\n세액공제 대상 한도를 초과한 납입액 {_won(result.excess_beyond_credit_limit)}은 "
                "세액공제액 계산에는 포함되지 않습니다."
            )
        if result.over_contribution_limit:
            lines.append(f"\n연금저축+IRP 합산 납입한도 {_won(TOTAL_CONTRIBUTION_LIMIT)}도 초과합니다.")
        return "\n".join(lines), _context(source, content)

    content = (
        f"세액공제액 계산에는 연금저축 납입액, IRP 납입액, 총급여 또는 종합소득금액이 필요합니다. "
        f"세액공제 대상 납입한도는 연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
        f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
        f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다."
    )
    draft = (
        "세액공제 금액은 납입액과 소득구간이 함께 있어야 계산할 수 있습니다.\n\n"
        "현재 질문에는 실제 계산에 필요한 입력값이 부족하므로, 세액공제액을 임의로 산출하지 않겠습니다.\n\n"
        "현재 답변 가능한 범위는 다음과 같습니다.\n"
        "- 세액공제 대상 한도는 연금저축 단독 연 600만원, 연금저축+IRP 합산 연 900만원입니다.\n"
        "- 세액공제율은 총급여 5,500만원 이하 또는 종합소득금액 4,500만원 이하이면 16.5%, "
        "그 초과 구간이면 13.2%입니다.\n\n"
        "정확한 계산을 위해 다음 정보를 한 번에 알려주세요.\n"
        "1. 올해 연금저축에 납입한 금액은 얼마인가요?\n"
        "2. 올해 IRP에 납입한 금액은 얼마인가요?\n"
        "3. 직장인이라면 총급여, 개인사업자라면 종합소득금액은 얼마인가요?\n"
        "4. 이미 회사 DC/IRP 추가납입 등 다른 연금계좌 납입액이 있다면 함께 알려주세요."
    )
    return draft, _context(source, content)


# DB(확정급여형) 급여 계산식. 실측(no.1/no.27): LLM이 "평균 임금의 60% x 근속연수"라는
# 근거에 없는 계산식을 지어냈다. 정답은 "퇴직 전 평균임금 30일분 x 계속근로기간"이다.
_DB_DC_COMPARISON_SOURCE = "DB DC 퇴직연금 산정 / 퇴직연금 가입대상 / 퇴직연금제도 기본"
_DB_DC_COMPARISON_CONTENT = (
    "퇴직연금제도는 확정급여형(DB, Defined Benefit)과 확정기여형(DC, Defined Contribution) "
    "두 가지로 나뉜다.\n\n"
    "확정급여형(DB): 근로자가 퇴직 시 받을 금액이 사전에 확정되어 있으며, 회사가 적립금을 "
    "운용한다. DB형 급여 계산식은 '퇴직 전 평균임금 30일분 x 계속근로기간'이다. "
    "평균임금 1일분은 퇴직 전 3개월간 지급된 임금 총액을 해당 기간의 총 일수로 나누어 "
    "산정한다.\n\n"
    "확정기여형(DC): 회사가 매년 일정 금액을 근로자의 계좌에 입금하고, 근로자가 직접 운용하여 "
    "수익률에 따라 최종 퇴직금이 달라진다. DC는 가입자 명부를 근거로 가입자 본인 명의의 "
    "실계좌가 개설되어 온라인 상품매매 등이 가능하다.\n\n"
    "제도 변경: 퇴직연금(DB)에서 퇴직연금(DC)으로는 변경할 수 있으나, 퇴직연금(DC)에서 "
    "퇴직연금(DB)으로는 변경할 수 없다."
)

# 구체적인 개인 계산을 요구하는 신호. "제가/저는"만으로는 판단하지 않는다 —
# "DB형은 제가 받을 퇴직금이 미리 확정돼 있는 게 맞나요?"(no.27)처럼 1인칭이지만
# 제도 사실을 묻는 질문이 흔해서, 그것까지 걸러내면 실측 사례를 놓친다. 실제 계산을
# 요구하는 신호(숫자 제시, "계산해줘")만 개인 판정으로 넘긴다.
_DB_DC_CALCULATION_SIGNAL_MARKERS = ("계산해", "계산해줘", "얼마나되나요", "얼마받나요", "얼마입니까")
_NUMBER_PRESENT_RE = re.compile(r"\d")


def _db_dc_comparison_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """DB형·DC형의 제도 차이(운용주체·계산방식)를 일반론으로 설명한다.

    실측(no.1 "DC와 DB, 퇴직금이 정해지는 방식이랑 운용 주체가 어떻게 다른가요?",
    no.27 "DB형은 제가 받을 퇴직금이 미리 확정돼 있는 게 맞나요?"): 근거를 7건씩
    갖고도 LLM이 DB 급여 계산식을 "평균 임금의 60% x 근속연수"로 창작했다. 정답은
    이미 문서에 있었다 — 정형 경로로 확정해 창작 여지를 아예 없앤다.

    ⚠️ 구체적인 개인 계산 질문은 여기서 답하지 않는다("근속 10년인데 퇴직금 얼마
    받나요" 등 숫자를 대입한 계산 요구). 이 카테고리는 "일반형만 확정한다" 원칙에
    따라 제도 설명만 다룬다 — 판정은 1인칭 여부가 아니라 숫자·계산 요구 여부다.
    """
    text = _compact(question)
    if _NUMBER_PRESENT_RE.search(text) or any(
        marker in text for marker in _DB_DC_CALCULATION_SIGNAL_MARKERS
    ):
        return None
    asks_db = "DB" in question or "확정급여" in text
    asks_dc = "DC" in question or "확정기여" in text
    asks_comparison = any(word in text for word in ("차이", "다른가", "다른가요", "다릅니", "비교"))
    asks_db_calc = asks_db and any(word in text for word in ("계산", "산정", "얼마로", "어떻게정해"))
    asks_operator = any(word in text for word in ("운용주체", "누가운용", "누가굴리", "직접운용"))
    asks_db_confirmation = asks_db and any(
        word in text for word in ("확정돼있는", "확정되어있는", "확정된게", "맞나요", "맞는")
    )
    if not (
        (asks_db and asks_dc and (asks_comparison or asks_operator))
        or asks_db_calc
        or asks_db_confirmation
    ):
        return None
    draft = (
        "퇴직연금제도는 확정급여형(DB)과 확정기여형(DC)으로 나뉩니다.\n\n"
        "**DB형(확정급여형)**\n"
        "- 회사가 적립금을 운용하고, 근로자가 퇴직 시 받을 금액은 사전에 확정되어 있습니다.\n"
        "- 급여 계산식: 퇴직 전 평균임금 30일분 x 계속근로기간\n\n"
        "**DC형(확정기여형)**\n"
        "- 회사가 매년 일정 금액을 근로자 명의 계좌에 입금하고, 근로자 본인이 직접 운용합니다.\n"
        "- 최종 퇴직금은 운용 수익률에 따라 달라집니다.\n\n"
        "즉 운용 주체가 DB는 회사, DC는 근로자 본인이라는 점이 핵심 차이입니다. "
        "제도 변경은 DB에서 DC로는 가능하지만 DC에서 DB로는 불가능합니다."
    )
    return draft, _context(_DB_DC_COMPARISON_SOURCE, _DB_DC_COMPARISON_CONTENT)


# 실물이전(상품 매도 없이 금융기관만 변경)과 이체(현금으로 다른 종류 계좌로 옮김)는
# 서로 다른 제도다. 근거: [퇴직연금 실물이전제도 안내], [IRP 중도인출·계약해지·이체
# 및 연금인출 안내].
_ACCOUNT_TRANSFER_SOURCE = "퇴직연금 실물이전제도 안내 / IRP 중도인출·계약해지·이체 및 연금인출 안내"
_ACCOUNT_TRANSFER_CONTENT = (
    "실물이전제도(2024년 10월 31일 시행): 보유 중인 상품을 매도하지 않고 퇴직연금 "
    "금융기관을 변경하는 제도. 동일 제도 간에만 가능하다 — DB제도→DB제도, "
    "DC제도→DC제도, IRP계좌→IRP계좌. DB/DC제도는 재직 중인 회사를 통해서만 이전 "
    "신청이 가능하고, IRP계좌는 영업점 또는 모바일(M-STOCK: 연금>타사연금가져오기/"
    "실물이전 경로)로 직접 신청할 수 있다.\n\n"
    "개인형 IRP 이체(소득세법 시행령 40조): 세액공제·과세이연 등 세제혜택을 유지하며 "
    "다른 연금계좌로 이체하는 것으로, 전액 이체만 가능하다.\n"
    "- IRP 상호간 이체: 가입자 부담금·이연퇴직소득이 있는 모든 IRP계좌 대상. "
    "연령 제한 없이 가능하며 실물이전도 가능하다.\n"
    "- IRP ↔ 연금저축계좌 간 이체: 가입자 연령 55세 이상 and 연금계좌 가입일로부터 "
    "5년 경과 시 가능(이연퇴직소득이 있으면 연령·경과기간 요건 완화). 실물이전은 "
    "불가하고 현금이전만 가능하다."
)


def _account_transfer_procedure_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """IRP·연금저축 계좌를 다른 금융기관·다른 종류로 옮기는 절차를 일반론으로 설명한다.

    실측(no.56 "IRP 계좌를 다른 증권사로 옮기려면 어떻게 해야 하나요?"): 근거를
    10건 확보하고도 LLM이 "2013년 3월 1일 이후 가입한 연금계좌는 그 이전 가입
    계좌로 옮길 수 없다"는 규정을 창작했다. 근거를 다시 확인한 결과 2013.03.01은
    실재하는 날짜이지만 **완전히 다른 제도**(연금수령연차 계산 시작점 — 그 이전
    가입한 구 연금저축계좌는 6년차부터 시작)에 관한 것이었다. 계좌 이전 가능 여부와는
    무관한데, LLM이 절반쯤 기억한 날짜를 엉뚱한 맥락에 갖다 붙인 전형적인 창작이다.

    ⚠️ 개인 상황(본인이 보유한 상품 유형·연령 등을 대입한 개별 판정)은 다루지
    않는다. "일반형만 확정한다" 원칙에 따라 이전 절차·조건의 일반론만 설명한다.
    """
    text = _compact(question)
    mentions_irp_or_pension_account = any(
        word in text for word in ("IRP", "irp", "연금저축", "퇴직연금", "연금계좌")
    )
    asks_transfer = any(
        word in text for word in ("이전", "옮기", "옮겨", "이체", "증권사를바꾸", "금융기관을바꾸")
    )
    if not (mentions_irp_or_pension_account and asks_transfer):
        return None
    # 개인 판정 신호(구체적 상품 상태·연령 등을 대입한 질문)는 여기서 답하지 않는다.
    if any(word in text for word in ("제가보유", "제보유", "실물이전가능한가요", "MMF", "RP상품")):
        return None
    draft = (
        "IRP·연금저축 계좌를 다른 금융기관으로 옮기는 방법은 두 가지입니다.\n\n"
        "**1. 실물이전** (2024년 10월 31일 시행)\n"
        "- 보유 중인 상품을 매도하지 않고 금융기관만 변경합니다.\n"
        "- 동일 제도 간에만 가능합니다: IRP계좌→IRP계좌, DC제도→DC제도, DB제도→DB제도.\n"
        "- IRP계좌는 영업점 방문 또는 모바일 앱으로 직접 신청할 수 있습니다.\n"
        "- DB/DC제도는 재직 중인 회사를 통해서만 이전 신청이 가능합니다.\n\n"
        "**2. 이체** (다른 종류의 연금계좌로 옮길 때)\n"
        "- 세액공제·과세이연 혜택을 유지하며 다른 연금계좌로 전액 이체합니다.\n"
        "- IRP 상호간 이체는 연령 제한 없이 가능하고 실물이전도 가능합니다.\n"
        "- IRP와 연금저축계좌 간 이체는 만 55세 이상이고 가입일로부터 5년이 지나야 "
        "가능하며, 이 경우 실물이전은 안 되고 현금이전만 가능합니다.\n\n"
        "본인이 보유한 상품 유형이나 구체적인 조건에 따라 적용되는 방식이 달라질 수 "
        "있으니, 정확한 절차는 이전받을 금융기관에 문의하시기 바랍니다."
    )
    return draft, _context(_ACCOUNT_TRANSFER_SOURCE, _ACCOUNT_TRANSFER_CONTENT)


def _tax_benefit_overview_response(question: str) -> tuple[str, list[RetrievedItem]]:
    compact = _compact(question)
    if "개인사업자" in compact or "자영업" in compact:
        source = "doc41 세액공제 규칙 및 개인사업자 IRP 가입대상"
        content = (
            "개인사업 대표는 퇴직연금에는 가입할 수 없지만 일반 IRP를 통해 자영업자로 가입할 수 있습니다. "
            f"연금저축+IRP 합산 납입한도는 연 {_won(TOTAL_CONTRIBUTION_LIMIT)}이고, 세액공제 대상 한도는 "
            f"연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, 연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
            f"개인사업자 등 종합소득자는 종합소득금액 {_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 "
            f"{_pct(CREDIT_RATE_LOW)}, 초과이면 {_pct(CREDIT_RATE_HIGH)} 세액공제율을 적용합니다. "
            "연금계좌 운용수익은 인출 전 과세이연되고, 세액공제 받은 금액과 운용수익을 연금으로 수령할 때는 "
            "연령별 연금소득세율과 사적연금소득 1,500만원 기준을 함께 확인합니다."
        )
        draft = (
            "개인사업자라면 연금 상담 범위에서는 주로 **연금저축·IRP를 활용한 절세**를 확인할 수 있습니다.\n\n"
            "- 개인사업 대표는 퇴직연금에는 가입할 수 없지만, 일반 IRP를 통해 자영업자로 가입할 수 있습니다.\n"
            "- 세액공제 대상 한도는 연금저축 단독 연 600만원, 연금저축+IRP 합산 연 900만원입니다.\n"
            "- 개인사업자처럼 종합소득 기준으로 보는 경우, 종합소득금액 4,500만원 이하이면 16.5%, 초과이면 13.2% 세액공제율이 적용됩니다.\n"
            "- 연금계좌 안의 운용수익은 인출 전까지 과세이연되며, 나중에 연금으로 받을 때는 재원·수령방식·연간 수령액에 따라 과세가 달라집니다.\n\n"
            "따라서 구체적인 세액공제액을 계산하려면 올해 연금저축 납입액, IRP 납입액, 종합소득금액을 알려주시면 됩니다."
        )
        return draft, _context(source, content)

    source = "doc38~doc41 연금계좌 세금혜택 규칙"
    content = (
        f"연금계좌의 세금혜택은 납입 시 세액공제, 운용 중 과세이연, 연금수령 시 저율 과세, "
        f"퇴직금을 연금으로 받을 때 이연퇴직소득세 감면으로 정리됩니다. "
        f"세액공제 대상 납입한도는 연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
        f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
        f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다. "
        f"과세대상 사적연금소득은 연 {_won(ANNUAL_THRESHOLD)} 초과 여부가 종합과세 판단 기준이며, "
        f"초과 시 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세를 선택할 수 있습니다. "
        f"이 판정 대상은 세액공제 받은 납입금과 운용수익 재원이며, 세액공제 받지 않은 원금과 "
        f"퇴직금(이연퇴직소득) 재원은 제외합니다. "
        f"세액공제 받은 납입금·운용수익 재원의 연금소득세율은 만 55세 이상 70세 미만 5.5%, "
        f"70세 이상 80세 미만 4.4%, 80세 이상 3.3%입니다. "
        f"종신연금은 연령과 무관하게 3.3%입니다. "
        f"퇴직금을 연금으로 받을 때 이연퇴직소득세는 연금실제수령연차 1~10년차 30%, "
        f"11~20년차 40%, 21년차 이상 50% 감면됩니다."
    )
    # 사용자가 나이를 밝혔으면 그 값으로 적용 구간을 확정해 준다 — 알고 있는 것을
    # 되묻지 않는다. 단 **세액공제 재원에 한정**해서만 확정한다: 같은 나이라도 퇴직금
    # 재원은 이연퇴직소득세 체계라 이 세율표가 적용되지 않으므로, 나이만으로 전체
    # 절세액을 단정하면 재원이 뒤섞인 오답이 된다(_pension_income_tax_rate_response의
    # docstring이 경고하는 실측 사고와 같은 구조).
    age = _extract_age(question)
    if age is not None and age >= 55:
        age_line = (
            f"- 말씀하신 만 {age}세는 '{_age_bracket_label(age)}' 구간이라, 이 재원에 대해서는 "
            f"{_pct(get_pension_income_tax_rate(age))}가 적용됩니다.\n"
        )
    else:
        age_line = ""

    draft = (
        "연금계좌의 세금혜택은 크게 네 가지로 볼 수 있습니다.\n\n"
        "1. 납입할 때 세액공제\n"
        "- 연금저축 단독 세액공제 대상 한도는 연 600만원입니다.\n"
        "- 연금저축과 IRP를 합치면 세액공제 대상 한도는 연 900만원입니다.\n"
        "- 세액공제율은 총급여 5,500만원 이하 또는 종합소득금액 4,500만원 이하이면 16.5%, "
        "그 초과 구간이면 13.2%입니다.\n\n"
        "2. 운용 중 과세이연\n"
        "- 계좌 안에서 발생한 운용수익에 대해 매년 바로 과세하지 않고, 나중에 인출할 때 과세하는 구조입니다.\n\n"
        "3. 연금으로 받을 때 낮은 세율 적용\n"
        "- 세액공제 받은 납입금과 운용수익을 연금으로 받으면 연령에 따라 5.5%, 4.4%, 3.3% 세율이 적용될 수 있습니다.\n"
        f"{age_line}"
        "- 종신연금은 연령과 무관하게 3.3%입니다.\n"
        "- 과세대상 사적연금소득이 연 1,500만원을 초과하면 종합과세 또는 16.5% 분리과세 선택 문제가 생길 수 있습니다. "
        "이 1,500만원 판정에는 세액공제 받은 납입금과 운용수익만 포함하고, 세액공제 받지 않은 원금과 "
        "퇴직금 재원은 제외합니다.\n\n"
        "4. 퇴직금을 연금으로 받을 때 퇴직소득세 감면\n"
        "- 퇴직금을 연금으로 받으면 이연퇴직소득세가 연금실제수령연차에 따라 30%, 40%, 50% 감면될 수 있습니다.\n"
        "- 위 3번의 연령별 세율(5.5%/4.4%/3.3%)은 이 퇴직금 재원에는 적용되지 않습니다 — 재원별로 과세 체계가 다릅니다.\n\n"
        "정리하면, 연금계좌는 납입 시점에는 세액공제, 운용 중에는 과세이연, 수령 시점에는 저율 과세 또는 "
        "퇴직소득세 감면을 기대할 수 있는 구조입니다.\n\n"
        "본인에게 적용될 세액을 재원별로 나눠 보려면 퇴직금 규모, 기존 연금저축·IRP 보유 여부, "
        "연간 예상 인출액을 알려주세요."
    )
    return draft, _context(source, content)


def _tax_credit_rate_for_income(values: dict[str, int | None]) -> str | None:
    """납입액 없이 소득만 주어졌을 때 적용 세액공제율을 확정해 답한다. 소득이 없으면 None.

    총급여와 종합소득금액이 함께 주어지면 총급여를 우선한다 — 근로소득자 기준이 우선
    적용되고, 실무 질문에서도 총급여를 먼저 말한다.
    """
    salary = values["total_salary"]
    comprehensive = values["comprehensive_income"]
    if salary is not None:
        threshold, label = INCOME_THRESHOLD_SALARY, "총급여"
        income = salary
    elif comprehensive is not None:
        threshold, label = INCOME_THRESHOLD_COMPREHENSIVE, "종합소득금액"
        income = comprehensive
    else:
        return None

    # 경계값은 "이하"가 낮은 소득 구간(16.5%)에 포함된다 — no.309(5499만원)/no.310(5501만원)
    # 처럼 경계 근처를 묻는 질문이 실제로 들어온다.
    within = income <= threshold
    rate = CREDIT_RATE_LOW if within else CREDIT_RATE_HIGH
    comparison = "이하" if within else "초과"
    return (
        f"{label} {_won(income)}이면 세액공제율은 **{_pct(rate)}**입니다.\n\n"
        f"- 적용 기준: {label} {_won(threshold)} {comparison} 구간\n"
        f"- 세액공제율: {label} {_won(threshold)} 이하는 {_pct(CREDIT_RATE_LOW)}, "
        f"초과는 {_pct(CREDIT_RATE_HIGH)}\n\n"
        f"세액공제 대상 납입한도는 연금저축 단독 연 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 연 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        f"합산 한도를 모두 채웠다면 세액공제액은 "
        f"{_won(COMBINED_CREDIT_LIMIT)} x {_pct(rate)} = "
        f"{_won_readable(round(COMBINED_CREDIT_LIMIT * rate))}입니다.\n\n"
        "실제 납입액을 알려주시면 정확한 세액공제액을 계산해 드릴 수 있습니다."
    )


def _tax_credit_limit_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc41 세액공제 규칙"
    content = (
        f"연금저축+IRP 합산 납입한도는 연 {_won(TOTAL_CONTRIBUTION_LIMIT)}입니다. "
        f"세액공제 대상 납입한도는 연금저축 단독 {_won(PENSION_SAVINGS_ONLY_LIMIT)}, "
        f"연금저축+IRP 합산 {_won(COMBINED_CREDIT_LIMIT)}입니다. "
        "연금저축 600만원과 IRP 900만원을 따로 더한 1,500만원이 세액공제 대상 한도가 아니라, "
        "연금저축 납입액을 포함한 IRP 합산 900만원이 세액공제 대상 한도입니다. "
        f"세액공제율은 총급여 {_won(INCOME_THRESHOLD_SALARY)} 이하 또는 종합소득금액 "
        f"{_won(INCOME_THRESHOLD_COMPREHENSIVE)} 이하이면 {_pct(CREDIT_RATE_LOW)}, "
        f"초과이면 {_pct(CREDIT_RATE_HIGH)}입니다. "
        f"최대 세액공제액은 {_won(COMBINED_CREDIT_LIMIT)} x {_pct(CREDIT_RATE_LOW)} = 148만 5천원, "
        f"{_won(COMBINED_CREDIT_LIMIT)} x {_pct(CREDIT_RATE_HIGH)} = 118만 8천원입니다."
    )
    values = extract_tax_credit_inputs(question)

    # 계산에 필요한 입력(납입액+소득)이 이미 질문에 있으면 일반론이 아니라 계산으로 답한다.
    # 왜 여기서 다시 판단하나: candidate_categories는 "세액공제"라는 단어만 보고
    # 세액공제_계산_입력부족과 세액공제_한도를 **항상 같이** 후보로 낸다. 둘 중 무엇이
    # 확정되는지는 라우터 LLM이 정하는데, 실측에서 같은 형태의 질문이 갈렸다 —
    # no.74/322(납입액+소득)는 계산 카테고리로 가서 정확히 계산됐지만,
    # no.323("종합소득금액 6천만원, 연금저축 600만원")은 한도 카테고리로 가서
    # 79.2만원 대신 일반론이 나갔다. 어느 쪽으로 분류되든 답이 같아야 하므로,
    # 분류에 답을 맡기지 않고 입력값을 기준으로 코드가 판단한다.
    if _has_sufficient_tax_credit_inputs(values):
        return _tax_credit_calculation_missing_response(question)

    # 소득만 주어진 경우("총급여 5499만원이면 세액공제율이 몇 %인가요?")는 납입액이 없어도
    # 세율을 확정할 수 있다. 실측 no.309/310은 5499/5501만원으로 경계를 물었는데도
    # 두 답변이 글자까지 동일한 일반론이었다 — 질문한 값이 답에 반영되지 않았다.
    income_rate_draft = _tax_credit_rate_for_income(values)
    if income_rate_draft is not None:
        return income_rate_draft, _context(source, content)

    pension_savings_paid = values["pension_savings_paid"]
    asks_all_credited = any(word in _compact(question) for word in ("전부", "모두", "다세액공제", "전체"))
    if pension_savings_paid and values["irp_paid"] is None and asks_all_credited:
        credited = min(pension_savings_paid, PENSION_SAVINGS_ONLY_LIMIT)
        excess = max(0, pension_savings_paid - PENSION_SAVINGS_ONLY_LIMIT)
        if excess:
            draft = (
                f"아니요. 연금저축에 {_won(pension_savings_paid)}을 납입했더라도, **연금저축만으로는 "
                f"{_won(PENSION_SAVINGS_ONLY_LIMIT)}까지만 세액공제 대상**입니다.\n\n"
                f"- 세액공제 대상 연금저축 납입액: {_won(credited)}\n"
                f"- 연금저축 단독 한도를 넘는 금액: {_won(excess)}\n\n"
                f"연금저축과 IRP를 함께 활용하면 두 계좌 합산으로 {_won(COMBINED_CREDIT_LIMIT)}까지 "
                "세액공제 대상이 될 수 있지만, 연금저축 단독 한도 자체가 900만원으로 늘어나는 구조는 아닙니다.\n\n"
                "세액공제율은 소득 기준에 따라 16.5% 또는 13.2%가 적용됩니다."
            )
            return draft, _context(source, content)

    draft = (
        "연금저축과 IRP를 합쳐서 볼 때 핵심은 **세액공제 대상 한도는 합산 900만원**이라는 점입니다.\n\n"
        "- 연금저축만 납입하는 경우: 세액공제 대상은 연 600만원까지\n"
        "- 연금저축 + IRP를 함께 납입하는 경우: 두 계좌 합산 세액공제 대상은 연 900만원까지\n"
        "- 단순 납입 가능 한도: 연금저축과 IRP 합산 연 1,800만원까지\n\n"
        "따라서 연금저축 600만원과 IRP 900만원을 각각 따로 더해 1,500만원까지 세액공제되는 구조가 "
        "아니라, **IRP 한도 900만원은 연금저축 납입액을 포함한 합산 한도**입니다.\n\n"
        "세액공제율은 소득 기준에 따라 달라집니다.\n"
        "- 총급여 5,500만원 이하 또는 종합소득금액 4,500만원 이하: 16.5%\n"
        "- 총급여 5,500만원 초과 또는 종합소득금액 4,500만원 초과: 13.2%\n\n"
        "합산 한도 900만원을 모두 채웠다면 최대 세액공제액은 16.5% 적용 시 148만 5천원, "
        "13.2% 적용 시 118만 8천원입니다."
    )
    return draft, _context(source, content)

def _is_early_withdrawal_general_question(text: str) -> bool:
    return "중도인출" in text and any(word in text for word in ("가능", "경우", "사유", "요건", "언제"))


def _parse_korean_dates(text: str) -> list[date]:
    dates = []
    current_year = None
    pattern = re.compile(r"(?:(\d{4})년)?(\d{1,2})월(\d{1,2})일")
    for year, month, day in pattern.findall(text):
        if year:
            current_year = int(year)
        if current_year is None:
            continue
        try:
            dates.append(date(current_year, int(month), int(day)))
        except ValueError:
            continue
    return dates


def _parse_first_korean_date_mention(text: str) -> tuple[int | None, int, int] | None:
    match = re.search(r"(?:(\d{4})년)?(\d{1,2})월(\d{1,2})일", text)
    if not match:
        return None
    year, month, day = match.groups()
    parsed_year = int(year) if year else None
    parsed_month = int(month)
    parsed_day = int(day)
    try:
        date(parsed_year or 2001, parsed_month, parsed_day)
    except ValueError:
        return None
    return parsed_year, parsed_month, parsed_day


def _format_date(value: date) -> str:
    return f"{value.year}년 {value.month}월 {value.day}일"


def _format_month_day(value: date) -> str:
    return f"{value.month}월 {value.day}일"


# 사용자 표현 -> 중도인출 사유. WITHDRAWAL_DEADLINE_RULES의 키와 일치시킨다.
_WITHDRAWAL_REASON_ALIASES: dict[str, tuple[str, ...]] = {
    "요양": ("요양", "치료", "병원", "의료비"),
    "개인회생파산": ("개인회생", "파산", "회생절차"),
    "무주택전월세": ("전월세", "전세", "월세", "임차보증금", "임대차"),
    "무주택주택구입": ("주택구입", "주택매입", "집구입", "집을사", "집사", "주택을사", "소유권이전", "등기"),
    "재난피해": ("재난", "피해", "수해", "화재"),
}
_WITHDRAWAL_REASON_LABELS = {
    "요양": "요양",
    "개인회생파산": "개인회생·파산",
    "무주택전월세": "무주택 전월세보증금",
    "무주택주택구입": "무주택 주택구입",
    "재난피해": "재난피해",
}

_CONTEXT_REASON_TO_RULE_REASON = {
    "MEDICAL": "요양",
    "PERSONAL_REHABILITATION": "개인회생파산",
    "BANKRUPTCY": "개인회생파산",
    "HOUSING_DEPOSIT": "무주택전월세",
    "HOME_PURCHASE": "무주택주택구입",
    "DISASTER": "재난피해",
}
_CONTEXT_REASON_TO_DOCUMENT_REASON = {
    **_CONTEXT_REASON_TO_RULE_REASON,
    "PERSONAL_REHABILITATION": "개인회생",
    "BANKRUPTCY": "파산",
}


def _has_final_consonant(word: str) -> bool:
    """마지막 글자에 받침이 있는지 — 조사(이/가, 이라는/라는) 선택용."""
    if not word:
        return False
    ch = word[-1]
    if not ("가" <= ch <= "힣"):
        return False
    return (ord(ch) - 0xAC00) % 28 != 0


def _with_particle(word: str, with_final: str, without_final: str) -> str:
    """받침 여부에 맞는 조사를 붙인다. 예: _with_particle("잔금지급일", "이라는", "라는")"""
    return word + (with_final if _has_final_consonant(word) else without_final)


def _detect_withdrawal_reason(text: str) -> str | None:
    """질문에서 중도인출 사유를 인식한다 (사유별 핸들러 대신 공통 경로가 쓴다)."""
    for reason, aliases in _WITHDRAWAL_REASON_ALIASES.items():
        if any(alias in text for alias in aliases):
            return reason
    return None


_COMPOSITE_TAX_WORDS = ("세금", "세율", "과세", "얼마", "떼")
_COMPOSITE_DEADLINE_WORDS = ("언제까지", "기한", "신청기한", "시기")
_COMPOSITE_DOCUMENT_WORDS = (
    "서류",
    "필요서류",
    "구비서류",
    "징구서류",
    "제출서류",
    "준비서류",
    "준비할",
    "챙겨",
)
_COMPOSITE_ELIGIBILITY_WORDS = (
    "가능",
    "가능한지",
    "되나요",
    "되는지",
    "할수",
    "하려",
    "할래",
    "하고싶",
    "DB형",
    "DC형",
    "IRP",
)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _build_composite_info_tasks(question: str) -> list[str]:
    """한 질문에 여러 정보 작업이 섞였는지 판정한다.

    "기한+서류+세금", "중도인출분+연금수령분 각각 세금"처럼 사용자 요구가 여러 개인데
    단일 정형 카테고리가 전체 질문을 먹으면 나머지 요구가 누락된다. 여기서는 새 지식을
    만들지 않고, 이미 결정론 핸들러/문서로 답할 수 있는 하위 작업만 감지한다.
    """
    text = _compact(question)
    withdrawal_context = extract_withdrawal_context(question)
    tasks: list[str] = []

    if "중도인출" in text:
        if any(word in text for word in _COMPOSITE_ELIGIBILITY_WORDS):
            tasks.append("early_withdrawal_eligibility")
        if any(word in text for word in _COMPOSITE_DEADLINE_WORDS):
            tasks.append("early_withdrawal_deadline")
        if any(word in text for word in _COMPOSITE_DOCUMENT_WORDS):
            tasks.append("early_withdrawal_documents")
        if any(word in text for word in _COMPOSITE_TAX_WORDS):
            tasks.append("early_withdrawal_tax")

    if withdrawal_context and withdrawal_context.receipt_mode == "SPLIT" and "TAX" in withdrawal_context.explicit_topics:
        if "중도인출" in text or "일부" in text or "연금외" in text:
            tasks.append("retirement_benefit_non_pension_tax")
        if "나머지" in text or "연금으로" in text or "연금수령" in text:
            tasks.append("retirement_benefit_pension_tax")

    tasks = _dedupe_keep_order(tasks)
    if any(task.startswith("retirement_benefit_") for task in tasks):
        tasks = [
            task
            for task in tasks
            if task not in {"early_withdrawal_eligibility", "early_withdrawal_tax"}
        ]
    if tasks == ["early_withdrawal_documents"] and withdrawal_context and withdrawal_context.reason:
        # 사유가 명시된 서류 질문은 해당 사유 근거만 사용하는 정형 경로로 처리한다.
        # 사유별 근거가 없다고 해서 주택·요양 서류를 fallback으로 섞지 않는다.
        return tasks
    return tasks if len(tasks) >= 2 else []


_WITHDRAWAL_DOCUMENT_RULES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "무주택전월세": (
        "doc48 중도인출 무주택 전월세보증금 필요서류",
        "무주택 전월세보증금 중도인출 필요서류는 공통서류, 전월세 계약서류, 상황에 따른 추가서류입니다. "
        "공통서류에는 중도인출신청서, 현거주지 주민등록등본, 주민등록등본주소지의 건물등기사항증명서 "
        "또는 건축물관리대장, 지방세 세목별 과세증명서가 포함됩니다. 전월세 계약서류는 주택 "
        "임대차계약서 사본 또는 전월세계약서 사본입니다. 잔금일 이후 1개월 이내 신청하는 경우 "
        "계약금 입금확인증 또는 영수증이 필요할 수 있습니다.",
        (
            "공통: 중도인출신청서",
            "무주택 확인: 현거주지 주민등록등본, 등본주소지 건물등기사항증명서 또는 건축물관리대장, 지방세 세목별 과세증명서",
            "전월세 계약 확인: 주택 임대차계약서 사본 또는 전월세계약서 사본",
            "잔금일 이후 신청 시: 계약금 입금확인증 또는 영수증 등",
        ),
    ),
    "무주택주택구입": (
        "doc49 중도인출 무주택 주택구입 필요서류",
        "무주택 주택구입 중도인출 구비서류는 무주택 확인서류와 주택구입 유형별 계약서류입니다. "
        "무주택 확인서류에는 현거주지 주민등록등본, 주민등록등본주소지의 건물등기사항증명서, "
        "지방세 세목별 과세증명서가 포함됩니다. 구입은 매매계약서 사본, 분양은 분양계약서 "
        "또는 공급계약서 또는 분양권매매계약서 사본, 신축은 공사계약서와 건축허가서 또는 "
        "착공신고필증, 경매·공매는 입찰보증금 입금영수증과 사건검색 발급본 등이 필요합니다.",
        (
            "무주택 확인: 현거주지 주민등록등본, 등본주소지 건물등기사항증명서, 지방세 세목별 과세증명서",
            "구입: 매매계약서 사본",
            "분양: 분양계약서, 공급계약서 또는 분양권매매계약서 사본",
            "신축·경매·공매: 공사계약서, 건축허가서, 입찰보증금 입금영수증 등 유형별 서류",
        ),
    ),
    "개인회생": (
        "doc47 중도인출 개인회생 필요서류",
        "개인회생 중도인출에는 최근 5년 이내 개인회생절차 개시결정문 사본 또는 "
        "개인회생절차변제인가 확정증명원 등 개시결정을 확인할 수 있는 서류와 대법원 "
        "나의 사건검색 진행경과 전체 출력물이 필요합니다. 폐지결정 또는 면책결정으로 "
        "개인회생 효력이 종료된 경우에는 신청할 수 없습니다.",
        (
            "개인회생: 최근 5년 이내 개인회생절차 개시결정문 사본 또는 개인회생절차변제인가 확정증명원 등",
            "개인회생: 대법원 나의 사건검색 진행경과 전체 출력물",
        ),
    ),
    "파산": (
        "doc47 중도인출 파산선고 필요서류",
        "파산선고 중도인출에는 최근 5년 이내 파산선고문 사본이 필요합니다. 면책 또는 "
        "복권 결정 여부와 무관하게 신청할 수 있습니다.",
        ("파산선고: 최근 5년 이내 파산선고문 사본",),
    ),
    "재난피해": (
        "doc50 중도인출 재난피해 필요서류",
        "재난피해 중도인출 서류는 피해 유형별로 다릅니다. 물적피해는 건축물대장등본과 "
        "피해상황확인서 등이 필요하며, 임차 주거시설이면 임대차계약서를 함께 제출합니다. "
        "인적피해는 피해상황확인서 또는 특별재난지역 선포 확인자료 등이 필요합니다. "
        "가입자가 15일 이상 입원치료한 경우에는 진단서·소견서 등 입원기간을 확인할 수 "
        "있는 서류가 필요하며, 가족 피해는 가족관계증명서 또는 주민등록등본으로 관계를 확인합니다.",
        (
            "물적피해: 건축물대장등본, 피해상황확인서 또는 특별재난지역 선포 확인자료 등",
            "임차 주거시설 피해: 임대차계약서 추가",
            "인적피해: 피해상황확인서, 실종신고접수증 또는 사건사고사실확인원 등 해당 증빙",
            "15일 이상 입원치료: 진단서·소견서 등 입원기간 확인서류",
            "가족 피해: 가족관계증명서 또는 주민등록등본",
        ),
    ),
}


def _withdrawal_deadline_section(reason: str) -> tuple[str, list[RetrievedItem]]:
    rule = WITHDRAWAL_DEADLINE_RULES[reason]
    reason_label = _WITHDRAWAL_REASON_LABELS.get(reason, reason)
    period = f"{rule.years}년" if rule.years else f"{rule.months}개월"
    deadline_phrase = _withdrawal_deadline_phrase(reason, rule.basis_event, period)
    source = f"{rule.source_doc} 중도인출 {reason_label} 신청기한 규칙"
    content = (
        f"{reason_label} 사유의 중도인출 신청기한은 {deadline_phrase}입니다. "
        "제공 DB에는 해당 기간을 정확한 날짜로 환산하는 방식이 명시되어 있지 않습니다. "
        f"{rule.note}"
    )
    section = (
        f"**신청기한**\n"
        f"- {reason_label} 사유의 신청기한은 **{deadline_phrase}**입니다.\n"
        "- 제공 자료에는 정확한 날짜 환산 방식, 초일산입 여부, 말일·휴일 처리, 신청서 작성일/접수일 기준이 명확히 적혀 있지 않아 특정 마감일은 단정하지 않겠습니다."
    )
    return section, _context(source, content)


def _withdrawal_deadline_phrase(reason: str, basis_event: str, period: str) -> str:
    """원문 표현에 가까운 신청기한 문구를 만든다."""
    if reason == "요양":
        return f"요양종료일 이후 {period} 이내"
    if reason == "무주택전월세":
        return f"잔금지급일 이후 {period} 이내"
    if reason == "무주택주택구입":
        return f"소유권 이전 등기접수일 기준 {period} 이내"
    return f"{basis_event}로부터 {period} 이내"


def _withdrawal_documents_section(reason: str | None) -> tuple[str, list[RetrievedItem]] | None:
    if reason not in _WITHDRAWAL_DOCUMENT_RULES:
        return None
    source, content, bullets = _WITHDRAWAL_DOCUMENT_RULES[reason]
    lines = "\n".join(f"- {bullet}" for bullet in bullets)
    section = f"**필요서류**\n{lines}"
    return section, _context(source, content)


def _personal_workout_eligibility_result() -> tuple[str, list[RetrievedItem]]:
    source = "doc47 중도인출 개인워크아웃·신용회복 제외 규칙"
    content = (
        "개인워크아웃·신용회복 자체는 법정 중도인출 사유에 해당하지 않습니다. "
        "개인회생절차개시 결정 또는 파산선고는 별도의 법정 사유입니다."
    )
    draft = (
        "아니요. 개인워크아웃·신용회복 자체는 제공 자료상 법정 중도인출 사유에 해당하지 않습니다.\n\n"
        "개인회생절차개시 결정 또는 파산선고는 별도의 법정 사유이며, 개인워크아웃·신용회복과 구분됩니다."
    )
    return draft, _context(source, content)


def _withdrawal_eligibility_section(question: str, reason: str | None) -> tuple[str, list[RetrievedItem]]:
    reason_label = _WITHDRAWAL_REASON_LABELS.get(reason or "", "해당 사유")
    source = "doc46~doc50 중도인출 요건판정 규칙"
    content = (
        "DB형은 중도인출이 허용되지 않습니다. 중도인출 가능한 제도는 DC와 IRP이며, "
        "DC와 IRP는 법정 사유가 있으면 중도인출 대상 제도입니다. "
        "가능 사유는 6개월 이상 요양, 개인회생 또는 파산선고, 무주택자 전월세보증금, "
        "무주택자 주택구입, 재난피해입니다."
    )
    if "DB형" in question or re.search(r"\bDB\b", question, flags=re.IGNORECASE):
        reason_sentence = (
            f"{reason_label} 같은 법정 사유가 있더라도"
            if reason
            else "법정 사유가 있더라도"
        )
        section = (
            "**가능 여부**\n"
            "- DB형 퇴직연금은 중도인출이 허용되지 않습니다.\n"
            f"- {reason_sentence} 중도인출 가능한 제도는 DC와 IRP입니다.\n"
            "- 아래 신청기한과 필요서류는 DC 또는 IRP에서 해당 사유로 중도인출하는 경우의 기준입니다."
        )
    elif reason:
        section = (
            "**가능 여부**\n"
            f"- 중도인출 가능한 제도는 IRP와 DC이며, 법정 사유를 충족해야 합니다.\n"
            f"- 질문의 사유는 제공 DB상 중도인출 사유 중 **{reason_label}**에 해당할 수 있습니다. 실제 가능 여부는 무주택 여부, 계약 명의 등 세부 요건과 증빙으로 확인됩니다."
        )
    else:
        section = (
            "**가능 여부**\n"
            "- DB형은 중도인출이 허용되지 않고, DC와 IRP는 법정 사유가 있을 때 중도인출 대상 제도입니다."
        )
    return section, _context(source, content)


def _early_withdrawal_tax_section(reason: str | None) -> tuple[str, list[RetrievedItem]]:
    reason_label = _WITHDRAWAL_REASON_LABELS.get(reason or "", "중도인출")
    source = "doc38~doc40 중도인출 재원별 과세 규칙"
    content = (
        "무주택자인 가입자의 본인 명의 주택 구입과 주거 목적 전세보증금 부담은 근퇴법상 "
        "중도인출 사유이나 세법상 부득이한 사유는 아니며, 표에는 16.5% 기타소득세로 표시되어 "
        "있습니다. 주택구입·전월세보증금 중도인출 사유에 해당되면 퇴직금에 대해서는 "
        "퇴직소득세를 차감 후 지급합니다. 세액공제 받지 않은 원금과 퇴직금은 사적연금소득 "
        "1,500만원 초과 여부 판단에서 제외합니다."
    )
    if reason in {"무주택전월세", "무주택주택구입"}:
        section = (
            "**세금**\n"
            f"- {reason_label}은 근퇴법상 중도인출 사유이지만, 제공 자료의 세법상 부득이한 사유 표에서는 부득이한 사유가 아닌 것으로 표시되어 있습니다.\n"
            "- 세액공제 받은 납입금·운용수익 재원은 16.5% 기타소득세로 안내되어 있습니다.\n"
            "- 퇴직금 재원은 퇴직소득세를 차감 후 지급하는 것으로 안내되어 있습니다.\n"
            "- 세액공제 받지 않은 납입원금은 위 두 재원과 구분해야 하며, 제공 자료상 사적연금소득 1,500만원 초과 여부 판단에서는 제외됩니다.\n"
            "- 정확한 세액을 계산하려면 인출 재원의 종류와 재원별 금액을 확인해야 합니다."
        )
    else:
        section = (
            "**세금**\n"
            "- 중도인출 세금은 사유가 세법상 부득이한 사유에 해당하는지와 인출 재원에 따라 달라집니다.\n"
            "- 재원은 적어도 세액공제 받은 납입금·운용수익, 퇴직금, 세액공제 받지 않은 납입원금을 구분해야 합니다.\n"
            "- 정확한 세액을 계산하려면 중도인출 사유, 인출 재원, 재원별 금액이 필요합니다."
        )
    return section, _context(source, content)


def _retirement_benefit_split_tax_sections(question: str) -> tuple[list[str], list[RetrievedItem]]:
    non_pension = get_deferred_retirement_tax_rate(1, is_pension_receipt=False)
    pension_tiers = [
        get_deferred_retirement_tax_rate(year, is_pension_receipt=True)
        for year in (1, 11, 21)
    ]
    payment_percentages = [round(result.payment_ratio * 100) for result in pension_tiers]
    non_pension_percentage = round(non_pension.payment_ratio * 100)
    actual_receipt_year = extract_tax_context(question).actual_pension_year
    source = "doc39~doc40 이연퇴직소득세 감면 규칙"
    content = (
        "IRP 중도인출은 법정 사유를 충족하는 경우에만 가능합니다. "
        f"퇴직금을 연금으로 수령하면 연금실제수령연차 1~10년차는 이연퇴직소득세의 {payment_percentages[0]}%를 납부하고, "
        f"11~20년차는 {payment_percentages[1]}%를 납부하며, 21년차 이상은 {payment_percentages[2]}%를 납부합니다. "
        f"연금외수령은 감면 없이 이연퇴직소득세의 {non_pension_percentage}%를 납부합니다."
    )
    pension_lines = [
        "- 퇴직금 재원을 연금으로 수령하면 연금실제수령연차에 따라 이연퇴직소득세 납부 비율이 달라집니다.",
        f"- 1~10년차는 {payment_percentages[0]}%, 11~20년차는 {payment_percentages[1]}%, 21년차 이상은 {payment_percentages[2]}%를 납부합니다.",
    ]
    if actual_receipt_year:
        actual_result = get_deferred_retirement_tax_rate(actual_receipt_year, is_pension_receipt=True)
        pension_lines.append(
            f"- 입력한 연금실제수령연차 {actual_receipt_year}년차에는 이연퇴직소득세의 "
            f"{round(actual_result.payment_ratio * 100)}%를 납부합니다."
        )
        pension_lines.append("- 정확한 세액 계산에는 원래 납부해야 할 이연퇴직소득세 금액이 필요합니다.")
    else:
        pension_lines.append(
            "- 정확한 비율을 적용하려면 연금실제수령연차가 필요하고, 정확한 세액 계산에는 원래 납부해야 할 이연퇴직소득세 금액도 필요합니다."
        )

    sections = [
        (
            "**전제 확인**\n"
            "- IRP 중도인출은 법정 사유를 충족하는 경우에만 가능합니다. 해당 요건을 충족한다는 전제에서 세금은 다음과 같습니다."
        ),
        (
            "**중도인출하는 퇴직금 부분**\n"
            f"- 중도인출로 퇴직금 재원을 연금외수령하는 부분은 감면 없이 이연퇴직소득세를 전액 납부({non_pension_percentage}%)하는 구조입니다."
        ),
        (
            "**나머지를 연금으로 받는 부분**\n"
            + "\n".join(pension_lines)
        ),
    ]
    return sections, _context(source, content)


def _composite_info_task_plan_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    tasks = _build_composite_info_tasks(question)
    if not tasks:
        return None

    text = _compact(question)
    withdrawal_context = extract_withdrawal_context(question)
    if withdrawal_context and withdrawal_context.reason == "PERSONAL_WORKOUT":
        draft, context = _personal_workout_eligibility_result()
        if "DOCUMENTS" in withdrawal_context.explicit_topics:
            draft += "\n\n따라서 개인워크아웃·신용회복 자체를 사유로 한 중도인출 서류 안내 대상도 아닙니다."
        return draft, context
    reason = (
        _CONTEXT_REASON_TO_RULE_REASON.get(withdrawal_context.reason)
        if withdrawal_context and withdrawal_context.reason
        else _detect_withdrawal_reason(text)
    )
    document_reason = (
        _CONTEXT_REASON_TO_DOCUMENT_REASON.get(withdrawal_context.reason)
        if withdrawal_context and withdrawal_context.reason
        else reason
    )
    sections: list[str] = []
    context: list[RetrievedItem] = []

    if "early_withdrawal_eligibility" in tasks:
        section, ctx = _withdrawal_eligibility_section(question, reason)
        sections.append(section)
        context.extend(ctx)

    if "early_withdrawal_deadline" in tasks and reason in WITHDRAWAL_DEADLINE_RULES:
        section, ctx = _withdrawal_deadline_section(reason)
        sections.append(section)
        context.extend(ctx)

    if "early_withdrawal_documents" in tasks:
        result = _withdrawal_documents_section(document_reason)
        if result is not None:
            section, ctx = result
            sections.append(section)
            context.extend(ctx)
        else:
            sections.append("**필요서류**\n- 중도인출 필요서류는 사유별로 달라서, 먼저 중도인출 사유를 확정해야 합니다.")

    if "early_withdrawal_tax" in tasks:
        section, ctx = _early_withdrawal_tax_section(reason)
        sections.append(section)
        context.extend(ctx)

    if (
        "retirement_benefit_non_pension_tax" in tasks
        or "retirement_benefit_pension_tax" in tasks
    ):
        split_sections, ctx = _retirement_benefit_split_tax_sections(question)
        sections.append(split_sections[0])
        if "retirement_benefit_non_pension_tax" in tasks:
            sections.append(split_sections[1])
        if "retirement_benefit_pension_tax" in tasks:
            sections.append(split_sections[2])
        context.extend(ctx)

    if len(sections) < 2 and tasks != ["early_withdrawal_documents"]:
        return None

    draft = "\n\n".join(sections)
    return draft, context


def _early_withdrawal_deadline_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """중도인출 신청기한 규칙을 안내한다. DB-grounded 경로에서는 exact date를 만들지 않는다.

    사유별로 핸들러를 만들면(요양_신청기한판정, 주택구입_신청기한 ...) 사유가 늘 때마다
    같은 코드를 복제하게 되고, 실제로 "요양·주택구입만 있고 전월세·재난·개인회생은 없는"
    비대칭이 생겼다. 사유 인식(_detect_withdrawal_reason)과 기한 규정
    (WITHDRAWAL_DEADLINE_RULES)을 분리해 5개 사유가 같은 경로를 쓰게 한다.

    제공 DB에는 "1개월/3개월/5년 이내"라는 규칙은 있지만, 1개월을 30일로 볼지,
    달력 기준으로 볼지, 초일·말일·휴일을 어떻게 처리할지까지는 명시되어 있지 않다.
    따라서 일반 Agent 답변에서는 마감일이나 신청일 통과 여부를 계산하지 않고, 확인된
    기준일·기간 규칙과 계산 불가 사유만 안내한다.
    """
    text = _compact(question)
    if "중도인출" not in text and not _mentions_withdrawal_basis_event(text):
        return None
    if not any(word in text for word in ("언제까지", "기한", "이내", "언제", "신청")):
        return None

    reason = _detect_withdrawal_reason(text)
    if reason is None:
        return None
    rule = WITHDRAWAL_DEADLINE_RULES[reason]
    reason_label = _WITHDRAWAL_REASON_LABELS.get(reason, reason)
    period = f"{rule.years}년" if rule.years else f"{rule.months}개월"
    deadline_phrase = _withdrawal_deadline_phrase(reason, rule.basis_event, period)

    source = f"{rule.source_doc} 중도인출 {reason_label} 신청기한 규칙"
    content = (
        f"{reason_label} 사유의 중도인출 신청기한은 {deadline_phrase}입니다. "
        "제공 DB에는 해당 기간을 정확한 날짜로 환산하는 방식이 명시되어 있지 않습니다. "
        f"{rule.note} "
        "중도인출 원문에는 30일/90일 환산 여부, 초일산입 여부, 말일·휴일 처리, "
        "영업시간·신청 도달시점 처리 기준이 명시되어 있지 않습니다. "
        "calculation_basis=not_defined_in_source."
    )
    caveat = (
        "제공 자료에는 이 기간을 정확한 날짜로 계산하는 방식, 초일산입 여부, 말일·휴일 "
        "처리, 신청서 작성일/접수일 기준이 명확히 적혀 있지 않습니다. 따라서 DB 근거만으로는 "
        "특정 날짜가 정확한 마감일인지 또는 신청기한 안인지 단정하지 않겠습니다."
    )

    dates = _parse_korean_dates(text)
    if not dates:
        # 날짜가 없으면 규정만 안내한다(기준일을 지어내지 않는다).
        draft = (
            f"{reason_label} 사유로 중도인출을 신청하는 경우, 신청기한은 "
            f"**{deadline_phrase}**입니다.\n\n"
            f"{rule.note}\n\n{caveat}"
        )
        return draft, _context(source, content)

    basis_date = dates[0]
    request_dates = dates[1:]
    # 연도 없이 "3월 1일"로만 말한 경우 연도를 붙여 답하면 지어낸 정보가 된다.
    mention = _parse_first_korean_date_mention(text)
    has_year = bool(mention and mention[0] is not None)
    fmt = _format_date if has_year else _format_month_day

    if not request_dates:
        draft = (
            f"**{fmt(basis_date)}이 {_with_particle(rule.basis_event, '이라는', '라는')} 조건**이라면, "
            f"제공 자료에서 확인되는 신청기한 규칙은 **{deadline_phrase}**입니다.\n\n"
            f"{caveat}"
        )
        return draft, _context(source, content)

    request_date_text = ", ".join(fmt(d) for d in request_dates)
    draft = (
        f"{rule.basis_event}이 {fmt(basis_date)}이면 제공 자료에서 확인되는 신청기한 규칙은 "
        f"**{deadline_phrase}**입니다.\n\n"
        f"{request_date_text} 신청이 각각 기한 안인지 여부는 DB 근거만으로 정확히 판정하지 않겠습니다.\n\n"
        f"{caveat}"
    )
    return draft, _context(source, content)

def _early_withdrawal_general_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """중도인출 사유 목록을 나열한다. 더 구체적인 작업을 묻는 질문이면 양보한다.

    ⚠️ candidate_categories는 "중도인출"이라는 단어만 보고 일반/기한판정/요건판정
    셋을 함께 후보로 내고, 그중 무엇을 확정할지는 라우터 LLM의 몫이다. 그런데 이
    핸들러는 게이트가 없어 어떤 질문에도 일반 목록을 반환했다 — 후보 순서상 항상
    첫 번째라, 라우터가 잘못 고르면 "배우자 의료비도 포함되나요?" 같은 대상자 질문에
    사유 목록만 나가는 동문서답이 된다.

    같은 도메인 안에서는 더 구체적인 작업이 우선이므로, 대상자 범위·의료비 비율을
    묻는 질문이면 None을 내어 중도인출_요건판정으로 넘긴다.
    """
    text = _compact(question)
    if _asks_medical_eligible_persons(text) or _asks_medical_expense_ratio(text):
        return None

    source = "doc46~doc50 중도인출 규칙"
    content = (
        "DB는 중도인출이 허용되지 않습니다. DC와 IRP는 법정 사유가 있으면 중도인출 대상 제도입니다. "
        "가능 사유는 6개월 이상 요양, 개인회생 또는 파산선고, 무주택자 전월세보증금, "
        "무주택자 주택구입, 재난피해입니다. 요양은 DC에서 직전 1년 의료비가 직전년도 연간임금총액의 "
        "12.5%를 초과해야 하며, IRP에는 이 비율 기준이 적용되지 않습니다. 개인회생·파산은 결정일 또는 "
        "선고일로부터 5년 이내 요건이 있습니다. 전월세보증금과 주택구입은 잔금지급일 또는 소유권 이전 "
        "등기접수일로부터 1개월 이내 신청 요건이 있습니다. 재난피해는 피해발생일로부터 "
        "3개월 이내가 원칙입니다. 제공 DB에는 이 기간을 정확한 날짜로 환산하는 방식이 명시되어 있지 않습니다."
    )
    draft = (
        "IRP는 중도인출이 가능한 제도이지만, 아무 때나 인출할 수 있는 것은 아니고 법정 사유가 필요합니다.\n\n"
        "대표적인 중도인출 사유는 다음과 같습니다.\n"
        "- 6개월 이상 요양\n"
        "- 개인회생 또는 파산선고\n"
        "- 무주택자 전월세보증금\n"
        "- 무주택자 주택구입\n"
        "- 재난피해\n\n"
        "참고로 DB형은 중도인출이 허용되지 않고, DC와 IRP가 중도인출 대상 제도입니다. "
        "각 사유별로 신청기한이나 추가 요건이 다릅니다. 예를 들어 개인회생·파산은 5년 이내 요건, "
        "전월세보증금·주택구입은 1개월 이내 신청 요건, 재난피해는 3개월 이내 "
        "신청 요건이 문제될 수 있습니다."
    )
    return draft, _context(source, content)


# 중도인출 사유별 기준일 용어 — WITHDRAWAL_DEADLINE_RULES에서 그대로 끌어온다.
# 손으로 목록을 적으면 사유가 늘 때 여기만 빠져 같은 비대칭이 다시 생긴다.
_WITHDRAWAL_BASIS_EVENT_TERMS = tuple(
    _compact(rule.basis_event) for rule in WITHDRAWAL_DEADLINE_RULES.values()
)


def _mentions_withdrawal_basis_event(compact_text: str) -> bool:
    """"요양종료일", "잔금지급일" 같은 중도인출 기준일 용어를 쓴 질문인지."""
    return any(term in compact_text for term in _WITHDRAWAL_BASIS_EVENT_TERMS)


def _asks_account_level_transfer(compact_text: str) -> bool:
    """상품이 아니라 **계좌 자체**를 다른 금융사로 옮기는 방법을 묻는 질문인지.

    계좌이전 제도는 상품 단위 실물이전과 다른 제도이고, 보유 DB에는 관련 문서가 없다.
    "계좌"를 옮긴다고 명시했고 실물이전이라는 용어를 쓰지 않은 경우만 잡는다 —
    "실물이전으로 계좌를 옮길 수 있나요?"처럼 용어를 쓴 질문은 실물이전이 맞다.
    """
    if "실물이전" in compact_text:
        return False
    if "계좌" not in compact_text:
        return False
    return any(w in compact_text for w in ("금융사", "금융기관", "증권사", "은행", "회사로", "타사"))


_MEDICAL_EXPENSE_RATIO_LABEL = f"{MEDICAL_EXPENSE_RATIO_THRESHOLD * 100:g}%"


# 요양 사유 문맥을 알려주는 어휘. 대상자 범위 질문과 의료비 비율 질문이 같은 도메인이라
# 두 판정이 같은 목록을 봐야 한다 — 예전에는 각자 다른 목록을 갖고 있어 "배우자 의료비도
# 중도인출 되나요?"가 대상자 판정에서만 빠졌다("의료비"가 그쪽 목록에 없었음).
_MEDICAL_CONTEXT_WORDS = ("요양", "질병", "치료", "의료비", "치료비", "부상", "입원", "간병")


def _asks_medical_eligible_persons(compact_text: str) -> bool:
    """요양 사유의 **대상자 범위**(본인 외 가족도 되는지)를 묻는 질문인지."""
    if not any(w in compact_text for w in _MEDICAL_CONTEXT_WORDS):
        return False
    return any(
        w in compact_text
        for w in ("가족", "배우자", "부모", "자녀", "본인아닌", "본인외", "남편", "아내", "부양")
    )


def _asks_medical_expense_ratio(compact_text: str) -> bool:
    """요양 사유의 "의료비가 임금총액의 몇 % 이상이어야 하나"를 묻는 질문인지."""
    if not any(w in compact_text for w in _MEDICAL_CONTEXT_WORDS):
        return False
    return any(
        w in compact_text
        for w in ("몇%", "몇퍼센트", "몇프로", "비율", "12.5", "임금총액", "연봉의", "연봉대비")
    )


def _mentions_irp_not_dc(compact_text: str) -> bool:
    """IRP만 언급하고 DC는 언급하지 않은 질문인지 (제도가 특정되면 그 제도만 답한다)."""
    has_irp = re.search(r"(?<![a-z])irp(?![a-z])", compact_text, flags=re.IGNORECASE) is not None
    has_dc = re.search(r"(?<![a-z])dc(?![a-z])", compact_text, flags=re.IGNORECASE) is not None
    return has_irp and not has_dc


def _early_withdrawal_eligibility_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    text = _compact(question)
    if "중도인출" not in text:
        return None
    # "가능한가"를 묻는 어휘가 없어도, 요건 수치(의료비 비율)를 묻는 질문은 답할 수 있다 —
    # "몇 퍼센트 이상이어야 하나요?"에는 아래 어휘가 하나도 없다.
    if not any(
        word in text for word in ("가능", "되나요", "할수", "되냐", "되는지", "가능한가", "하려", "할래", "하고싶")
    ) and not _asks_medical_expense_ratio(text) and not _asks_medical_eligible_persons(text):
        return None

    source = "doc46~doc50 중도인출 요건판정 규칙"
    content = (
        "DB형은 중도인출이 허용되지 않습니다. DC와 IRP는 법정 사유가 있으면 중도인출 대상 제도입니다. "
        "가능 사유는 6개월 이상 요양, 개인회생 또는 파산선고, 무주택자 전월세보증금, "
        "무주택자 주택구입, 재난피해입니다. 개인워크아웃제도와 신용회복은 중도인출 사유에 해당하지 않습니다."
    )

    if "DB형" in question or re.search(r"\bDB\b", question, flags=re.IGNORECASE):
        if _asks_alternative_withdrawal_plan_types(text):
            draft = (
                "중도인출 가능한 제도는 DC와 IRP입니다.\n\n"
                "다만 DC와 IRP도 언제든 중도인출할 수 있는 것은 아니며, 법정 사유를 충족해야 합니다. "
                "DB형 퇴직연금은 중도인출이 허용되지 않습니다."
            )
            return draft, _context(source, content)
        reason = _detect_withdrawal_reason(text)
        reason_label = _WITHDRAWAL_REASON_LABELS.get(reason or "", "법정 사유")
        reason_sentence = (
            f"{reason_label} 같은 법정 사유가 있더라도"
            if reason
            else "법정 사유가 있더라도"
        )
        draft = (
            "아니요. DB형 퇴직연금은 중도인출이 허용되지 않습니다.\n\n"
            f"{reason_sentence} 중도인출 가능한 제도는 DC와 IRP입니다. "
            "따라서 질문 조건이 DB형이라면 중도인출할 수 없다고 보는 것이 맞습니다."
        )
        return draft, _context(source, content)

    has_workout = any(word in text for word in ("개인워크아웃", "워크아웃", "신용회복"))
    has_rehabilitation = "개인회생" in text or "회생절차" in text
    if has_workout and has_rehabilitation:
        comparison_source = "doc47 중도인출 개인회생·개인워크아웃 구분 규칙"
        comparison_content = (
            "개인회생절차개시 결정은 받은 날부터 5년 이내이고 효력이 진행 중이면 DC와 IRP의 "
            "중도인출 사유가 될 수 있습니다. 개인워크아웃·신용회복 자체는 법정 중도인출 "
            "사유에 해당하지 않습니다."
        )
        draft = (
            "개인회생과 개인워크아웃은 중도인출에서 다르게 처리됩니다.\n\n"
            "- 개인회생: 개인회생절차개시 결정을 받은 날부터 5년 이내이고 효력이 진행 중이면 "
            "DC와 IRP의 법정 중도인출 사유가 될 수 있습니다.\n"
            "- 개인워크아웃·신용회복: 그 자체는 제공 자료상 법정 중도인출 사유에 해당하지 않습니다."
        )
        return draft, _context(comparison_source, comparison_content)

    if has_workout:
        return _personal_workout_eligibility_result()

    if has_rehabilitation:
        rehabilitation_source = "doc47 중도인출 개인회생 요건 규칙"
        rehabilitation_content = (
            "DC와 IRP에서는 개인회생절차개시 결정을 받은 날부터 5년 이내이고 신청 시점에 "
            "개인회생절차의 효력이 진행 중이면 중도인출 사유가 될 수 있습니다. 폐지결정 또는 "
            "면책결정으로 효력이 종료된 경우에는 신청할 수 없습니다. DB형은 중도인출이 허용되지 않습니다."
        )
        draft = (
            "개인회생절차개시 결정은 DC와 IRP의 법정 중도인출 사유가 될 수 있습니다.\n\n"
            "- 신청 시기: 개인회생절차개시 결정을 받은 날부터 **5년 이내**\n"
            "- 추가 조건: 신청 시점에도 개인회생절차의 효력이 진행 중이어야 합니다.\n"
            "- 폐지결정 또는 면책결정으로 효력이 종료된 경우에는 이 사유로 신청할 수 없습니다.\n"
            "- DB형 퇴직연금은 사유와 관계없이 중도인출이 허용되지 않습니다."
        )
        return draft, _context(rehabilitation_source, rehabilitation_content)

    withdrawal_context = extract_withdrawal_context(question)
    if withdrawal_context and withdrawal_context.reason == "BANKRUPTCY":
        bankruptcy_source = "doc47 중도인출 파산선고 요건 규칙"
        bankruptcy_content = (
            "DC와 IRP에서는 파산선고를 받은 날부터 5년 이내이면 중도인출 사유가 될 수 있습니다. "
            "파산선고는 면책 또는 복권 결정 여부와 무관합니다. DB형은 중도인출이 허용되지 않습니다."
        )
        draft = (
            "파산선고는 DC와 IRP의 법정 중도인출 사유가 될 수 있습니다.\n\n"
            "- 신청 시기: 파산선고를 받은 날부터 **5년 이내**\n"
            "- 면책 또는 복권 결정 여부와는 무관합니다.\n"
            "- DB형 퇴직연금은 사유와 관계없이 중도인출이 허용되지 않습니다."
        )
        return draft, _context(bankruptcy_source, bankruptcy_content)

    if withdrawal_context and withdrawal_context.reason == "DISASTER":
        disaster_source = "doc50 중도인출 재난피해 요건 규칙"
        disaster_content = (
            "재난으로 피해를 입은 경우는 DC와 IRP의 법정 중도인출 사유가 될 수 있습니다. "
            "DB형은 중도인출이 허용되지 않으며, 실제 신청은 피해 유형과 증빙 요건을 확인해야 합니다."
        )
        draft = (
            "재난으로 피해를 입은 경우는 DC와 IRP의 법정 중도인출 사유가 될 수 있습니다.\n\n"
            "다만 실제 신청 가능 여부는 피해 유형과 해당 증빙 요건을 충족하는지 확인해야 합니다. "
            "DB형 퇴직연금은 사유와 관계없이 중도인출이 허용되지 않습니다."
        )
        return draft, _context(disaster_source, disaster_content)

    # 요양 사유의 대상자 범위 — 본인만이 아니라 배우자·부양가족의 요양도 사유가 된다.
    # 실측 no.390("본인 아닌 가족의 의료비도 포함되나요?")에서 LLM은 이 내용이 담긴
    # 문서를 검색하지 않고 check_early_withdrawal(기한 정보만 반환)만 호출한 뒤,
    # 근거 없이 "본인 외 가족의 의료비는 포함되지 않습니다"라고 정반대로 지어냈다.
    if _asks_medical_eligible_persons(text):
        persons = " · ".join(MEDICAL_TREATMENT_ELIGIBLE_PERSONS)
        persons_content = (
            f"6개월 이상 요양 사유의 대상자는 {persons}입니다 — 근로자 본인뿐 아니라 "
            "배우자와 부양가족의 요양도 중도인출 사유에 해당합니다. 부양가족의 요양으로 "
            "신청할 때는 가족관계증명서 또는 주민등록등본으로 관계를 증빙해야 합니다."
        )
        draft = (
            f"네, 포함됩니다. 요양 사유의 대상자는 **{persons}**입니다.\n\n"
            f"- 본인뿐 아니라 **배우자·부양가족**이 {MEDICAL_TREATMENT_MIN_MONTHS}개월 이상 "
            "질병 또는 부상으로 요양을 필요로 하는 경우도 중도인출 사유가 됩니다.\n"
            "- 부양가족의 요양으로 신청할 때는 가족관계증명서 또는 주민등록등본으로 "
            "관계를 증빙해야 합니다.\n"
            f"- 요양 사실은 병명과 {MEDICAL_TREATMENT_MIN_MONTHS}개월 이상 치료 필요가 "
            "명시된 진단서(소견서) 또는 장기요양확인서로 증빙합니다."
        )
        return draft, _context(source, persons_content)

    # 요양 사유의 의료비 비율(12.5%) 적용 여부 — 제도별로 답이 갈리는데, 근거 문서에서는
    # 그 구분이 "(개인형IRP는 불필요)"라는 괄호 한 줄에만 묻혀 있다. 실측 no.146에서 LLM은
    # 그 괄호를 놓쳐 "IRP도 12.5%를 초과해야 한다"고 답한 뒤, 바로 다음 문장에서
    # "개인형 IRP는 12.5%와 상관없이 가능하다"고 스스로 뒤집는 자기모순 답변을 냈다.
    # 규칙(check_medical_treatment_eligibility)은 이미 DC에만 비율을 적용하므로 그대로 따른다.
    if _asks_medical_expense_ratio(text):
        r = _MEDICAL_EXPENSE_RATIO_LABEL
        ratio_content = (
            f"6개월 이상 요양 사유의 의료비 비율 기준({r})은 DC형에만 적용됩니다. "
            "IRP는 이 비율 기준이 적용되지 않아 요양 사유와 신청기한 요건만 충족하면 됩니다. "
            "DC형은 직전 1년간 본인 부담 의료비 총액이 직전년도 연간임금총액의 "
            f"{r}를 초과해야 신청할 수 있습니다."
        )
        if _mentions_irp_not_dc(text):
            draft = (
                f"IRP는 의료비 비율 기준({r})이 적용되지 않습니다.\n\n"
                "- IRP: 6개월 이상 요양 사유에 해당하면 의료비가 연간임금총액에서 차지하는 "
                "비율과 무관하게 중도인출을 신청할 수 있습니다. 연간임금총액 확인서류도 "
                "제출 대상이 아닙니다.\n"
                "- DC형: 직전 1년간 본인이 부담한 의료비 총액이 직전년도 연간임금총액의 "
                f"{r}를 **초과**해야 신청할 수 있습니다.\n\n"
                "다만 요양 사유 자체(본인·배우자·부양가족의 6개월 이상 요양)와 "
                "신청기한(요양종료일로부터 1개월 이내)은 IRP에도 그대로 적용됩니다."
            )
        else:
            # 제도를 특정하지 않은 질문에만 되묻는다 — DC를 명시했는데 "어느 제도인지
            # 알려주세요"라고 답하면 이미 준 정보를 다시 묻는 꼴이 된다.
            mentions_plan = re.search(r"(?<![a-z])(?:dc|db)(?![a-z])", text, flags=re.IGNORECASE)
            draft = (
                f"의료비 비율 기준({r})은 제도에 따라 다릅니다.\n\n"
                "- DC형: 직전 1년간 본인이 부담한 의료비 총액이 직전년도 연간임금총액의 "
                f"{r}를 **초과**해야 신청할 수 있습니다.\n"
                "- IRP: 이 비율 기준이 적용되지 않습니다. 요양 사유에 해당하면 비율과 "
                "무관하게 신청할 수 있습니다.\n"
                "- DB형: 중도인출이 허용되지 않습니다."
            )
            if not mentions_plan:
                draft += "\n\n어느 제도인지 알려주시면 더 정확히 안내드릴 수 있습니다."
        return draft, _context(source, ratio_content)

    return None


_EXISTING_MEMBER_WORDS = ("기존가입자", "기존 가입자", "기존가입", "만기가 된", "만기된", "만기 도래")
_NEW_MEMBER_WORDS = ("신규가입자", "신규 가입자", "신규가입", "최초 가입", "처음 가입", "신규 납입")


def _default_option_auto_purchase_response(question: str) -> tuple[str, list[RetrievedItem]]:
    """기존/신규가입자가 질문에 이미 특정돼 있으면 그 케이스만, 아니면 역질문+두 케이스를 함께 낸다.

    질문에 이미 없는 정보(정보 손실)는 아니지만, 특정된 사용자에게 무관한 케이스까지 섞어
    답하면 실제로 필요한 정보를 찾기 어려워진다 — 실물이전 사고와 같은 클래스(개인화 미반영)
    의 더 가벼운 변종.
    """
    source = "doc29 디폴트옵션 자동매수 규칙"
    # 일수는 반드시 규칙엔진 상수를 참조한다 — 문구에 숫자를 손으로 적으면 상수만 고쳤을 때
    # 답변이 옛 값을 계속 노출한다(실제로 28/14 오류가 이 경로로 사용자에게 나갔다).
    notice = f"4주 뒤(만기일 + {NOTICE_DELAY_DAYS_EXISTING}일)"
    wait = f"2주 뒤(통지일 + {WAIT_DAYS_AFTER_NOTICE}일)"
    business_day_note = "통지일 또는 매수예정일이 비영업일이면 익영업일로 적용됩니다."
    continuity_note = (
        "다만 대기 중 전액을 다른 상품으로 이동해 연속성이 끊기면 다음 만기분부터 "
        "다시 통지와 대기 절차를 거칠 수 있습니다."
    )
    content = (
        f"기존가입자는 상품 만기일로부터 {notice}에 통지하고, {wait}에 자동매수합니다. "
        f"신규가입자는 최초 부담금 납입 다음 영업일 통지하고, {wait}에 자동매수합니다. "
        "동일 상품 반복 만기 등 연속성이 유지되는 경우에는 통지·대기 없이 즉시 적용됩니다. "
        f"{business_day_note}"
    )

    is_existing = any(word in question for word in _EXISTING_MEMBER_WORDS)
    is_new = any(word in question for word in _NEW_MEMBER_WORDS)

    if is_existing and not is_new:
        draft = (
            f"기존가입자 기준으로 안내드리면, 디폴트옵션 자동매수는 상품 만기일로부터 {notice}에 "
            f"통지하고, {wait}에 이뤄집니다.\n\n"
            f"{business_day_note} {continuity_note}"
        )
    elif is_new and not is_existing:
        draft = (
            "신규가입자 기준으로 안내드리면, 디폴트옵션 자동매수는 최초 부담금 납입 다음 영업일에 "
            f"통지하고, {wait}에 이뤄집니다.\n\n"
            f"{business_day_note}"
        )
    else:
        draft = (
            "기존가입자인지 신규가입자인지에 따라 자동매수 시점이 달라 정확히 안내드리려면 "
            "어느 쪽에 해당하시는지 알려주시면 좋습니다. 다만 일반적으로는 다음과 같습니다.\n\n"
            f"- 기존가입자: 상품 만기일로부터 {notice} 통지, {wait} 자동매수\n"
            f"- 신규가입자: 최초 부담금 납입 다음 영업일 통지, {wait} 자동매수\n"
            "- 동일 상품 반복 만기처럼 연속성이 유지되는 경우: 통지·대기 없이 즉시 적용\n\n"
            f"{business_day_note} {continuity_note}"
        )
    return draft, _context(source, content)


_ZERO_HOLDING_WORDS = ("보유하지않", "보유중인상품이없", "가입하지않", "없는상태")
_MULTIPLE_HOLDING_WORDS = ("2개이상", "2개 이상", "복수", "여러개", "여러 개", "둘다", "두개다")
_SAME_PRODUCT_WORDS = ("같은상품", "같은 상품", "동일상품", "동일 상품", "동일한상품", "그상품", "그 상품")
_DIFFERENT_PRODUCT_WORDS = (
    "다른상품", "다른 상품", "다른유형", "다른 유형", "새상품", "새 상품", "새로운상품", "새로운 상품",
)


def _extract_optin_holding_count(text: str) -> int | None:
    """질문에서 '현재 실제 보유 중인 디폴트옵션 개수'를 읽는다. 못 읽으면 None."""
    if any(word in text for word in _ZERO_HOLDING_WORDS):
        return 0
    if any(word in text for word in _MULTIPLE_HOLDING_WORDS):
        return 2  # 정확한 개수는 몰라도 check_optin_eligibility는 2 이상을 전부 동일하게 판정한다
    if re.search(r"1개|한개|한 개|하나", text):
        return 1
    return None


def _default_option_optin_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """디폴트옵션 옵트인(가입자 직접매수) 가능 여부 — 보유 개수 + 동일상품 여부로 판정한다.

    자동매수(_default_option_auto_purchase_response)와 다른 축이다: 저건 "언제
    자동으로 사지는가", 이건 "지금 내가 직접 살 수 있는가". 규칙 엔진에는 이미
    check_optin_eligibility로 분리돼 있었는데 카테고리가 없어 트리거되지 못했다.
    """
    text = _compact(question)
    holdings = _extract_optin_holding_count(text)
    if holdings is None:
        return None  # 보유 개수를 읽을 수 없으면 임의로 가정하지 않고 LLM+툴 경로로 넘긴다

    is_same_product = any(word in text for word in _SAME_PRODUCT_WORDS)
    is_different_product = any(word in text for word in _DIFFERENT_PRODUCT_WORDS)

    source = "doc29 디폴트옵션 옵트인 규칙"
    content = (
        "실제 보유 중인 디폴트옵션이 0개면 1개 상품을 직접 매수(옵트인)할 수 있습니다. "
        "1개 보유 중이고 매수하려는 상품이 그 보유 상품과 동일하면 추가 매수가 예외적으로 "
        "허용됩니다. 1개 보유 중인데 다른 유형으로 옵트인하려면 기존 상품을 전량 매도해야 "
        "합니다. 디폴트옵션을 2개 이상(복수) 보유 중이면 먼저 1개 상품만 남도록 전량 정리해야 "
        "추가 옵트인이 가능합니다(일부 매도로는 해소되지 않습니다)."
    )

    result = check_optin_eligibility(
        current_holdings_count=holdings,
        target_is_same_as_only_holding=is_same_product and not is_different_product,
    )

    if holdings == 1 and not is_same_product and not is_different_product:
        # 1개 보유 상태에서 매수하려는 상품이 기존 것과 같은지 다른지가 불명확하면
        # 결과가 정반대로 갈리므로(허용 vs 전량매도 필요) 임의로 단정하지 않고 되묻는다.
        draft = (
            "디폴트옵션을 1개 보유 중이시군요. 다만 매수하려는 상품이 **지금 보유 중인 상품과 "
            "같은 상품인지, 다른 상품인지**에 따라 결과가 달라집니다.\n\n"
            "- 같은 상품 추가 매수: 예외적으로 허용됩니다.\n"
            "- 다른 상품으로 옵트인: 기존 상품을 전량 매도해야 가능합니다.\n\n"
            "어느 쪽에 해당하는지 알려주시면 정확히 안내드리겠습니다."
        )
        return draft, _context(source, content)

    verdict = "네, 가능합니다." if result.eligible else "아니요, 지금은 불가능합니다."
    draft = f"{verdict}\n\n{result.reason}"
    return draft, _context(source, content)


_PLAN_TYPE_WORDS: dict[PlanType, tuple[str, ...]] = {
    PlanType.DB: ("DB형", "DB제도", "확정급여형"),
    PlanType.DC: ("DC형", "DC제도", "확정기여형"),
    PlanType.IRP: ("IRP",),
}


def _extract_plan_type(text: str) -> PlanType | None:
    for plan, words in _PLAN_TYPE_WORDS.items():
        if any(word in text for word in words):
            return plan
    return None


def _pct_int(rate: float) -> str:
    return f"{rate * 100:g}%"


# 사용자가 쓰는 표현 -> PRODUCT_RISK_TIER의 상품유형 키.
# 규칙 테이블의 키는 원문 용어("국내상장주식")라 실제 질문 표현("개별주식", "삼성전자
# 주식", "국내 주식 직접")과 그대로는 매칭되지 않는다. 이 매핑이 그 간극만 메운다 —
# 판정 자체는 전부 investment_limit.py가 한다.
_INVESTMENT_PRODUCT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("비상장주식", ("비상장주식", "비상장주", "장외주식")),
    ("국내상장주식", ("개별주식", "개별종목", "국내주식", "상장주식", "직접투자", "종목투자", "주식직접")),
    ("사모펀드", ("사모펀드", "사모펀")),
    ("증권예탁증권(DR)", ("증권예탁증권", "dr", "예탁증권")),
    ("전환사채·신주인수권부사채·교환사채·후순위채권", ("전환사채", "신주인수권부사채", "교환사채", "후순위채")),
    ("적격 해외상장주식", ("해외주식", "해외상장주식", "미국주식")),
)


def _detect_investment_product_type(compact_text: str) -> str | None:
    """질문에서 투자 가부를 판정할 상품유형을 찾는다. 없으면 None.

    앞에 놓인 항목이 우선한다 — "비상장주식"이 "상장주식"을 부분문자열로 포함하므로
    더 구체적인 쪽을 먼저 본다.
    """
    for product_type, aliases in _INVESTMENT_PRODUCT_ALIASES:
        for alias in aliases:
            if _alias_matches(alias, compact_text):
                return product_type
    return None


def _investment_eligibility_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """특정 상품유형을 그 제도에서 담을 수 있는지 판정한다 (한도가 아니라 가부).

    ⚠️ 판정은 전적으로 investment_limit.PRODUCT_RISK_TIER를 따른다. 여기서 사실을
    새로 쓰지 않는다 — 대표적으로 국내상장주식은 DB만 직접투자 가능하고 DC/IRP는
    투자금지인데, 이 구분이 답변에서 빠지면 "IRP로 개별주식 추천해달라"는 요청에
    금지 사실을 말하지 못한 채 조건만 되묻게 된다(실측 no.459).
    """
    compact = _compact(question)
    product_type = _detect_investment_product_type(compact)
    if product_type is None:
        return None

    source = "doc56·doc58 퇴직연금 투자가능 상품유형"
    tiers = PRODUCT_RISK_TIER[product_type]
    tier_label = {
        RiskTier.SAFE: "안전자산(투자 가능)",
        RiskTier.RISKY: "위험자산(투자 가능, 한도 적용)",
        RiskTier.FORBIDDEN: "투자금지",
    }
    content = f"{product_type}의 제도별 투자 가능 여부: " + ", ".join(
        f"{plan.value} {tier_label[tiers[plan]]}" for plan in (PlanType.DB, PlanType.DC, PlanType.IRP)
    ) + f". 위험자산은 적립금의 {_pct_int(RISKY_ASSET_LIMIT)} 한도가 적용됩니다."

    plan = _extract_plan_type(question)
    if plan is not None:
        tier = tiers[plan]
        if tier == RiskTier.FORBIDDEN:
            allowed = [p.value for p in (PlanType.DB, PlanType.DC, PlanType.IRP) if tiers[p] != RiskTier.FORBIDDEN]
            allowed_note = (
                f"\n\n같은 상품이라도 {' · '.join(allowed)}에서는 투자할 수 있습니다."
                if allowed
                else "\n\n이 상품은 DB·DC·IRP 어느 제도에서도 투자할 수 없습니다."
            )
            draft = (
                f"{plan.value}에서는 **{product_type}에 투자할 수 없습니다**(투자금지 상품).\n\n"
                f"따라서 {plan.value} 계좌로는 해당 상품을 담거나 추천드릴 수 없습니다."
                f"{allowed_note}"
            )
            return draft, _context(source, content)
        if tier == RiskTier.SAFE:
            draft = (
                f"{plan.value}에서 **{product_type}은 안전자산으로 투자할 수 있습니다**.\n\n"
                f"안전자산이므로 위험자산 한도({_pct_int(RISKY_ASSET_LIMIT)})의 적용을 받지 않습니다."
            )
            return draft, _context(source, content)
        draft = (
            f"{plan.value}에서 **{product_type}은 위험자산으로 투자할 수 있습니다**.\n\n"
            f"다만 위험자산은 적립금의 {_pct_int(RISKY_ASSET_LIMIT)}까지만 담을 수 있어, "
            "이 한도 안에서 비중을 조절해야 합니다."
        )
        return draft, _context(source, content)

    lines = [f"**{product_type}**의 제도별 투자 가능 여부는 다음과 같습니다.", ""]
    for p in (PlanType.DB, PlanType.DC, PlanType.IRP):
        lines.append(f"- {p.value}: {tier_label[tiers[p]]}")
    lines.append("")
    lines.append("어느 제도인지 알려주시면 더 정확히 안내드릴 수 있습니다.")
    return "\n".join(lines), _context(source, content)


def _asks_irp_mandatory_transfer(compact_text: str) -> bool:
    """퇴직·이직 시 퇴직급여가 어디로 가는지(IRP 의무이전)를 묻는 질문인지."""
    if not any(w in compact_text for w in ("퇴직", "퇴사", "이직", "회사를옮", "직장을옮")):
        return False
    # 세금 질문은 이 카테고리가 아니다 — "퇴직할 때 세금이 어떻게 되나요?"의
    # "어떻게되"에 걸려 IRP 이전 안내가 나가면 묻지 않은 답을 하게 된다.
    if any(w in compact_text for w in ("세금", "세율", "소득세", "과세", "세액")):
        return False
    return any(
        w in compact_text
        for w in (
            "irp", "IRP", "계좌", "이전", "옮겨", "새로생", "새로만", "만들어야", "개설",
            "통장", "어떻게되", "수령", "받을수", "받나요",
        )
    )


def _irp_mandatory_transfer_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """퇴직 시 IRP 의무이전 — 원칙과 예외를 함께 답한다.

    ⚠️ 원칙(의무이전)과 예외(직접수령) 중 한쪽만 말하면 실측 오답이 그대로 재현된다:
    no.17은 "DC 퇴직금은 나이와 상관없이 반드시 IRP로"라고 예외를 빠뜨렸고,
    no.361은 "이직할 때마다 DC 계좌가 새로 생긴다"고 이전 구조 자체를 뒤집었다.
    """
    text = _compact(question)
    if not _asks_irp_mandatory_transfer(text):
        return None

    source = "퇴직연금제도 기본 · 사무담당자 업무 매뉴얼 (IRP 의무이전)"
    exceptions = "; ".join(IRP_MANDATORY_TRANSFER_EXCEPTIONS)
    content = (
        "근로자 퇴직 시 법정 예외사유를 제외하고 퇴직급여 전액을 IRP로 이전해야 합니다. "
        f"의무이전 예외사유: {exceptions}. 예외사유에 해당하면 퇴직급여를 개인(예금)계좌 "
        "등으로 직접 지급받을 수 있으며, 이 경우에도 퇴직급여 수령일부터 "
        f"{IRP_POST_RECEIPT_DEPOSIT_DAYS}일 이내에 IRP를 개설해 전부 또는 일부를 납입할 수 "
        "있습니다. DC 적립금도 퇴직 시 이 경로를 따라 IRP로 이전되며, 이직할 때마다 DC "
        "계좌가 새로 생기는 구조가 아닙니다."
    )

    exception_lines = "\n".join(f"  - {item}" for item in IRP_MANDATORY_TRANSFER_EXCEPTIONS)

    # "새로 생기나" 유형은 이전 구조 자체를 오해한 질문이라 그 전제부터 바로잡는다.
    if any(w in text for w in ("새로생", "새로만", "매번", "때마다")):
        draft = (
            "아니요. 이직할 때마다 DC 계좌가 새로 생기는 구조가 아닙니다.\n\n"
            "퇴직하면 그 회사의 DC 적립금은 **원칙적으로 IRP 계좌로 이전**됩니다. "
            "새 직장에서 DC에 가입하면 그 회사의 부담금을 받을 계좌가 새로 설정되지만, "
            "이전 직장의 적립금이 거기로 따라오는 것이 아니라 IRP로 옮겨져 이어집니다.\n\n"
            "다만 아래 법정 예외사유에 해당하면 퇴직급여를 개인(예금)계좌로 직접 받을 수 "
            "있습니다.\n"
            f"{exception_lines}\n\n"
            f"예외로 직접 받은 경우에도 수령일부터 {IRP_POST_RECEIPT_DEPOSIT_DAYS}일 이내에 "
            "IRP에 납입하면 과세이연을 이어갈 수 있습니다."
        )
        return draft, _context(source, content)

    draft = (
        "퇴직 시 퇴직급여는 **원칙적으로 IRP 계좌로 이전**됩니다. 따라서 대부분의 경우 "
        "IRP 계좌가 필요합니다.\n\n"
        "다만 아래 법정 예외사유에 해당하면 개인(예금)계좌 등으로 직접 지급받을 수 있어, "
        "IRP 개설이 의무는 아닙니다.\n"
        f"{exception_lines}\n\n"
        f"예외사유로 직접 받았더라도 수령일부터 {IRP_POST_RECEIPT_DEPOSIT_DAYS}일 이내에 "
        "IRP를 개설해 전부 또는 일부를 납입할 수 있습니다(과세이연 유지)."
    )
    return draft, _context(source, content)


def _investment_limit_response(question: str) -> tuple[str, list[RetrievedItem]]:
    """위험자산 투자한도(70%, TDF 조건충족 시 DC/IRP 100%) — 제도유형이 확정되면 그 제도만,
    아니면 세 제도를 모두 안내한다.

    ⚠️ investment_limit.py 규칙: 위험자산 한도 70%는 DB/DC/IRP **공통**이다. TDF 특례로
    DC/IRP만 100%까지 늘어나는 것이지, DB의 기본 한도가 다른 게 아니다. 실측 오답
    ("DB형도 위험자산 한도가 70%인가요?" -> "아니다"라고 반대로 답함)이 정확히 이
    지점에서 나왔다 — 집중투자한도(발행자별 10%/15%)와 위험자산 한도(70%)를 혼동하면
    같은 오답이 재현되므로, 여기서는 위험자산 한도만 다룬다(집중투자한도는 별도 규칙).
    """
    source = "doc58 퇴직연금 적립금 운용 및 투자한도 안내"
    content = (
        f"위험자산 투자한도는 DB·DC·IRP 공통으로 적립금의 {_pct_int(RISKY_ASSET_LIMIT)}까지입니다. "
        "다만 감독원장이 정한 조건(주식비중 80%, 예상은퇴시점 이후 40% 이내 등)을 충족한 TDF에 "
        f"전액 투자하는 경우, DC·IRP는 {_pct_int(TDF_QUALIFIED_LIMIT[PlanType.DC])}까지 허용됩니다. "
        "이 TDF 특례는 DC·IRP 전용이며, DB는 예상은퇴시점을 특정할 수 없어 특례 대상이 아니고 "
        f"기본 한도인 {_pct_int(RISKY_ASSET_LIMIT)}가 그대로 적용됩니다. 위험자산 한도와는 별개로 "
        "발행자·계열기업군 단위 집중투자한도(예: 동일법인 증권 DB 10%/DC·IRP 30%)가 추가로 적용됩니다."
    )

    plan = _extract_plan_type(question)
    compact = _compact(question)
    # "TDF 아닌"처럼 TDF를 명시적으로 배제한 질문은 특례 대상이 아니다 — 부정 표현을
    # 무시하고 "TDF" 단어만 보면 정반대로 "TDF에 투자하면 100%"라고 답하게 된다.
    mentions_tdf = "TDF" in question.upper() and not _is_negated(compact, "TDF")

    if plan is None:
        draft = (
            f"위험자산 투자한도는 **DB·DC·IRP 공통으로 {_pct_int(RISKY_ASSET_LIMIT)}**까지입니다.\n\n"
            f"다만 감독원장이 정한 조건을 충족한 TDF에 전액 투자하면 **DC·IRP는 "
            f"{_pct_int(TDF_QUALIFIED_LIMIT[PlanType.DC])}**까지 허용됩니다. "
            f"이 특례는 DC·IRP 전용이며, **DB는 조건 충족 여부와 무관하게 {_pct_int(RISKY_ASSET_LIMIT)} "
            "한도가 그대로 적용**됩니다(예상은퇴시점을 특정할 수 없어 특례 대상이 아닙니다).\n\n"
            "위험자산 한도와 별개로 발행자·계열기업군 단위 집중투자한도가 추가로 적용된다는 점도 "
            "참고해 주세요."
        )
        return draft, _context(source, content)

    limit = TDF_QUALIFIED_LIMIT[plan] if mentions_tdf else RISKY_ASSET_LIMIT
    if plan == PlanType.DB:
        draft = (
            f"{plan.value}형의 위험자산 투자한도는 **{_pct_int(RISKY_ASSET_LIMIT)}**입니다.\n\n"
            "TDF 조건충족 특례는 DC·IRP 전용이라 DB에는 적용되지 않으며(예상은퇴시점을 특정할 수 "
            f"없기 때문), TDF 여부와 무관하게 {_pct_int(RISKY_ASSET_LIMIT)} 한도가 그대로 적용됩니다."
        )
    elif mentions_tdf:
        draft = (
            f"네, {plan.value}에서 감독원장이 정한 조건(주식비중 80%, 예상은퇴시점 이후 40% 이내 등)을 "
            f"충족한 TDF에 전액 투자하면 위험자산 비중을 **{_pct_int(limit)}**까지 채울 수 있습니다.\n\n"
            f"조건을 충족하지 못한 일반 상품이라면 다른 위험자산과 마찬가지로 "
            f"{_pct_int(RISKY_ASSET_LIMIT)} 한도가 적용됩니다."
        )
    else:
        draft = (
            f"{plan.value}의 위험자산 투자한도는 **{_pct_int(RISKY_ASSET_LIMIT)}**입니다.\n\n"
            f"다만 감독원장이 정한 조건을 충족한 TDF에 전액 투자하는 경우에 한해 "
            f"{_pct_int(TDF_QUALIFIED_LIMIT[plan])}까지 늘어날 수 있습니다."
        )
    return draft, _context(source, content)


def _in_kind_transfer_block_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc34 실물이전 불가사유 코드"
    definite_codes = [code for code, info in TRANSFER_BLOCK_CODES.items() if not info.get("directional")]
    manual_codes = [code for code, info in TRANSFER_BLOCK_CODES.items() if info.get("directional")]
    content = "확정 불가사유: " + "; ".join(
        f"{code}. {TRANSFER_BLOCK_CODES[code]['name']} - {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in definite_codes
    )
    content += " 추가 확인 필요 사유: " + "; ".join(
        f"{code}. {TRANSFER_BLOCK_CODES[code]['name']} - {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in manual_codes
    )
    # ⚠️ 답변 목록을 손으로 고른 하드코딩 리스트로 만들면 안 된다 — 실제로 그렇게 돼 있던
    # 동안 "21. 만기(상환)"이 목록에서 빠져, 만기 때문에 이전이 막힌 사용자에게 근거(content)
    # 에는 있는 사유가 답변 본문에서는 안 보이는 사고가 났다. 근거와 답변이 같은 원천
    # (TRANSFER_BLOCK_CODES)에서 나오도록 해서 둘이 어긋날 수 없게 한다.
    definite_lines = "\n".join(
        f"- {code}. {TRANSFER_BLOCK_CODES[code]['name']}: {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in definite_codes
    )
    manual_lines = "\n".join(
        f"- {code}. {TRANSFER_BLOCK_CODES[code]['name']}: {TRANSFER_BLOCK_CODES[code]['desc']}"
        for code in manual_codes
    )
    draft = (
        "퇴직연금 실물이전이 제한될 수 있는 상품·상황은 다음과 같습니다.\n\n"
        f"[실물이전 불가사유]\n{definite_lines}\n\n"
        f"[상대 금융기관 확인이 필요한 사유]\n{manual_lines}\n\n"
        # doc34는 25개 코드 외에 "99.기타"를 두어 목록에 없는 사유도 포괄한다. 이 문장을 빼면
        # 25개가 전부인 것처럼 읽혀 근거를 과대 표현하게 된다(만기지정식 예금, 제도불일치 등이 여기 속한다).
        "이 밖에도 만기지정식 예금, 제도 불일치 등 개별 사유(99. 기타)로 이전이 제한될 수 있습니다.\n\n"
        "보유하신 상품이 어디에 해당하는지 알려주시면(예: MMF, 사모펀드, 만기 도래 여부, "
        "디폴트옵션 상품 여부) 실제 이전 가능 여부를 더 정확히 확인해 드릴 수 있습니다."
    )
    return draft, _context(source, content)


# 사용자가 쓰는 표현 -> doc34 불가사유 코드. 코드표(TRANSFER_BLOCK_CODES)의 name을 그대로
# 쓰되, 실제 질문에서 나오는 축약·구어 표현만 추가로 매핑한다.
_TRANSFER_CODE_ALIASES: dict[str, tuple[str, ...]] = {
    "01": ("소규모펀드", "소규모"),
    "03": ("사모펀드", "사모"),
    "04": ("mmf", "머니마켓", "단기금융"),
    "05": ("환매수수료",),
    "08": ("압류", "질권"),
    "09": ("만기매칭형",),
    "10": ("지분증권", "리츠"),
    "11": ("rp", "환매조건부"),
    "12": ("발행어음",),
    "13": ("금리연동형보험", "금리연동"),
    "14": ("실적배당형보험", "변액보험"),
    "16": ("규약미체결", "규약"),
    "18": ("한도초과", "예금자보호한도"),
    "19": ("자사상품",),
    "21": ("만기", "만기도래", "만기상환", "상환"),
    "22": ("환매불가",),
    "23": ("디폴트옵션",),
    "24": ("상장투자회사", "맥쿼리인프라"),
}


def _alias_matches(alias: str, text: str) -> bool:
    """별칭이 텍스트에 나타나는지 본다 — 짧은 영문 약어는 단어 경계를 요구한다.

    ⚠️ 실측 사고: "rp"(환매조건부채권)를 단순 부분문자열로 찾다가 **"IRP"의 뒤 두
    글자**에 걸려, 상품을 전혀 언급하지 않은 "IRP 계좌를 다른 증권사로 옮기려면?",
    "연금저축을 IRP로 실물이전할 수 있나요?" 같은 질문에 "RP라서 실물이전 불가"라는
    정반대 답이 나갔다(no.56/405). IRP는 이 도메인에서 가장 흔한 단어라 영향이 컸다.

    한글 별칭은 그 자체로 충분히 길고 조사가 붙어 경계를 못 쓰므로 부분문자열 매칭을
    유지하고, 영문 약어(rp 등)만 앞뒤에 다른 영문자가 없을 때 매칭한다.
    """
    if alias.isascii() and alias.isalpha():
        return re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text) is not None
    return alias in text


def _detect_transfer_codes(question: str) -> list[str]:
    """질문에서 실물이전 불가사유 코드를 인식한다 (코드표 name + 구어 표현)."""
    text = _compact(question).lower()
    detected: list[str] = []
    for code, info in TRANSFER_BLOCK_CODES.items():
        name_key = _compact(info["name"]).lower()
        aliases = _TRANSFER_CODE_ALIASES.get(code, ())
        if name_key in text or any(_alias_matches(a, text) for a in aliases):
            detected.append(code)
    return detected


def in_kind_transfer_judgement_response(question: str) -> tuple[str, list[RetrievedItem]] | None:
    """보유 상품이 특정된 실물이전 질문에 가능/불가를 판정한다 (목록 나열이 아님).

실물이전이 안 되는 경우는?"(목록답변)과 "MMF인데 옮길 수 있나요?"(개별판정)는
    같은 도메인이지만 다른 작업이다. 개별판정용 정형 경로가 없던 동안 라우터가 이런
    질문을 기각했고, LLM이 자유롭게 툴을 고르다 실측에서 실물이전 질문에
    check_product_pension_eligibility(연금계좌 투자 가능 여부)를 호출해 "네, 실물이전
    문제없이 진행 가능합니다"라는 정반대 답을 냈다.

    상품 유형이 인식되지 않으면 None을 돌려 목록답변이나 일반 경로로 넘긴다.
    """
    codes = _detect_transfer_codes(question)
    if not codes:
        return None

    source = "doc34 실물이전 불가사유 코드"
    blocking = [c for c in codes if not TRANSFER_BLOCK_CODES[c].get("directional")]
    manual = [c for c in codes if TRANSFER_BLOCK_CODES[c].get("directional")]

    def _describe(code: str) -> str:
        info = TRANSFER_BLOCK_CODES[code]
        return f"{code}. {info['name']}: {info['desc']}"

    content = "; ".join(_describe(c) for c in codes)

    if blocking:
        reasons = "\n".join(f"- {_describe(c)}" for c in blocking)
        draft = (
            "아니요, 말씀하신 상품은 실물이전이 제한됩니다.\n\n"
            f"해당하는 불가사유는 다음과 같습니다.\n{reasons}\n\n"
            "실물이전이 불가한 상품은 매도(환매) 후 현금으로 이전하는 방법을 검토하실 수 "
            "있습니다. 다만 매도 시점의 손익과 환매 조건은 상품마다 다르므로 가입 금융기관에 "
            "확인하시기 바랍니다."
        )
    else:
        reasons = "\n".join(f"- {_describe(c)}" for c in manual)
        draft = (
            "조건에 따라 다릅니다. 말씀하신 상품은 상대 금융기관 확인이 필요한 사유에 "
            f"해당합니다.\n\n{reasons}\n\n"
            "이관은 가능하더라도 수관 기관의 협약·상품라인업 여부에 따라 최종 가능 여부가 "
            "달라지므로, 옮기려는 금융기관에 확인이 필요합니다."
        )
    return draft, _context(source, content)


_ACCOUNT_VALUE_RE = re.compile(
    r"(?:평가액|잔고|적립금|계좌)[^\d]{0,6}([0-9,]+(?:\s*(?:천|백|십))?)\s*(억\s*원|만\s*원|억|만|원)"
)
_PAYMENT_YEAR_RE = re.compile(r"(?:연금수령|수령)\s*(\d{1,2})\s*년차")


def _extract_withdrawal_limit_inputs(question: str) -> tuple[Optional[int], Optional[int]]:
    """질문에서 (연금계좌 평가액, 연금수령연차)를 뽑는다. 못 읽으면 None.

    ⚠️ "연금수령연차"(한도 산정용, 개시 가능 시점부터 자동 누적)와 "연금실제수령연차"
    (이연퇴직소득세 감면율용, 실제 인출한 해만 누적)는 서로 다른 값이다 — 감면율 쪽
    추출기(tax_context._extract_actual_pension_year)를 재사용하면 안 된다.
    """
    compact = _compact(question)
    if "실제수령연차" in compact:
        return None, None  # 감면율 질문이므로 이 핸들러가 다룰 대상이 아니다

    value_match = _ACCOUNT_VALUE_RE.search(compact)
    account_value = (
        _parse_korean_amount(value_match.group(1), value_match.group(2)) if value_match else None
    )
    year_match = _PAYMENT_YEAR_RE.search(compact)
    payment_year = int(year_match.group(1)) if year_match else None
    return account_value, payment_year


_OVERAGE_TAX_QUESTION_MARKERS = ("넘겨서", "넘게", "초과해서", "초과하면", "초과인출", "한도넘")
_OVERAGE_TAX_NOTE = (
    "\n\n한도를 초과해 인출하는 부분은 연금수령이 아니라 연금외수령으로 분류되어, "
    "세액공제 받은 납입금·운용수익 재원이라면 전액 16.5% 기타소득세가 부과됩니다 "
    "(한도 이내 인출분에 적용되는 연령별 연금소득세율보다 높습니다)."
)


def _withdrawal_limit_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc39 연금수령한도 규칙"
    content = (
        "연금수령한도 = 연금계좌 평가액 ÷ (11 - 연금수령연차) × 120%입니다. "
        f"연금수령연차 {UNLIMITED_FROM_YEAR}년차 이상부터는 한도 자체가 사라집니다. "
        f"2013.3.1 이전 가입한 연금계좌는 {SIX_YEAR_EXCEPTION_START}년차부터 기산하는 특례가 있습니다. "
        "연금수령 요건은 가입기간 5년 이상, 만 55세 이후, 한도 이내 인출입니다."
    )
    formula_note = (
        "다만 연금수령연차 11년차 이상부터는 한도가 없어져 전액 인출해도 연금수령으로 인정될 수 있습니다. "
        "또 2013.3.1 이전 가입한 연금계좌는 1년차가 아니라 6년차부터 기산하는 특례가 있습니다."
    )
    # ⚠️ 이 핸들러는 원래 "한도가 얼마인지"만 답했다. 그런데 "한도를 넘겨서 인출하면
    # 세금이 얼마나 더 나오나요"(실측 no.142)처럼 **초과분의 세금**을 묻는 질문도
    # 같은 카테고리(연금수령한도)로 들어온다 — 정답(16.5% 기타소득세)이 이미 doc38
    # 근거 문서에 있는데(no.106 수정 때 확인한 것과 같은 문서), 한도 공식만 반복
    # 답하고 정작 질문의 핵심(초과 시 세금)에는 답하지 못했다.
    asks_overage_tax = any(marker in question for marker in _OVERAGE_TAX_QUESTION_MARKERS)
    overage_note = _OVERAGE_TAX_NOTE if asks_overage_tax else ""
    if asks_overage_tax:
        content += " 한도 초과 인출분은 연금외수령으로 분류되어 전액 16.5% 기타소득세가 부과됩니다."

    # 질문에 평가액과 연차가 모두 있으면 규칙엔진으로 실제 금액을 계산한다.
    # ⚠️ 예전에는 question을 아예 읽지 않고 항상 공식만 안내해서, 계산에 필요한 값이
    # 다 주어진 질문에도 "구체적인 금액 계산을 하려면 평가액과 연차가 필요합니다"라고
    # 되물었다(실측 no.103: 5000만원·3년차 -> 750만원, no.334: 1억·5년차 -> 2000만원).
    account_value, payment_year = _extract_withdrawal_limit_inputs(question)
    if account_value is not None and payment_year is not None:
        result = calculate_withdrawal_limit(
            account_value=account_value, pension_payment_year=payment_year
        )
        if result.is_unlimited:
            draft = (
                f"연금수령연차 {payment_year}년차는 {UNLIMITED_FROM_YEAR}년차 이상이라 "
                "연금수령한도가 적용되지 않습니다. 전액 인출해도 연금수령으로 인정될 수 있습니다.\n\n"
                f"{formula_note}"
            )
        else:
            draft = (
                f"입력해주신 조건(평가액 {_won(account_value)}, 연금수령 {payment_year}년차)의 "
                f"올해 연금수령한도는 **{_won(result.limit_amount)}**입니다.\n\n"
                f"계산식: {_won(account_value)} ÷ (11 - {payment_year}) × 120% = "
                f"{_won(result.limit_amount)}\n\n{formula_note}{overage_note}"
            )
        content += (
            f" 입력 조건에서는 평가액 {_won(account_value)}, 연금수령 {payment_year}년차이며 "
            + (
                f"{UNLIMITED_FROM_YEAR}년차 이상이라 한도가 없습니다."
                if result.is_unlimited
                else f"한도는 {_won(result.limit_amount)}입니다."
            )
        )
        return draft, _context(source, content)

    draft = (
        "연금수령한도는 다음 공식으로 계산합니다.\n\n"
        "연금수령한도 = 연금계좌 평가액 ÷ (11 - 연금수령연차) × 120%\n\n"
        f"{formula_note}{overage_note}\n\n"
        "구체적인 금액 계산을 하려면 연금계좌 평가액과 현재 연금수령연차가 필요합니다."
    )
    return draft, _context(source, content)


def _retirement_tax_reduction_response(question: str) -> tuple[str, list[RetrievedItem]]:
    r1 = get_deferred_retirement_tax_rate(1)
    r11 = get_deferred_retirement_tax_rate(11)
    r21 = get_deferred_retirement_tax_rate(21)
    source = "doc39~doc40 이연퇴직소득세 감면 규칙"
    content = (
        f"연금실제수령연차 1~10년차는 이연퇴직소득세의 {_pct(r1.payment_ratio)}를 납부하고 "
        f"{_pct(r1.reduction_ratio)}를 감면합니다. 11~20년차는 {_pct(r11.payment_ratio)}를 납부하고 "
        f"{_pct(r11.reduction_ratio)}를 감면합니다. 21년차 이상은 {_pct(r21.payment_ratio)}를 납부하고 "
        f"{_pct(r21.reduction_ratio)}를 감면합니다. 연금외수령은 감면 없이 전액 납부합니다."
    )
    # ⚠️ 예전에는 question을 받고도 전혀 읽지 않아, 사용자가 연금실제수령연차를 밝혀도
    # 항상 같은 일반표만 반환했다. 같은 질문이 라우터의 선택에 따라 개인세금_입력충분성으로
    # 가면 연차를 정직하게 되묻는데(personal_tax_response), 이 카테고리로 오면 되묻지도
    # 확정하지도 않고 일반표로 끝나 답변 완결성이 라우터의 비결정적 선택에 좌우됐다
    # (실측 T09/T10 vs T12/T18). 다른 결정론 핸들러(_pension_income_tax_rate_response,
    # _withdrawal_limit_response)는 이미 질문에서 값을 읽어 확정하는 패턴을 쓴다.
    actual_receipt_year = extract_tax_context(question).actual_pension_year

    draft = (
        "퇴직금을 연금으로 수령하면 이연퇴직소득세가 연차에 따라 감면됩니다.\n\n"
        "- 연금실제수령연차 1~10년차: 이연퇴직소득세의 70% 납부, 30% 감면\n"
        "- 연금실제수령연차 11~20년차: 60% 납부, 40% 감면\n"
        "- 연금실제수령연차 21년차 이상: 50% 납부, 50% 감면\n\n"
        "주의할 점은 여기서 쓰는 기준이 '연금수령연차'가 아니라 실제로 인출한 해만 세는 "
        "'연금실제수령연차'라는 점입니다. 연금외수령이면 감면 없이 이연퇴직소득세 전액을 납부합니다."
    )

    if actual_receipt_year is not None:
        applied = get_deferred_retirement_tax_rate(actual_receipt_year, is_pension_receipt=True)
        draft += (
            f"\n\n말씀하신 연금실제수령연차 {actual_receipt_year}년차는 이연퇴직소득세의 "
            f"{_pct(applied.payment_ratio)}를 납부하고 {_pct(applied.reduction_ratio)}를 감면받는 구간입니다.\n"
            "실제 납부세액은 원래 부과될 이연퇴직소득세 금액에 이 비율을 적용해 계산합니다 — "
            "그 금액은 퇴직 시점에 확정되므로 퇴직금 수령 기관에서 확인하실 수 있습니다."
        )
        content += (
            f" 입력 조건의 연금실제수령연차 {actual_receipt_year}년차는 "
            f"{_pct(applied.payment_ratio)} 납부, {_pct(applied.reduction_ratio)} 감면 구간입니다."
        )
    else:
        # 연차를 모르면 감면율을 확정할 수 없다 — 되묻되, 위 일반 규칙은 이미 답했으므로
        # 답변 자체를 막지는 않는다(알고 있는 것은 답하고 모르는 것만 묻는다).
        draft += (
            "\n\n본인에게 적용될 감면율을 확정하려면 연금실제수령연차가 몇 년차인지 알려주세요. "
            "정확한 세액까지 계산하려면 원래 부과될 이연퇴직소득세 금액도 함께 필요합니다."
        )

    return draft, _context(source, content)


def _pension_income_tax_response(question: str) -> tuple[str, list[RetrievedItem]]:
    source = "doc38 연금소득 종합과세 규칙"
    content = (
        f"사적연금소득은 연 {_won(ANNUAL_THRESHOLD)} 초과 여부가 종합과세 판단 기준입니다. "
        f"초과 시 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세를 선택할 수 있습니다. "
        "1,500만원 판정에는 세액공제 받은 납입금과 운용수익 재원만 포함하고, 세액공제 안 받은 원금과 퇴직금은 제외합니다. "
        "1,500만원 이내 연금소득세율은 만 55세 이상 70세 미만 5.5%, 70세 이상 80세 미만 4.4%, 80세 이상 3.3%입니다."
    )
    draft = (
        "사적연금소득의 종합과세 여부는 연 1,500만원 초과 여부가 기준입니다.\n\n"
        "- 연 1,500만원 이내: 연령별 연금소득세율로 분리과세\n"
        "- 연 1,500만원 초과: 종합과세 또는 16.5% 분리과세 선택 가능\n\n"
        "다만 이 1,500만원 기준에는 세액공제 받은 납입금과 운용수익 재원만 포함됩니다. "
        "세액공제 받지 않은 원금과 퇴직금 재원은 이 판정에서 제외됩니다.\n\n"
        "1,500만원 이내일 때 연금소득세율은 만 55세 이상 70세 미만 5.5%, "
        "70세 이상 80세 미만 4.4%, 80세 이상 3.3%입니다."
    )
    return draft, _context(source, content)


_AGE_RE = re.compile(r"(?:만\s*)?(\d{1,3})\s*세")
_LIFETIME_ANNUITY_WORDS = ("종신연금", "종신형", "종신 연금")


def _extract_age(question: str) -> Optional[int]:
    """질문에서 나이를 뽑는다. 연금 수령 가능 범위(55~120) 밖이면 무시한다.

70세 이상 80세 미만" 같은 제도 설명 인용이 섞이면 오탐이 나므로, 나이가 여러 개
    등장하면 특정하지 않는다(None) — 하나로 단정하는 것보다 조건 부족으로 처리하는 편이 안전하다.
    """
    text = question or ""
    ages = [int(m.group(1)) for m in _AGE_RE.finditer(text)]
    if not ages:
        # 숫자 표기가 없을 때만 한글 나이를 본다 — 숫자가 이미 있으면 그쪽이 더 정확하고,
        # "일흔이면 4.4%인데 저는 76세예요" 같은 문장에서 둘을 섞으면 오히려 오탐이 난다.
        ages = [_KOREAN_AGE_WORDS[m.group()] for m in _KOREAN_AGE_RE.finditer(text)]
    valid = [a for a in ages if 55 <= a <= 120]
    if len(valid) != 1:
        return None
    return valid[0]


def _age_bracket_label(age: int) -> str:
    if age < 70:
        return "만 55세 이상 70세 미만"
    if age < 80:
        return "만 70세 이상 80세 미만"
    return "만 80세 이상"


def _pension_income_tax_rate_response(question: str) -> tuple[str, list[RetrievedItem]]:
    """연령별 연금소득세율 — 조건이 충분하면 확정, 부족하면 구간 확정 + 분기 + 역질문.

    ⚠️ 나이만으로 세율을 단정하면 안 된다(실측 사고: "만 74세 → 3.3%"라고 답했으나 정답은
    4.4%). 같은 나이라도 ①종신연금 여부 ②연 1,500만원 초과 여부 ③인출 재원(퇴직금 재원은
    이연퇴직소득세 체계)에 따라 적용 세율이 달라지기 때문이다. 그래서 확정 가능한 부분만
    규칙엔진으로 확정하고, 나머지는 분기를 그대로 제시한 뒤 부족한 조건을 되묻는다 —
    답변을 포기하지 않으면서 틀린 단정도 하지 않는 방식.
    """
    source = "doc38 연금소득세율 규칙"
    content = (
        f"1,500만원 이내 구간의 연금소득세율은 만 55세 이상 70세 미만 "
        f"{_pct(get_pension_income_tax_rate(55))}, 70세 이상 80세 미만 "
        f"{_pct(get_pension_income_tax_rate(70))}, 80세 이상 "
        f"{_pct(get_pension_income_tax_rate(80))}입니다. 종신연금을 수령하면 연령과 무관하게 "
        f"{_pct(get_pension_income_tax_rate(55, is_lifetime_annuity=True))}가 적용됩니다. "
        f"사적연금소득이 연 {_won(ANNUAL_THRESHOLD)}을 초과하면 종합과세 또는 "
        f"{_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세를 선택합니다. "
        "이 세율은 세액공제 받은 납입금과 운용수익 재원에 적용되며, 퇴직금(이연퇴직소득) "
        "재원은 이연퇴직소득세 감면 체계가 별도로 적용됩니다."
    )

    age = _extract_age(question)
    # "종신연금 아니고 확정기간형으로 받으면"처럼 명시적으로 배제한 경우를 제외한다 —
    # 단어만 보고 종신연금으로 판정하면 3.3%(종신)와 4.4%(70대 일반수령)가 뒤바뀐다.
    compact_question = _compact(question)
    is_lifetime = any(word in question for word in _LIFETIME_ANNUITY_WORDS) and not any(
        _is_negated(compact_question, word) for word in _LIFETIME_ANNUITY_WORDS
    )

    if age is None:
        draft = (
            "연금소득세율은 연금을 받는 시점의 나이에 따라 달라집니다.\n\n"
            f"- {_age_bracket_label(55)}: {_pct(get_pension_income_tax_rate(55))}\n"
            f"- {_age_bracket_label(70)}: {_pct(get_pension_income_tax_rate(70))}\n"
            f"- {_age_bracket_label(80)}: {_pct(get_pension_income_tax_rate(80))}\n\n"
            f"다만 종신연금으로 받으시면 나이와 관계없이 "
            f"{_pct(get_pension_income_tax_rate(55, is_lifetime_annuity=True))}가 적용됩니다.\n\n"
            f"또한 위 세율은 사적연금소득이 연 {_won(ANNUAL_THRESHOLD)} 이내일 때 적용되며, "
            f"초과하면 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세 중 "
            "선택하게 됩니다.\n\n"
            "정확한 적용 세율을 확인하시려면 연금 수령 시점의 나이, 종신연금 여부, 연간 "
            "예상 수령액을 알려주세요."
        )
        return draft, _context(source, content)

    bracket = _age_bracket_label(age)
    base_rate = get_pension_income_tax_rate(age)
    lifetime_rate = get_pension_income_tax_rate(age, is_lifetime_annuity=True)

    if is_lifetime:
        draft = (
            f"만 {age}세에 종신연금으로 수령하시는 경우 연금소득세율은 "
            f"**{_pct(lifetime_rate)}**입니다. 종신연금은 연령과 무관하게 동일한 세율이 적용됩니다.\n\n"
            f"다만 이 세율은 사적연금소득이 연 {_won(ANNUAL_THRESHOLD)} 이내일 때 적용되며, "
            f"초과하면 종합과세 또는 {_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세 중 "
            "선택하게 됩니다."
        )
    else:
        draft = (
            f"만 {age}세는 '{bracket}' 구간에 해당하며, 이 구간의 연금소득세율은 "
            f"**{_pct(base_rate)}**입니다.\n\n"
            "다만 아래 조건에 해당하면 적용 세율이 달라집니다.\n"
            f"- 종신연금으로 수령하는 경우: 나이와 무관하게 {_pct(lifetime_rate)}\n"
            f"- 사적연금소득이 연 {_won(ANNUAL_THRESHOLD)}을 초과하는 경우: 종합과세 또는 "
            f"{_pct(SEPARATE_TAXATION_RATE_OVER_THRESHOLD)} 분리과세 중 선택\n\n"
            "또한 이 세율은 세액공제를 받은 납입금과 운용수익 재원에 적용됩니다. 퇴직금을 "
            "연금으로 받는 경우에는 이연퇴직소득세 감면 체계가 별도로 적용됩니다.\n\n"
            "종신연금 여부와 연간 예상 수령액을 알려주시면 더 정확히 안내드릴 수 있습니다."
        )
    return draft, _context(source, content)


_CATEGORY_HANDLERS = {
    "복합정보_태스크플랜": _composite_info_task_plan_response,
    "제도비교_DB_DC": _db_dc_comparison_response,
    "계좌이전_절차": _account_transfer_procedure_response,
    "세액공제_계산_입력부족": _tax_credit_calculation_missing_response,
    "세액공제_한도": _tax_credit_limit_response,
    "세금혜택_개요": _tax_benefit_overview_response,
    "개인세금_입력충분성": personal_tax_response,
    "중도인출_기한판정": _early_withdrawal_deadline_response,
    "중도인출_요건판정": _early_withdrawal_eligibility_response,
    "중도인출_일반": _early_withdrawal_general_response,
    "디폴트옵션_자동매수": _default_option_auto_purchase_response,
    "디폴트옵션_옵트인판정": _default_option_optin_response,
    "투자한도_위험자산": _investment_limit_response,
    "투자가능여부_상품유형": _investment_eligibility_response,
    "퇴직시_IRP의무이전": _irp_mandatory_transfer_response,
    "실물이전_불가사유": _in_kind_transfer_block_response,
    "실물이전_개별판정": in_kind_transfer_judgement_response,
    "연금수령한도": _withdrawal_limit_response,
    "퇴직소득세감면": _retirement_tax_reduction_response,
    "연금소득세_종합과세": _pension_income_tax_response,
    "연금소득세율_연령별": _pension_income_tax_rate_response,
}
