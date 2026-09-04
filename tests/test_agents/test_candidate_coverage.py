"""정형 카테고리 어휘 커버리지 회귀 테스트 (패러프레이즈 테스트셋).

## 왜 이 파일이 따로 있는가

`candidate_categories`는 키워드 열거 방식이라 **표현이 달라지면 무너진다** — 이 함수의
docstring 스스로가 경고하는 실패 유형인데, 실제로 같은 클래스의 사고가 반복됐다:

  1. no.279  "일흔 넘었는데 연금소득세율이..."   → 후보 0건 (한글 나이 표현)
  2. V03/V05 "세액공제 대박으로 받는 방법"       → 후보 0건 → 700만원(폐지 한도) 생성
  3. no.383  "연금저축 600만원 채우고 IRP 300만원" → 후보 0건 → 700만원 생성
  4. 실사용  "65세 정년 은퇴... 절세 방법"        → 후보 0건 → 700만원 생성

넷 다 **정답을 가진 핸들러가 이미 있었는데** 어휘 조건이 좁아 도달하지 못한 것이다.

## 이 테스트가 막는 것

핵심 비대칭: 후보를 **넓게** 잡는 오탐은 라우터와 `deterministic_response_for`의
후보 재확인 게이트가 걸러내지만, **누락은 아무도 막지 못한다**. 그래서 누락 방향만
집중적으로 고정한다 — 같은 의도를 사용자가 실제로 쓸 법한 여러 표현으로 물었을 때
후보가 나오는지 검사한다.

LLM을 호출하지 않고 순수 함수만 돌리므로 전체가 1초 내에 끝난다. 조건을 좁히는
변경이 들어오면 여기서 즉시 실패한다.

## 유지보수 규칙

- 새 표현 때문에 사고가 나면 **먼저 여기 한 줄 추가**하고(실패 확인) 그 다음 고친다.
- 카테고리를 새로 만들면 PARAPHRASES에 항목을 추가한다.
- 여기 없는 표현이 실패하는 것은 이 테스트의 한계다(열거는 유한하다). 근본 해결은
  의미 기반 라우팅이며, 이 파일은 그때까지의 안전망이다.
"""

import pytest

from src.agents.deterministic_info import (
    _CATEGORY_HANDLERS,
    candidate_categories,
    deterministic_miss_signal,
)

# 카테고리 → 같은 의도를 표현하는 질문들. 전부 그 카테고리를 후보로 내야 한다.
# 표현은 "사용자가 실제로 쓸 법한 말"을 기준으로 고른다 — 제도 용어를 그대로 쓰는
# 질문뿐 아니라, 용어를 모르는 사람이 풀어 쓰는 표현을 반드시 포함한다.
PARAPHRASES: dict[str, list[str]] = {
    "세금혜택_개요": [
        "연금 세금혜택이 뭐가 있나요",
        "절세 방법 알려줘",
        "65세인데 절세하고 싶어",
        "나는 올해 나이가 65세로 정년 은퇴를 앞두고 있어. 이런 내가 절세를 하고자하는데 방법 알려줘",
        "세금 줄이는 법 알려줘",
        "세금 아끼는 방법 있나요",
    ],
    "세액공제_한도": [
        "세액공제 한도가 얼마인가요",
        "세액공제 얼마까지 받을 수 있나요",
        "연금저축이랑 IRP에 최대한 많이 넣고 싶어요. 얼마까지 되나요?",
        "IRP에 얼마까지 넣을 수 있나요",
        "연금저축 납입한도가 궁금해요",
        "세액공제 대박으로 받을 수 있는 방법 있나요?",
        "연금저축을 먼저 600만원 채우고 IRP로 300만원 추가하는 순서가 맞나요?",
    ],
    "연금소득세율_연령별": [
        "연금소득세율 알려줘",
        "연금소득세율이 어떻게 되나요",
        "70세면 연금 세율이 몇 퍼센트야",
        "일흔 넘었는데 연금소득세율이 어떻게 되나요",
        "연령별 연금소득세율 표 알려줘",
    ],
    "중도인출_일반": [
        "중도인출 되나요",
        "중도인출 사유가 뭐가 있나요",
        "퇴직연금 중도인출 가능한가요",
        "퇴직연금 중간에 빼서 쓸 수 있나요",
    ],
    "실물이전_불가사유": [
        "실물이전 안 되는 경우가 뭔가요",
        "팔지 않고 옮길 수 있나요",
        "보유 상품 그대로 이전 가능한가요",
    ],
    "투자한도_위험자산": [
        "위험자산 한도가 얼마인가요",
        "주식형 비중 70%까지 되나요",
        "위험자산 투자한도 알려줘",
    ],
    "연금수령한도": [
        "연금수령한도가 얼마인가요",
        "연금수령한도 계산법 알려줘",
    ],
    "퇴직소득세감면": [
        "퇴직소득세 감면이 얼마나 되나요",
        "이연퇴직소득세 감면율 알려줘",
    ],
    "디폴트옵션_자동매수": [
        "디폴트옵션 언제 자동매수되나요",
        "디폴트옵션 자동매수 시점이 언제인가요",
        "기존가입자인데 언제 자동으로 매수되나요",
    ],
    "퇴직시_IRP의무이전": [
        "퇴직하면 IRP로 꼭 옮겨야 하나요",
        "퇴직급여 IRP 의무이전인가요",
    ],
    "연금소득세_종합과세": [
        "연금소득이 1500만원 넘으면 어떻게 되나요",
        "사적연금 종합과세 기준이 뭔가요",
    ],
    # ⚠️ 이 카테고리는 _INVESTMENT_PRODUCT_ALIASES에 등재된 상품유형만 판정한다
    # (비상장주식·국내상장주식·사모펀드·DR·전환사채류·해외상장주식). 레버리지 ETF처럼
    # 목록에 없는 상품은 핸들러가 답할 근거 자체가 없으므로 후보로 내지 않는 것이 옳다 —
    # 여기에는 **핸들러가 실제로 답할 수 있는** 표현만 넣는다.
    "투자가능여부_상품유형": [
        "IRP로 개별주식 담을 수 있나요",
        "연금계좌에서 국내주식 직접투자 되나요",
        "IRP에서 사모펀드 살 수 있나요",
    ],
    "제도비교_DB_DC": [
        "DC와 DB, 퇴직금이 정해지는 방식이랑 운용 주체가 어떻게 다른가요?",
        "DB형은 제가 받을 퇴직금이 미리 확정돼 있는 게 맞나요?",
        "DB와 DC 차이가 뭔가요?",
        "DB형은 회사가 운용하고 DC형은 제가 직접 운용한다는데 맞나요?",
    ],
    "계좌이전_절차": [
        "IRP 계좌를 다른 증권사로 옮기려면 어떻게 해야 하나요?",
        "연금저축을 IRP로 이체할 수 있나요",
        "퇴직연금 다른 회사로 옮기는 방법",
        "연금저축 계좌를 해지하지 않고 다른 금융사로 옮기는 방법이 있나요?",
    ],
    "계좌선택_가이드": [
        "직장인이면 IRP만 만들어도 되나요, 연금저축도 같이 만들어야 하나요?",
        "연금저축과 IRP 뭐가 다른가요",
        "연금저축이랑 IRP 둘 다 필요한가요",
    ],
}

# 정형 주제와 무관한 질문 — 후보가 나오면 안 된다(과잉 확장 방지).
NEGATIVES: list[str] = [
    "점심 뭐 먹지",
    "안녕하세요",
    "오늘 날씨 어때요",
    "주식 시장 전망이 어떤가요",
    "비트코인 사도 될까요",
]


@pytest.mark.parametrize(
    ("category", "question"),
    [(category, question) for category, questions in PARAPHRASES.items() for question in questions],
)
def test_paraphrase_reaches_category(category: str, question: str) -> None:
    """같은 의도를 다르게 표현해도 해당 카테고리가 후보에 올라야 한다."""
    assert category in candidate_categories(question), (
        f"{category} 누락: {question!r} -> {candidate_categories(question)}"
    )


@pytest.mark.parametrize("question", NEGATIVES)
def test_unrelated_question_has_no_candidate(question: str) -> None:
    """정형 주제와 무관한 질문은 후보가 없어야 한다."""
    assert candidate_categories(question) == [], question


def test_every_paraphrase_category_has_handler() -> None:
    """테스트셋의 카테고리가 실제 핸들러와 일치하는지 확인한다(오타·삭제 방지)."""
    for category in PARAPHRASES:
        assert category in _CATEGORY_HANDLERS, category


def test_miss_signal_is_silent_for_covered_questions() -> None:
    """커버된 질문에는 미탐지 신호가 뜨지 않아야 한다."""
    for questions in PARAPHRASES.values():
        for question in questions:
            assert deterministic_miss_signal(question) is None, question


def test_miss_signal_fires_when_topic_present_but_no_candidate() -> None:
    """정형 주제어가 있는데 후보가 0건이면 신호를 남긴다 (관측용).

    이 신호는 '결함 확정'이 아니라 '점검 대상'이다 — 정형 카테고리가 아직 없는
    주제도 여기 걸린다.
    """
    signal = deterministic_miss_signal("이 펀드 판매클래스별로 과세특징이 다르다고 들었는데 어떻게 다른가요?")

    assert signal is not None and "과세" in signal
    # 주제어 자체가 없으면 신호도 없다
    assert deterministic_miss_signal("점심 뭐 먹지") is None
