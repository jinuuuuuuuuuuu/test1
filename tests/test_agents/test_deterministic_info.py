from src.agents.deterministic_info import candidate_categories, deterministic_response_for


def test_tax_credit_question_candidates_both_calc_and_limit():
    # 후보 단계는 주제어("세액공제")만 보고 둘 다 낸다 — 확정은 router의 LLM 몫이다.
    candidates = candidate_categories("연금저축이랑 IRP 다 합쳐서 세액공제 얼마까지 되나요?")

    assert "세액공제_한도" in candidates
    assert "세액공제_계산_입력부족" in candidates


def test_tax_credit_limit_response_content():
    draft, context = deterministic_response_for(
        "세액공제_한도", "연금저축이랑 IRP 다 합쳐서 세액공제 얼마까지 되나요?"
    )

    assert "합산 900만원" in draft
    assert "1,500만원까지 세액공제되는 구조가 아니라" in draft
    assert context
    assert "연금저축+IRP 합산 900만원" in context[0]["content"]


def test_tax_benefit_overview_candidate_and_response():
    question = "연금계좌의 세금혜택에 대해 알려주세요"
    assert "세금혜택_개요" in candidate_categories(question)

    draft, context = deterministic_response_for("세금혜택_개요", question)
    assert "납입할 때 세액공제" in draft
    assert "운용 중 과세이연" in draft
    assert "연금으로 받을 때 낮은 세율" in draft
    assert "퇴직소득세 감면" in draft
    assert context
    assert "세액공제 대상 납입한도" in context[0]["content"]
    assert "1,500만원" in context[0]["content"]


def test_tax_credit_calculation_missing_response_content():
    question = "제가 받을 수 있는 세액공제 금액을 계산해 주세요."
    assert "세액공제_계산_입력부족" in candidate_categories(question)

    draft, context = deterministic_response_for("세액공제_계산_입력부족", question)
    assert "임의로 산출하지 않겠습니다" in draft
    assert "연금저축에 납입한 금액" in draft
    assert "IRP에 납입한 금액" in draft
    assert "총급여" in draft
    assert "종합소득금액" in draft
    assert context


def test_early_withdrawal_general_candidate_and_response():
    question = "IRP에서 중도인출은 어떤 경우에 가능한가요?"
    assert "중도인출_일반" in candidate_categories(question)

    draft, context = deterministic_response_for("중도인출_일반", question)
    assert "6개월 이상 요양" in draft
    assert "개인회생" in draft
    assert "무주택자 주택구입" in draft
    assert context


def test_default_option_auto_purchase_candidate_and_response():
    question = "디폴트옵션은 언제 자동으로 매수되나요?"
    assert "디폴트옵션_자동매수" in candidate_categories(question)

    draft, context = deterministic_response_for("디폴트옵션_자동매수", question)
    assert "4주(28일)" in draft
    assert "2주(14일)" in draft
    assert context


def test_default_option_existing_member_gets_only_that_case():
    draft, _ = deterministic_response_for(
        "디폴트옵션_자동매수", "저는 기존가입자인데 언제 자동매수되나요?"
    )
    assert "4주(28일)" in draft
    assert "신규가입자" not in draft


def test_default_option_new_member_gets_only_that_case():
    draft, _ = deterministic_response_for(
        "디폴트옵션_자동매수", "저는 신규가입자인데 언제 자동매수되나요?"
    )
    assert "최초 부담금 납입 다음 영업일" in draft
    assert "기존가입자" not in draft


def test_default_option_unspecified_asks_back_and_shows_both():
    draft, _ = deterministic_response_for(
        "디폴트옵션_자동매수", "디폴트옵션은 언제 자동으로 매수되나요?"
    )
    assert "어느 쪽에 해당하시는지" in draft
    assert "기존가입자" in draft and "신규가입자" in draft


def test_in_kind_transfer_block_candidate_and_response():
    question = "퇴직연금 실물이전이 안 되는 상품은 뭐가 있나요?"
    assert "실물이전_불가사유" in candidate_categories(question)

    draft, context = deterministic_response_for("실물이전_불가사유", question)
    assert "사모펀드" in draft
    assert "디폴트옵션" in draft
    assert context


def test_in_kind_transfer_draft_lists_every_block_code():
    """답변 본문이 근거(content)와 같은 원천을 써서 어떤 코드도 누락하지 않아야 한다.

    실측 사고: 답변 목록이 손으로 고른 11개 하드코딩 리스트였던 동안 "21. 만기(상환)"이
    빠져, 만기 때문에 이전이 막힌 사용자에게 근거에는 있는 사유가 답변에서는 안 보였다.
    """
    from src.rules.in_kind_transfer import TRANSFER_BLOCK_CODES

    draft, context = deterministic_response_for(
        "실물이전_불가사유", "퇴직연금 실물이전이 안 되는 상품은?"
    )

    missing = [code for code in TRANSFER_BLOCK_CODES if f"{code}." not in draft]
    assert missing == [], f"답변 본문에서 누락된 불가사유 코드: {missing}"
    assert "21. 만기(상환)" in draft
    # 근거에 있는 코드는 전부 답변에도 있어야 한다(근거-답변 불일치 방지).
    for code in TRANSFER_BLOCK_CODES:
        assert f"{code}." in context[0]["content"]


def test_pension_income_tax_candidate_and_response():
    question = "연금소득세 종합과세 기준 1500만원이 뭐야?"
    assert "연금소득세_종합과세" in candidate_categories(question)

    draft, context = deterministic_response_for("연금소득세_종합과세", question)
    assert "1,500만원" in draft
    assert "16.5%" in draft
    assert context


def test_no_candidates_for_unrelated_question():
    # 주제어 자체가 없으면 후보가 비어야 한다 — router가 LLM 호출 없이도 "해당없음"으로 확정 가능.
    assert candidate_categories("오늘 점심 메뉴 추천해줘") == []


def test_deterministic_response_for_returns_none_when_no_handler():
    assert deterministic_response_for("해당없음", "아무 질문") is None
    assert deterministic_response_for("존재하지않는카테고리", "아무 질문") is None


def test_candidate_hint_does_not_force_wrong_category_alone():
    """후보에 여러 카테고리가 있어도 candidate_categories 자체는 확정하지 않는다.

    "실물이전이 안 되는 상품은?"과 "디폴트옵션은 언제 매수되나요?"가 한 질문에 섞이는
    식의 혼동을 방지하는 최종 확정은 router LLM의 몫이며, 이 테스트는 후보 함수가 실제로
    "판정"이 아니라 "힌트"만 낸다는 계약을 문서화한다.
    """
    # "실물이전이 되는 상품은?"은 실물이전_불가사유 후보가 뜨지만(주제어만 봄),
    # 실제로는 허용 목록을 묻는 질문이라 router가 기각해야 하는 케이스 — 후보 단계에서는
    # 걸러지지 않는 것이 설계상 정상이다(2단계에서 LLM이 처리).
    assert "실물이전_불가사유" in candidate_categories("실물이전이 되는 상품은 뭐가 있나요?")


def test_age_based_tax_rate_uses_correct_bracket():
    """나이가 주어지면 그 나이의 구간 세율을 규칙엔진에서 확정해야 한다.

    실측 사고: "만 74세 → 3.3%"라고 답했으나 정답은 4.4%(70~80세 구간)였다. 근거 표에는
    정답이 있었는데 LLM이 구간을 잘못 골랐고, 수치 자체는 근거에 존재하므로 L0/L1 검증도
    통과해버렸다.
    """
    draft, context = deterministic_response_for(
        "연금소득세율_연령별", "제가 만 74세인데 연금 받으면 세율이 몇 퍼센트인가요?"
    )
    assert "4.4%" in draft
    assert "만 70세 이상 80세 미만" in draft
    assert context


def test_age_based_tax_rate_lifetime_annuity_overrides_age():
    draft, _ = deterministic_response_for(
        "연금소득세율_연령별", "만 74세이고 종신연금으로 받는데 세율이 몇 %인가요?"
    )
    assert "3.3%" in draft
    assert "연령과 무관" in draft


def test_age_based_tax_rate_without_age_shows_all_brackets_and_asks_back():
    """나이가 없으면 하나로 단정하지 말고 구간 전체 + 역질문을 낸다(답변 포기 아님)."""
    draft, _ = deterministic_response_for("연금소득세율_연령별", "연금 받으면 세율이 얼마인가요?")
    assert "5.5%" in draft and "4.4%" in draft and "3.3%" in draft
    assert "알려주세요" in draft


def test_age_based_tax_rate_always_notes_other_conditions():
    """나이가 특정돼도 종신연금·1,500만원 초과라는 다른 조건을 반드시 함께 고지해야 한다."""
    draft, _ = deterministic_response_for(
        "연금소득세율_연령별", "만 60세인데 연금 세율이 얼마인가요?"
    )
    assert "5.5%" in draft
    assert "종신연금" in draft
    assert "1,500만원" in draft


def test_age_based_tax_rate_candidate_matches_natural_phrasing():
    """"연금소득세"라는 정확한 단어 없이 물어도 후보로 잡혀야 한다."""
    assert "연금소득세율_연령별" in candidate_categories("연금 받으면 세율이 몇 퍼센트인가요?")
    assert "연금소득세율_연령별" in candidate_categories("만 74세인데 연금 세금이 얼마인가요?")
