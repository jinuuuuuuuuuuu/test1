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


def test_self_employed_tax_benefit_overview_uses_current_limits():
    question = "개인사업자인데 절세 방법 알려줘."

    assert "세금혜택_개요" in candidate_categories(question)
    draft, context = deterministic_response_for("세금혜택_개요", question)

    assert "연금저축·IRP를 활용한 절세" in draft
    assert "연금저축+IRP 합산 연 900만원" in draft
    assert "종합소득금액 4,500만원 이하이면 16.5%" in draft
    assert "700만원" not in draft
    assert "개인사업 대표는 퇴직연금에는 가입할 수 없지만" in context[0]["content"]


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


def test_tax_credit_calculation_runs_when_inputs_are_sufficient():
    question = "연금저축 600만원, IRP 300만원 넣었고 총급여 5,000만원이면 세액공제 얼마야?"

    draft, context = deterministic_response_for("세액공제_계산_입력부족", question)

    assert "추가로 필요한 정보" not in draft
    assert "900만원 x 16.5% = 148만 5천원" in draft
    assert "입력 조건에서는 세액공제 대상 납입액 900만원 x 16.5% = 148만 5천원" in context[0]["content"]


def test_tax_credit_limit_answers_pension_savings_only_excess_directly():
    draft, _ = deterministic_response_for("세액공제_한도", "연금저축 900만원 넣었는데 전부 세액공제 되나요?")

    assert "아니요" in draft
    assert "연금저축만으로는 600만원까지만 세액공제 대상" in draft
    assert "연금저축 단독 한도를 넘는 금액: 300만원" in draft


def test_personal_tax_question_uses_input_sufficiency_gate_before_general_tax_rules():
    question = "나 74세인데 세금 어떻게 내?"
    assert "개인세금_입력충분성" in candidate_categories(question)

    draft, context = deterministic_response_for("개인세금_입력충분성", question)

    assert "수령 방식" in draft
    assert "돈의 출처" in draft
    assert "4.4%" not in draft
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
    assert "만기일 + 29일" in draft
    assert "통지일 + 15일" in draft
    assert context


def test_early_withdrawal_medical_date_question_does_not_invent_exact_judgement():
    question = (
        "IRP 가입자입니다. 요양종료일이 2026년 1월 31일이고 신청일이 2026년 2월 28일이면 "
        "중도인출 신청기한 안인가요? 같은 조건에서 3월 1일이면 기한이 지난 건가요?"
    )
    assert "중도인출_기한판정" in candidate_categories(question)

    draft, context = deterministic_response_for("중도인출_기한판정", question)

    assert "요양종료일로부터 1개월 이내" in draft
    assert "DB 근거만으로 정확히 판정하지 않겠습니다" in draft
    assert "2026년 2월 28일 신청: 신청기한 안" not in draft
    assert "2026년 3월 1일 신청: 신청기한이 지난" not in draft
    assert "30일" not in draft
    assert "12.5" not in draft
    assert context


def test_early_withdrawal_home_purchase_deadline_question_gets_calendar_deadline():
    question = "중도인출 시 주택구입은 달력 기준 1개월 이내 신청 요건을 가지잖아. 만약 3월 1일에 한 경우 언제까지야?"
    assert "중도인출_기한판정" in candidate_categories(question)

    draft, context = deterministic_response_for("중도인출_기한판정", question)

    # 연도 없이 "3월 1일"이라고만 하면 기준일을 확정할 수 없으므로 규정을 안내한다 —
    # 연도를 임의로 채워 "2026년 4월 1일까지"라고 답하면 지어낸 정보가 된다.
    assert "소유권 이전 등기접수일로부터 1개월 이내" in draft
    assert "정확한 날짜로 계산하는 방식" in draft
    assert "대표적인 중도인출 사유" not in draft
    assert context


def test_early_withdrawal_home_purchase_general_deadline_question_is_specific():
    question = "주택구입 중도인출은 언제까지 신청해야 해?"
    assert "중도인출_기한판정" in candidate_categories(question)

    draft, context = deterministic_response_for("중도인출_기한판정", question)

    assert "소유권 이전 등기접수일로부터 1개월 이내" in draft
    assert "정확한 날짜로 계산하는 방식" in draft
    assert "대표적인 중도인출 사유" not in draft
    assert "6개월 이상 요양" not in draft
    assert context


def test_default_option_existing_member_gets_only_that_case():
    draft, _ = deterministic_response_for(
        "디폴트옵션_자동매수", "저는 기존가입자인데 언제 자동매수되나요?"
    )
    assert "만기일 + 29일" in draft
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


# ── 실물이전 개별판정 (task_type 공통화, 2026-08-27) ────────────────────
#
# "실물이전이 안 되는 경우는?"(목록답변)과 "MMF인데 옮길 수 있나요?"(개별판정)는
# 같은 도메인이지만 다른 작업이다. 개별판정 경로가 없던 동안 라우터가 이런 질문을
# 기각했고, LLM이 자유롭게 툴을 고르다 실물이전 질문에 투자 가능 여부 툴을 호출해
# "네, 실물이전 문제없이 진행 가능합니다"라는 정반대 답을 냈다.


def test_transfer_judgement_answers_no_for_blocked_product():
    """불가 상품은 '아니요'로 시작해야 한다 — '네, 불가능합니다'는 반대로 읽힌다."""
    draft, context = deterministic_response_for("실물이전_개별판정", "MMF인데 실물이전 옮길 수 있나요?")

    assert draft.startswith("아니요")
    assert "MMF" in draft
    assert context


def test_transfer_judgement_recognizes_colloquial_expressions():
    """코드표 name 그대로가 아닌 구어 표현도 인식해야 한다."""
    for question in (
        "만기가 이미 도래한 상품인데 실물이전 되나요?",
        "사모펀드 실물이전 되나요?",
        "디폴트옵션 상품도 옮길 수 있나요?",
    ):
        result = deterministic_response_for("실물이전_개별판정", question)
        assert result is not None, question
        assert result[0].startswith("아니요"), question


def test_transfer_judgement_returns_none_when_product_not_identified():
    """상품이 특정되지 않으면 개별판정하지 않고 목록/일반 경로로 넘긴다."""
    assert deterministic_response_for("실물이전_개별판정", "실물이전이 안 되는 경우는?") is None


def test_transfer_judgement_cites_source_code():
    """근거에 코드 번호와 설명이 함께 담겨야 답변이 지어내지 않는다."""
    _, context = deterministic_response_for("실물이전_개별판정", "MMF인데 실물이전 되나요?")

    assert "04" in context[0]["content"]
    assert "MMF" in context[0]["content"]


def test_both_transfer_categories_are_candidates():
    """목록/개별판정 둘 다 후보로 올라가야 라우터가 고를 수 있다."""
    candidates = candidate_categories("MMF인데 실물이전 되나요?")

    assert "실물이전_불가사유" in candidates
    assert "실물이전_개별판정" in candidates


# ── 후보 목록 확장 (2026-08-27) ────────────────────────────────────────

def test_age_tax_candidates_survive_paraphrase():
    """같은 의도의 다른 표현에서도 후보가 나와야 한다.

    "연금 + 세율" AND 조건을 요구하던 시절, 7개 표현 중 3개가 후보 0건이었다.
    후보를 놓치면 라우터는 그 카테고리를 고려조차 못 한다.
    """
    for question in (
        "나 74세인데 세금 어떻게 내?",
        "74세인데 얼마나 떼나요?",
        "제 나이가 74인데 세율 알려주세요",
        "만 74세인데 연금 세율이 몇 %인가요?",
    ):
        assert "연금소득세율_연령별" in candidate_categories(question), question


def test_personal_tax_candidates_positive_negative_collision():
    positive = [
        "나 74세인데 세금 어떻게 내?",
        "내 연금 세금 계산해줘",
        "74세 세율이 궁금한데 제 경우 실제 얼마 내나요?",
    ]
    for question in positive:
        assert "개인세금_입력충분성" in candidate_categories(question), question

    negative = [
        "연령별 연금소득세율 표 알려줘",
        "연금소득세 제도를 설명해줘",
        "70세 이상 세율 기준이 어떻게 돼?",
    ]
    for question in negative:
        assert "개인세금_입력충분성" not in candidate_categories(question), question


def test_broadened_candidates_do_not_overtrigger():
    """넓혔다고 무관한 질문까지 후보가 생기면 안 된다."""
    for question in ("IRP가 뭔가요?", "솔로몬 국공채 위험등급 알려줘", "DC와 DB 차이가 뭔가요?"):
        assert candidate_categories(question) == [], question


def test_withdrawal_deadline_covers_every_reason_without_exact_dates():
    """5개 사유 전부 같은 경로로 기한 규칙을 답해야 한다 — 사유별 핸들러 시절의 비대칭 방지.

    요양·주택구입만 전용 핸들러가 있던 동안 전월세·재난·개인회생 날짜 질문은
    사유 목록 답변으로 빠졌다.
    """
    cases = {
        "요양": "요양종료일이 2026년 1월 31일인데 중도인출 언제까지 신청하나요?",
        "전월세": "전월세보증금 잔금을 2026년 1월 31일에 지급했는데 중도인출 언제까지 신청 가능한가요?",
        "주택구입": "소유권 이전 등기접수일이 2026년 3월 20일인데 중도인출 기한이 언제까지인가요?",
        "재난": "재난피해가 2026년 5월 10일에 발생했는데 중도인출 언제까지 신청할 수 있나요?",
        "개인회생": "개인회생 결정일이 2026년 3월 10일인데 중도인출 언제까지 신청 가능한가요?",
    }
    expected_rules = {
        "요양": "요양종료일로부터 1개월 이내",
        "전월세": "잔금지급일로부터 1개월 이내",
        "주택구입": "소유권 이전 등기접수일로부터 1개월 이내",
        "재난": "피해발생일로부터 3개월 이내",
        "개인회생": "개인회생절차개시 결정일 또는 파산선고일로부터 5년 이내",
    }
    for label, question in cases.items():
        assert "중도인출_기한판정" in candidate_categories(question), label
        result = deterministic_response_for("중도인출_기한판정", question)
        assert result is not None, label
        assert expected_rules[label] in result[0], f"{label}: {result[0][:160]}"
        assert "정확한 날짜로 계산하는 방식" in result[0], label


def test_withdrawal_deadline_does_not_judge_multiple_request_dates_from_db_only():
    """신청일 후보가 여럿이어도 DB에 없는 exact-date 계산으로 판정하지 않는다."""
    question = (
        "요양종료일이 2026년 1월 31일이고 신청일이 2026년 2월 28일이면 중도인출 신청기한 "
        "안인가요? 같은 조건에서 3월 1일이면 기한이 지난 건가요?"
    )
    draft, _ = deterministic_response_for("중도인출_기한판정", question)

    assert "요양종료일로부터 1개월 이내" in draft
    assert "DB 근거만으로 정확히 판정하지 않겠습니다" in draft
    assert "신청기한 안에 들어갑니다" not in draft
    assert "신청기한이 지난" not in draft


def test_withdrawal_eligibility_answers_db_plan_directly():
    draft, _ = deterministic_response_for("중도인출_요건판정", "DB형 퇴직연금인데 전월세보증금 때문에 중도인출 가능해?")

    assert "DB형 퇴직연금은 중도인출이 허용되지 않습니다" in draft
    assert "대표적인 중도인출 사유" not in draft


def test_withdrawal_eligibility_rejects_personal_workout():
    draft, _ = deterministic_response_for("중도인출_요건판정", "개인워크아웃 중인데 퇴직연금 중도인출 가능한가요?")

    assert "개인워크아웃이나 신용회복은 퇴직연금 중도인출 사유에 해당하지 않습니다" in draft
    assert "대표적인 중도인출 사유" not in draft


# ── 정형 답변의 적합성 게이트 (2026-08-27 구조 수정) ──────────────────
#
# 원래 설계는 "후보 생성 → 라우터 확정 → 핸들러 실행(신뢰)"이라, 핸들러 13개 중 10개는
# 반환 타입에 None이 없어 물러날 방법 자체가 없었다("오늘 점심 뭐 먹지"에도 세액공제
# 한도표를 반환). 라우터가 유일한 관문일 때는 성립했지만, 라우터를 우회하는 코드
# 오버라이드(router._restore_rejected_category)가 생기면서 검증 없이 정형 답변이
# 나가는 구멍이 열렸다.
#
# 핸들러 13개에 각각 가드를 붙이는 대신 dispatch 한 곳에서 후보 목록을 재확인한다 —
# 핸들러가 몇 개든, 앞으로 몇 개가 더 생기든 같은 보호를 받는다.


def test_no_category_answers_unrelated_questions():
    """어떤 카테고리도 무관한 질문에 정형 답변을 내면 안 된다.

    이 테스트는 카테고리 목록 전체를 훑으므로, 새 카테고리가 추가돼도 자동으로
    같은 검사를 받는다(사람이 목록을 관리할 필요가 없다).
    """
    from src.agents.deterministic_info import DETERMINISTIC_CATEGORIES

    unrelated = [
        "오늘 점심 뭐 먹지",
        "삼성전자 주가 얼마야?",
        "부동산 양도세 계산해주세요",
        "여행지 추천해줘",
        "파이썬 리스트 정렬 방법 알려줘",
    ]
    leaking = [
        (category, question)
        for category in DETERMINISTIC_CATEGORIES
        if category != "해당없음"
        for question in unrelated
        if deterministic_response_for(category, question) is not None
    ]

    assert leaking == [], f"무관 질문에 정형 답변을 내는 카테고리: {leaking}"


def test_dispatch_rejects_category_outside_candidates():
    """후보에 없는 카테고리로 호출하면 핸들러를 실행하지 않는다."""
    # "세액공제 한도"는 중도인출 질문의 후보가 아니다.
    assert deterministic_response_for("세액공제_한도", "중도인출이 어떤 경우에 가능한가요?") is None


def test_dispatch_still_runs_for_valid_category():
    """정상 경로는 그대로 동작해야 한다 (과잉 차단 방지)."""
    assert deterministic_response_for("세액공제_한도", "세액공제 한도가 얼마인가요?") is not None
    assert deterministic_response_for("중도인출_일반", "중도인출이 어떤 경우에 가능한가요?") is not None


def test_age_rate_category_requires_age_or_receipt_context():
    """연령별 세율은 나이나 '연금 수령' 문맥이 있어야 후보가 된다.

    "연금"이라는 단어만으로 후보에 넣으면 "연금저축 600만원 납입하고 총급여
    5000만원인데 세액공제 얼마?"가 나이도 없이 연령별 세율표를 답하게 된다(실측).
    """
    assert "연금소득세율_연령별" not in candidate_categories(
        "연금저축 600만원 납입하고 총급여 5000만원인데 세액공제 얼마?"
    )
    assert "연금소득세율_연령별" in candidate_categories("나 74세인데 세금 어떻게 내?")
    assert "연금소득세율_연령별" in candidate_categories("연금 수령할 때 세금 얼마나 나가요?")


def test_candidates_cover_paraphrases_without_domain_keyword():
    """제도명을 그대로 쓰지 않는 표현도 후보로 잡아야 한다."""
    assert "디폴트옵션_자동매수" in candidate_categories("저는 기존가입자인데 언제 자동매수되나요?")
    assert "실물이전_개별판정" in candidate_categories("디폴트옵션 상품도 옮길 수 있나요?")


def test_composite_withdrawal_deadline_documents_and_tax_are_answered_together():
    question = "전세보증금 때문에 IRP 중도인출하려고 하는데, 언제까지 신청해야 하고 필요한 서류랑 세금은 어떻게 되나요?"

    candidates = candidate_categories(question)
    assert candidates[0] == "복합정보_태스크플랜"

    draft, context = deterministic_response_for("복합정보_태스크플랜", question)

    assert "**가능 여부**" in draft
    assert "**신청기한**" in draft
    assert "**필요서류**" in draft
    assert "**세금**" in draft
    assert "잔금지급일" in draft
    assert "임대차계약서" in draft or "전월세계약서" in draft
    assert "16.5%" in draft
    assert "수령 방식" not in draft
    assert {item["source"] for item in context} >= {
        "doc46~doc50 중도인출 요건판정 규칙",
        "doc48 중도인출 무주택 전월세보증금 필요서류",
        "doc40 중도인출 사유별 세법상 부득이한 사유 및 과세 규칙",
    }


def test_composite_retirement_benefit_split_tax_answers_each_part():
    question = "퇴직금을 IRP로 받은 뒤 일부는 중도인출하고 나머지는 연금으로 받으려고 해요. 각각 세금이 어떻게 되나요?"

    assert candidate_categories(question)[0] == "복합정보_태스크플랜"
    draft, context = deterministic_response_for("복합정보_태스크플랜", question)

    assert "중도인출하는 퇴직금 부분" in draft
    assert "나머지를 연금으로 받는 부분" in draft
    assert "연금외수령" in draft
    assert "감면 없이" in draft
    assert "연금실제수령연차" in draft
    assert "1~10년차" in draft and "11~20년차" in draft and "21년차" in draft
    assert context[0]["source"] == "doc39~doc40 이연퇴직소득세 감면 규칙"


def test_composite_house_purchase_db_type_still_includes_deadline_and_documents():
    question = "무주택자인데 집을 사려고 퇴직연금 중도인출하려고 해요. 신청기한, 필요한 서류, DB형에서도 가능한지 알려주세요."

    draft, context = deterministic_response_for("복합정보_태스크플랜", question)

    assert "DB형 퇴직연금은 중도인출이 허용되지 않습니다" in draft
    assert "무주택 주택구입" in draft
    assert "소유권 이전 등기접수일" in draft
    assert "매매계약서" in draft
    assert "전월세보증금 같은" not in draft
    assert {item["source"] for item in context} >= {
        "doc46~doc50 중도인출 요건판정 규칙",
        "doc49 중도인출 무주택 주택구입 필요서류",
    }
