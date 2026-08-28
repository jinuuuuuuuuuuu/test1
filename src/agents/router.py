"""① 라우터/가드레일 노드 — 질문을 정보형/상품형(복합형은 둘 다)으로 다중분류하고
안전성을 사전 필터링한다.

복합형일 때는 항상 ②정보 Agent → ③상품 Agent 순차 실행이다 (병렬 아님) — ③이 ②의
retrieved_context를 State에서 그대로 읽어 쓸 수 있어야 하기 때문 (설계 결정, 2026-08-11).

⚠️ Structured Outputs는 네이버 공식 문서 기준 HCX-007에서만 지원된다(HCX-DASH-002는
"Unsupported function" 에러). HCX-007은 기본적으로 Thinking이 켜져 있어 Structured
Outputs와 동시 사용이 안 되므로 thinking_effort="none"으로 꺼서 써야 한다.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.agents.context import format_conversation_history
from src.agents.deterministic_info import (
    CODE_OVERRIDABLE_CATEGORIES,
    candidate_categories,
    deterministic_response_for,
)
from src.agents.llm import get_llm, invoke_with_retry
from src.agents.state import PensionAgentState
from src.storage.queries import find_asset_overlap

ROUTER_MODEL = "HCX-007"

ROUTER_SYSTEM_PROMPT = """당신은 연금 상담 AI의 질문 분류 게이트입니다. 사용자 질문을 보고 다음을 판단하세요.

1. intent (해당하는 것 전부 선택, 복합형이면 둘 다):
   - 정보형: 연금 제도, 세금(세액공제/연금소득세/퇴직소득세), 인출·이전 규정 등 지식성 질문
   - 상품형: 특정 펀드/상품 추천, 비교, 매수 가능 여부 등 상품 관련 질문
   (예: "IRP에 넣을 펀드 추천해줘"는 상품 추천이지만 계좌 유형별 투자한도 판단이 필요하므로
   상품형만으로 충분합니다 — 투자한도는 상품 Agent가 자체 툴로 확인합니다.)

   ⚠️ "펀드"라는 단어가 들어갔다고 무조건 상품형이 아닙니다. 특정 상품을 지목하지 않고
   제도·규정·절차 일반을 묻는 질문은 "펀드"가 언급돼도 정보형입니다 — 상품 Agent는 개별
   상품 데이터만 조회할 수 있어 제도 근거를 찾을 수단이 없고, 그런 질문이 상품형으로 가면
   임의의 펀드 몇 개를 뽑아 그 데이터로 제도 일반론을 지어내게 됩니다.
   - 정보형(제도 질문): "연금저축 펀드 환매 제한기간이 있나요", "펀드 환매수수료가 뭔가요",
     "연금저축 펀드는 중도인출이 되나요" — 규정 자체를 묻는 것이라 상품명이 없어도 답이 나온다
   - 상품형(상품 질문): "○○펀드 환매수수료가 얼마인가요", "수익률 높은 펀드 추천해줘" —
     특정 상품이 지목됐거나 상품 후보를 골라야 답이 나온다

   [보유 상품 조회] 힌트를 함께 보세요. 조회 결과가 비어 있고 질문이 상품 추천·비교를
   요구하지도 않는다면 상품형을 붙이지 마세요 — 상품 Agent가 할 일이 없는데 실행되면
   임의의 펀드를 뽑아 근거로 삼는 경로가 열립니다. 특히 "퇴직금을 언제 얼마나 인출할 수
   있고 세금은 얼마인가" 같은 질문은 제도·세제 계산이라 정보형만으로 충분합니다.

2. scope: 이 서비스의 상담 범위는 ⓐ 연금 제도(DB/DC/IRP·연금저축, 디폴트옵션·실물이전·
   중도인출 등), ⓑ 연금 세제(세액공제·연금소득세·퇴직소득세 등), ⓒ 연금계좌에서 투자하는
   상품(펀드)의 설명·비교·추천 **및 상품 속성 일반론**입니다. 질문이 이 범위에 속하는지
   판정하세요.

   ⓒ에는 특정 상품을 지목한 질문뿐 아니라 **상품 데이터 항목 자체를 묻는 일반 질문**도
   포함됩니다 — 우리 상품 DB가 판매클래스(온라인/오프라인)·총보수·수익률·위험등급·
   환매수수료·설정일·시장잔고를 보유하고 있어 상품명 없이도 답할 수 있기 때문입니다.
   (예: "온라인 클래스와 오프라인 클래스는 보수가 얼마나 차이 나나요?", "위험등급 6등급이
   1등급보다 안전한가요?", "총보수가 뭔가요?" → 모두 범위내)

   ★ 판정 원칙 — "보유 데이터로 답할 수 있으면 거부하지 않는다":
   범위 판정은 "연금스러운 표현인가"가 아니라 "우리 자료로 답할 수 있는가"입니다.
   [보유 상품 조회] 힌트가 함께 주어지는데, 이는 질문에 언급된 이름이 **우리 상품 DB에
   실재하는지 코드가 직접 조회한 결과**입니다. 여기에 상품이 나왔다면 그 상품의
   위험등급·총보수·수익률·자산유형·판매클래스를 실제로 조회해 답할 수 있으므로,
   질문에 "연금"·"IRP" 같은 제도 단어가 없더라도 반드시 "범위내"입니다.
   사용자는 상품명만 알고 제도 용어를 모를 수 있으며, 그런 질문을 거부하면 답할 수 있는
   질문을 못 답한 것이 됩니다. 조회 결과가 비어 있을 때만 아래 기준으로 판단하세요.

   - 범위내: 질문 전체가 위 범위에 속함
   - 부분관련: 질문의 핵심은 범위 밖이지만 연금 관점에서 답할 가치가 있는 부분이 있음.
     scope_note에 연금 관점에서 답할 방향을 적으세요. (예: "부모님이 개인 사업을 하셔,
     절세법 알려줘" → 사업소득 절세 일반론은 범위 밖이지만, 종합소득이 있는 개인사업자의
     연금저축·IRP 세액공제는 안내 가능 → scope_note: "개인사업자(종합소득자)의 연금계좌
     세액공제 관점으로 답변")
   - 범위외: 연금과 접점이 없음 (예: 주식 종목 추천, 부동산 양도세, 일반 상식).
     scope_note에 짧은 사유를 적으세요.

3. is_safe: 아래에 해당하는 경우에만 False로 표시하세요. 애매하면 True로 두고 넘기세요 —
   과도한 차단이 더 큰 문제입니다.
   - 확정 수익률/원금 보장을 요구하거나 암시하는 질문
   - 탈세(소득 은닉, 허위서류 등 명백히 불법인 방법)를 구체적으로 묻는 질문
   - 개인정보·명의도용 관련 요청. **주민등록번호·계좌번호·카드번호 등 민감정보를 질문에
     직접 적어 보내거나, 그 정보로 본인 가입내역·잔고를 조회해 달라는 요청은 "애매한 경우"가
     아니라 명확한 차단 대상입니다** — 우리는 개인 계좌 데이터를 보유하지 않아 조회 자체가
     불가능하고, 민감정보를 입력받는 흐름을 열어주면 안 되기 때문입니다.
     (예: "제 주민등록번호는 900101-1234567인데 제 연금 가입 내역을 조회해 주세요"
      → is_safe=False, safety_reason에 조회 불가·정보 미저장 사유를 적으세요)

   ⚠️ 절세(세액공제·과세이연·연금수령 시기 조정처럼 세법이 허용하는 합법적 세금 최적화)는
   이 서비스의 핵심 주제이며 전혀 위험하지 않습니다 — "절세법 알려주세요", "세금 어떻게
   줄여요?" 같은 질문에 is_safe=False를 주면 안 됩니다. 질문에 "세금감면이 어마어마하다던데"
   처럼 과장되거나 잘못된 전제가 섞여 있어도 그 자체로는 차단 사유가 아닙니다 — 그 전제를
   바로잡아 답하면 되는 정상적인 정보형/종합 질문입니다.

4. deterministic_category: [후보 카테고리] 목록으로 힌트가 함께 주어집니다. 이 힌트는
   질문에 특정 주제어(예: "세액공제")가 있다는 것만 알려줄 뿐, 그 카테고리가 실제로 맞다는
   보장이 아닙니다 — 반드시 질문의 실제 의도를 읽고 아래 기준으로 확정하거나 기각하세요.
   후보가 비어 있으면 무조건 "해당없음"입니다. 후보가 있어도 아래 기준에 안 맞으면
   "해당없음"을 선택하세요 (과잉 확정보다 기각이 안전합니다 — 기각되면 일반 검색·추론
   경로로 넘어가 답변 자체는 계속 시도됩니다).

   ★ 모든 카테고리에 우선 적용되는 판정 원칙 — "일반형만 확정한다":
   이 카테고리들의 정형 답변은 "제도 일반론"(한도표·사유 목록·절차 요약)입니다. 따라서
   질문이 사용자 개인의 구체적 상황·조건을 제시하고 그에 대한 판정·계산을 요구하면,
   후보에 있더라도 반드시 "해당없음"으로 기각하세요. 그런 질문은 전용 판정·계산 툴
   (세액공제 계산기, 중도인출 사유 판정기, 실물이전 가능여부 판정기 등)이 정확히 답할 수
   있는데, 여기서 정형 답변으로 확정해버리면 그 툴이 아예 호출되지 못하고 사용자는
   자기 상황과 무관한 일반 목록만 받게 됩니다.
   - 개인 상황 신호의 예: "제가 가진/보유 중인", "제 소득이 ~인데", "저는 ~형인데",
     "MMF인데", "만기가 됐는데", "요양 때문에", "5년차인데", 구체적 금액·나이·날짜 제시 등
   - 반대로 "~은 무엇인가요", "어떤 경우에 ~되나요", "한도가 얼마인가요"처럼 제도 자체를
     묻는 질문은 개인 상황이 없으므로 정상적으로 해당 카테고리로 확정합니다.

   - 세액공제_계산_입력부족: "얼마 받을 수 있나요/계산해주세요"처럼 본인 세액공제액을
     계산해달라는 의도이지만, 질문에 (연금저축 또는 IRP 납입액) + (총급여 또는
     종합소득금액) 중 하나 이상이 빠져 있는 경우.
     ⚠️ 납입액과 소득금액이 질문에 둘 다 숫자로 나와 있으면 이 카테고리가 아니라
     "해당없음"입니다 — 계산 가능한 질문은 정형 답변으로 가로채지 말고 계산 툴로
     넘겨야 합니다.
   - 세액공제_한도: 세액공제 대상 납입한도(600만원/900만원)나 공제율이 몇 %인지 등
     제도 자체의 한도·기준을 묻는 질문 (본인 수치를 대입한 계산 요청이 아님).
   - 세금혜택_개요: 연금계좌의 세금 혜택 전반을 개괄적으로 설명해달라는 질문.
   - 개인세금_입력충분성: 사용자가 본인 나이·수령금액·재원·수령방식 일부를 제시하고
     "세금은/얼마 내/계산"처럼 개인 세금 판정·계산을 묻는 질문. 이 카테고리는 입력값을
     임의 가정하지 않고, 충분하면 규칙으로 세율·세액을 계산하며 부족하면 필요한 정보만 묻습니다.
   - 중도인출_기한판정: 중도인출 **신청기한**을 묻거나("언제까지 신청해야 하나요"),
     특정 날짜가 기한 안인지 판정을 요구하는 질문("2월 28일에 신청하면 되나요").
     5개 사유(요양·개인회생파산·전월세보증금·주택구입·재난피해) 모두 이 카테고리입니다.
     제공 DB가 확인하는 기준일·기간 규칙까지만 안내하고, DB에 명시되지 않은 exact date
     환산이나 신청일 통과 여부는 단정하지 않습니다. ⚠️ 사유별 가능 여부·요건을 묻는
     질문(기한과 무관)은 여기가 아닙니다.
   - 중도인출_요건판정: "DB형인데 중도인출 가능한가요", "개인워크아웃 중인데 가능한가요",
     "전월세보증금 때문에 가능한가요"처럼 사용자의 제도유형·사유 조건이 주어지고 가능/불가
     판정을 묻는 질문. 목록을 나열하지 말고 그 조건에 대해 판정합니다.
   - 중도인출_일반: "중도인출이 어떤 경우에 가능한가요"처럼 중도인출 가능 사유 전체
     목록을 묻는 질문. ⚠️ 질문이 이미 요양/개인회생/파산/전월세보증금/주택구입/재난피해
     같은 특정 사유 하나를 콕 짚어 그 사유로 가능한지 묻고 있다면 이 카테고리가 아니라
     "해당없음"입니다(특정 사유 질문에 5개 사유 전체를 나열하면 안 됩니다). "중도인출
     제도가 언제 도입됐나요" 같은 연혁 질문도 "해당없음"입니다.
   - 디폴트옵션_자동매수: 디폴트옵션이 언제/어떤 절차로 자동매수되는지 묻는 질문.
     ⚠️ "실물이전이 안 되는 상품"의 예시로 디폴트옵션이 언급된 것처럼 다른 주제의
     일부로만 나온 경우는 "해당없음"입니다.
   - 실물이전_불가사유: 실물이전이 안 되거나 제한되는 상품·사유에 어떤 것들이 있는지
     **목록**을 묻는 질문 (보유 상품이 특정되지 않음).
     ⚠️ "실물이전이 되는 상품은?"처럼 허용되는 것을 묻거나, 실물이전 절차 자체를 묻는
     질문은 "해당없음"입니다(이 카테고리는 불가사유 전용입니다).
   - 실물이전_개별판정: "MMF인데 이전 되나요", "만기가 됐는데 왜 안 되죠", "사모펀드도
     옮길 수 있나요"처럼 **보유 상품의 종류·상태가 제시된** 실물이전 가능여부 질문.
     목록을 나열하는 게 아니라 그 상품에 대해 가능/불가를 판정합니다.
   - 연금수령한도: 연금수령한도 계산식이나 그 개념 자체를 묻는 질문.
   - 퇴직소득세감면: 퇴직금을 연금으로 받을 때 이연퇴직소득세 감면율을 묻는 질문.
     ⚠️ 연금소득세(사적연금소득 종합과세) 질문과 혼동하지 마세요 — 퇴직소득세감면은
     "퇴직소득세"/"이연퇴직소득세"라는 단어가 명시된 경우만입니다.
   - 연금소득세_종합과세: 사적연금소득이 연 1,500만원을 넘을 때 종합과세/분리과세
     선택 기준을 묻는 질문.
   - 연금소득세율_연령별: 연금을 받을 때 적용되는 연금소득세율이 몇 %인지 묻는 질문
     (예: "연금 받으면 세율이 얼마인가요", "만 74세인데 몇 % 떼나요").
     ⚠️ 이 카테고리는 나이가 질문에 있어도 기각하지 않습니다 — 다른 카테고리와 달리
     정형 응답이 나이를 직접 읽어 해당 구간 세율을 확정하고, 종신연금·1,500만원 초과
     여부 같은 남은 조건은 분기로 안내하기 때문입니다.
     ⚠️ 다만 "퇴직금을 연금으로 받을 때 세금"을 묻는 것이면 퇴직소득세감면입니다.
   - 복합정보_태스크플랜: 한 질문 안에 서로 다른 정보 작업이 2개 이상 명시된 경우.
     예: 중도인출 신청기한+필요서류+세금, DB형 가능여부+주택구입 신청기한+서류,
     퇴직금 중도인출분 세금+나머지 연금수령 세금. 이 경우 단일 세금/기한/요건
     카테고리로 전체 질문을 대표하지 말고, 복합정보_태스크플랜을 우선 선택합니다.
   - 해당없음: 위 어디에도 명확히 안 속하거나, 판단이 애매한 경우.

[이전 대화]가 함께 주어지면, "그거 다시 설명해줘", "방금 말한 상품 중 두 번째는?"처럼 현재
질문만 봐서는 의도가 불분명한 후속 질문을 이전 대화 맥락으로 해석해서 분류하세요. 이전 대화가
없으면 현재 질문만으로 판단하세요.
"""


class RouterDecision(BaseModel):
    """사용자 질문의 의도 분류와 안전성 판정 결과."""

    intent: list[Literal["정보형", "상품형"]] = Field(
        description="해당하는 의도 전부. 정보+상품이 모두 필요하면 둘 다 포함(복합형)."
    )
    scope: Literal["범위내", "부분관련", "범위외"] = Field(
        default="범위내",
        description="질문이 연금 상담 범위(제도/세제/연금계좌 상품)에 속하는지 판정",
    )
    scope_note: Optional[str] = Field(
        default=None,
        description="부분관련이면 연금 관점에서 답할 방향, 범위외면 짧은 사유 (범위내면 비움)",
    )
    is_safe: bool = Field(description="질문이 안전 가이드라인을 위반하지 않으면 True")
    safety_reason: Optional[str] = Field(default=None, description="is_safe=False일 때만 사유를 적는다")
    deterministic_category: Literal[
        "복합정보_태스크플랜",
        "세액공제_계산_입력부족",
        "세액공제_한도",
        "세금혜택_개요",
        "개인세금_입력충분성",
        "중도인출_기한판정",
        "중도인출_요건판정",
        "중도인출_일반",
        "디폴트옵션_자동매수",
        "실물이전_불가사유",
        "실물이전_개별판정",
        "연금수령한도",
        "퇴직소득세감면",
        "연금소득세_종합과세",
        "연금소득세율_연령별",
        "해당없음",
    ] = Field(
        default="해당없음",
        description="후보 힌트를 참고해 확정한 정형 답변 카테고리. 후보가 비었거나 안 맞으면 해당없음.",
    )


def _apply_asset_scope_override(
    scope: str, scope_note: Optional[str], intent: list[str], matched_funds: list[str]
) -> tuple[str, Optional[str], list[str]]:
    """보유 상품이 조회된 질문을 '범위외'로 판정한 결과를 코드가 뒤집는다.

    ④검증의 L0 오버라이드와 같은 사상이다 — 반드시 지켜야 하는 것은 프롬프트 순종에
    맡기지 않는다. 다만 오버라이드 조건은 **코드가 사실로 확정한 경우로만** 한정한다:
    상품 DB 조회에서 실제 매칭이 나왔다면 그 상품의 위험등급·총보수·수익률을 조회해
    답할 수 있다는 뜻이므로, "답할 수 없다"는 판정은 사실과 다르다.

    범위외 오판은 답할 수 있는 질문을 통째로 거부하게 만들어 요구사항 충족이 0점이 된다
    (실측: 대회 공식 참고 질의 "솔로몬 국공채 3종 비교"가 3/3 거부됨). 반대로 조회가
    비었을 때는 개입하지 않는다 — 게이트를 무력화하면 범위 밖 질문까지 통과한다.
    """
    if not matched_funds or scope != "범위외":
        return scope, scope_note, intent

    note = (
        f"질문에 언급된 상품이 보유 DB에 있어 상담 범위로 확정 "
        f"(조회 결과: {', '.join(matched_funds[:3])})"
    )
    # 상품을 지목한 질문이므로 상품 Agent가 처리해야 한다. 범위외 판정과 함께 intent가
    # 비어 나오는 경우가 있어(판정을 포기한 상태) 여기서 함께 보정한다.
    return "범위내", note, (intent or ["상품형"])


def _drop_product_intent_for_deterministic_info(
    intent: list[str], category: str, matched_funds: list[str]
) -> list[str]:
    """정형 제도 답변으로 확정된 질문에서 불필요한 상품형을 떼어낸다.

    실측: "MMF인데 실물이전 옮길 수 있나요?"가 deterministic_category=실물이전_개별판정
    으로 정확히 확정되고도 intent=['정보형','상품형']이 붙어 ③상품 Agent까지 실행됐다.
    그 결과 최종 답변에 "[제도·세제 관련 답변] ... [상품 관련 답변] ..."이 함께 붙어
    같은 질문에 두 번 답하는 형태가 됐다.

    상품 유형 단어(MMF·펀드 등)가 들어갔다는 이유로 상품형이 붙지만, 정형 카테고리가
    확정됐다는 것은 이 질문이 제도 판정으로 답해진다는 뜻이다. 보유 상품 조회까지
    비어 있으면 ③이 할 일이 없다 — 실행되면 임의의 펀드를 근거로 끌어올 뿐이다.
    프롬프트에 같은 취지의 지시가 있으나 지켜지지 않아 코드로 강제한다.
    """
    if category == "해당없음" or matched_funds:
        return intent
    remaining = [i for i in intent if i != "상품형"]
    return remaining or ["정보형"]


def _restore_rejected_category(category: str, candidates: list[str], question: str) -> str:
    """라우터가 잘못 기각한 정형 카테고리를 코드가 되살린다 — 안전이 확인된 것만.

    실측: "개인세금_입력충분성"이 후보로 정확히 주어져도 3/3 "해당없음"으로 기각됐고,
    프롬프트 문구 조정으로는 안정적으로 고쳐지지 않았다(상위 판정 원칙 블록을 어떤
    형태로 남겨도 재현). 프롬프트 순종에 의존할 수 없는 유형이라 코드로 강제한다.

    적용 범위는 CODE_OVERRIDABLE_CATEGORIES로 좁힌다. 핸들러가 "이 카테고리가 실제로
    맞는 상황인지"를 스스로 판단하는 카테고리만 담겨 있어(아니면 None), 되살려도
    엉뚱한 정형 답변이 나가지 않는다.
    """
    if category != "해당없음":
        return category
    for candidate in candidates:
        if candidate not in CODE_OVERRIDABLE_CATEGORIES:
            continue
        if deterministic_response_for(candidate, question) is not None:
            return candidate
    return category


def _prioritize_collision_category(category: str, candidates: list[str], question: str) -> str:
    """둘 이상의 정형 카테고리가 맞아 보일 때 더 보수적인 작업을 선택한다.

    복합정보_태스크플랜은 단일 카테고리가 아니라, 검증 가능한 하위 작업 여러 개를 묶는
    결정론 플랜이다. 후보에 올라왔다는 것은 사용자가 이미 "기한+서류+세금"처럼 여러
    요구를 명시했다는 뜻이므로, 단일 세금/기한/요건 카테고리보다 우선한다.

    대표 충돌: "나 74세인데 세금 어떻게 내?"는 개인세금_입력충분성과
    연금소득세율_연령별 후보가 함께 뜬다. 라우터가 후자를 고르면 나이 구간 세율(4.4%)만
    보고 실제 세금 질문에 답하는 경로가 열린다. 금융 상담에서는 애매하면 개인 계산
    Gate로 보내 부족정보를 확인하는 쪽이 더 안전하다.
    """
    if (
        "복합정보_태스크플랜" in candidates
        and deterministic_response_for("복합정보_태스크플랜", question) is not None
    ):
        return "복합정보_태스크플랜"
    if (
        category in {"중도인출_일반", "중도인출_기한판정"}
        and "중도인출_요건판정" in candidates
        and deterministic_response_for("중도인출_요건판정", question) is not None
    ):
        return "중도인출_요건판정"
    if (
        category == "연금소득세율_연령별"
        and "개인세금_입력충분성" in candidates
        and deterministic_response_for("개인세금_입력충분성", question) is not None
    ):
        return "개인세금_입력충분성"
    return category


def _enforce_candidate_scope(category: str, candidates: list[str]) -> str:
    """라우터가 후보 밖 카테고리를 고르면 "해당없음"으로 되돌린다.

    후보 목록은 코드가 질문을 보고 만든 "이 카테고리를 검토할 근거가 있다"는 사실이다.
    후보에 없다는 것은 그 카테고리의 트리거 신호가 질문에 없다는 뜻이므로, 그럼에도
    확정하면 정형 답변이 엉뚱한 질문에 붙는다.

    ⚠️ 이 강제는 candidate_categories가 **넓게** 후보를 잡는다는 전제 위에서만 안전하다.
    후보를 좁게 잡으면서 밖을 막으면, 표현이 조금 다른 정상 질문까지 정형 경로를 잃는다
    (실측: "나 74세인데 세금 어떻게 내?"가 후보 0건이었는데 라우터는 연금소득세율_연령별을
    정확히 골랐다 — 그때 이 강제를 걸었다면 맞는 판단을 되돌렸을 것이다). 그래서
    candidate_categories를 먼저 넓히고 이 강제를 함께 도입한다. 둘은 한 쌍이다.
    """
    if category == "해당없음" or category in candidates:
        return category
    return "해당없음"


def build_router_node():
    # ⚠️ method 미지정 시 langchain_openai(ChatClovaX가 상속)의 기본값 "function_calling"이
    # 적용되는데, CLOVA Structured Outputs는 method="json_schema"만 지원한다 — 미지정 시
    # 간헐적으로 400 "Unsupported function"(40009) 오류가 난다(실측 확인).
    llm = get_llm(ROUTER_MODEL, thinking_effort="none").with_structured_output(
        RouterDecision, method="json_schema"
    )

    def router_node(state: PensionAgentState) -> dict:
        history_text = format_conversation_history(state.get("conversation_history"))
        candidates = candidate_categories(state["question"])
        candidate_hint = (
            f"[후보 카테고리 힌트] {', '.join(candidates)}" if candidates else "[후보 카테고리 힌트] (없음)"
        )

        # scope 판정에 필요한 "우리가 무엇을 보유했는가"를 코드가 조회해 사실로 넘긴다.
        # LLM에게 맡기면 자기 상식으로 추측하다 틀린다 — 실측: DB에 실재하는 펀드를 두고
        # "개별 상품 정보는 제공 불가능"이라며 범위외로 판정(3/3 재현). candidate_hint가
        # 정형 카테고리에 대해 하는 일을 scope 축에 동일하게 적용한 것이다.
        matched_funds = find_asset_overlap(state["question"])
        asset_hint = (
            f"[보유 상품 조회] 질문에 언급된 이름이 상품 DB에 있음: {', '.join(matched_funds)}"
            if matched_funds
            else "[보유 상품 조회] 질문에서 보유 상품명을 찾지 못함"
        )

        hints = f"{candidate_hint}\n{asset_hint}"
        user_content = (
            f"[이전 대화]\n{history_text}\n\n{hints}\n\n[현재 질문]\n{state['question']}"
            if history_text
            else f"{hints}\n\n[현재 질문]\n{state['question']}"
        )
        decision: RouterDecision = invoke_with_retry(llm, [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])
        scope, scope_note, intent = _apply_asset_scope_override(
            decision.scope, decision.scope_note, decision.intent, matched_funds
        )
        deterministic_category = _enforce_candidate_scope(
            decision.deterministic_category, candidates
        )
        deterministic_category = _restore_rejected_category(
            deterministic_category, candidates, state["question"]
        )
        deterministic_category = _prioritize_collision_category(
            deterministic_category, candidates, state["question"]
        )
        intent = _drop_product_intent_for_deterministic_info(
            intent, deterministic_category, matched_funds
        )
        return {
            "intent": intent,
            "scope": scope,
            "scope_note": scope_note,
            "is_safe": decision.is_safe,
            "safety_reason": decision.safety_reason,
            "deterministic_category": deterministic_category,
        }

    return router_node
