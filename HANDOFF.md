# 인수인계 — 연금 Agent (제10회 미래에셋증권 AI Festival)

검수를 부탁드리기 위한 요약 문서입니다. 전체 맥락은 `README.md`에 있고, 이 문서는 구조와
설계 판단 위주로 핵심만 정리했습니다.

## 2026-09-02 Cost Guard / Guardian 통합 메모

- Cost Guard 비용 데이터는 `fund_class_pension` 전용 canonical dataset으로 분리했습니다. 총보수·비용, 합성총보수·비용, 1,000만원 3년 비용 예시는 서로 다른 지표로 보존하고, lower-cost 비교는 pair 단위로 동일 metric끼리만 수행합니다.
- `data/processed/fund_class_pension_review.csv`의 P0 review 판정을 반영해 `FROZEN_V1 / cost_guard_v1` dataset을 생성했습니다. 현재 canonical은 64펀드 / 210행이고, STANDARD lower-cost pair는 93건입니다.
- review 적용으로 167행에서 210행으로 늘어난 이유를 추적하기 위해 `data/processed/fund_class_pension_review_provenance.csv`를 생성합니다. 현재 `RESTORED_TO_CANONICAL` 55건, `EXCLUDED_FROM_CANONICAL` 1건으로 순증 +43행입니다.
- `prospectus.db`에는 `fund_class_pension`과 `cost_guard_dataset_manifest`를 적재합니다. CSV row count와 DB row count는 210건으로 일치하고, DB manifest hash도 canonical manifest와 일치합니다.
- Guardian에는 Cost Guard C1을 연결했습니다. 특정 상품/클래스/계좌가 명확하고, 사용자가 비용을 직접 묻지 않았으며, frozen STANDARD lower-cost pair가 있을 때만 `🛡️ 파수꾼 체크`를 최대 1건 추가합니다.
- Product Agent는 사용자가 `KR...`, `C-P2`, `IRP`처럼 현재 보유 상품 맥락을 명시한 경우 해당 상품을 lock해서 Core 답변을 생성합니다. 명시적으로 "비슷한 다른 상품 추천/비교"를 요청한 경우에만 기존 recommender로 넘깁니다.
- 직접 비용 질문은 Core가 처리하고 Guardian은 `EXPLICIT_USER_TOPIC`으로 꺼집니다. 예: `IRP KR514X450008 C-P2 보수 더 낮은 클래스 있어?`
- 검증: `pytest tests -q` 기준 556 passed, 19 skipped. `scripts/chat.py` 대화형 E2E는 로컬 API 연결 문제로 `Connection error`가 발생해 네트워크/API 연결 확인 후 재실행이 필요합니다.

## 무엇을 만드는가

연금 제도(DB/DC/IRP, 연금저축)·세제·펀드상품 질의에 답하는 에이전트입니다. LLM은 **HyperCLOVA
X만 사용 가능**(대회 규정)하고, 평가는 `GET /answer?question_id=&question=` 호출 1회로
`{question_id, question, retrieved_context, think_trace, answer}` JSON을 돌려주는 방식입니다.
**대회 평가가 싱글턴 기준으로 최종 확정**됐습니다 — 이전 대화를 이어받는 메커니즘이 평가 API
자체에 없다는 뜻이라, 이후 설계는 전부 "한 번의 질문에 최선의 답을 낸다"를 전제로 갑니다.

## 아키텍처 — 5노드 파이프라인 (LangGraph)

```
사용자 질의
    │
    ▼
① 라우터 / 가드레일 (HCX-007, thinking 끔, Structured Outputs)
    - 정보형/상품형 다중 분류 + scope 판정(범위내/부분관련/범위외) + 안전성 필터
    │
    ├─ 정보형 ──▶ ② 정보 Agent (HCX-005, ReAct)
    │              tools: 세제 규칙엔진 5개 + search_pension_docs(RAG)
    │
    └─ 상품형 ──▶ ③ 상품 Agent (HCX-005, ReAct)
                   tools: check_product_pension_eligibility + search_funds + get_fund_detail
                   (복합형은 ②→③ 순차 실행 — 병렬 아님)
    │
    ▼
④ 검증 / Grounding (HCX-007, thinking 끔, Structured Outputs)
    - L0: 답변 속 숫자를 근거 텍스트와 기계적으로 대사(코드 레벨, LLM 판단에 의존 안 함)
    - L1: 근거부합(grounded) + 전제교정(premise_issues)
    - L3: 요구사항충족(requirements_met) — 틀리면 해당 에이전트로 1회 repair 재실행
    │
    ▼
⑤ 응답 생성기 (HCX-005)
    - 최종 답변 조립 + think_trace(시간순 서사) 포맷
```

**왜 이렇게 나눴나**: HCX-007만 Structured Outputs(구조화 출력)를 지원하고, HCX-007은
Thinking이 기본 켜져 있어 그것 때문에 Function calling과 충돌한다(자세한 이유는 아래
"CLOVA API 특이사항" 참고). 그래서 구조화 출력이 필요한 ①④만 HCX-007+thinking 끔, 툴
호출이 필요한 ②③은 별도 설정 없이 되는 HCX-005로 나눴습니다.

## 데이터 자산 (전부 git에 포함, 재적재 불필요)

| 저장소 | 내용 | 위치 |
|---|---|---|
| SQLite (`fund_master`/`fund_class`) | 투자설명서 100개 펀드/198개 판매클래스, 사람이 원문 대조 검수 완료 | `data/processed/prospectus.db` |
| Chroma (제도문서) | docs.zip 58개 문서 → 708청크, 원문 대조 검수 완료 | `data/processed/chroma_docs/` |
| Chroma (투자설명서 서술형) | 투자전략·위험 등 서술 섹션 430문서 — 단일 상품 설명 질의 대응용 | `data/processed/chroma_docs/`(별도 컬렉션) |

원본 xlsm 트래커(파싱 검수본)는 용량 문제로 git에 없고 팀원 개인 로컬에만 있습니다 — 이 DB들을
다시 만들 방법이 사실상 없으니 **그대로 유지**하시면 됩니다.

## 핵심 설계 판단과 이유

- **싱글턴 대응(`response_mode`, `src/agents/state.py`)**: 정보 부족 시 예전엔 "역질문하고
  멈춤"이었는데, 평가가 싱글턴이라 역질문은 죽은 턴이 됩니다. 지금은 `complete`(바로 답변
  가능) / `conditional`(조건별로 답변, 가정 명시) / `clarification_included`(질문에 답하되
  추가로 확인할 점을 답변 안에 포함) 세 모드로 나눠 항상 뭔가 유용한 답을 냅니다.
- **L0 결정론적 게이트**: LLM(grounding)이 "이 정도면 근거 있다고 볼 수 있다"고 스스로 면제해
  버리는 경로가 있었습니다(예: "순수 설명형이면 grounded=True"). 이제 답변 속 숫자를 근거
  원문과 코드로 직접 대조해서, LLM 판단과 무관하게 근거 없는 숫자는 걸러냅니다.
- **scope 축 분리**: 연금과 무관한 질문(예: 개인사업자 일반 세금 절세)에 일반 지식으로 답해버리는
  버그가 있었습니다. `is_safe`(안전성)와 별도로 `scope`(범위내/부분관련/범위외) 축을 추가해
  범위 밖 질문엔 한계를 고지하도록 분리했습니다.
- **멀티턴은 코드엔 있지만 평가엔 안 씀**: `conversation_history`는 `scripts/chat.py` 로컬
  데모용으로 남겨뒀고, 평가 API 경로에서는 어차피 매 요청이 독립적이라 실질적으로 안 쓰입니다.

## CLOVA Studio API 특이사항 (재발견하면 시간 낭비하는 것들)

`README.md`의 "CLOVA Studio 모델 제약" 절에 실측 기반으로 정리돼 있습니다 — 요약하면 Thinking/
Function calling/Structured Outputs는 동시 사용 불가, Structured Outputs는 HCX-007에서만
지원, `disabled_params`/`thinking={"effort":"none"}` 안 하면 400 에러. 코드 건드리기 전에
꼭 한 번 읽어보시길 권합니다.

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux — Windows는 README 참고
pip install -r requirements.txt
cp .env.sample .env              # CLOVASTUDIO_API_KEY 채우기 (본인 키 발급 필요)

pytest -q                        # 223 passed, 2 skipped 나오면 정상
python scripts/chat.py           # 터미널에서 직접 질문해보기
```

## 알려진 갭 (우선순위순)

1. **평가용 API 서버(Phase 4) 미착수** — `src/api/`가 빈 폴더입니다. `Dockerfile`의 CMD가
   존재하지 않는 `src.api.main:app`을 가리키고 있어서 지금 `docker run`하면 바로 실패합니다.
   제출 전 반드시 만들어야 하는 가장 중요한 남은 작업입니다.
2. **자체 평가셋 미실행** — `eval/eval_questions_100.csv`에 100문항(대주제/난이도/질문유형/
   중점평가지표/점검포인트까지 구조화)을 만들어뒀는데 아직 실제로 돌려서 정답률을 뽑아보진
   않았습니다.
3. **팀원 개별 브랜치 정리 필요** — `feature/dana-agent-grounding-recommendation`,
   `feature/haein` 브랜치에 이 통합 브랜치(`integration/agent-best-of-three`)로 fast-forward
   병합되지 않은 커밋이 각각 남아있습니다. 내용을 비교해보면 통합 브랜치가 더 최신/포괄적이라
   유실된 작업은 아닌 것으로 보이지만, 최종 제출 전 각 작성자가 한 번씩 확인하는 걸 권합니다.
4. 그 외 세부 이슈는 별도 카카오톡으로 전달 예정.

## 파일 지도

- `src/agents/` — LangGraph 노드(router/info_agent/product_agent/grounding/generator) + 툴
- `src/storage/` — SQLite/Chroma 조회 함수
- `src/rules/` — 세제/제도 판정 순수 함수(결정론적 계산, 평가 정확성의 핵심)
- `scripts/chat.py` — 터미널 데모
- `eval/eval_questions_100.csv` — 자체 평가 문항셋
- `tests/` — 223 tests, 실행: `pytest -q`
