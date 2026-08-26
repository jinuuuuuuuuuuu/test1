# 연금 Agent — 제10회(2026) 미래에셋증권 AI Festival

연금 제도(DB/DC/IRP, 연금저축)·세제·상품(펀드) 질의에 답하는 HyperCLOVA X 기반 에이전트.

## 아키텍처

```
사용자 질의
    │
    ▼
① 라우터 / 가드레일 (HCX-007, thinking 끔)
    - 정보형 / 상품형 / 복합형 다중 분류
    - 복합형일 경우 순차·병렬·조건부 게이팅 판단
    - 안전성 사전 필터
    │
    ├─ 정보형 ──▶ ② 정보 Agent (HCX-005)
    │              tools: RAG검색(벡터DB) / 세제 규칙엔진 / 제도 판정기(디폴트옵션·실물이전·중도인출)
    │
    └─ 상품형 ──▶ ③ 상품 Agent (HCX-005)
                   tools: 펀드 필터/비교(구조화DB) / 슬롯필링 상태
                   (복합형은 ②→③ 순차 실행, 필요시 ③ 스킵하고 역질문)
    │
    ▼
④ 검증 / Grounding (HCX-007, thinking 끔)
    - ②③가 실제 호출한 툴 결과와 답변 초안 대조
    - 제도 부합 / 계산 일치 / 근거 존재 / 투자제한 위반 여부
    │
    ▼
⑤ 응답 생성기 (HCX-005)
    - 최종 답변 조립 + think_trace 포맷
```

**CLOVA Studio 모델 제약 (네이버 공식 문서 + 실측, 2026-08-13):**
이미지입력/튜닝/Function calling/Structured Outputs/추론(Thinking)은 동시 이용 불가.
- Structured Outputs(①④가 씀)는 **HCX-007에서만** 지원 (HCX-DASH-002는 "Unsupported function").
- HCX-007은 기본적으로 Thinking이 켜져 있어, Function calling이든 Structured Outputs든 쓰려면
  `thinking={"effort": "none"}`으로 꺼야 함 (안 그러면 400 "tools, reasoning").
- `with_structured_output()`은 LangChain이 기본으로 `parallel_tool_calls`를 얹는데 CLOVA가
  이 파라미터를 모름 — `disabled_params={"parallel_tool_calls": None}`으로 꺼야 함.
- Pydantic 응답 스키마 클래스에 **docstring이 없으면** "tools[].function.description" 400 에러.
- HCX-005/HCX-DASH-002는 Thinking이 없어 bind_tools()는 바로 되지만, Structured Outputs는
  HCX-005만 됨(DASH-002는 전혀 안 됨) — 그래서 ②③(tool 호출)은 HCX-005, ①④(구조화 출력)는
  HCX-007+thinking 끔으로 나눴다.

## 데이터 자산

- `docs.zip` (58개, PDF/DOCX/XLSX/PPTX) — 제도·세제·업무 매뉴얼/FAQ. 다중 카테고리 라벨링 완료(`data/labels/`).
  - doc29(디폴트옵션 Q&A), doc34(실물이전 25코드)는 RAG 대상에서 제외, 구조화 DB로 별도 처리
- `투자설명서.zip` (100개 PDF) — 펀드 투자설명서. 6축(상품분류/위험등급/판매클래스/총보수/수익률/AUM) 구조화 추출 대상.

## 폴더 구조

```
data/
  raw/        원본 문서 (git 추적 안 함, 용량 큼)
  processed/  파싱된 청크/구조화 데이터
  labels/     데이터 라벨링.xlsx 등 메타데이터
src/
  parsing/    문서 파서 (pdf/docx/pptx/xlsx → 텍스트/표)
  storage/    벡터DB(Chroma) / 구조화DB(SQLite) 클라이언트
  rules/      세제 규칙엔진, 제도 판정기 (결정론적 계산)
  agents/     LangGraph 노드 (router / info_agent / product_agent / grounding / generator)
  api/        평가용 FastAPI 서버
tests/        pytest 테스트 (특히 rules/ 는 정확성이 평가 핵심이라 테스트 필수)
scripts/      배치 실행 스크립트 (파싱, 색인 등)
```

## 진행 단계

- [x] **Phase 0** — Python 환경, 레포 스캐폴드, NCP 크레딧/Clova Studio API 키 발급
- [x] **Phase 0.5** — 원본 데이터 검수/라벨링 QA (파싱·색인의 입력 신뢰도 확보)
  - [x] `docs.zip`(58개) 다중 카테고리 라벨링 + 전수 원문 대조 검증
  - [x] `투자설명서.zip`(100개) 전수 원문 대조 검증, AUM 98/100 수기검수 반영
- [x] **Phase 1** — `docs.zip`·`투자설명서.zip` 파싱 + 벡터DB(Chroma)/구조화DB(SQLite) 색인
  - `data/processed/prospectus.db`(fund_master/fund_class, 100펀드/198클래스) +
    `data/processed/chroma_docs`(제도문서 708청크) + 투자설명서 서술형(투자전략·위험 등) 벡터
    컬렉션(430문서) — 3개 데이터 저장소 완료, git에 포함
- [x] **Phase 2** — 세제 규칙엔진(세액공제/연금수령한도/감면율/종합과세) + 제도 판정기 + 투자한도
  판정기 (`src/rules/`, 8개 모듈)
- [x] **Phase 3** — LangGraph 에이전트 (①라우터 → ②정보/③상품 Agent → ④검증 → ⑤생성기) 배선 완료
  - ②③ 규칙엔진 툴 6개 + RAG/구조화DB 툴 3개(`search_pension_docs`, `search_funds`,
    `get_fund_detail`) 전부 연결
  - L0 결정론적 검증 게이트(수치 대사) + scope 축(범위내/부분관련/범위외) + 1회 repair 루프 반영
  - 대회 평가가 싱글턴 기준으로 확정되어, 역질문 대신 조건부 답변(`response_mode`)으로
    대응하도록 ②③ 응답 전략 조정 (2026-08-20)
  - 멀티턴 대화(`conversation_history`)는 코드상 지원되지만 싱글턴 평가 API에서는 사용되지
    않음 — 로컬 데모(`scripts/chat.py`)용 기능으로 남겨둠
- [~] **Phase 4** — 평가용 API 서버 **완료** / NCP 배포 **미착수**
  - `src/api/main.py` — 요강 p8 스키마(`GET /answer` → `question_id`/`question`/
    `retrieved_context`/`think_trace`/`answer`) 구현. 로컬 기동·공식 질의 응답 확인 완료.
  - 파이프라인이 예외로 죽어도 500 대신 200 + 한계 고지를 반환한다 (무응답은 그 문항이
    0점이므로). 원인은 `think_trace`에 남는다.
  - **남은 것**: 주최측에 제출할 것은 코드가 아니라 **접속 가능한 End-point URL**이다.
    `localhost`는 제출용이 될 수 없으므로 NCP 등에 배포하고 URL을 확보해야 한다.
- [ ] **Phase 5** — 자체 평가 반복, 기술제안서 — `eval/eval_questions_100.csv`(100문항 자체
  평가셋) 작성 완료, 실제 회귀 실행/결과 정리는 미착수

## 셋업

⚠️ **`.venv` 폴더는 절대 다른 사람과 공유하지 마세요** — Windows에서 만든 `.venv`는 컴파일된
바이너리가 들어있어 Mac/Linux에서 절대 실행되지 않습니다(반대도 마찬가지). 각자 자기 컴퓨터에서
아래 명령으로 새로 만들어야 합니다. `.venv`는 `.gitignore`에 이미 포함돼 있어 git으로는
공유되지 않습니다 — 공유되는 건 `requirements.txt`(설치 목록)뿐입니다.

```bash
python -m venv .venv

# 가상환경 활성화 (OS별로 다름, 자기 OS에 맞는 것 하나만 실행)
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (cmd)
.venv\Scripts\Activate.ps1       # Windows (PowerShell)

pip install -r requirements.txt
cp .env.sample .env              # 키 채워넣기 (CLOVASTUDIO_API_KEY)
```

**실행 확인**:
```bash
pytest -q                        # 전체 테스트 (276 passed, 19 skipped면 정상)
python scripts/chat.py           # 터미널에서 직접 질문해보기 (.venv 활성화 상태에서 python만 쓰면 됨,
                                  #  .venv/Scripts/python.exe처럼 OS별 경로를 직접 안 써도 됨)
```

**제출 전 회귀 확인 (필수)**: 위 `pytest -q`는 실제 모델을 부르지 않으므로,
"답할 수 있는 질문을 거부한다" 같은 파이프라인 전체 결함은 잡지 못합니다
(실제로 단위 테스트 276개가 전부 통과하는 상태에서 대회 공식 질의가 거부되고
있었습니다). 대회 참고 질의 5건을 실제로 통과시키는 E2E 회귀 테스트를 돌리세요:
```bash
RUN_LIVE_AGENT_TESTS=1 pytest tests/test_e2e -v   # 약 3~5분, API 크레딧 소모
```

**평가용 API 서버 실행**:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
확인 (요강 p8 스펙):
```bash
curl -G "http://localhost:8000/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=연금저축이랑 IRP에 넣으면 세액공제 얼마까지 되나요?"
```
`GET /health`로 기동 여부만 따로 확인할 수 있습니다(모델 호출 없음).

⚠️ Windows Git Bash의 `curl`은 한글 인자를 CP949로 보내 질문이 깨집니다. 한글 질의를
테스트할 때는 요강 예시대로 Python `requests`(또는 `urllib`)를 쓰세요 — 서버 문제가
아니라 클라이언트 인코딩 문제입니다.

**데이터 자산(`data/processed/prospectus.db`, `data/processed/chroma_docs/`)은 git에
포함돼 있어 별도로 다시 만들 필요가 없습니다** — `git pull` 받으면 바로 있습니다. 원본 xlsm
트래커 파일(파싱 검수본)은 크기가 커서 git에 안 올렸으니, 그 원본 자체를 새로 처리해야 하는
경우에만 별도로 공유가 필요합니다.

**Docker로 실행 (OS 무관, 가장 안전한 방법)**:
```bash
docker build -t pension-agent .
docker run -p 8000:8000 --env-file .env pension-agent
```
`.env`는 이미지에 넣지 않고 실행 시 주입합니다 — API 키가 이미지 레이어에 박히면
이미지를 받는 사람 모두에게 키가 노출됩니다. 데이터 자산(`data/processed/`)은 이미지에
포함되므로 별도 볼륨 마운트가 필요 없습니다.

## 대회 제출 요건 (요약)

- LLM은 HyperCLOVA X만 사용 가능
- 예선 마감: 2026-09-06 / 평가기간: 09-07~09-30
- 평가용 API 응답 스키마: `question_id`, `question`, `retrieved_context`, `think_trace`, `answer`
- 제출: 주최측 Github Organization 내 Private Repository (마감 후 수정 시 실격)
