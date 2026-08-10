# 연금 Agent — 제10회(2026) 미래에셋증권 AI Festival

연금 제도(DB/DC/IRP, 연금저축)·세제·상품(펀드) 질의에 답하는 HyperCLOVA X 기반 에이전트.

## 아키텍처

```
사용자 질의
    │
    ▼
① 라우터 / 가드레일 (HCX-DASH)
    - 정보형 / 상품형 / 복합형 다중 분류
    - 복합형일 경우 순차·병렬·조건부 게이팅 판단
    - 안전성 사전 필터
    │
    ├─ 정보형 ──▶ ② 정보 Agent (HCX-007)
    │              tools: RAG검색(벡터DB) / 세제 규칙엔진 / 제도 판정기(디폴트옵션·실물이전·중도인출)
    │
    └─ 상품형 ──▶ ③ 상품 Agent (HCX-007)
                   tools: 펀드 필터/비교(구조화DB) / 슬롯필링 상태
                   (복합형은 ②→③ 순차 실행, 필요시 ③ 스킵하고 역질문)
    │
    ▼
④ 검증 / Grounding (경량모델)
    - ②③가 실제 호출한 툴 결과와 답변 초안 대조
    - 제도 부합 / 계산 일치 / 근거 존재 / 투자제한 위반 여부
    │
    ▼
⑤ 응답 생성기 (HCX-005)
    - 최종 답변 조립 + think_trace 포맷
```

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

- [x] Phase 0: Python 환경, 레포 스캐폴드
- [ ] Phase 0: NCP 크레딧 신청 / Clova Studio API 키
- [ ] Phase 1: docs.zip·투자설명서.zip 파싱 + 벡터DB/구조화DB 색인
- [x] Phase 2: 세제 규칙엔진 (세액공제/연금수령한도/감면율/종합과세) + 제도 판정기 (`src/rules/`, 73 tests passing)
- [ ] Phase 3: LangGraph 에이전트 (①~⑤)
- [ ] Phase 4: 평가용 API 서버 + NCP 배포
- [ ] Phase 5: 자체 평가 반복, 기술제안서

## 셋업

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
cp .env.sample .env          # 키 채워넣기
```

## 대회 제출 요건 (요약)

- LLM은 HyperCLOVA X만 사용 가능
- 예선 마감: 2026-09-06 / 평가기간: 09-07~09-30
- 평가용 API 응답 스키마: `question_id`, `question`, `retrieved_context`, `think_trace`, `answer`
- 제출: 주최측 Github Organization 내 Private Repository (마감 후 수정 시 실격)
