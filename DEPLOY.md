# 평가용 API 서버 배포 절차 (NCP)

평가기간 **09.07 ~ 09.20** 동안 이 서버가 계속 떠 있어야 한다. 이 기간 중 다운되면
그 시간에 들어온 질의는 실패하고, 해당 문항은 0점이 된다(주최측은 타임아웃·5xx에
한해 최대 2회 재시도).

제출해야 하는 것은 코드가 아니라 **접속 가능한 End-point URL**이다.

---

## 0. 사전 확인

- [x] NCP 크레딧 승인 및 등록 완료 (₩200,000, 유효기간 2026.09.30)
- [x] CLOVA Studio API 키 발급 완료 (이미 로컬에서 호출 중)
- [ ] GitHub private repo 접근 수단 (서버에서 clone하려면 필요 — 아래 1-5 참고)

> ⚠️ 크레딧 유효기간이 **09.30**이고 평가는 09.20에 끝난다. 기간 자체는 문제없으나,
> 잔액이 먼저 소진되면 그 시점부터 과금된다. 평가기간 중 콘솔에서 잔액을 주기적으로
> 확인할 것 (주최측은 초과분을 보전해주지 않는다).

---

## 1. 서버 인프라 생성 (NCP 콘솔)

### 1-1. VPC
`Console → Networking → VPC → VPC Management → VPC 생성`
- IP 주소 범위: `192.168.0.0/16`

### 1-2. Subnet
`Console → Networking → VPC → Subnet Management → Subnet 생성`
- 주소범위: `192.168.1.0/24`
- 가용 Zone: `KR-2`
- **Internet Gateway 전용 여부: Y** ← Public Subnet이어야 공인 IP를 붙일 수 있다
- 용도: 일반

### 1-3. Server
`Console → Compute → Server → Server 생성`
- 이미지: **Ubuntu 22.04** (또는 Rocky Linux)
- 스펙: **High-CPU 2vCPU / 4GB** — 벡터DB 로드 + 파이프라인 실행에 충분
- VPC/Subnet: 위에서 만든 것 선택
- 공인 IP: **새로운 공인 IP 할당** ← 이게 제출할 End-point의 주소가 된다
- 스토리지: 20GB
- 인증키: 새로 생성 후 `.pem` 파일 **분실 주의** (재발급 불가, 서버 접속 불가능해짐)

### 1-4. ACG (방화벽)
`Console → Compute → Server → ACG`

Inbound에 아래를 추가:

| 프로토콜 | 접근 소스 | 허용 포트 | 용도 |
|---|---|---|---|
| TCP | `0.0.0.0/0` | 22 | SSH 접속 |
| TCP | `0.0.0.0/0` | 8000 | 평가 API |

Outbound: TCP `0.0.0.0/0` `1-65535` (CLOVA API 호출에 필요)

> 주최측 평가 서버의 IP를 모르므로 8000은 `0.0.0.0/0`으로 열어야 한다.
> 22번은 가능하면 본인 IP로 좁히는 게 안전하다.

### 1-5. 서버 접속
`Console → Compute → Server → 서버 선택 → 서버 관리 및 설정 변경 → 관리자 비밀번호 확인`
(1-3에서 받은 `.pem` 파일을 업로드하면 root 비밀번호가 나온다)

```bash
ssh root@{공인IP}
```

---

## 2. 서버 환경 세팅

```bash
# Docker 설치
apt-get update && apt-get install -y docker.io git
systemctl enable --now docker

# 코드 가져오기 (private repo이므로 인증 필요)
# 방법 A: GitHub Personal Access Token 사용
git clone https://{TOKEN}@github.com/jinuuuuuuuuuuu/test1.git pension-agent
cd pension-agent
git checkout integration/agent-best-of-three   # 배포할 브랜치

# data/processed(벡터DB·SQLite)는 git에 커밋돼 있으므로 clone만으로 함께 받아진다.
# 별도 데이터 전송이 필요 없다.
ls -la data/processed/   # chroma_docs, chroma_prospectus, prospectus.db 확인
```

---

## 3. 환경변수 파일 생성

`.env`는 git에 없다(비밀정보). 서버에서 직접 만든다:

```bash
cat > /app-env <<'EOF'
CLOVASTUDIO_API_KEY=여기에_실제_키
CLOVASTUDIO_EMBEDDING_API_KEY=여기에_실제_키
LANGSMITH_TRACING=false
EOF
chmod 600 /app-env
```

> `LANGSMITH_TRACING=false`로 두는 이유: 평가 중 외부 트레이싱 호출이 지연을 만들
> 이유가 없다. 디버깅이 필요하면 그때 켠다.

---

## 4. 빌드 및 실행

```bash
cd ~/pension-agent
docker build -t pension-agent .

docker run -d \
  --name pension-api \
  --restart=always \
  --env-file /app-env \
  -p 8000:8000 \
  pension-agent
```

**`--restart=always`가 핵심이다.** 서버가 재부팅되거나 컨테이너가 죽어도 자동으로
다시 뜬다. 이게 없으면 새벽에 한 번 죽는 것으로 평가 문항 여러 개를 잃는다.

---

## 5. 배포 검증

```bash
# 1) 컨테이너 상태 — STATUS가 "healthy"가 될 때까지 기다린다 (첫 기동 1~2분)
docker ps

# 2) 서버 내부에서 헬스체크
curl http://127.0.0.1:8000/health
# 기대: {"status":"ok","graph_ready":true}

# 3) 서버 내부에서 실제 질의 (60~120초 소요)
curl -G "http://127.0.0.1:8000/answer" \
  --data-urlencode "question_id=Q-TEST" \
  --data-urlencode "question=연금저축이랑 IRP에 넣으면 세액공제 얼마까지 되나요?"
```

**그리고 반드시 외부(로컬 PC)에서도 확인한다** — ACG가 안 열려 있으면 서버 안에서만
되고 밖에서는 안 되는 상태가 되는데, 이게 가장 흔한 사고다:

```bash
# 본인 PC에서 실행
curl "http://{공인IP}:8000/health"

python -c "
import requests
r = requests.get('http://{공인IP}:8000/answer',
    params={'question_id':'Q-001','question':'DC와 DB 차이가 뭔가요?'}, timeout=300)
print(r.status_code); print(r.json()['answer'][:200])
"
```

---

## 6. 운영 (09.07 ~ 09.20)

```bash
docker logs -f pension-api        # 실시간 로그
docker logs --tail 100 pension-api
docker stats pension-api          # 메모리/CPU
docker restart pension-api        # 재시작
```

**체크리스트**
- [ ] 평가 시작 전날(09.06) 외부에서 `/health` 200 확인
- [ ] 평가기간 중 NCP 콘솔에서 크레딧 잔액 주기적 확인
- [ ] **09.06 마감 이후 코드 변경·재배포 금지** (요강: 마감 후 변경 발견 시 실격)

---

## 7. 제출 정보

제출 항목 3번(평가용 API 서버 정보)에 기입할 내용:

```
End-point URL: http://{공인IP}:8000/answer

요청 (GET):
  question_id : string, 선택 — 생략 시 서버가 생성
  question    : string, 필수

응답 (JSON, 모든 필드 string):
  question_id       : 요청에서 받은 식별자
  question          : 평가 질의 원문
  retrieved_context : 답변 생성에 참고한 검색 문서 ([출처]\n내용 형식)
  think_trace       : 사고·추론·도구 사용 과정 (노드별 시간순 서사)
  answer            : 최종 생성 답변

예시:
  curl -G "http://{공인IP}:8000/answer" \
    --data-urlencode "question_id=Q-001" \
    --data-urlencode "question=평가 질의"
```

> HTTPS가 필요하면 도메인(Global DNS) + Certificate Manager + nginx 리버스 프록시가
> 추가로 필요하다. 요강은 "Public 망 통신 가능"만 요구하므로 HTTP로도 제출 가능하다.

---

## 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| 서버 안에서는 되는데 밖에서 안 됨 | ACG Inbound 8000 미개방. 1-4 확인 |
| `graph_ready: false` | `docker logs`에서 startup 예외 확인. 보통 API 키 누락 |
| 응답이 300초 넘음 | CLOVA API 지연. `docker logs`로 재시도 로그 확인 |
| 컨테이너가 계속 재시작 | `docker logs`로 크래시 원인 확인. 메모리 부족이면 서버 스펙 상향 |
| CLOVA 401/403 | API 키 오타 또는 크레딧 소진. 콘솔에서 잔액 확인 |
