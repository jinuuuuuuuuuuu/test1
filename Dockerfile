FROM python:3.12-slim

WORKDIR /app

# 의존성을 먼저 설치한다 — src/ 가 바뀌어도 이 레이어는 캐시되어 재빌드가 빠르다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/processed/ ./data/processed/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 컨테이너가 살아있는지가 아니라 "그래프가 준비됐는지"를 본다. uvicorn 프로세스는
# 떠 있는데 startup(build_graph)에서 실패한 상태면 요청은 전부 실패하므로,
# 프로세스 생존만 보는 헬스체크는 그 상황을 놓친다. /health는 모델을 호출하지
# 않으므로 주기적으로 찔러도 크레딧이 소모되지 않는다.
#
# start-period가 긴 이유: startup에서 벡터DB(Chroma)를 로드하므로 첫 기동이
# 수십 초 걸릴 수 있다. 이 시간 동안의 실패는 unhealthy로 세지 않는다.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys,json; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5)); sys.exit(0 if r.get('graph_ready') else 1)"

# 워커는 1개로 둔다. 평가는 팀당 순차 1건씩 들어오므로(주최측 답변) 다중 워커가
# 필요 없고, 워커마다 그래프와 벡터DB를 따로 로드해 메모리만 배로 쓴다.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
