"""평가용 API 서버 — 대회 제출물 3종 중 하나(요강 p8 "평가용 API 서버 정보").

주최측이 정한 스키마를 그대로 따른다. 우리가 바꿀 수 있는 부분이 아니다:

    GET /answer?question_id={id}&question={평가 질의}
    -> {"question_id", "question", "retrieved_context", "think_trace", "answer"}

이 모듈은 파이프라인을 감싸는 얇은 껍데기다. 답변 품질에 관여하는 로직은 전부
①~⑤ 노드에 있고, 여기서는 그래프를 호출해 응답 스키마로 옮기는 일만 한다 —
API 계층에 판단이 들어가면 chat.py로 테스트한 동작과 평가 시 동작이 갈라진다.

## 설계 판단

- 그래프는 startup에서 한 번만 build한다. 요청마다 build하면 노드·툴 바인딩을 매번
  다시 만들어 첫 응답이 수 초 느려진다. build 자체는 네트워크 호출이 없어 안전하다.
- 파이프라인이 예외로 죽어도 500을 던지지 않고 200 + 한계 고지 답변을 반환한다.
  평가 기준상 무응답은 그 문항이 0점이지만, 한계를 고지한 답변은 "정보한계 대응"으로
  읽힐 여지가 있다. 예외 내용은 think_trace에 남겨 디버깅 가능하게 한다.
- 그래프 호출은 동기(blocking)이므로 async def가 아닌 def로 선언한다. FastAPI가
  def 핸들러를 스레드풀에서 실행해주기 때문에, 한 요청이 이벤트 루프를 막아 동시
  요청이 직렬화되는 것을 피할 수 있다 (평가 시 여러 문항이 동시에 들어올 수 있다).
"""

import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

load_dotenv()

from src.agents.context import dedupe_context  # noqa: E402
from src.agents.graph import build_graph  # noqa: E402
from src.agents.text import normalize_user_text  # noqa: E402

logger = logging.getLogger(__name__)

# 파이프라인이 통째로 실패했을 때 내보내는 답변. 침묵(500)보다 한계 고지가 낫다.
_FAILURE_ANSWER = (
    "죄송합니다. 요청을 처리하는 중 오류가 발생해 답변을 드리지 못했습니다. "
    "잠시 후 다시 질문해 주시기 바랍니다."
)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["graph"] = build_graph()
    logger.info("연금 Agent 그래프 준비 완료")
    yield
    _state.clear()


app = FastAPI(
    title="연금 Agent 평가용 API",
    description="제10회(2026) 미래에셋증권 AI Festival — 연금 제도·세제·상품 질의응답",
    version="1.0.0",
    lifespan=lifespan,
)


class EvaluationResponse(BaseModel):
    """주최측 제출 스키마 (요강 p8). 필드명·구성을 임의로 바꾸지 않는다."""

    question_id: str = Field(description="요청에서 받은 식별자 (없으면 서버가 생성)")
    question: str = Field(description="평가 질의 원문")
    retrieved_context: str = Field(description="답변 생성에 참고한 검색 문서")
    think_trace: str = Field(description="사고·추론·도구 사용 과정")
    answer: str = Field(description="최종 생성 답변")


def _format_retrieved_context(context: list) -> str:
    """State의 근거 리스트를 평가 스키마의 문자열 필드로 옮긴다.

    출처와 내용을 함께 남긴다 — 심사자가 "이 답변이 어느 문서에서 나왔는가"를
    확인할 수 있어야 하고, 그것이 근거 완전성 평가의 대상이기 때문이다.
    """
    deduped = dedupe_context(context or [])
    if not deduped:
        return ""
    return "\n\n".join(f"[{item['source']}]\n{item['content']}" for item in deduped)


@app.get("/answer", response_model=EvaluationResponse)
def answer(
    question: str = Query(description="평가 질의"),
    question_id: str | None = Query(default=None, description="질의 식별자"),
) -> EvaluationResponse:
    """평가 질의 하나를 받아 파이프라인을 1회 실행하고 결과를 반환한다.

    평가는 문항당 단발 호출이므로 대화 이력은 넘기지 않는다 (싱글턴).
    """
    qid = question_id or str(uuid.uuid4())[:8]
    graph = _state.get("graph")
    if graph is None:  # lifespan 밖에서 직접 호출된 경우 방어
        graph = _state["graph"] = build_graph()

    try:
        result = graph.invoke({
            "question_id": qid,
            "question": normalize_user_text(question),
            "conversation_history": [],
            "recommendation_profile": {},
        })
    except Exception as exc:
        # 무응답(500)은 그 문항을 0점으로 만든다 — 한계를 고지한 답변으로 대체하고
        # 원인은 think_trace에 남긴다.
        logger.exception("파이프라인 실행 실패 (question_id=%s)", qid)
        return EvaluationResponse(
            question_id=qid,
            question=question,
            retrieved_context="",
            think_trace=f"[오류] 파이프라인 실행 실패: {type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()}",
            answer=_FAILURE_ANSWER,
        )

    return EvaluationResponse(
        question_id=qid,
        question=question,
        retrieved_context=_format_retrieved_context(result.get("retrieved_context")),
        think_trace=result.get("think_trace") or "",
        answer=result.get("answer") or _FAILURE_ANSWER,
    )


@app.get("/health")
def health() -> dict:
    """배포 후 상태 확인용 — 그래프가 준비됐는지만 본다 (모델 호출 없음)."""
    return {"status": "ok", "graph_ready": _state.get("graph") is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
