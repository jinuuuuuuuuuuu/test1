"""모델 이름별 ChatClovaX(HyperCLOVA X) 인스턴스 생성 팩토리.

CLOVASTUDIO_API_KEY 환경변수가 필요하다 (.env.sample 참고). 이 함수는 각 노드 함수 "안에서"
호출하도록 설계돼 있다 — 그래프를 짜고 import/build_graph()/compile()하는 것 자체는 이 키
없이도 전부 가능하고, 실제로 .invoke()를 호출해 모델에 요청을 보내는 순간에만 키가 필요하다.
(ChatClovaX 생성자는 키가 없으면 즉시 에러를 내므로, 모듈 최상단에서 미리 만들어두지 않는다.)

⚠️ 실제 API 키로 테스트해서 확인한 CLOVA Studio 제약(네이버 공식 문서: 이미지입력/튜닝/
Function calling/Structured Outputs/추론(Thinking)은 동시 이용 불가):
  - HCX-007은 기본적으로 Thinking이 켜져 있어, bind_tools()나 with_structured_output()을
    쓰려면 반드시 thinking={"effort": "none"}으로 꺼야 한다 (안 그러면 400 "tools, reasoning").
  - with_structured_output()은 (모델 무관) LangChain이 기본으로 parallel_tool_calls를
    같이 보내는데 CLOVA가 이 파라미터 자체를 모른다 — disabled_params={"parallel_tool_calls":
    None}으로 꺼줘야 한다.
  - HCX-DASH-002는 bind_tools()는 되지만 with_structured_output()은 (function_calling/
    json_mode 둘 다) "Unsupported function"으로 아예 안 된다 — 구조화 출력이 필요한 노드에는
    쓸 수 없다.
"""

from typing import Optional

from langchain_naver import ChatClovaX, ClovaXEmbeddings

# with_structured_output()이 기본으로 얹는 parallel_tool_calls를 CLOVA가 인식하지 못해 꺼야 한다.
_DISABLE_PARALLEL_TOOL_CALLS = {"parallel_tool_calls": None}

# ⚠️ 타임아웃을 명시하지 않으면 openai SDK 기본값(read 600초)이 그대로 적용된다.
# 실측: CLOVA가 응답하지 않는 호출 하나 때문에 질문 1건이 629초(600초 대기 + 재시도 성공)
# 걸린 사례가 있었다. 대회 평가는 GET 엔드포인트 1회 호출이라 이 정도면 타임아웃 처리된다.
# 정상 응답은 실측 1~4초라 30초면 충분한 여유다.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


def get_llm(
    model_name: str,
    temperature: float = 0.0,
    thinking_effort: Optional[str] = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> ChatClovaX:
    """모델명(예: HCX-DASH-002, HCX-007, HCX-005)으로 ChatClovaX 인스턴스를 만든다.

    thinking_effort: HCX-007처럼 Thinking이 기본 활성화된 모델에서 bind_tools()/
    with_structured_output()을 쓰려면 "none"을 넘겨 꺼야 한다.

    timeout: 호출 1건의 상한(초). max_retries=0으로 SDK 자체 재시도를 끄고 재시도는
    call_with_retry 한 곳에서만 하도록 모은다 — 안 그러면 SDK 재시도(기본 2회)와
    우리 재시도(3회)가 곱해져 최악의 경우 수십 분까지 늘어난다.
    """
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "disabled_params": _DISABLE_PARALLEL_TOOL_CALLS,
        "request_timeout": timeout,
        "max_retries": 0,
    }
    if thinking_effort is not None:
        kwargs["thinking"] = {"effort": thinking_effort}
    return ChatClovaX(**kwargs)


def get_embeddings(model_name: str = "clir-emb-dolphin") -> ClovaXEmbeddings:
    """CLOVA Studio 임베딩 모델 인스턴스를 만든다. ChatClovaX와 동일하게 CLOVASTUDIO_API_KEY를 쓴다."""
    return ClovaXEmbeddings(model=model_name)


def call_with_retry(fn, *args, max_retries: int = 3, backoff_seconds: float = 1.5, **kwargs):
    """CLOVA Studio가 간헐적으로 내는 일시적 오류(같은 요청인데 가끔 나는 "Unsupported
    function" 400, 429 rate limit, 응답 없음으로 인한 타임아웃 등 — openai.APIError 계열)를
    재시도로 흡수한다. 실측상 코드/설정 문제가 아니라 같은 입력으로 재요청하면 성공하는
    경우가 많다 (APITimeoutError도 APIError 하위라 여기서 함께 잡힌다).

    LLM invoke뿐 아니라 검색 시점 임베딩 호출(similarity_search 내부) 같은 일반 함수도
    감쌀 수 있다. APIError가 아닌 예외(코드 버그)는 재시도 없이 그대로 전파한다.

    ⚠️ 호출 1건의 상한은 get_llm(timeout=...)이 정한다. 이 함수는 그 상한을 넘겨 실패한
    호출을 몇 번 더 시도할지만 정한다 — 최악의 경우 시간은 (timeout × max_retries)다.
    평가 API 서버(Phase 4)에서는 이와 별개로 요청 전체에 대한 데드라인이 필요하다.
    """
    import time

    from openai import APIError

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (attempt + 1))
    raise last_error


def invoke_with_retry(runnable, input_, max_retries: int = 3, backoff_seconds: float = 1.5):
    """runnable.invoke(input_) 형태 호출용 call_with_retry 래퍼 — 파이프라인의 모든 모델
    호출(①~⑤, ReAct 에이전트 포함)은 이걸 거쳐야 한다 (평가 기간 상시 가동 요건)."""
    return call_with_retry(
        runnable.invoke, input_, max_retries=max_retries, backoff_seconds=backoff_seconds
    )
