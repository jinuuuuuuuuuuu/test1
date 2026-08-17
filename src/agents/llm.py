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


def get_llm(
    model_name: str,
    temperature: float = 0.0,
    thinking_effort: Optional[str] = None,
) -> ChatClovaX:
    """모델명(예: HCX-DASH-002, HCX-007, HCX-005)으로 ChatClovaX 인스턴스를 만든다.

    thinking_effort: HCX-007처럼 Thinking이 기본 활성화된 모델에서 bind_tools()/
    with_structured_output()을 쓰려면 "none"을 넘겨 꺼야 한다.
    """
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": 2048,
        "disabled_params": _DISABLE_PARALLEL_TOOL_CALLS,
    }
    if thinking_effort is not None:
        kwargs["thinking"] = {
            "effort": thinking_effort
        }

    return ChatClovaX(**kwargs)


def get_embeddings(model_name: str = "clir-emb-dolphin") -> ClovaXEmbeddings:
    """CLOVA Studio 임베딩 모델 인스턴스를 만든다. ChatClovaX와 동일하게 CLOVASTUDIO_API_KEY를 쓴다."""
    return ClovaXEmbeddings(model=model_name)


def invoke_with_retry(runnable, input_, max_retries: int = 3, backoff_seconds: float = 1.5):
    """CLOVA Studio(특히 테스트 키)가 간헐적으로 내는 일시적 오류(예: 동일 요청인데도 가끔
    나는 "Unsupported function" 400)에 대응해 재시도한다. with_structured_output()을 쓰는
    HCX-007 노드(router/grounding)에서 이 플레이크가 실측됨 — 코드/설정 문제가 아니라
    같은 입력으로 재요청하면 성공하는 경우가 많아 재시도로 흡수한다.
    """
    import time

    from openai import APIError

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_)
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (attempt + 1))
    raise last_error
