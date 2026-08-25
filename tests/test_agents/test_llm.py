"""call_with_retry — CLOVA 플레이크(간헐 400/429) 흡수 로직 검증. 네트워크 호출 없음."""

from unittest import mock

import httpx
import pytest
from openai import APIError, APITimeoutError, RateLimitError

from src.agents.llm import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    RATE_LIMIT_BACKOFF_SECONDS,
    call_with_retry,
    get_embeddings,
    get_llm,
)


def _transient_error() -> APIError:
    return APIError("flaky backend", request=httpx.Request("GET", "http://test"), body=None)


def test_recovers_from_transient_errors():
    calls = {"count": 0}

    def flaky(value):
        calls["count"] += 1
        if calls["count"] < 3:
            raise _transient_error()
        return value * 2

    assert call_with_retry(flaky, 21, backoff_seconds=0) == 42
    assert calls["count"] == 3


def test_raises_after_max_retries():
    def always_failing():
        raise _transient_error()

    with pytest.raises(APIError):
        call_with_retry(always_failing, max_retries=2, backoff_seconds=0)


def test_programming_errors_are_not_retried():
    # 코드 버그(APIError 아님)는 재시도로 가려지면 안 된다 — 즉시 전파.
    calls = {"count": 0}

    def buggy():
        calls["count"] += 1
        raise ValueError("bug")

    with pytest.raises(ValueError):
        call_with_retry(buggy, backoff_seconds=0)
    assert calls["count"] == 1


def test_timeout_is_explicit_and_sdk_retries_are_disabled():
    """타임아웃 미설정 회귀 방지 — openai SDK 기본값(600초)에 맡기면 안 된다.

    실측 사례: 응답 없는 호출 하나 때문에 질문 1건이 629초(600초 대기 후 재시도 성공)
    걸렸다. 재시도도 SDK(기본 2회)와 call_with_retry(3회)가 곱해지지 않도록
    SDK 쪽은 0으로 꺼두고 한 곳에서만 재시도해야 한다.
    """
    llm = get_llm("HCX-005")

    assert llm.request_timeout == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert llm.request_timeout < 600, "openai SDK 기본 600초에 의존하면 평가에서 타임아웃된다"
    assert llm.max_retries == 0, "SDK 재시도가 켜져 있으면 최악의 경우 시간이 곱해진다"


def test_embeddings_timeout_is_explicit():
    """임베딩에도 타임아웃이 필요하다 — RAG 검색이 질의 임베딩에 이 API를 쓰기 때문에,
    여기가 막히면 LLM 타임아웃과 무관하게 SDK 기본값 600초까지 요청이 멈춘다.

    필드명이 timeout이 아니라 request_timeout이라, timeout=으로 넘기면 조용히 무시된다.
    """
    embeddings = get_embeddings()

    assert embeddings.request_timeout == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert embeddings.max_retries == 0


def test_rate_limit_errors_wait_longer_than_generic_errors():
    """429는 짧은 백오프로 재시도하면 그대로 다시 걸린다 — 더 오래 기다려야 한다."""
    waits: list[float] = []

    def fake_sleep(seconds):
        waits.append(seconds)

    def always_rate_limited():
        raise RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=httpx.Request("POST", "http://test")),
            body=None,
        )

    with mock.patch("time.sleep", fake_sleep):
        with pytest.raises(RateLimitError):
            call_with_retry(always_rate_limited, max_retries=2, backoff_seconds=0.1)

    assert waits, "429에서도 재시도 전 대기가 있어야 한다"
    assert waits[0] >= RATE_LIMIT_BACKOFF_SECONDS


def test_timeout_errors_are_retried():
    """APITimeoutError도 APIError 하위라 재시도 대상이어야 한다 (행 발생 시 복구 경로)."""
    calls = {"count": 0}

    def hangs_once(value):
        calls["count"] += 1
        if calls["count"] == 1:
            raise APITimeoutError(request=httpx.Request("POST", "http://test"))
        return value

    assert call_with_retry(hangs_once, "ok", backoff_seconds=0) == "ok"
    assert calls["count"] == 2
