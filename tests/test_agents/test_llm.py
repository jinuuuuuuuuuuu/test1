"""call_with_retry — CLOVA 플레이크(간헐 400/429) 흡수 로직 검증. 네트워크 호출 없음."""

import httpx
import pytest
from openai import APIError

from src.agents.llm import call_with_retry


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
