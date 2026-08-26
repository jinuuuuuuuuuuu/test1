"""call_with_retry — CLOVA 플레이크(간헐 400/429) 흡수 로직 검증. 네트워크 호출 없음."""

import httpx
import pytest
from openai import APIError, RateLimitError

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


# ── 429(rate limit) 별도 예산 (F-6) ───────────────────────────────────
#
# 429는 호출량 윈도우가 열려야 성공하므로 간헐적 400과 회복 시간이 다르다.
# 실측: 라우터 연속 30회 호출 시 429가 반복 발생, 기존 정책(총 4.5초 대기)으로는
# 전부 소진돼 무응답이 됐다. 평가는 단발 호출이라 무응답 = 그 문항 0점이다.

def _rate_limit_error() -> RateLimitError:
    response = httpx.Response(
        429, request=httpx.Request("POST", "http://test"), json={"error": {"code": "42901"}}
    )
    return RateLimitError("rate exceeded", response=response, body=None)


def test_recovers_from_rate_limit_with_longer_budget(monkeypatch):
    """429는 5회까지 재시도한다 — 기존 3회 예산으로는 못 넘기던 실패."""
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 5:
            raise _rate_limit_error()
        return "ok"

    assert call_with_retry(flaky) == "ok"
    assert calls["count"] == 5
    # 백오프가 점증해야 윈도우가 열릴 시간을 번다
    assert slept == [8.0, 16.0, 24.0, 32.0]


def test_rate_limit_budget_is_separate_from_transient(monkeypatch):
    """429가 400 재시도 기회를 잡아먹으면 안 된다 — 각자 예산을 쓴다."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"count": 0}

    def mixed():
        calls["count"] += 1
        # 429 두 번 → 400 두 번 → 성공. 어느 쪽 예산도 단독으로는 소진되지 않는다.
        if calls["count"] <= 2:
            raise _rate_limit_error()
        if calls["count"] <= 4:
            raise _transient_error()
        return "ok"

    assert call_with_retry(mixed) == "ok"
    assert calls["count"] == 5


def test_gives_up_after_rate_limit_budget_exhausted(monkeypatch):
    """무한 재시도는 없다 — 예산이 끝나면 마지막 오류를 던진다."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    def always_limited():
        raise _rate_limit_error()

    with pytest.raises(RateLimitError):
        call_with_retry(always_limited)
