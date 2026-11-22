# Tests for how the rate limiter retries an overloaded Gemini (503). Run from
# the project root:
#     python -m pytest tests
# No test calls Gemini: the real model call is replaced by one that fails on cue,
# and the waits are recorded instead of slept.
import asyncio
import io
import logging
import urllib.error

import pytest
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors as genai_errors
from google.genai import types

import rate_limit
from rate_limit import RateLimitedGemini
from rate_limit import RateLimiter
from rate_limit import overload_delay
from search_agent import FetchError
from search_agent import http_get


def overloaded():
    return genai_errors.ServerError(503, {"error": {
        "code": 503, "message": "This model is currently experiencing high demand.",
        "status": "UNAVAILABLE"}})


# Gemini's own call, replaced: fails with a 503 `failures` times, then answers
def fake_gemini(monkeypatch, failures):
    calls = []

    async def generate(self, llm_request, stream=False):
        calls.append(llm_request)
        if len(calls) <= failures:
            raise overloaded()
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="ok")]))

    monkeypatch.setattr(Gemini, "generate_content_async", generate)
    return calls


# Records every wait instead of sleeping, with no spacing between calls and no randomness
@pytest.fixture
def waits(monkeypatch):
    recorded = []

    async def sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(rate_limit.asyncio, "sleep", sleep)
    monkeypatch.setattr(rate_limit, "model_limiter", lambda: RateLimiter(0))
    monkeypatch.setattr(rate_limit.random, "random", lambda: 0.5)
    return recorded


def ask(model):
    async def go():
        return [response async for response in model.generate_content_async(LlmRequest(model=model.model))]
    return asyncio.run(go())


def test_overload_waits_are_about_10_20_and_40_seconds_within_20_percent():
    for retry, wait in ((1, 10), (2, 20), (3, 40)):
        assert overload_delay(retry, rng=lambda: 0.0) == pytest.approx(wait * 0.8)
        assert overload_delay(retry, rng=lambda: 1.0) == pytest.approx(wait * 1.2)


def test_an_overload_that_clears_is_retried_with_a_plain_line(monkeypatch, waits, caplog):
    calls = fake_gemini(monkeypatch, failures=2)
    with caplog.at_level(logging.WARNING, logger="rate_limit"):
        responses = ask(RateLimitedGemini(model="gemini-3.5-flash-lite"))
    assert len(responses) == 1
    assert len(calls) == 3
    assert waits == [10.0, 20.0]
    assert "Gemini is overloaded right now (503); retry 1 of 3 in 10s" in caplog.text
    assert "Gemini is overloaded right now (503); retry 2 of 3 in 20s" in caplog.text


def test_an_overload_gives_up_after_3_retries(monkeypatch, waits):
    calls = fake_gemini(monkeypatch, failures=10)
    with pytest.raises(genai_errors.ServerError):
        ask(RateLimitedGemini(model="gemini-3.5-flash-lite"))
    assert len(calls) == 4
    assert waits == [10.0, 20.0, 40.0]


# ----- SRS 3.10.1: a 429 is retried with exponential backoff and jitter, up to a
# configurable maximum; a used-up daily quota stops at once -----

def too_many_requests(retry_delay=None, per_day=False):
    details = []
    if retry_delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    if per_day:
        details.append({"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
             "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]})
    return genai_errors.ClientError(429, {"error": {
        "code": 429, "message": "You exceeded your current quota.", "status": "RESOURCE_EXHAUSTED",
        "details": details}})


# Gemini's own call, replaced: raises each error in turn, then answers
def failing_gemini(monkeypatch, errors):
    calls = []
    errors = list(errors)

    async def generate(self, llm_request, stream=False):
        calls.append(llm_request)
        if errors:
            raise errors.pop(0)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="ok")]))

    monkeypatch.setattr(Gemini, "generate_content_async", generate)
    return calls


def test_3_10_1_a_429_is_retried_with_doubling_waits_and_jitter(monkeypatch, waits):
    calls = failing_gemini(monkeypatch, [too_many_requests(), too_many_requests(), too_many_requests()])
    responses = ask(RateLimitedGemini(model="gemini-3.5-flash-lite"))
    assert len(responses) == 1 and len(calls) == 4
    # Full jitter: each wait is random, between 0 and a ceiling that doubles (4, 8, 16 s)
    assert len(waits) == 3
    for wait, ceiling in zip(waits, (4.0, 8.0, 16.0)):
        assert 0.0 <= wait <= ceiling


def test_3_10_1_jitter_stays_between_zero_and_the_doubling_ceiling():
    policy = rate_limit.model_retry_policy()
    for attempt, ceiling in ((1, 4.0), (2, 8.0), (3, 16.0), (5, 60.0)):
        assert policy.delay(attempt, rng=lambda: 0.0) == 0.0
        assert policy.delay(attempt, rng=lambda: 1.0) == pytest.approx(ceiling)


def test_3_10_1_geminis_retry_hint_is_followed_but_capped(monkeypatch, waits):
    failing_gemini(monkeypatch, [too_many_requests("37s"), too_many_requests("500s")])
    ask(RateLimitedGemini(model="gemini-3.5-flash-lite"))
    assert waits == [37.0, 60.0]


def test_3_10_1_the_maximum_attempts_is_configurable(monkeypatch, waits):
    monkeypatch.setenv("SP201_MODEL_MAX_ATTEMPTS", "2")
    calls = failing_gemini(monkeypatch, [too_many_requests()] * 5)
    with pytest.raises(genai_errors.ClientError):
        ask(RateLimitedGemini(model="gemini-3.5-flash-lite"))
    assert len(calls) == 2 and len(waits) == 1


def test_3_10_1_a_used_up_daily_quota_stops_without_retrying(monkeypatch, waits):
    calls = failing_gemini(monkeypatch, [too_many_requests("37s", per_day=True)])
    with pytest.raises(rate_limit.DailyQuotaExceeded, match="daily request quota is used up"):
        ask(RateLimitedGemini(model="gemini-3.5-flash-lite"))
    assert len(calls) == 1 and waits == []


def test_3_10_1_search_sources_retry_a_429_and_honor_retry_after():
    attempts = []

    def opener(request, timeout):
        attempts.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "7"},
                                     io.BytesIO(b""))

    slept = []
    policy = rate_limit.RetryPolicy(max_attempts=3, base_delay=2.0, max_delay=30.0)
    with pytest.raises(FetchError, match="failed after 3 attempts: HTTP 429"):
        http_get("https://example.org/api", {"q": "x"}, RateLimiter(0), policy=policy, opener=opener,
                 sleep=slept.append)
    assert len(attempts) == 3
    assert slept == [7.0, 7.0]
