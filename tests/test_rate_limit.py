# Tests for how the rate limiter retries an overloaded Gemini (503). Run from
# the project root:
#     python -m pytest tests
# No test calls Gemini: the real model call is replaced by one that fails on cue,
# and the waits are recorded instead of slept.
import asyncio
import logging

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
