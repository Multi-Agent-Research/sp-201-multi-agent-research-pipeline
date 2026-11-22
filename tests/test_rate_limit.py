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
from pydantic import BaseModel

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


# ----- A JSON reply that comes back cut off or broken is asked for once more.
# Seen live on 2026-11-22: a Synthesizer draft looped on junk characters until
# the reply was cut off, and the JSON check failed with a stack trace -----

class Reply(BaseModel):
    answer: str


GOOD_JSON = '{"answer": "ok"}'
CUT_OFF_JSON = '{"answer": "o\\u0307\\u0302\\u0307\\u0302'


def text_reply(*parts):
    return LlmResponse(content=types.Content(role="model", parts=list(parts)))


# Gemini's own call, replaced: sends back each reply in turn
def replying_gemini(monkeypatch, replies):
    calls = []
    replies = list(replies)

    async def generate(self, llm_request, stream=False):
        calls.append(llm_request)
        yield replies.pop(0)

    monkeypatch.setattr(Gemini, "generate_content_async", generate)
    return calls


def ask_for_json(model, stream=False):
    request = LlmRequest(model=model.model)
    request.set_output_schema(Reply)

    async def go():
        return [response async for response in model.generate_content_async(request, stream)]
    return asyncio.run(go())


def test_a_cut_off_json_reply_is_dropped_and_asked_for_once_more(monkeypatch, waits, caplog):
    good = text_reply(types.Part(text=GOOD_JSON))
    calls = replying_gemini(monkeypatch, [text_reply(types.Part(text=CUT_OFF_JSON)), good])
    with caplog.at_level(logging.WARNING, logger="rate_limit"):
        responses = ask_for_json(RateLimitedGemini(model="gemini-3.5-flash-lite"))
    # Only the good reply reaches the agent, so only it is counted as a model call
    assert responses == [good]
    assert len(calls) == 2
    assert "cut off or did not parse; asking once more" in caplog.text


def test_two_broken_json_replies_stop_with_a_plain_message(monkeypatch, waits):
    calls = replying_gemini(monkeypatch, [text_reply(types.Part(text=CUT_OFF_JSON)),
                                          text_reply(types.Part(text="not json at all")),
                                          text_reply(types.Part(text=GOOD_JSON))])
    with pytest.raises(rate_limit.MalformedReply, match="cut off or malformed twice, so no report was written"):
        ask_for_json(RateLimitedGemini(model="gemini-3.5-flash-lite"))
    assert len(calls) == 2


def test_a_json_reply_in_a_code_fence_is_accepted_as_adk_accepts_it(monkeypatch, waits):
    fenced = text_reply(types.Part(text="```json\n" + GOOD_JSON + "\n```"))
    calls = replying_gemini(monkeypatch, [fenced])
    assert ask_for_json(RateLimitedGemini(model="gemini-3.5-flash-lite")) == [fenced]
    assert len(calls) == 1


def test_the_models_thoughts_are_not_read_as_its_json(monkeypatch, waits):
    reply = text_reply(types.Part(text="Let me think { about this", thought=True), types.Part(text=GOOD_JSON))
    calls = replying_gemini(monkeypatch, [reply])
    assert ask_for_json(RateLimitedGemini(model="gemini-3.5-flash-lite")) == [reply]
    assert len(calls) == 1


def test_a_plain_text_reply_is_never_checked_as_json(monkeypatch, waits):
    reply = text_reply(types.Part(text="not json {"))
    calls = replying_gemini(monkeypatch, [reply])
    assert ask(RateLimitedGemini(model="gemini-3.5-flash-lite")) == [reply]
    assert len(calls) == 1


def test_a_streamed_json_reply_is_passed_through_unchecked(monkeypatch, waits):
    # A streamed reply arrives in pieces, and a piece on its own is not valid JSON
    piece = text_reply(types.Part(text=CUT_OFF_JSON))
    calls = replying_gemini(monkeypatch, [piece])
    assert ask_for_json(RateLimitedGemini(model="gemini-3.5-flash-lite"), stream=True) == [piece]
    assert len(calls) == 1
