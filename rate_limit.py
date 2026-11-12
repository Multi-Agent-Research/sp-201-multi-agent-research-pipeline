import asyncio
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
 
from google.adk.models import Gemini
from google.genai import errors as genai_errors
 
logger = logging.getLogger(__name__)
 
# Free-tier requests per minute for the model; spacing is 60 / RPM seconds.
# Set it to your key's limit for SP201_MODEL_ID (AI Studio shows it per model).
MODEL_RPM_ENV = "SP201_MODEL_RPM"
DEFAULT_MODEL_RPM = 10
MODEL_MAX_ATTEMPTS_ENV = "SP201_MODEL_MAX_ATTEMPTS"
DEFAULT_MODEL_MAX_ATTEMPTS = 4
 
# Status codes worth another attempt. Everything else in 4xx is our fault.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
 
# Gemini answers 503 when Google's servers for the model are overloaded. That
# takes longer to clear than a blip, so it gets its own schedule: 3 retries,
# about 10, 20 and 40 seconds apart, each nudged up to 20% either way. A run
# gives up after about 70 seconds.
OVERLOADED_STATUS = 503
OVERLOAD_RETRIES = 3
OVERLOAD_FIRST_WAIT = 10.0
OVERLOAD_JITTER = 0.2
 
 
class RetryableError(Exception):
    """A failure another attempt might fix. retry_after is the server's hint, in seconds."""
 
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after
 
 
class RetriesExhausted(Exception):
    """Every attempt failed with a retryable error."""
 
 
class DailyQuotaExceeded(RuntimeError):
    """The free tier's requests-per-day quota is used up. Retrying will not help."""
 
 
# --------------------------------------------------------------------------
# Spacing
# --------------------------------------------------------------------------
 
class RateLimiter:
    """Keeps at least min_interval seconds between the starts of two requests.
 
    Callers reserve the next free slot under a lock and then sleep outside it,
    so concurrent callers queue up one interval apart instead of all firing
    when the first gap closes.
    """
 
    def __init__(self, min_interval, clock=time.monotonic):
        self.min_interval = max(0.0, float(min_interval))
        self._clock = clock
        self._lock = threading.Lock()
        self._next_slot = 0.0
 
    def reserve(self):
        """Claims the next slot and returns how many seconds to wait for it."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval
            return slot - now
 
    def wait(self, sleep=time.sleep):
        delay = self.reserve()
        if delay > 0:
            sleep(delay)
        return delay
 
    async def wait_async(self):
        delay = self.reserve()
        if delay > 0:
            await asyncio.sleep(delay)
        return delay
 
 
def overload_delay(retry, rng=None):
    """Seconds to wait before overload retry number `retry` (1-based): about 10, 20, 40."""
    if rng is None:
        rng = random.random
    wait = OVERLOAD_FIRST_WAIT * (2 ** (retry - 1))
    return wait * (1 - OVERLOAD_JITTER + 2 * OVERLOAD_JITTER * rng())
 
 
def interval_for_rpm(requests_per_minute):
    rpm = float(requests_per_minute)
    if rpm <= 0:
        return 0.0
    return 60.0 / rpm
 
 
# --------------------------------------------------------------------------
# Backoff
# --------------------------------------------------------------------------
 
@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3        # total tries, including the first
    base_delay: float = 2.0      # seconds; doubles each attempt
    max_delay: float = 30.0      # cap on any single wait, server hint included
 
    def delay(self, attempt, retry_after=None, rng=random.random):
        """Seconds to wait after failed attempt number `attempt` (1-based).
 
        A server hint wins over our own schedule, but is still capped, so one
        bad header cannot stall the run. Without a hint, full jitter: a random
        wait in [0, base * 2^(attempt-1)], which keeps parallel callers from
        retrying in lockstep.
        """
        if retry_after is not None and retry_after >= 0:
            return min(float(retry_after), self.max_delay)
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return ceiling * rng()
 
 
def retry_call(func, policy, sleep=time.sleep, describe="request"):
    """Calls func() until it returns, retrying only on RetryableError.
 
    Any other exception passes straight through. After the last attempt the
    final RetryableError is wrapped in RetriesExhausted.
    """
    attempt = 1
    while True:
        try:
            return func()
        except RetryableError as error:
            if attempt >= policy.max_attempts:
                raise RetriesExhausted(
                    f"{describe} failed after {attempt} attempts: {error}"
                ) from error
            wait = policy.delay(attempt, error.retry_after)
            logger.warning("%s failed (%s); retry %d in %.1fs", describe, error, attempt, wait)
            sleep(wait)
            attempt += 1
 
 
def parse_retry_after(value, now=None):
    """Reads an HTTP Retry-After header (seconds or an HTTP date). None if absent or unreadable."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if now is None:
        now = time.time()
    return max(0.0, when.timestamp() - now)
 
 
# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
 
def model_rpm():
    try:
        return float(os.environ.get(MODEL_RPM_ENV, DEFAULT_MODEL_RPM))
    except ValueError:
        return float(DEFAULT_MODEL_RPM)
 
 
def model_retry_policy():
    try:
        attempts = int(os.environ.get(MODEL_MAX_ATTEMPTS_ENV, DEFAULT_MODEL_MAX_ATTEMPTS))
    except ValueError:
        attempts = DEFAULT_MODEL_MAX_ATTEMPTS
    # Gemini's per-minute 429s usually say "retry in ~40s", so the cap is a minute
    return RetryPolicy(max_attempts=max(1, attempts), base_delay=4.0, max_delay=60.0)
 
 
# One limiter for the whole process: the free-tier quota is per project, not per agent
_model_limiter = None
_model_limiter_lock = threading.Lock()
 
 
def model_limiter():
    global _model_limiter
    with _model_limiter_lock:
        if _model_limiter is None:
            _model_limiter = RateLimiter(interval_for_rpm(model_rpm()))
        return _model_limiter
 
 
def _error_details(error):
    """The 'details' list of a Gemini error body, or [] if the body has none."""
    body = getattr(error, "details", None)
    if not isinstance(body, dict):
        return []
    inner = body.get("error", body)
    if not isinstance(inner, dict):
        return []
    details = inner.get("details")
    return details if isinstance(details, list) else []
 
 
def gemini_retry_delay(error):
    """Seconds from the RetryInfo detail of a Gemini 429 ("retryDelay": "37s"), or None."""
    for detail in _error_details(error):
        if not isinstance(detail, dict):
            continue
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            match = re.match(r"^\s*([0-9.]+)s\s*$", str(detail.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None
 
 
def is_daily_quota_error(error):
    """True when a 429 names a per-day quota, which no amount of waiting fixes today."""
    for detail in _error_details(error):
        if not isinstance(detail, dict):
            continue
        for violation in detail.get("violations", []) or []:
            if not isinstance(violation, dict):
                continue
            quota_id = str(violation.get("quotaId", "")) + str(violation.get("quotaMetric", ""))
            if "PerDay" in quota_id or "per_day" in quota_id.lower():
                return True
    return False
 
 
def is_retryable_model_error(error):
    return isinstance(error, genai_errors.APIError) and error.code in RETRYABLE_STATUS_CODES
 
 
class RateLimitedGemini(Gemini):
    """Gemini with request spacing and 429/5xx backoff.
 
    Retries only when the failed attempt yielded nothing, so a caller never
    sees part of one response followed by another. Retries are logged, not
    counted in model_call_count, which counts responses the agents received.
    """
 
    async def generate_content_async(self, llm_request, stream=False):
        policy = model_retry_policy()
        attempt = 1
        while True:
            await model_limiter().wait_async()
            yielded = False
            try:
                async for response in super().generate_content_async(llm_request, stream):
                    yielded = True
                    yield response
                return
            except genai_errors.APIError as error:
                if yielded or not is_retryable_model_error(error):
                    raise
                if error.code == 429 and is_daily_quota_error(error):
                    raise DailyQuotaExceeded(
                        "Gemini free-tier daily request quota is used up; it resets at "
                        "midnight Pacific. Retrying now would only waste attempts."
                    ) from error
                overloaded = error.code == OVERLOADED_STATUS
                retries = OVERLOAD_RETRIES if overloaded else policy.max_attempts - 1
                if attempt > retries:
                    raise
                if overloaded:
                    wait = overload_delay(attempt)
                    logger.warning("Gemini is overloaded right now (503); retry %d of %d in %.0fs",
                                   attempt, retries, wait)
                else:
                    wait = policy.delay(attempt, gemini_retry_delay(error))
                    logger.warning(
                        "Gemini %s returned %s; retry %d of %d in %.1fs",
                        llm_request.model, error.code, attempt, retries, wait,
                    )
                await asyncio.sleep(wait)
                attempt += 1
 
 
def use_rate_limited_models(agent):
    """Gives every LlmAgent in the tree a RateLimitedGemini in place of its model-ID string.
 
    Agents that already hold a model object are left alone, so a test can still
    plug in a fake model. Returns the agent for chaining.
    """
    model = getattr(agent, "model", None)
    if isinstance(model, str) and model:
        agent.model = RateLimitedGemini(model=model)
    for sub_agent in getattr(agent, "sub_agents", []) or []:
        use_rate_limited_models(sub_agent)
    return agent
