import argparse
import asyncio
import json
import logging
import re
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple
 
from dotenv import load_dotenv
from google.adk.tools import FunctionTool
 
from evidence_store import DEFAULT_PASSAGE_CHARS
from evidence_store import EvidenceStore
from evidence_store import EvidenceStoreError
from evidence_store import split_into_passages
from evidence_store import utc_now
from rate_limit import RETRYABLE_STATUS_CODES
from rate_limit import RateLimiter
from rate_limit import RetriesExhausted
from rate_limit import RetryableError
from rate_limit import RetryPolicy
from rate_limit import parse_retry_after
from rate_limit import retry_call
from schemas import EvidenceRecord
from schemas import ManagerPlan
from schemas import STATE_PLAN
from schemas import STATE_SEARCH_RESULTS
from schemas import STATE_USER_QUESTION
from schemas import SearchResults
from schemas import SearchStatus
from schemas import SourceType
from schemas import SubQuestion
from schemas import SubQuestionEvidence
from schemas import max_sub_questions
 
try:
    from schemas import STATE_QUESTION_ID
except ImportError:
    # Until schemas.py defines it; then this fallback can go
    STATE_QUESTION_ID = "question_id"
 
logger = logging.getLogger(__name__)
 
SEARCH_TOOL_NAME = "search_sub_question"
 
# Wikipedia rejects the default Python User-Agent; its API policy asks for one
# that names the project and a contact. arXiv and DuckDuckGo get the same one.
USER_AGENT = (
    "SP201-multi-agent-research/0.1 "
    "(KSU CS 4850 senior project; https://github.com/Multi-Agent-Research/sp-201-multi-agent-research)"
)
HTTP_TIMEOUT_SECONDS = 30
 
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
 
# How much one sub-question may add to the Synthesizer's prompt. Every passage
# stored here is placed in that prompt, so this bounds input tokens per minute.
MAX_EVIDENCE_PER_SUB_QUESTION = 4
WIKIPEDIA_CANDIDATE_PAGES = 3      # search hits tried, skipping disambiguation pages
ARXIV_MAX_RESULTS = 2
WEB_MAX_RESULTS = 2
PASSAGES_PER_DOCUMENT = {
    SourceType.WIKIPEDIA: 2,       # the lead of one article, in up to two passages
    SourceType.ARXIV: 1,           # one abstract per paper
    SourceType.WEB: 1,             # one result snippet per page
}
MIN_PASSAGE_CHARS = 20             # shorter pieces are headings or debris
MAX_ERROR_CHARS = 200              # per source; the error goes back into the Manager's context
MAX_BUZZ_WORDS = 5
# A call the tool refused before searching starts its error with this, so the
# pipeline can tell "not searched" from "searched and found nothing"
NOT_SEARCHED_PREFIX = "not searched: "
 
# Spacing per source. arXiv's API terms ask for no more than one request every
# three seconds. Wikipedia asks for serial requests; one a second is polite.
# DuckDuckGo has no official API and blocks bursts, so it gets the widest gap
# that does not slow a run noticeably.
SOURCE_LIMITERS = {
    SourceType.WIKIPEDIA: RateLimiter(1.0),
    SourceType.ARXIV: RateLimiter(3.0),
    SourceType.WEB: RateLimiter(2.0),
}
SOURCE_RETRY_POLICY = RetryPolicy(max_attempts=3, base_delay=2.0, max_delay=30.0)
 
STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from has have how in into is it its "
    "of on or that the their there these this to was were what when where which who whom why "
    "will with would current currently".split()
)
 
 
class FetchError(Exception):
    """A source could not be searched. The message says why, for SubQuestionEvidence.error."""
 
 
class RawDocument(NamedTuple):
    title: str
    url: str
    published_date: str
    content: str
 
 
# --------------------------------------------------------------------------
# HTTP with spacing and retry
# --------------------------------------------------------------------------
 
def http_get(url, params, limiter, policy=SOURCE_RETRY_POLICY, opener=None, sleep=time.sleep):
    """GETs url?params and returns the body bytes.
 
    Waits for the limiter before every attempt, retries 408/429/5xx, timeouts and
    connection errors under policy (honouring Retry-After), and raises FetchError
    for anything else or once the retries run out.
    """
    if opener is None:
        opener = urllib.request.urlopen
    full_url = url + "?" + urllib.parse.urlencode(params)
    host = urllib.parse.urlparse(url).netloc
 
    def attempt():
        limiter.wait(sleep)
        request = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
        try:
            with opener(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in RETRYABLE_STATUS_CODES:
                retry_after = parse_retry_after(error.headers.get("Retry-After") if error.headers else None)
                raise RetryableError(f"HTTP {error.code} from {host}", retry_after) from error
            raise FetchError(f"HTTP {error.code} from {host}") from error
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as error:
            reason = getattr(error, "reason", error)
            raise RetryableError(f"could not reach {host}: {reason}") from error
 
    try:
        return retry_call(attempt, policy, sleep=sleep, describe=f"GET {host}")
    except RetriesExhausted as error:
        raise FetchError(str(error)) from error
 
 
def http_get_json(url, params, limiter, **kwargs):
    body = http_get(url, params, limiter, **kwargs)
    try:
        return json.loads(body)
    except ValueError as error:
        raise FetchError(f"{urllib.parse.urlparse(url).netloc} returned something that is not JSON") from error
 
 
# --------------------------------------------------------------------------
# Sources. Each returns a list of RawDocument ([] means searched, nothing found)
# or raises FetchError.
# --------------------------------------------------------------------------
 
def wikipedia_page(title, http=http_get_json):
    """One article's plain text, following redirects.

    None if the article is missing, a disambiguation page or empty.
    """
    limiter = SOURCE_LIMITERS[SourceType.WIKIPEDIA]
    page_data = http(WIKIPEDIA_API_URL, {
        "action": "query", "prop": "extracts|info|pageprops", "explaintext": 1,
        "inprop": "url", "ppprop": "disambiguation", "redirects": 1,
        "titles": title, "format": "json", "formatversion": 2,
    }, limiter)
    if "error" in page_data:
        raise FetchError(f"Wikipedia page error: {page_data['error'].get('info', page_data['error'])}")
    pages = page_data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    if page.get("missing") or "disambiguation" in page.get("pageprops", {}):
        return None
    content = (page.get("extract") or "").strip()
    if not content:
        return None
    url = page.get("fullurl") or "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
    return RawDocument(page.get("title", title), url, "", content)
 
 
def fetch_wikipedia(query, http=http_get_json):
    """Searches English Wikipedia: its top hits in rank order, skipping disambiguation pages."""
    limiter = SOURCE_LIMITERS[SourceType.WIKIPEDIA]
    found = http(WIKIPEDIA_API_URL, {
        "action": "query", "list": "search", "srsearch": query, "srnamespace": 0,
        "srlimit": WIKIPEDIA_CANDIDATE_PAGES, "format": "json", "formatversion": 2,
    }, limiter)
    if "error" in found:
        raise FetchError(f"Wikipedia search error: {found['error'].get('info', found['error'])}")
    documents = []
    for hit in found.get("query", {}).get("search", []):
        document = wikipedia_page(hit["title"], http)
        if document is not None:
            documents.append(document)
    return documents
 
 
def fetch_wikipedia_article(name, http=http_get_json):
    """Looks one article up by its exact name, following redirects. [] if there is none."""
    document = wikipedia_page(name, http)
    return [] if document is None else [document]
 
 
def arxiv_keywords(text, limit=5):
    """Content words of a sub-question, for arXiv's field query syntax."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", text.lower())
    keywords = []
    for word in words:
        if word in STOPWORDS or len(word) < 2 or word in keywords:
            continue
        keywords.append(word)
    return keywords[:limit]
 
 
def parse_arxiv_feed(body):
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise FetchError("arXiv returned malformed XML") from error
    documents = []
    for entry in root.findall("atom:entry", ATOM):
        entry_id = (entry.findtext("atom:id", "", ATOM) or "").strip()
        title = " ".join((entry.findtext("atom:title", "", ATOM) or "").split())
        # arXiv reports a bad query as a single entry whose id is an error URL
        if "/api/errors" in entry_id:
            summary = " ".join((entry.findtext("atom:summary", "", ATOM) or "").split())
            raise FetchError(f"arXiv rejected the query: {summary or title}")
        summary = " ".join((entry.findtext("atom:summary", "", ATOM) or "").split())
        published = (entry.findtext("atom:published", "", ATOM) or "")[:10]
        if entry_id and summary:
            documents.append(RawDocument(title or entry_id, entry_id, published, summary))
    return documents
 
 
def fetch_arxiv(query, http=http_get):
    """Searches arXiv abstracts: all keywords first, then any keyword if that finds nothing."""
    keywords = arxiv_keywords(query)
    if not keywords:
        return []
    limiter = SOURCE_LIMITERS[SourceType.ARXIV]
    for joiner in (" AND ", " OR "):
        search = joiner.join("all:" + word for word in keywords)
        body = http(ARXIV_API_URL, {
            "search_query": search, "start": 0, "max_results": ARXIV_MAX_RESULTS,
            "sortBy": "relevance", "sortOrder": "descending",
        }, limiter)
        documents = parse_arxiv_feed(body)
        if documents or len(keywords) == 1:
            return documents
    return []
 
 
def fetch_web(query, ddgs_factory=None, policy=SOURCE_RETRY_POLICY, sleep=time.sleep):
    """DuckDuckGo text search. The library is unofficial, so every failure mode is mapped explicitly."""
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import DuckDuckGoSearchException
        from duckduckgo_search.exceptions import RatelimitException
        from duckduckgo_search.exceptions import TimeoutException
    except ImportError as error:
        raise FetchError("duckduckgo-search is not installed") from error
    if ddgs_factory is None:
        ddgs_factory = lambda: DDGS(headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT_SECONDS)
    limiter = SOURCE_LIMITERS[SourceType.WEB]
 
    def attempt():
        limiter.wait(sleep)
        try:
            return ddgs_factory().text(query, max_results=WEB_MAX_RESULTS) or []
        except RatelimitException as error:
            raise RetryableError(f"DuckDuckGo rate limit: {error}") from error
        except TimeoutException as error:
            raise RetryableError(f"DuckDuckGo timed out: {error}") from error
        except DuckDuckGoSearchException as error:
            raise FetchError(f"DuckDuckGo error: {error}") from error
 
    try:
        results = retry_call(attempt, policy, sleep=sleep, describe="DuckDuckGo search")
    except RetriesExhausted as error:
        raise FetchError(str(error)) from error
 
    documents = []
    for result in results:
        url = (result.get("href") or "").strip()
        body = (result.get("body") or "").strip()
        if url and body:
            documents.append(RawDocument((result.get("title") or url).strip(), url, "", body))
    return documents
 
 
DEFAULT_FETCHERS = {
    SourceType.WIKIPEDIA: fetch_wikipedia,
    SourceType.ARXIV: fetch_arxiv,
    SourceType.WEB: fetch_web,
}
 
# Sources that can look an article up by its exact name; tried before any search
DEFAULT_ARTICLE_FETCHERS = {
    SourceType.WIKIPEDIA: fetch_wikipedia_article,
}
 
 
# --------------------------------------------------------------------------
# One sub-question, end to end (blocking; runs in a worker thread)
# --------------------------------------------------------------------------
 
def normalize_sources(preferred_sources):
    """Valid SourceTypes in the Manager's order, without repeats. Unknown names are dropped."""
    sources = []
    for name in preferred_sources or []:
        try:
            source = SourceType(str(getattr(name, "value", name)).strip().lower())
        except ValueError:
            logger.warning("ignoring unknown source %r", name)
            continue
        if source not in sources:
            sources.append(source)
    if not sources:
        sources.append(SourceType.WIKIPEDIA)
    return sources
 
 
def clean_buzz_words(buzz_words):
    """The buzz words, trimmed, without repeats, at most MAX_BUZZ_WORDS. The first is the main name."""
    words = []
    for word in buzz_words or []:
        word = " ".join(str(word).split())
        if word and word.lower() not in [kept.lower() for kept in words]:
            words.append(word)
    return words[:MAX_BUZZ_WORDS]
 
 
def mentions(text, phrase):
    """True when text contains phrase as whole words, ignoring case, allowing a plural
    ending, and treating hyphens and spaces alike ("retrieval augmented generation"
    matches "Retrieval-augmented generation")."""
    words = [re.escape(word) for word in re.split(r"[\s\-]+", phrase or "") if word]
    if not words:
        return False
    pattern = r"(?<![A-Za-z0-9])" + r"[\s\-]+".join(words) + r"(?:s|es)?(?![A-Za-z0-9])"
    return re.search(pattern, text or "", re.IGNORECASE) is not None
 
 
def passages_of(document, source, buzz_words=()):
    """The passages of a document worth saving, at most PASSAGES_PER_DOCUMENT[source],
    in their original order. With buzz words, the best-matching passages anywhere in
    the document are chosen, not just the first ones: 3 points for the main name and 1
    for each other buzz word. If none matches, or there are no buzz words, the first
    passages are kept."""
    pieces = [piece for piece in split_into_passages(document.content, DEFAULT_PASSAGE_CHARS)
              if len(piece.strip()) >= MIN_PASSAGE_CHARS]
    limit = PASSAGES_PER_DOCUMENT[source]
    if not buzz_words:
        return pieces[:limit]
    main, others = buzz_words[0], buzz_words[1:]
    scores = []
    for position, piece in enumerate(pieces):
        score = 3 * mentions(piece, main) + sum(mentions(piece, word) for word in others)
        if score > 0:
            scores.append((score, position))
    if not scores:
        return pieces[:limit]
    best = sorted(scores, key=lambda item: (-item[0], item[1]))[:limit]
    return [pieces[position] for _, position in sorted(best, key=lambda item: item[1])]
 
 
def search_steps(source, text, buzz_words, fetchers, article_fetchers):
    """(step name, fetch, query, gated) in the order a source is tried: the main name as
    an exact article (sources that can look one up), then the buzz words, then the full
    sub-question. A gated step's documents must mention the main name; the exact
    article is the one the Manager named, so it is not gated."""
    steps = []
    main = buzz_words[0] if buzz_words else ""
    if main and source in article_fetchers:
        steps.append(("article name", article_fetchers[source], main, False))
    if source in fetchers:
        if buzz_words:
            steps.append(("buzz words", fetchers[source], " ".join(buzz_words), True))
        steps.append(("sub-question", fetchers[source], text, True))
    return steps
 
 
def register_sub_question(store, question_id, sub_question_id, text):
    """Adds the sub_question row add_evidence needs. A row that already exists is fine."""
    plan = ManagerPlan(
        user_question="(registered by Search)",
        sub_questions=[SubQuestion(
            sub_question_id=sub_question_id, text=text,
            preferred_sources=[SourceType.WIKIPEDIA], priority=1,
        )],
    )
    try:
        store.add_sub_questions(question_id, plan)
    except EvidenceStoreError:
        pass  # UNIQUE (question_id, plan_number): already registered
 
 
def run_search(store, question_id, sub_question_id, text, preferred_sources, buzz_words=(),
               fetchers=None, article_fetchers=None):
    """Searches one sub-question, saves its passages, and returns its SubQuestionEvidence.
 
    Each source is tried in the order search_steps gives and stops at the first step
    that finds something usable. A document from a search is used only if it
    mentions the main name (the first buzz word), so an off-topic article is never
    saved; Wikipedia keeps one article, the other sources what passed.
    """
    if fetchers is None:
        fetchers = DEFAULT_FETCHERS
    if article_fetchers is None:
        article_fetchers = {}
    buzz_words = clean_buzz_words(buzz_words)
    main = buzz_words[0] if buzz_words else ""
    register_sub_question(store, question_id, sub_question_id, text)
 
    evidence_ids = []
    errors = []
    tried = []
 
    def save(documents, source):
        retrieved_at = utc_now()
        for document in documents:
            for passage in passages_of(document, source, buzz_words):
                if len(evidence_ids) >= MAX_EVIDENCE_PER_SUB_QUESTION:
                    return
                record = EvidenceRecord(
                    sub_question_id=sub_question_id,
                    source_url=document.url,
                    source_title=document.title or document.url,
                    published_date=document.published_date or "",
                    passage=passage,
                    retrieved_at=retrieved_at,
                    source_type=source,
                )
                evidence_ids.append(store.add_evidence(question_id, record))
 
    def search_source(source):
        tried.append(source)
        steps = search_steps(source, text, buzz_words, fetchers, article_fetchers)
        if not steps:
            errors.append(f"{source.value}: no fetcher configured")
            return
        for step, fetch, query, gated in steps:
            try:
                documents = list(fetch(query))
            except FetchError as error:
                # the service itself failed, so its later steps would too
                errors.append(f"{source.value}: {short_error(error)}")
                return
            except Exception as error:  # a bug in a fetcher must not take the run down
                logger.exception("%s fetcher crashed", source.value)
                errors.append(f"{source.value}: unexpected {type(error).__name__}: {short_error(error)}")
                return
            if gated and main:
                documents = [document for document in documents if mentions(document.content, main)]
            if source == SourceType.WIKIPEDIA:
                documents = documents[:1]
            if documents:
                logger.info("%s, sub-question %s: %s step used %s", source.value, sub_question_id,
                            step, [document.title for document in documents])
                save(documents, source)
                return
            logger.info("%s, sub-question %s: %s step found nothing that mentions %r",
                        source.value, sub_question_id, step, main)
 
    for source in normalize_sources(preferred_sources):
        if len(evidence_ids) >= MAX_EVIDENCE_PER_SUB_QUESTION:
            break
        search_source(source)
    if not evidence_ids and SourceType.WEB not in tried:
        search_source(SourceType.WEB)  # README: web search is the fallback
 
    error_text = "; ".join(errors)
    if evidence_ids:
        return SubQuestionEvidence(
            sub_question_id=sub_question_id, status=SearchStatus.FOUND,
            evidence_ids=evidence_ids, error=error_text,
        )
    if errors:
        return SubQuestionEvidence(
            sub_question_id=sub_question_id, status=SearchStatus.FAILED, error=error_text,
        )
    return SubQuestionEvidence(sub_question_id=sub_question_id, status=SearchStatus.EMPTY)
 
 
def short_error(error):
    text = " ".join(str(error).split())
    if len(text) > MAX_ERROR_CHARS:
        text = text[:MAX_ERROR_CHARS].rstrip() + "..."
    return text
 
 
def failed(sub_question_id, message):
    return SubQuestionEvidence(
        sub_question_id=max(1, int(sub_question_id)), status=SearchStatus.FAILED, error=message,
    )
 
 
# --------------------------------------------------------------------------
# The tool and the Manager callbacks around it
# --------------------------------------------------------------------------
 
SEARCH_INSTRUCTION = (
    "\n\nSearching:\n"
    "7. After planning, call search_sub_question once for each sub-question in "
    "your plan, with its sub_question_id, its text, its preferred_sources and "
    "2 to 5 buzz_words. The first buzz word is the main topic's full name, as "
    "Wikipedia would title its article, never an abbreviation (\"User Datagram "
    "Protocol\", not \"UDP\"); the others are the key terms a search engine "
    "needs to find the answer. Search at most LIMIT sub-questions and never the "
    "same one twice, unless the tool says it was not searched.\n"
    "8. The tool returns a status and evidence ids only. You do not read or "
    "judge evidence. A status of empty or failed is recorded for the report, so "
    "do not search again or rephrase to get a different answer.\n"
    "9. When every sub-question has been searched, return the plan JSON, with "
    "the same sub_question_id values you searched."
)
 
 
class SearchTool:
    """Holds the evidence store, the fetchers, and each question's search outcomes.
 
    One SearchTool belongs to one Manager. Its registry maps question_id to
    {sub_question_id: asyncio.Task}; a task per sub-question means a repeated
    call waits for the first one instead of fetching and storing twice.
    """
 
    def __init__(self, store=None, fetchers=None, article_fetchers=None):
        self.store = store if store is not None else EvidenceStore()
        self.fetchers = fetchers if fetchers is not None else DEFAULT_FETCHERS
        if article_fetchers is None:
            # Caller-supplied fetchers (tests) get no real lookups unless they ask
            article_fetchers = DEFAULT_ARTICLE_FETCHERS if fetchers is None else {}
        self.article_fetchers = article_fetchers
        self.registry = {}
 
    # ----- callbacks -----
 
    def register_question(self, callback_context):
        """before_agent_callback: records the user's question so evidence has a question_id."""
        state = callback_context.state
        question_id = self.store.add_question(state[STATE_USER_QUESTION])
        state[STATE_QUESTION_ID] = question_id
        self.registry[question_id] = {}
        return None
 
    def finalize_search_results(self, callback_context):
        """after_agent_callback: writes SearchResults for exactly the sub-questions in the capped plan."""
        state = callback_context.state
        runs = self.registry.pop(state.get(STATE_QUESTION_ID), {})
        raw_plan = state.get(STATE_PLAN)
        if raw_plan is None:
            return None
        plan = raw_plan if isinstance(raw_plan, ManagerPlan) else ManagerPlan.model_validate(raw_plan)
        results = []
        for sub_question in plan.sub_questions:
            task = runs.pop(sub_question.sub_question_id, None)
            results.append(task_outcome(task, sub_question.sub_question_id))
        if runs:
            logger.info("dropping searches for sub-questions cut from the plan: %s", sorted(runs))
        state[STATE_SEARCH_RESULTS] = SearchResults(results=results).model_dump(mode="json")
        return None
 
    # ----- the tool body -----
 
    async def search(self, sub_question_id, text, preferred_sources, buzz_words, tool_context):
        question_id = tool_context.state.get(STATE_QUESTION_ID)
        if question_id is None:
            raise RuntimeError(
                f"no {STATE_QUESTION_ID} in state; build the Manager with attach_search() "
                "so register_question runs first"
            )
        try:
            sub_question_id = int(sub_question_id)
        except (TypeError, ValueError):
            message = f"sub_question_id must be an integer, got {sub_question_id!r}"
            return failed(1, NOT_SEARCHED_PREFIX + message).model_dump(mode="json")
        if sub_question_id < 1:
            message = "sub_question_id must be 1 or more"
            return failed(1, NOT_SEARCHED_PREFIX + message).model_dump(mode="json")
        text = (text or "").strip()
        if len(text) < 5:
            return failed(sub_question_id, NOT_SEARCHED_PREFIX + "sub-question text is too short to search"
                          ).model_dump(mode="json")
        buzz_words = clean_buzz_words(buzz_words)
        if not buzz_words:
            return failed(sub_question_id, NOT_SEARCHED_PREFIX + "buzz_words is empty; call again with the "
                          "main topic's full name first").model_dump(mode="json")
 
        runs = self.registry.setdefault(question_id, {})
        task = runs.get(sub_question_id)
        if task is None:
            limit = max_sub_questions()
            if len(runs) >= limit:
                # Not recorded: the plan will be capped to `limit` anyway
                return failed(
                    sub_question_id,
                    NOT_SEARCHED_PREFIX + f"search limit reached: at most {limit} sub-questions are "
                    "searched per question",
                ).model_dump(mode="json")
            task = asyncio.ensure_future(asyncio.to_thread(
                run_search, self.store, question_id, sub_question_id, text,
                preferred_sources, buzz_words, self.fetchers, self.article_fetchers,
            ))
            runs[sub_question_id] = task
        try:
            result = await asyncio.shield(task)
        except Exception as error:
            logger.exception("search for sub-question %s failed", sub_question_id)
            result = failed(sub_question_id, f"search crashed: {type(error).__name__}: {error}")
        return result.model_dump(mode="json")
 
    def as_function_tool(self):
        async def search_sub_question(
            sub_question_id: int, text: str, preferred_sources: list[str], buzz_words: list[str],
            tool_context
        ) -> dict:
            """Searches the given sources for one sub-question and stores the passages.
 
            Args:
              sub_question_id: The sub-question's id from your plan (1, 2, 3...).
              text: The sub-question text, exactly as in your plan.
              preferred_sources: Sources to search, from wikipedia, arxiv and web.
              buzz_words: 2 to 5 search terms. The first is the main topic's full
                name as Wikipedia would title its article, never an abbreviation.
 
            Returns:
              sub_question_id, status (found, empty or failed), evidence_ids, and
              error (what went wrong, if anything).
            """
            return await self.search(sub_question_id, text, preferred_sources, buzz_words, tool_context)
 
        return FunctionTool(search_sub_question)
 
 
def task_outcome(task, sub_question_id):
    """The SubQuestionEvidence a finished search task produced, or a failure explaining why not."""
    if task is None:
        return failed(sub_question_id, "the Manager did not search this sub-question")
    if not task.done():
        task.cancel()
        return failed(sub_question_id, "search did not finish before the Manager returned its plan")
    if task.cancelled():
        return failed(sub_question_id, "search was cancelled")
    if task.exception() is not None:
        error = task.exception()
        return failed(sub_question_id, f"search crashed: {type(error).__name__}: {error}")
    return task.result()
 
 
def as_callback_list(existing):
    if existing is None:
        return []
    if isinstance(existing, list):
        return list(existing)
    return [existing]
 
 
def attach_search(manager, store=None, fetchers=None, article_fetchers=None):
    """Gives the Manager the search tool, its instructions, and the two callbacks around it.
 
    register_question runs before the Manager's own before_agent_callback.
    finalize_search_results runs after the Manager's cap callback, so it sees the
    capped plan. Returns the SearchTool, whose store the pipeline should also
    read evidence from.
    """
    search_tool = SearchTool(store=store, fetchers=fetchers, article_fetchers=article_fetchers)
    manager.tools = list(manager.tools or []) + [search_tool.as_function_tool()]
    if isinstance(manager.instruction, str):
        manager.instruction += SEARCH_INSTRUCTION.replace("LIMIT", str(max_sub_questions()))
    manager.before_agent_callback = (
        [search_tool.register_question] + as_callback_list(manager.before_agent_callback)
    )
    manager.after_agent_callback = (
        as_callback_list(manager.after_agent_callback) + [search_tool.finalize_search_results]
    )
    return search_tool
 
 
# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
 
def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Search one sub-question into a scratch evidence store.")
    parser.add_argument("--live", metavar="SUB_QUESTION",
                        help="hit the real sources (network needed, no API key)")
    parser.add_argument("--buzz", default="",
                        help="comma-separated buzz words, main name first (default: none)")
    parser.add_argument("--sources", default="wikipedia,arxiv",
                        help="comma-separated preferred sources (default wikipedia,arxiv)")
    args = parser.parse_args()
 
    with tempfile.TemporaryDirectory() as scratch:
        store = EvidenceStore(Path(scratch) / "demo.db")
        question_id = store.add_question("Demo question")
 
        article_fetchers = {}
        if args.live:
            fetchers = DEFAULT_FETCHERS
            article_fetchers = DEFAULT_ARTICLE_FETCHERS
            text = args.live
        else:
            # Canned fetchers, so the demo runs offline: wikipedia finds something,
            # arxiv errors, and the error is reported, not saved as a passage
            def fake_wikipedia(query):
                return [RawDocument("Transformer (deep learning)", "https://example.org/transformer", "",
                                    "A transformer is a neural network architecture based on attention.")]
 
            def fake_arxiv(query):
                raise FetchError("HTTP 503 from export.arxiv.org (demo)")
 
            fetchers = {SourceType.WIKIPEDIA: fake_wikipedia, SourceType.ARXIV: fake_arxiv,
                        SourceType.WEB: lambda query: []}
            text = "What is a transformer model?"
 
        sources = [name.strip() for name in args.sources.split(",") if name.strip()]
        buzz_words = [word.strip() for word in args.buzz.split(",") if word.strip()]
        result = run_search(store, question_id, 1, text, sources, buzz_words, fetchers, article_fetchers)
        print(f"\nSub-question: {text}")
        print("Returned to the Manager:", json.dumps(result.model_dump(mode="json"), indent=1))
        for row in store.read_evidence(result.evidence_ids):
            print(f"\n[{row.evidence_id}] {row.source_type} | {row.source_title} | {row.source_url}")
            print("   " + row.passage[:300].replace("\n", " ") + ("..." if len(row.passage) > 300 else ""))
 
 
if __name__ == "__main__":
    main()
