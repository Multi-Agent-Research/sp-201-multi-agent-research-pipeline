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
 
def fetch_wikipedia(query, http=http_get_json):
    """Searches English Wikipedia and returns the plain-text body of the best non-disambiguation hit."""
    limiter = SOURCE_LIMITERS[SourceType.WIKIPEDIA]
    found = http(WIKIPEDIA_API_URL, {
        "action": "query", "list": "search", "srsearch": query, "srnamespace": 0,
        "srlimit": WIKIPEDIA_CANDIDATE_PAGES, "format": "json", "formatversion": 2,
    }, limiter)
    if "error" in found:
        raise FetchError(f"Wikipedia search error: {found['error'].get('info', found['error'])}")
    titles = [hit["title"] for hit in found.get("query", {}).get("search", [])]
 
    for title in titles:
        page_data = http(WIKIPEDIA_API_URL, {
            "action": "query", "prop": "extracts|info|pageprops", "explaintext": 1,
            "inprop": "url", "ppprop": "disambiguation", "redirects": 1,
            "titles": title, "format": "json", "formatversion": 2,
        }, limiter)
        if "error" in page_data:
            raise FetchError(f"Wikipedia page error: {page_data['error'].get('info', page_data['error'])}")
        pages = page_data.get("query", {}).get("pages", [])
        if not pages:
            continue
        page = pages[0]
        if page.get("missing") or "disambiguation" in page.get("pageprops", {}):
            continue
        content = (page.get("extract") or "").strip()
        if not content:
            continue
        url = page.get("fullurl") or "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        return [RawDocument(page.get("title", title), url, "", content)]
    return []
 
 
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
 
 
def passages_of(document, source):
    pieces = split_into_passages(document.content, DEFAULT_PASSAGE_CHARS)
    kept = []
    for piece in pieces:
        if len(piece.strip()) >= MIN_PASSAGE_CHARS:
            kept.append(piece)
        if len(kept) >= PASSAGES_PER_DOCUMENT[source]:
            break
    return kept
 
 
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
 
 
def run_search(store, question_id, sub_question_id, text, preferred_sources, fetchers=None):
    """Searches one sub-question, saves its passages, and returns its SubQuestionEvidence."""
    if fetchers is None:
        fetchers = DEFAULT_FETCHERS
    register_sub_question(store, question_id, sub_question_id, text)
 
    evidence_ids = []
    errors = []
    tried = []
 
    def search_source(source):
        tried.append(source)
        fetch = fetchers.get(source)
        if fetch is None:
            errors.append(f"{source.value}: no fetcher configured")
            return
        try:
            documents = fetch(text)
        except FetchError as error:
            errors.append(f"{source.value}: {short_error(error)}")
            return
        except Exception as error:  # a bug in a fetcher must not take the run down
            logger.exception("%s fetcher crashed", source.value)
            errors.append(f"{source.value}: unexpected {type(error).__name__}: {short_error(error)}")
            return
        retrieved_at = utc_now()
        for document in documents:
            for passage in passages_of(document, source):
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
    "your plan, with its sub_question_id, its text and its preferred_sources. "
    "Search at most LIMIT sub-questions and never the same one twice.\n"
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
 
    def __init__(self, store=None, fetchers=None):
        self.store = store if store is not None else EvidenceStore()
        self.fetchers = fetchers if fetchers is not None else DEFAULT_FETCHERS
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
 
    async def search(self, sub_question_id, text, preferred_sources, tool_context):
        question_id = tool_context.state.get(STATE_QUESTION_ID)
        if question_id is None:
            raise RuntimeError(
                f"no {STATE_QUESTION_ID} in state; build the Manager with attach_search() "
                "so register_question runs first"
            )
        try:
            sub_question_id = int(sub_question_id)
        except (TypeError, ValueError):
            return failed(1, f"sub_question_id must be an integer, got {sub_question_id!r}").model_dump(mode="json")
        if sub_question_id < 1:
            return failed(1, "sub_question_id must be 1 or more").model_dump(mode="json")
        text = (text or "").strip()
        if len(text) < 5:
            return failed(sub_question_id, "sub-question text is too short to search").model_dump(mode="json")
 
        runs = self.registry.setdefault(question_id, {})
        task = runs.get(sub_question_id)
        if task is None:
            limit = max_sub_questions()
            if len(runs) >= limit:
                # Not recorded: the plan will be capped to `limit` anyway
                return failed(
                    sub_question_id,
                    f"search limit reached: at most {limit} sub-questions are searched per question",
                ).model_dump(mode="json")
            task = asyncio.ensure_future(asyncio.to_thread(
                run_search, self.store, question_id, sub_question_id, text,
                preferred_sources, self.fetchers,
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
            sub_question_id: int, text: str, preferred_sources: list[str], tool_context
        ) -> dict:
            """Searches the given sources for one sub-question and stores the passages.
 
            Args:
              sub_question_id: The sub-question's id from your plan (1, 2, 3...).
              text: The sub-question text, exactly as in your plan.
              preferred_sources: Sources to search, from wikipedia, arxiv and web.
 
            Returns:
              sub_question_id, status (found, empty or failed), evidence_ids, and
              error (what went wrong, if anything).
            """
            return await self.search(sub_question_id, text, preferred_sources, tool_context)
 
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
 
 
def attach_search(manager, store=None, fetchers=None):
    """Gives the Manager the search tool, its instructions, and the two callbacks around it.
 
    register_question runs before the Manager's own before_agent_callback.
    finalize_search_results runs after the Manager's cap callback, so it sees the
    capped plan. Returns the SearchTool, whose store the pipeline should also
    read evidence from.
    """
    search_tool = SearchTool(store=store, fetchers=fetchers)
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
    parser.add_argument("--sources", default="wikipedia,arxiv",
                        help="comma-separated preferred sources (default wikipedia,arxiv)")
    args = parser.parse_args()
 
    with tempfile.TemporaryDirectory() as scratch:
        store = EvidenceStore(Path(scratch) / "demo.db")
        question_id = store.add_question("Demo question")
 
        if args.live:
            fetchers = DEFAULT_FETCHERS
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
        result = run_search(store, question_id, 1, text, sources, fetchers)
        print(f"\nSub-question: {text}")
        print("Returned to the Manager:", json.dumps(result.model_dump(mode="json"), indent=1))
        for row in store.read_evidence(result.evidence_ids):
            print(f"\n[{row.evidence_id}] {row.source_type} | {row.source_title} | {row.source_url}")
            print("   " + row.passage[:300].replace("\n", " ") + ("..." if len(row.passage) > 300 else ""))
 
 
if __name__ == "__main__":
    main()
