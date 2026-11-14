import argparse
import asyncio
import json
import logging
import re
import socket
import tempfile
import threading
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
# stored here is placed in that prompt, so this bounds input tokens per minute:
# the main Wikipedia article's 6, one opening passage for each of up to 4 other
# buzz words, 2 arXiv abstracts, and up to 3 passages from each of 2 web pages,
# all under this one cap. At 1,500 characters a passage, 3 sub-questions stay
# under about 14,000 tokens.
MAX_EVIDENCE_PER_SUB_QUESTION = 12
WIKIPEDIA_CANDIDATE_PAGES = 3      # search hits tried, skipping disambiguation pages
ARXIV_MAX_RESULTS = 2
WEB_MAX_RESULTS = 8                # search hits asked for (one request); the MAX_WEB_PAGES best are read
MAX_WEB_PAGES = 2
WEB_TIMEOUT_SECONDS = 20
MAX_PAGE_CHARS = 200_000           # of a web page's text; the rest is ignored
PASSAGES_PER_DOCUMENT = {
    SourceType.WIKIPEDIA: 6,       # the main article: its opening passage and the 5 best matches
    SourceType.ARXIV: 1,           # one abstract per paper
    SourceType.WEB: 3,             # the best matches in the page (or its snippet if the page cannot be read)
}
# A web page passage is kept only with at least this score (3 for the main name,
# 1 for each other buzz word and each word of the sub-question it mentions), so a
# menu that happens to contain one word of the question is not saved
MIN_WEB_PASSAGE_SCORE = 2
# Sources whose first passage is always kept. A Wikipedia article opens with its
# summary, which holds most of the facts a question asks for (60% of the eval
# set's facts in the articles Search opened, checked 2026-11-12).
OPENING_KEPT = frozenset({SourceType.WIKIPEDIA})
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
# Reading a web page goes to the page's own site, not the search engine; one a
# second is polite, and a page that cannot be read falls back to its snippet,
# so it is tried once
PAGE_LIMITER = RateLimiter(1.0)
SOURCE_RETRY_POLICY = RetryPolicy(max_attempts=3, base_delay=2.0, max_delay=30.0)

STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from has have how in into is it its "
    "of on or that the their there these this to was were what when where which who whom why "
    "will with would current currently".split()
)
# Words a question uses that say nothing about its topic; left out of question_words
QUESTION_FILLER = frozenset(
    "about after also any been before between both but differ different difference during each "
    "kind kinds many more most much not only other over same should since some such than then they "
    "them type types use used uses using very way ways well work works".split()
)
WEB_NO_RESULTS = "No results found"  # ddgs raises this when every engine answered with nothing
# The engines ddgs may ask. Its "auto" adds Grokipedia, an AI-written encyclopedia,
# which a fact-checking pipeline must not cite as evidence. ddgs asks Wikipedia first
# and the rest in a random order, two at a time, until it has enough results; an
# engine that blocks or rate-limits answers nothing and the next is tried.
WEB_BACKENDS = "wikipedia,brave,duckduckgo,mojeek,startpage,yahoo"
 
 
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
 
 
# One engine's parser sometimes glues several results' titles into one string with no
# space between them ("...Loops & Jump StatementsLoops in Programming - GeeksforGeeks...").
# The record's address and snippet are the first title's. Engines cut a real title near
# 60 to 65 characters, so only a title longer than GLUED_TITLE_CHARS is treated as glued.
GLUED_TITLE_CHARS = 100
MIN_FIRST_TITLE_CHARS = 20
# A glue point: a small letter, a digit, a closing bracket or a sentence end, then a
# capital followed by a small letter. Capital then small leaves names like LoRA alone.
TITLE_GLUE = re.compile(r"(?<=[a-z0-9).?!\]])(?=[A-Z][a-z])")


def first_title(title):
    """The record's own title out of a glued one (see GLUED_TITLE_CHARS): the text before
    the first glue point at least MIN_FIRST_TITLE_CHARS in. The other pieces are copies
    of other results' titles, which arrive on their own when they are results at all, so
    they are dropped, not split out. A camel-case name past that point in the first
    title (JavaScript) would cut it short; that only shortens a label, and the address
    is never changed. On 2026-11-13, 3 of 17 web titles were glued (Q03, Q19, E07), and
    Q03's report listed its source under the whole 359-character string."""
    if len(title) <= GLUED_TITLE_CHARS:
        return title
    for glue in TITLE_GLUE.finditer(title, MIN_FIRST_TITLE_CHARS):
        return title[:glue.start()].rstrip()
    return title


def fetch_web(query, ddgs_factory=None, policy=SOURCE_RETRY_POLICY, sleep=time.sleep):
    """Web text search through ddgs, which asks several search engines and merges their
    answers. Each result is its title, its address and the engine's snippet; the page
    itself is read later (read_web_page), only if the result is on topic.

    The library is unofficial, so every failure mode is mapped explicitly. It replaced
    duckduckgo-search 8.1.1, which by 2026-11-13 answered most queries with nothing or
    with dictionary pages for words like "does" (1 web step in 33 saved anything on
    the full evaluation run)."""
    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
        from ddgs.exceptions import RatelimitException
        from ddgs.exceptions import TimeoutException
    except ImportError as error:
        raise FetchError("ddgs is not installed") from error
    if ddgs_factory is None:
        ddgs_factory = lambda: DDGS(timeout=WEB_TIMEOUT_SECONDS)
    limiter = SOURCE_LIMITERS[SourceType.WEB]

    def attempt():
        limiter.wait(sleep)
        try:
            return ddgs_factory().text(query, region="us-en", max_results=WEB_MAX_RESULTS,
                                       backend=WEB_BACKENDS) or []
        except RatelimitException as error:
            raise RetryableError(f"web search rate limit: {error}") from error
        except TimeoutException as error:
            raise RetryableError(f"web search timed out: {error}") from error
        except DDGSException as error:
            if WEB_NO_RESULTS in str(error):
                return []  # searched, nothing found
            raise FetchError(f"web search error: {error}") from error

    try:
        results = retry_call(attempt, policy, sleep=sleep, describe="web search")
    except RetriesExhausted as error:
        raise FetchError(str(error)) from error
 
    documents = []
    for result in results:
        url = (result.get("href") or "").strip()
        body = (result.get("body") or "").strip()
        if url and body:
            title = first_title((result.get("title") or url).strip())
            documents.append(RawDocument(title, url, "", body))
    return documents


MIN_PROSE_WORDS = 7                # a shorter line is a menu item, a button or a heading
MIN_PROSE_STOPWORDS = 2            # "the", "of", "is"...: prose has them, a menu of names does not
SENTENCE_END = re.compile(r"[.!?:;\"')\]]\s*$")


def is_prose(line):
    words = re.findall(r"[A-Za-z']+", line.lower())
    stopwords = sum(word in STOPWORDS for word in words)
    return len(line.split()) >= MIN_PROSE_WORDS and stopwords >= MIN_PROSE_STOPWORDS


def readable_text(text):
    """The prose of a page's text, one paragraph per run of prose lines, menus dropped.
    A page's text comes one line per menu item and wrapped at a fixed width, so a line
    is kept if it is prose (is_prose), or if it is the one short line that ends a
    wrapped sentence. Checked 2026-11-13 on 4 pages: the ADK LoopAgent page keeps all
    three facts the eval set takes from it, and a Google Cloud docs page whose first
    500 lines are menus goes from 12,303 characters to 2,940."""
    paragraphs, current, previous_prose = [], [], False
    for line in (text or "").split("\n"):
        line = " ".join(line.split())
        prose = is_prose(line)
        tail = not prose and previous_prose and bool(line) and not SENTENCE_END.search(current[-1])
        if prose or tail:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
        previous_prose = prose
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def read_web_page(url, ddgs_factory=None, sleep=time.sleep):
    """A web page's readable text (readable_text), or None if it has none: ddgs fetches
    the page the way a browser would and strips the markup. Raises FetchError if the
    page cannot be fetched. A PDF is not read (its text would come back as debris)."""
    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme not in ("http", "https") or parsed.path.lower().endswith(".pdf"):
        return None
    try:
        from ddgs import DDGS
    except ImportError as error:
        raise FetchError("ddgs is not installed") from error
    if ddgs_factory is None:
        ddgs_factory = lambda: DDGS(timeout=WEB_TIMEOUT_SECONDS)
    PAGE_LIMITER.wait(sleep)
    try:
        content = ddgs_factory().extract(url, fmt="text_plain").get("content")
    except Exception as error:  # the library raises its own errors and the HTTP client's
        raise FetchError(f"could not read {parsed.netloc}: {type(error).__name__}: {error}") from error
    if not isinstance(content, str):
        return None
    return readable_text(content[:MAX_PAGE_CHARS]).strip() or None


DEFAULT_PAGE_READER = read_web_page


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


TRAILING_BRACKET = re.compile(r"\s*\(([^()]*)\)\s*$")


def bare_name(name):
    """The name without a trailing bracketed qualifier: "BERT (language model)" -> "BERT"."""
    bare = TRAILING_BRACKET.sub("", name or "").strip()
    return bare or (name or "").strip()


def bracket_words(name):
    """The content words of a trailing bracketed qualifier: "BERT (language model)" ->
    ["language", "model"]. [] when the name has none."""
    match = TRAILING_BRACKET.search(name or "")
    if not match:
        return []
    return [word for word in re.split(r"[\s\-]+", match.group(1))
            if word and word.lower() not in STOPWORDS]


def names_topic(text, main):
    """True when a search result is about the main topic: it mentions the main name, and
    if the name carries a bracketed qualifier, it mentions the name without it plus at
    least one word from inside it. No page or paper contains "Transformer (machine
    learning architecture)" word for word, so the bracket must not be matched as written;
    Wikipedia's ranking for the full name tells meanings apart, and the bracket word
    guards sources whose ranking is weaker (checked on 10 names, 2026-11-12)."""
    words = bracket_words(main)
    if not words:
        return mentions(text, main)
    return mentions(text, bare_name(main)) and any(mentions(text, word) for word in words)


def question_words(text, limit=12):
    """The words of a sub-question that say what it is about, for scoring: no stopwords,
    no filler such as "kinds" or "work", at least 3 letters, without a plural s so that
    mentions() matches the singular and the plural alike. "In Google's Agent Development
    Kit (ADK), how does a LoopAgent decide when to stop looping?" gives google, agent,
    development, kit, adk, loopagent, decide, stop, looping."""
    words = []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", (text or "").lower()):
        if word in STOPWORDS or word in QUESTION_FILLER or len(word) < 3:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if word not in words:
            words.append(word)
    return words[:limit]


def about_sub_question(text, main, words):
    """True when a web result is on topic: it names the main topic (names_topic), or it
    mentions at least half the sub-question's own words. Buzz words are Wikipedia
    article names, and a web page about a topic Wikipedia lacks rarely uses them: the
    ADK's LoopAgent page never says "Software development kit"."""
    if main and names_topic(text, main):
        return True
    hits = sum(mentions(text, word) for word in words)
    return len(words) >= 2 and 2 * hits >= len(words)


def best_web_results(documents, main, words):
    """The web results worth reading: those on topic (about_sub_question), best first by
    how many of the sub-question's own words their title and snippet mention, then in
    the engine's order; at most MAX_WEB_PAGES. The engines' order alone is not enough:
    on 2026-11-13 the same Q19 query returned the ADK LoopAgent page first once and not
    in the top 3 twice, while a page that merely named "Software development kit" (the
    Manager's buzz word) came first and was read."""
    scored = []
    for rank, document in enumerate(documents):
        text = document.title + " " + document.content
        if about_sub_question(text, main, words):
            scored.append((-sum(mentions(text, word) for word in words), rank, document))
    return [document for _, _, document in sorted(scored, key=lambda item: item[:2])][:MAX_WEB_PAGES]


# ----- the original paper: what the main Wikipedia article cites -----

ARXIV_LINK = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
MAX_CITED_PAPERS = 100             # arXiv ids looked up in one request
MAX_LINK_PAGES = 5                 # continuation pages of an article's external links


def names_paper(text, main):
    """True when a paper's title and abstract name the main topic: the bare name as a
    phrase, or every word of it in any order ("residual networks ... deep neural
    networks" names "Residual neural network"), plus a word from inside the bracket
    when the name has one. Looser than names_topic because the candidates are already
    the papers the topic's own article cites."""
    name = bare_name(main)
    words = [word for word in re.split(r"[\s\-]+", name) if word and word.lower() not in STOPWORDS]
    named = mentions(text, name) or (len(words) > 1 and all(mentions(text, word) for word in words))
    brackets = bracket_words(main)
    return named and (not brackets or any(mentions(text, word) for word in brackets))


def wikipedia_arxiv_citations(title, http=http_get_json):
    """The arXiv ids a Wikipedia article links to, in the article's order, without repeats."""
    limiter = SOURCE_LIMITERS[SourceType.WIKIPEDIA]
    params = {"action": "query", "prop": "extlinks", "titles": title, "ellimit": "max",
              "redirects": 1, "format": "json", "formatversion": 2}
    ids = []
    more = {}
    for _ in range(MAX_LINK_PAGES):
        data = http(WIKIPEDIA_API_URL, {**params, **more}, limiter)
        if "error" in data:
            raise FetchError(f"Wikipedia links error: {data['error'].get('info', data['error'])}")
        for page in data.get("query", {}).get("pages", []):
            for link in page.get("extlinks", []):
                match = ARXIV_LINK.search(link.get("url", ""))
                if match and match.group(1) not in ids:
                    ids.append(match.group(1))
        if "continue" not in data:
            break
        more = data["continue"]
    return ids


def fetch_arxiv_papers(ids, http=http_get):
    """The arXiv records for the given ids. arXiv serves each paper's latest version;
    published_date is when it was first posted."""
    if not ids:
        return []
    body = http(ARXIV_API_URL, {"id_list": ",".join(ids[:MAX_CITED_PAPERS]),
                                "max_results": MAX_CITED_PAPERS}, SOURCE_LIMITERS[SourceType.ARXIV])
    return parse_arxiv_feed(body)


def original_paper(article_title, names, http_json=http_get_json, http=http_get):
    """The paper that introduced the topic, or None: of the arXiv papers the topic's
    Wikipedia article cites, the oldest whose title or abstract names the topic. names
    is a name or a list tried in order (the focus topic first, then the main name), all
    against the same cited papers. arXiv's keyword search ranks later papers that build
    on a topic above the one that started it; the article's own citations do not have
    that problem. Checked 2026-11-12 on 8 of the eval set's topics: the original paper
    every time, holding all 18 of the facts the eval set takes from arXiv (the newest
    naming paper held 4)."""
    if isinstance(names, str):
        names = [names]
    cited = fetch_arxiv_papers(wikipedia_arxiv_citations(article_title, http_json), http)
    for name in names:
        papers = [paper for paper in cited if names_paper(paper.title + " " + paper.content, name)]
        if papers:
            return min(papers, key=lambda paper: (paper.published_date or "9999", paper.url))
    return None


DEFAULT_PAPER_FINDER = original_paper


# ----- the focus topic: a narrower topic that lives inside the main article -----

MIN_FOCUS_CHARS = 4                # shorter redirect names (abbreviations) never become the focus


def section_redirects(title, http=http_get_json):
    """The names that redirect to a section of a Wikipedia article: the narrower topics
    it covers inside it ("Chain-of-thought prompting" leads to a section of "Prompt
    engineering"). A redirect to the whole article is just another name for it, so it
    is left out."""
    limiter = SOURCE_LIMITERS[SourceType.WIKIPEDIA]
    params = {"action": "query", "prop": "redirects", "titles": title, "rdlimit": "max",
              "rdprop": "title|fragment", "rdnamespace": 0, "redirects": 1,
              "format": "json", "formatversion": 2}
    titles = []
    more = {}
    for _ in range(MAX_LINK_PAGES):
        data = http(WIKIPEDIA_API_URL, {**params, **more}, limiter)
        if "error" in data:
            raise FetchError(f"Wikipedia redirects error: {data['error'].get('info', data['error'])}")
        for page in data.get("query", {}).get("pages", []):
            for redirect in page.get("redirects", []):
                if redirect.get("fragment") and redirect.get("title") not in titles:
                    titles.append(redirect["title"])
        if "continue" not in data:
            break
        more = data["continue"]
    return titles


DEFAULT_REDIRECT_FINDER = section_redirects


def focus_topic(section_titles, text):
    """The narrower topic a sub-question asks about, or None: the longest of the main
    article's section redirects that the sub-question's own text names (its bare name,
    plus a word from inside its bracket when it has one; at least MIN_FOCUS_CHARS long).
    Whatever the Manager named, the article itself says which narrower topics live in it.
    Checked 2026-11-12 on all 20 eval questions: it fired only on chain-of-thought
    prompting (Q05), where it took the original paper from the wrong one to the right
    one, and changed nothing else."""
    named = []
    for title in section_titles:
        bracket = [word for word in bracket_words(title) if len(word) > 2]
        if len(bare_name(title)) < MIN_FOCUS_CHARS or not mentions(text, bare_name(title)):
            continue
        if bracket and not any(mentions(text, word) for word in bracket):
            continue
        named.append(title)
    return max(named, key=len) if named else None

 
def document_passages(document):
    """A document cut into passages the way the evidence store keeps them, debris dropped."""
    return [piece for piece in split_into_passages(document.content, DEFAULT_PASSAGE_CHARS)
            if len(piece.strip()) >= MIN_PASSAGE_CHARS]


def passage_score(piece, buzz_words, focus=None):
    """3 points for the main name (its bare name), 1 for each other buzz word, and 3 for
    the focus topic, if any."""
    main, others = bare_name(buzz_words[0]), buzz_words[1:]
    score = 3 * mentions(piece, main) + sum(mentions(piece, word) for word in others)
    if focus:
        score += 3 * mentions(piece, bare_name(focus))
    return score


def passages_of(document, source, buzz_words=(), focus=None, words=()):
    """The passages of a document worth saving, at most PASSAGES_PER_DOCUMENT[source],
    in their original order. With buzz words, the best-matching passages anywhere in
    the document are chosen, not just the first ones: 3 points for the main name and 1
    for each other buzz word, and a passage that matches nothing is not chosen. For a
    source in OPENING_KEPT the first passage is always kept and the rest of the limit
    goes to the best matches. Without buzz words, or for another source when nothing
    matches, the first passages are kept. A focus topic (focus_topic) scores 3 points
    too, so the section about it is chosen even when the main name is everywhere.

    words are the sub-question's own words (question_words). They only break ties:
    between passages with the same score, the one that mentions more of them wins,
    then the earlier one. On Q10 (2026-11-13) "Perihelion" opened the Apsis article
    and position alone picked its passages on the word's history over its passage on
    Earth's perihelion in January."""
    pieces = document_passages(document)
    limit = PASSAGES_PER_DOCUMENT[source]
    if not buzz_words or not pieces:
        return pieces[:limit]
    keep_opening = source in OPENING_KEPT
    scores = []
    for position, piece in enumerate(pieces):
        if keep_opening and position == 0:
            continue
        score = passage_score(piece, buzz_words, focus)
        if score > 0:
            hits = sum(mentions(piece, word) for word in words)
            scores.append((score, hits, position))
    if not scores and not keep_opening:
        return pieces[:limit]
    best = sorted(scores, key=lambda item: (-item[0], -item[1], item[2]))[:limit - (1 if keep_opening else 0)]
    chosen = ([0] if keep_opening else []) + [position for _, _, position in best]
    return [pieces[position] for position in sorted(chosen)]


# Marks of a bibliography or a references section: dense with names and topic words,
# so they outscore prose, but they state no fact
REFERENCE_MARK = re.compile(r"\bRetrieved\b|\bArchived\b|\bdoi\b|\barXiv:|\bet al\.|\bpp\.|\[\d+\]|\bISBN\b|"
                            r"\bvol\.|\bProceedings\b|↑")
MAX_REFERENCE_MARKS = 3


def is_reference_list(passage):
    return len(REFERENCE_MARK.findall(passage)) > MAX_REFERENCE_MARKS


def page_passages(document, buzz_words, words):
    """The best passages of a web page, at most PASSAGES_PER_DOCUMENT[WEB], in their
    original order; [] if none scores MIN_WEB_PASSAGE_SCORE. Unlike passages_of, the
    sub-question's own words count as points here, not just as a tie-break, because a
    web page rarely uses Wikipedia's names; and nothing is kept just for being first,
    because a page opens with its menus."""
    scores = []
    for position, piece in enumerate(document_passages(document)):
        if is_reference_list(piece):
            continue
        score = (passage_score(piece, buzz_words) if buzz_words else 0) + \
            sum(mentions(piece, word) for word in words)
        if score >= MIN_WEB_PASSAGE_SCORE:
            scores.append((score, position, piece))
    best = sorted(scores, key=lambda item: (-item[0], item[1]))[:PASSAGES_PER_DOCUMENT[SourceType.WEB]]
    return [piece for _, _, piece in sorted(best, key=lambda item: item[1])]
 
 
def search_steps(source, text, buzz_words, fetchers, article_fetchers):
    """(step name, fetch, query, gated) in the order a source is tried. For a source
    that can look an article up: the main name as an exact article, then a search on
    the main name alone (catches a near miss such as "Ten-percent of the brain myth"
    for "Ten-percent-of-the-brain myth"). Then, for every source, the buzz words, then
    the full sub-question; for the web, the other way round, because a web search
    engine answers a plain question well and a string of article names badly (both
    checked live 2026-11-13). A gated step's documents must name the topic
    (names_topic; for the web, about_sub_question); the exact article is the one the
    Manager named, so it is not gated."""
    steps = []
    main = buzz_words[0] if buzz_words else ""
    name_searched = False
    if main and source in article_fetchers:
        steps.append(("article name", article_fetchers[source], main, False))
        if source in fetchers:
            steps.append(("name search", fetchers[source], main, True))
            name_searched = True
    if source in fetchers:
        buzz_step = None
        if buzz_words and not (name_searched and len(buzz_words) == 1):
            buzz_step = ("buzz words", fetchers[source], " ".join(buzz_words), True)
        question_step = ("sub-question", fetchers[source], text, True)
        order = [question_step, buzz_step] if source == SourceType.WEB else [buzz_step, question_step]
        steps.extend(step for step in order if step is not None)
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
 
 
class SavedPassages:
    """The passages already saved for one question, shared by its sub-questions' searches,
    so the Synthesizer never reads the same passage twice.

    Each sub-question searches in its own worker thread at the same time as the others,
    so every check-and-save holds the lock. A passage is known by its document's URL and
    its text, which is the same whichever sub-question fetched the article.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.evidence_ids = {}  # (url, passage) -> evidence_id


def run_search(store, question_id, sub_question_id, text, preferred_sources, buzz_words=(),
               fetchers=None, article_fetchers=None, saved=None, paper_finder=None,
               redirect_finder=None, page_reader=None):
    """Searches one sub-question, saves its passages, and returns its SubQuestionEvidence.

    With a page_reader (read_web_page in real runs), each web result on topic is read
    as a whole page and its best passages saved (page_passages); a page that cannot be
    read, or holds no passage on topic, gives its search snippet instead. A web result
    is on topic if about_sub_question holds for its title and snippet.

    If the listed sources save nothing and Wikipedia was not among them, Wikipedia is
    tried next (its exact article first, then its branches), before the web fallback:
    the buzz words are Wikipedia article names, so an empty arXiv search must not mean
    the article the Manager named is never looked up (Q02, 2026-11-13).

    Wikipedia, when listed, is searched first, so its main article can point arXiv at
    the original paper: with a paper_finder (original_paper in real runs), arXiv saves
    the paper the article cites that introduced the topic, and its keyword search runs
    only when there is none.

    With a redirect_finder (section_redirects in real runs), the main article's section
    redirects are checked against the sub-question's text: a narrower topic it names
    (focus_topic) also scores passages and is tried first by the paper finder.

    Each source is tried in the order search_steps gives and stops at the first step
    that finds something usable. A document from a search is used only if it names
    the main topic (the first buzz word; see names_topic), so an off-topic article is
    never saved; Wikipedia keeps one main article, the other sources what passed.

    Then, for a source that can look an article up (Wikipedia), every other buzz word
    is looked up as its own article and that article's opening passage is saved: a
    small branch beside the main one. A name with no article, a disambiguation page,
    or an article already saved for this sub-question adds nothing.

    saved is the question's SavedPassages: a passage another sub-question already
    saved is skipped, not saved again. If every passage this sub-question picked was
    a repeat, it points at those passages instead, so it is never reported as having
    found nothing. None means this sub-question is searched on its own.
    """
    if fetchers is None:
        fetchers = DEFAULT_FETCHERS
    if article_fetchers is None:
        article_fetchers = {}
    if saved is None:
        saved = SavedPassages()
    buzz_words = clean_buzz_words(buzz_words)
    main = buzz_words[0] if buzz_words else ""
    words = question_words(text)
    register_sub_question(store, question_id, sub_question_id, text)
    sources = normalize_sources(preferred_sources)
    if SourceType.WIKIPEDIA in sources:
        sources = [SourceType.WIKIPEDIA] + [source for source in sources if source != SourceType.WIKIPEDIA]
    logger.info("sub-question %s: buzz words %s, sources %s", sub_question_id, buzz_words,
                [source.value for source in sources])

    evidence_ids = []
    repeats = []  # picked, but another sub-question had already saved them
    errors = []
    tried = []
    focus = None  # a narrower topic inside the main article that the sub-question names

    def save_passages(document, source, passages):
        retrieved_at = utc_now()
        for passage in passages:
            if len(evidence_ids) >= MAX_EVIDENCE_PER_SUB_QUESTION:
                return
            key = (document.url, passage)
            with saved.lock:
                existing = saved.evidence_ids.get(key)
                if existing is None:
                    record = EvidenceRecord(
                        sub_question_id=sub_question_id,
                        source_url=document.url,
                        source_title=document.title or document.url,
                        published_date=document.published_date or "",
                        passage=passage,
                        retrieved_at=retrieved_at,
                        source_type=source,
                    )
                    evidence_id = store.add_evidence(question_id, record)
                    saved.evidence_ids[key] = evidence_id
            if existing is not None:
                if existing not in repeats:
                    repeats.append(existing)
                continue
            evidence_ids.append(evidence_id)

    def web_passages(document):
        """The best passages of the page a web result points to, or its snippet."""
        page_text = None
        try:
            page_text = page_reader(document.url)
        except FetchError as error:
            logger.info("web, sub-question %s: %s; keeping the snippet", sub_question_id, short_error(error))
        except Exception:  # a bug in the reader must not take the run down
            logger.exception("page reader crashed")
        if page_text:
            page = RawDocument(document.title, document.url, document.published_date, page_text)
            chosen = page_passages(page, buzz_words, words)
            if chosen:
                logger.info("web, sub-question %s: read %r, kept %d passages", sub_question_id,
                            document.title, len(chosen))
                return chosen
        return passages_of(document, SourceType.WEB, buzz_words, focus, words)[:1]

    def save(documents, source):
        for document in documents:
            if source == SourceType.WEB and page_reader is not None:
                passages = web_passages(document)
            else:
                passages = passages_of(document, source, buzz_words, focus, words)
            save_passages(document, source, passages)

    def find_focus(article_title):
        """Sets focus from the main article's section redirects; a failed lookup only means no focus."""
        nonlocal focus
        try:
            titles = redirect_finder(article_title)
        except FetchError as error:
            errors.append(f"wikipedia redirects: {short_error(error)}")
            return
        except Exception as error:  # a bug in the finder must not take the run down
            logger.exception("redirect finder crashed")
            errors.append(f"wikipedia redirects: unexpected {type(error).__name__}: {short_error(error)}")
            return
        focus = focus_topic(titles, text)
        if focus:
            logger.info("wikipedia, sub-question %s: the sub-question names %r, a section of %r",
                        sub_question_id, focus, article_title)

    def search_source(source):
        """Tries the source's steps; returns the documents it saved ([] if none)."""
        tried.append(source)
        steps = search_steps(source, text, buzz_words, fetchers, article_fetchers)
        if not steps:
            errors.append(f"{source.value}: no fetcher configured")
            return []
        for step, fetch, query, gated in steps:
            try:
                documents = list(fetch(query))
            except FetchError as error:
                # the service itself failed, so its later steps would too
                errors.append(f"{source.value}: {short_error(error)}")
                return []
            except Exception as error:  # a bug in a fetcher must not take the run down
                logger.exception("%s fetcher crashed", source.value)
                errors.append(f"{source.value}: unexpected {type(error).__name__}: {short_error(error)}")
                return []
            if gated and source == SourceType.WEB:
                documents = best_web_results(documents, main, words)
            elif gated and main:
                documents = [document for document in documents if names_topic(document.content, main)]
            if source == SourceType.WIKIPEDIA:
                documents = documents[:1]
            if documents:
                logger.info("%s, sub-question %s: %s step used %s", source.value, sub_question_id,
                            step, [document.title for document in documents])
                if source == SourceType.WIKIPEDIA and redirect_finder is not None:
                    find_focus(documents[0].title)  # before saving, so it can steer the passages
                save(documents, source)
                return documents
            logger.info("%s, sub-question %s: %s step found nothing that mentions %r",
                        source.value, sub_question_id, step, main)
        return []

    def search_branches(source, saved_documents):
        """Saves the opening passage of each other buzz word's own article."""
        fetch = article_fetchers[source]
        seen = {document.title.lower() for document in saved_documents}
        for word in buzz_words[1:]:
            if len(evidence_ids) >= MAX_EVIDENCE_PER_SUB_QUESTION:
                return
            try:
                documents = list(fetch(word))[:1]
            except FetchError as error:
                # the service itself failed, so the other branches would too
                errors.append(f"{source.value} branch {word!r}: {short_error(error)}")
                return
            except Exception as error:  # a bug in a fetcher must not take the run down
                logger.exception("%s branch fetcher crashed", source.value)
                errors.append(f"{source.value} branch {word!r}: unexpected {type(error).__name__}: "
                              f"{short_error(error)}")
                return
            if not documents:
                logger.info("%s, sub-question %s: branch %r has no article",
                            source.value, sub_question_id, word)
                continue
            document = documents[0]
            if document.title.lower() in seen:
                logger.info("%s, sub-question %s: branch %r is %r, already saved", source.value,
                            sub_question_id, word, document.title)
                continue
            seen.add(document.title.lower())
            logger.info("%s, sub-question %s: branch %r used %s", source.value, sub_question_id,
                        word, [document.title])
            save_passages(document, source, document_passages(document)[:1])

    def search_original_paper(article_title):
        """Saves the paper the main article cites that introduced the topic. True if one
        was found (even one another sub-question already saved), so arXiv's keyword
        search is not needed."""
        tried.append(SourceType.ARXIV)
        try:
            names = [name for name in (focus, main) if name]
            paper = paper_finder(article_title, names)
        except FetchError as error:
            errors.append(f"arxiv original paper: {short_error(error)}")
            return False
        except Exception as error:  # a bug in the finder must not take the run down
            logger.exception("original paper finder crashed")
            errors.append(f"arxiv original paper: unexpected {type(error).__name__}: {short_error(error)}")
            return False
        if paper is None:
            logger.info("arxiv, sub-question %s: %r cites no arXiv paper that names %s",
                        sub_question_id, article_title, names)
            return False
        logger.info("arxiv, sub-question %s: original paper used %s", sub_question_id, [paper.title])
        save([paper], SourceType.ARXIV)
        return True

    main_article = None
    for source in sources:
        if len(evidence_ids) >= MAX_EVIDENCE_PER_SUB_QUESTION:
            break
        if source == SourceType.ARXIV and paper_finder is not None and main_article is not None:
            if search_original_paper(main_article):
                continue
        saved_documents = search_source(source)
        if source == SourceType.WIKIPEDIA and saved_documents:
            main_article = saved_documents[0].title
        if source in article_fetchers:
            search_branches(source, saved_documents)
    can_search_wikipedia = SourceType.WIKIPEDIA in fetchers or SourceType.WIKIPEDIA in article_fetchers
    if not evidence_ids and not repeats and SourceType.WIKIPEDIA not in tried and can_search_wikipedia:
        logger.info("sub-question %s: the listed sources found nothing; trying Wikipedia", sub_question_id)
        saved_documents = search_source(SourceType.WIKIPEDIA)
        if SourceType.WIKIPEDIA in article_fetchers:
            search_branches(SourceType.WIKIPEDIA, saved_documents)
    if not evidence_ids and not repeats and SourceType.WEB not in tried:
        search_source(SourceType.WEB)  # README: web search is the fallback
    if not evidence_ids and repeats:
        logger.info("sub-question %s: every passage it picked was already saved by another "
                    "sub-question; pointing at evidence %s", sub_question_id, repeats)
        evidence_ids = list(repeats)

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
    "2 to 5 buzz_words. Each buzz word is the name of a Wikipedia article, "
    "written the way Wikipedia titles it, never an abbreviation (\"User "
    "Datagram Protocol\", not \"UDP\"). The first is the article most likely to "
    "hold the answer: the most specific topic, not the broad subject it belongs "
    "to (for \"Why is the sky blue?\", \"Rayleigh scattering\", not \"Sky\"). If "
    "the sub-question asks about a technique, event or idea that Wikipedia covers "
    "only as a section of a broader article, the first buzz word is still that "
    "technique's own name, worded the way the question words it, and the broader "
    "article comes later. If the question rests on a false belief or a "
    "popular myth, the first is the article about that belief. The others name "
    "the people, things or ideas the answer also depends on, so each can be "
    "looked up on its own. If the question compares two or more things, plan one "
    "sub-question for each thing, with that thing's name first. Search at most "
    "LIMIT sub-questions and never the same one twice, unless the tool says it "
    "was not searched.\n"
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
    call waits for the first one instead of fetching and storing twice. saved maps
    question_id to the SavedPassages its sub-questions share, so no passage is
    saved twice for one question.
    """

    def __init__(self, store=None, fetchers=None, article_fetchers=None, paper_finder=None,
                 redirect_finder=None, page_reader=None):
        self.store = store if store is not None else EvidenceStore()
        self.fetchers = fetchers if fetchers is not None else DEFAULT_FETCHERS
        if article_fetchers is None:
            # Caller-supplied fetchers (tests) get no real lookups unless they ask
            article_fetchers = DEFAULT_ARTICLE_FETCHERS if fetchers is None else {}
        self.article_fetchers = article_fetchers
        if paper_finder is None and fetchers is None:
            paper_finder = DEFAULT_PAPER_FINDER
        self.paper_finder = paper_finder
        if redirect_finder is None and fetchers is None:
            redirect_finder = DEFAULT_REDIRECT_FINDER
        self.redirect_finder = redirect_finder
        if page_reader is None and fetchers is None:
            page_reader = DEFAULT_PAGE_READER
        self.page_reader = page_reader
        self.registry = {}
        self.saved = {}

    # ----- callbacks -----

    def register_question(self, callback_context):
        """before_agent_callback: records the user's question so evidence has a question_id."""
        state = callback_context.state
        question_id = self.store.add_question(state[STATE_USER_QUESTION])
        state[STATE_QUESTION_ID] = question_id
        self.registry[question_id] = {}
        self.saved[question_id] = SavedPassages()
        return None

    def finalize_search_results(self, callback_context):
        """after_agent_callback: writes SearchResults for exactly the sub-questions in the capped plan."""
        state = callback_context.state
        runs = self.registry.pop(state.get(STATE_QUESTION_ID), {})
        self.saved.pop(state.get(STATE_QUESTION_ID), None)
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
                self.saved.setdefault(question_id, SavedPassages()), self.paper_finder,
                self.redirect_finder, self.page_reader,
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
              buzz_words: 2 to 5 Wikipedia article names, never abbreviations. The
                first is the article most likely to hold the answer; the others are
                the people, things or ideas the answer also depends on.
 
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
