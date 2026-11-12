# Tests for how Search picks what to save (search_agent.run_search). Run from the
# project root:
#     python -m pytest tests
# No test touches the network: every source is a fake that records its queries.
from evidence_store import EvidenceStore
from schemas import SourceType
from search_agent import NOT_SEARCHED_PREFIX
from search_agent import PASSAGES_PER_DOCUMENT
from search_agent import RawDocument
from search_agent import mentions
from search_agent import passages_of
from search_agent import run_search

SUB_QUESTION = "What kinds of applications use UDP?"
BUZZ = ["User Datagram Protocol", "applications", "DNS"]

UDP = RawDocument("User Datagram Protocol", "https://example.org/udp", "",
                  "The User Datagram Protocol (UDP) is a core protocol of the Internet. "
                  "Applications such as DNS and streaming use it.")
WAP = RawDocument("Wireless Application Protocol", "https://example.org/wap", "",
                  "The Wireless Application Protocol is a standard for mobile phones. "
                  "Its datagram layer looks a bit like UDP.")


# A source that answers each query from a table, [] for anything else, and keeps every query
class FakeSource:
    def __init__(self, answers):
        self.answers = answers
        self.queries = []

    def __call__(self, query):
        self.queries.append(query)
        return list(self.answers.get(query, []))


def search(tmp_path, wikipedia_search, article=None, buzz_words=BUZZ, sources=("wikipedia",)):
    store = EvidenceStore(tmp_path / "evidence.db")
    question_id = store.add_question("A question")
    fetchers = {SourceType.WIKIPEDIA: wikipedia_search, SourceType.ARXIV: FakeSource({}),
                SourceType.WEB: FakeSource({})}
    article_fetchers = {} if article is None else {SourceType.WIKIPEDIA: article}
    result = run_search(store, question_id, 1, SUB_QUESTION, list(sources), buzz_words,
                        fetchers, article_fetchers)
    rows = store.read_evidence(result.evidence_ids)
    return result, rows


BUZZ_QUERY = "User Datagram Protocol applications DNS"


def test_the_full_name_is_looked_up_first_and_nothing_else_is_searched(tmp_path):
    article = FakeSource({"User Datagram Protocol": [UDP]})
    wikipedia_search = FakeSource({})
    result, rows = search(tmp_path, wikipedia_search, article=article)
    assert article.queries == ["User Datagram Protocol"]
    assert wikipedia_search.queries == []
    assert result.status.value == "found"
    assert [row.source_title for row in rows] == ["User Datagram Protocol"]


def test_the_buzz_words_are_next_and_an_article_that_never_names_the_topic_is_skipped(tmp_path):
    article = FakeSource({})
    wikipedia_search = FakeSource({BUZZ_QUERY: [WAP, UDP]})
    result, rows = search(tmp_path, wikipedia_search, article=article)
    assert wikipedia_search.queries == [BUZZ_QUERY]
    assert [row.source_title for row in rows] == ["User Datagram Protocol"]


def test_the_full_sub_question_is_the_last_resort(tmp_path):
    wikipedia_search = FakeSource({SUB_QUESTION: [UDP]})
    result, rows = search(tmp_path, wikipedia_search, article=FakeSource({}))
    assert wikipedia_search.queries == [BUZZ_QUERY, SUB_QUESTION]
    assert [row.source_title for row in rows] == ["User Datagram Protocol"]


def test_when_nothing_names_the_topic_nothing_is_saved(tmp_path):
    wikipedia_search = FakeSource({BUZZ_QUERY: [WAP], SUB_QUESTION: [WAP]})
    result, rows = search(tmp_path, wikipedia_search, article=FakeSource({}),
                          sources=("wikipedia", "web"))
    assert rows == []
    assert result.status.value == "empty"


def test_the_named_article_is_kept_even_if_its_text_words_the_name_differently(tmp_path):
    # A redirect: the Manager asked for one name and Wikipedia answered with another page
    apsis = RawDocument("Apsis", "https://example.org/apsis", "",
                        "Perihelion is the closest point to the Sun.")
    article = FakeSource({"Perihelion and aphelion": [apsis]})
    result, rows = search(tmp_path, FakeSource({}), article=article,
                          buzz_words=["Perihelion and aphelion", "perihelion"])
    assert [row.source_title for row in rows] == ["Apsis"]


def test_arxiv_and_web_search_the_buzz_words_then_the_sub_question(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    question_id = store.add_question("A question")
    arxiv = FakeSource({})
    run_search(store, question_id, 1, SUB_QUESTION, ["arxiv"], BUZZ,
               {SourceType.ARXIV: arxiv, SourceType.WEB: FakeSource({}),
                SourceType.WIKIPEDIA: FakeSource({})},
               {SourceType.WIKIPEDIA: FakeSource({})})
    assert arxiv.queries == [BUZZ_QUERY, SUB_QUESTION]


def test_the_passages_that_match_the_buzz_words_are_saved_not_just_the_first_ones():
    filler = "Orbits are ellipses with the Sun at one focus. " * 40
    answer = "The axial tilt of the Earth accounts for most of the seasonal variation. "
    document = RawDocument("Earth's orbit", "https://example.org/orbit", "", filler + "\n\n" + answer * 3)
    chosen = passages_of(document, SourceType.WIKIPEDIA, ["Axial tilt", "seasons"])
    assert len(chosen) <= PASSAGES_PER_DOCUMENT[SourceType.WIKIPEDIA]
    assert any("axial tilt" in passage for passage in chosen)
    assert not chosen[0].startswith("Orbits are ellipses")


def test_names_match_as_whole_words_with_plurals_and_hyphens_or_spaces():
    assert mentions("Retrieval-augmented generation helps.", "retrieval augmented generation")
    assert mentions("Transformers rely on attention.", "Transformer")
    assert not mentions("The WAP datagram layer looks a bit like UDP.", "User Datagram Protocol")
    assert not mentions("A transformational idea.", "Transformer")


def test_a_search_with_no_buzz_words_is_refused_as_not_searched():
    import asyncio

    from search_agent import SearchTool

    class Context:
        state = {"question_id": 1}

    tool = SearchTool(store=object(), fetchers={})
    result = asyncio.run(tool.search(1, SUB_QUESTION, ["wikipedia"], [], Context()))
    assert result["status"] == "failed"
    assert result["error"].startswith(NOT_SEARCHED_PREFIX + "buzz_words is empty")
    assert tool.registry == {}  # refused before a search task exists, so a retry can run
