# Tests for how Search picks what to save (search_agent.run_search). Run from the
# project root:
#     python -m pytest tests
# No test touches the network: every source is a fake that records its queries.
from evidence_store import EvidenceStore
from rate_limit import RetryPolicy
from schemas import STATE_QUESTION_ID
from schemas import STATE_USER_QUESTION
from schemas import SourceType
from search_agent import MAX_EVIDENCE_PER_SUB_QUESTION
from search_agent import NOT_SEARCHED_PREFIX
from search_agent import PASSAGES_PER_DOCUMENT
from search_agent import SEARCH_INSTRUCTION
from search_agent import FetchError
from search_agent import RawDocument
from search_agent import SavedPassages
from search_agent import SearchTool
from search_agent import bare_name
from search_agent import bracket_words
from search_agent import focus_topic
from search_agent import mentions
from search_agent import names_paper
from search_agent import names_topic
from search_agent import original_paper
from search_agent import section_redirects
from search_agent import passages_of
from search_agent import about_sub_question
from search_agent import fetch_web
from search_agent import first_title
from search_agent import page_passages
from search_agent import question_words
from search_agent import read_web_page
from search_agent import readable_text
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
    def __init__(self, answers, fail_on=None):
        self.answers = answers
        self.fail_on = fail_on
        self.queries = []

    def __call__(self, query):
        self.queries.append(query)
        if query == self.fail_on:
            raise FetchError("HTTP 503 from en.wikipedia.org")
        return list(self.answers.get(query, []))


def search(tmp_path, wikipedia_search, article=None, buzz_words=BUZZ, sources=("wikipedia",), arxiv=None,
           paper_finder=None):
    store = EvidenceStore(tmp_path / "evidence.db")
    question_id = store.add_question("A question")
    fetchers = {SourceType.WIKIPEDIA: wikipedia_search, SourceType.ARXIV: arxiv or FakeSource({}),
                SourceType.WEB: FakeSource({})}
    article_fetchers = {} if article is None else {SourceType.WIKIPEDIA: article}
    result = run_search(store, question_id, 1, SUB_QUESTION, list(sources), buzz_words,
                        fetchers, article_fetchers, paper_finder=paper_finder)
    rows = store.read_evidence(result.evidence_ids)
    return result, rows


# One paragraph of about `length` characters built from one sentence, so each is its own passage
def paragraph(sentence, length=1400):
    text = ""
    while len(text) + len(sentence) + 1 <= length:
        text += sentence + " "
    return text.strip()


def long_document(title, sentences):
    return RawDocument(title, "https://example.org/" + title.replace(" ", "_"), "",
                       "\n\n".join(paragraph(sentence) for sentence in sentences))


BUZZ_QUERY = "User Datagram Protocol applications DNS"


# ----- the order a source is searched in -----

def test_the_full_name_is_looked_up_first_and_nothing_is_searched_when_it_exists(tmp_path):
    article = FakeSource({"User Datagram Protocol": [UDP]})
    wikipedia_search = FakeSource({})
    result, rows = search(tmp_path, wikipedia_search, article=article)
    # the other buzz words are then looked up as articles of their own (none exists here)
    assert article.queries == ["User Datagram Protocol", "applications", "DNS"]
    assert wikipedia_search.queries == []
    assert result.status.value == "found"
    assert [row.source_title for row in rows] == ["User Datagram Protocol"]


def test_the_name_alone_is_searched_next_then_the_buzz_words(tmp_path):
    article = FakeSource({})
    wikipedia_search = FakeSource({BUZZ_QUERY: [WAP, UDP]})
    result, rows = search(tmp_path, wikipedia_search, article=article)
    assert wikipedia_search.queries == ["User Datagram Protocol", BUZZ_QUERY]
    # WAP came first but never names the topic, so it is skipped
    assert [row.source_title for row in rows] == ["User Datagram Protocol"]


def test_a_near_miss_name_is_found_by_searching_the_name_alone(tmp_path):
    myth = RawDocument("Ten-percent-of-the-brain myth", "https://example.org/myth", "",
                       "The ten-percent-of-the-brain myth says people use a tenth of their brains.")
    article = FakeSource({})  # "Ten-percent of the brain myth" is not an exact title
    wikipedia_search = FakeSource({"Ten-percent of the brain myth": [myth]})
    result, rows = search(tmp_path, wikipedia_search, article=article,
                          buzz_words=["Ten-percent of the brain myth", "Human brain"])
    assert wikipedia_search.queries == ["Ten-percent of the brain myth"]
    assert [row.source_title for row in rows] == ["Ten-percent-of-the-brain myth"]


def test_the_full_sub_question_is_the_last_resort(tmp_path):
    wikipedia_search = FakeSource({SUB_QUESTION: [UDP]})
    result, rows = search(tmp_path, wikipedia_search, article=FakeSource({}))
    assert wikipedia_search.queries == ["User Datagram Protocol", BUZZ_QUERY, SUB_QUESTION]
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


# ----- the small branches: one article per other buzz word -----

def test_each_other_buzz_word_adds_the_opening_passage_of_its_own_article(tmp_path):
    dns = long_document("Domain Name System", ["The Domain Name System names computers.",
                                               "A second passage about DNS records."])
    article = FakeSource({"User Datagram Protocol": [UDP], "Domain Name System": [dns]})
    result, rows = search(tmp_path, FakeSource({}), article=article,
                          buzz_words=["User Datagram Protocol", "Domain Name System", "Port number"])
    assert article.queries == ["User Datagram Protocol", "Domain Name System", "Port number"]
    assert [row.source_title for row in rows] == ["User Datagram Protocol", "Domain Name System"]
    assert rows[1].passage.startswith("The Domain Name System names computers.")
    assert "second passage" not in rows[1].passage


def test_a_branch_that_lands_on_an_article_already_saved_adds_nothing(tmp_path):
    apsis = RawDocument("Apsis", "https://example.org/apsis", "",
                        "Perihelion is the closest point to the Sun and aphelion the farthest.")
    article = FakeSource({"Perihelion": [apsis], "Aphelion": [apsis]})
    result, rows = search(tmp_path, FakeSource({}), article=article, buzz_words=["Perihelion", "Aphelion"])
    assert [row.source_title for row in rows] == ["Apsis"]


def test_wikipedia_is_not_searched_when_a_listed_source_found_something(tmp_path):
    study = RawDocument("A UDP study", "https://arxiv.org/abs/2", "",
                        "The User Datagram Protocol under load.")
    article = FakeSource({"User Datagram Protocol": [UDP]})
    result, rows = search(tmp_path, FakeSource({}), article=article, sources=("arxiv",),
                          arxiv=FakeSource({BUZZ_QUERY: [study]}))
    assert article.queries == []
    assert [row.source_title for row in rows] == ["A UDP study"]


def test_when_the_listed_sources_find_nothing_wikipedia_is_tried_before_the_web(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    question_id = store.add_question("A question")
    calls = []

    class Recording(FakeSource):
        def __init__(self, name, answers):
            super().__init__(answers)
            self.name = name

        def __call__(self, query):
            calls.append(self.name)
            return super().__call__(query)

    fetchers = {SourceType.ARXIV: Recording("arxiv", {}), SourceType.WEB: Recording("web", {}),
                SourceType.WIKIPEDIA: Recording("wikipedia search", {})}
    article = Recording("wikipedia article", {"User Datagram Protocol": [UDP]})
    result = run_search(store, question_id, 1, SUB_QUESTION, ["arxiv"], BUZZ, fetchers,
                        {SourceType.WIKIPEDIA: article})
    rows = store.read_evidence(result.evidence_ids)
    # arXiv (the Manager's choice) first; then the article the Manager named; the web is never needed
    assert calls[:3] == ["arxiv", "arxiv", "wikipedia article"]
    assert "web" not in calls
    assert result.status.value == "found"
    assert [row.source_title for row in rows] == ["User Datagram Protocol"]


def test_a_failed_branch_lookup_keeps_what_was_found(tmp_path):
    article = FakeSource({"User Datagram Protocol": [UDP]}, fail_on="Domain Name System")
    result, rows = search(tmp_path, FakeSource({}), article=article,
                          buzz_words=["User Datagram Protocol", "Domain Name System", "Port number"])
    # the service failed, so the branches after it are not tried
    assert article.queries == ["User Datagram Protocol", "Domain Name System"]
    assert result.status.value == "found"
    assert [row.source_title for row in rows] == ["User Datagram Protocol"]
    assert "branch 'Domain Name System'" in result.error


def test_a_sub_question_never_stores_more_than_the_cap(tmp_path):
    # distinct paragraphs: identical text would count as one passage
    main = long_document("Season", [f"A season is one of the year's divisions, part {n}." for n in range(10)])
    branches = {name: [RawDocument(name, "https://example.org/" + name, "", f"{name} opening passage.")]
                for name in ("Axial tilt", "Earth's orbit", "Equinox", "Solstice")}
    article = FakeSource({"Season": [main], **branches})
    papers = [RawDocument(f"Paper {n}", f"https://arxiv.org/abs/{n}", "2020-01-01",
                          f"Season study number {n}.") for n in range(3)]
    buzz = ["Season", "Axial tilt", "Earth's orbit", "Equinox", "Solstice"]
    arxiv = FakeSource({" ".join(buzz): papers})
    result, rows = search(tmp_path, FakeSource({}), article=article, buzz_words=buzz,
                          sources=("wikipedia", "arxiv"), arxiv=arxiv)
    # 6 from the main article, 4 branch openings, then arXiv until the cap
    assert len(rows) == MAX_EVIDENCE_PER_SUB_QUESTION
    assert [row.source_title for row in rows][-2:] == ["Paper 0", "Paper 1"]


# ----- no passage is saved twice for one question -----

def two_sub_questions(tmp_path, first, second, article, web=None):
    """Searches two sub-questions of one question, sharing one SavedPassages, as SearchTool does."""
    store = EvidenceStore(tmp_path / "evidence.db")
    question_id = store.add_question("A question")
    saved = SavedPassages()
    fetchers = {SourceType.WIKIPEDIA: FakeSource({}), SourceType.ARXIV: FakeSource({}),
                SourceType.WEB: web or FakeSource({})}
    results = []
    for number, (main_and_buzz, sources) in enumerate((first, second), start=1):
        result = run_search(store, question_id, number, f"Sub-question {number} text", list(sources),
                            main_and_buzz, fetchers, {SourceType.WIKIPEDIA: article}, saved)
        results.append((result, store.read_evidence(result.evidence_ids)))
    return results


def test_a_second_sub_question_skips_the_passages_the_first_already_saved(tmp_path):
    sentences = ["Opening summary of the seasons."] + \
                ["A season comes from the axial tilt."] * 5 + ["A season begins at an equinox."] * 4
    sentences = [f"{sentence} Paragraph {n}." for n, sentence in enumerate(sentences)]
    season = long_document("Season", sentences)
    article = FakeSource({"Season": [season]})
    (first, first_rows), (second, second_rows) = two_sub_questions(
        tmp_path, (["Season", "Axial tilt"], ["wikipedia"]), (["Season", "Equinox"], ["wikipedia"]), article)
    # the first takes the opening and the 5 axial tilt passages; the second picks the opening,
    # one axial tilt passage and the 4 equinox ones, and keeps only the 4 it did not repeat
    assert [row.passage for row in first_rows] == [paragraph(sentences[n]) for n in range(6)]
    assert [row.passage for row in second_rows] == [paragraph(sentences[n]) for n in range(6, 10)]
    assert not set(first.evidence_ids) & set(second.evidence_ids)


def test_a_branch_another_sub_question_already_saved_is_skipped(tmp_path):
    fission = RawDocument("Nuclear fission", "https://example.org/fission", "",
                          "Nuclear fission splits a heavy nucleus into lighter ones.")
    sun = RawDocument("Sun", "https://example.org/sun", "", "The Sun is powered by nuclear fusion.")
    article = FakeSource({"Nuclear fission": [fission], "Sun": [sun]})
    (_, first_rows), (_, second_rows) = two_sub_questions(
        tmp_path, (["Nuclear fission"], ["wikipedia"]), (["Sun", "Nuclear fission"], ["wikipedia"]), article)
    assert [row.source_title for row in first_rows] == ["Nuclear fission"]
    assert [row.source_title for row in second_rows] == ["Sun"]


def test_a_sub_question_whose_every_pick_was_a_repeat_points_at_those_passages(tmp_path):
    article = FakeSource({"User Datagram Protocol": [UDP]})
    web = FakeSource({})
    (first, _), (second, second_rows) = two_sub_questions(
        tmp_path, (BUZZ, ["wikipedia"]), (BUZZ, ["wikipedia"]), article, web=web)
    # nothing new for the second, so it shares the first's evidence instead of reporting none
    assert second.status.value == "found"
    assert second.evidence_ids == first.evidence_ids
    assert [row.source_title for row in second_rows] == ["User Datagram Protocol"]
    assert web.queries == []  # found something, so the web fallback never runs


def test_each_question_starts_with_nothing_saved_and_its_record_is_dropped_at_the_end(tmp_path):
    class Context:
        def __init__(self, question):
            self.state = {STATE_USER_QUESTION: question}

    tool = SearchTool(store=EvidenceStore(tmp_path / "evidence.db"), fetchers={})
    first, second = Context("First question"), Context("Second question")
    tool.register_question(first)
    tool.register_question(second)
    first_id, second_id = first.state[STATE_QUESTION_ID], second.state[STATE_QUESTION_ID]
    assert tool.saved[first_id] is not tool.saved[second_id]
    tool.finalize_search_results(first)  # no plan in state: nothing to write, the record still goes
    assert first_id not in tool.saved
    assert second_id in tool.saved


# ----- which passages of an article are kept -----

def test_the_opening_passage_is_always_kept_with_the_passages_that_match():
    sentences = ["This article opens with a summary of the topic."] + \
                ["Filler text with nothing useful in it."] * 9
    sentences[3] = "A season follows from the axial tilt."
    sentences[5] = "Each season is a quarter of the year."
    sentences[8] = "The axial tilt is about 23 degrees."
    document = long_document("Season", sentences)
    chosen = passages_of(document, SourceType.WIKIPEDIA, ["Season", "Axial tilt"])
    # the opening, then only the passages that match, in their original order
    assert chosen == [paragraph(sentences[n]) for n in (0, 3, 5, 8)]


def test_at_most_the_opening_and_five_best_are_kept_from_a_wikipedia_article():
    sentences = ["Opening summary."] + [f"Season fact number {n}." for n in range(1, 10)]
    document = long_document("Season", sentences)
    chosen = passages_of(document, SourceType.WIKIPEDIA, ["Season"])
    assert len(chosen) == PASSAGES_PER_DOCUMENT[SourceType.WIKIPEDIA] == 6
    assert chosen == [paragraph(sentence) for sentence in sentences[:6]]


def test_only_wikipedia_keeps_its_opening_passage_regardless():
    document = long_document("A paper", ["Nothing here.", "Still nothing.", "Season results are reported."])
    chosen = passages_of(document, SourceType.ARXIV, ["Season"])
    assert chosen == [paragraph("Season results are reported.")]


def test_the_sub_questions_own_words_break_a_tie_between_passages():
    sentences = ["Apsis opening summary.",
                 "The word perihelion comes from Greek roots.",
                 "Earth reaches perihelion in early January, when the Northern Hemisphere has winter."]
    document = long_document("Apsis", sentences)
    words = ["earth", "closest", "sun", "northern", "hemisphere", "summer"]
    # a source that keeps one passage, so only one of the two tied passages fits; both score 3
    assert passages_of(document, SourceType.ARXIV, ["Perihelion"]) == [paragraph(sentences[1])]
    assert passages_of(document, SourceType.ARXIV, ["Perihelion"], words=words) == [paragraph(sentences[2])]


def test_the_sub_questions_words_never_beat_a_higher_score():
    sentences = ["Earth and the Sun and the Northern Hemisphere summer, at length.",
                 "Perihelion and aphelion are the two apsides."]
    document = long_document("Apsis", sentences)
    chosen = passages_of(document, SourceType.ARXIV, ["Perihelion", "Aphelion"],
                         words=["earth", "sun", "northern", "hemisphere", "summer"])
    assert chosen == [paragraph(sentences[1])]


def test_question_words_drop_stopwords_filler_and_plurals():
    assert question_words("In Google's Agent Development Kit (ADK), how does a LoopAgent decide "
                          "when to stop looping?") == \
        ["google", "agent", "development", "kit", "adk", "loopagent", "decide", "stop", "looping"]
    assert question_words("What kinds of applications use UDP?") == ["application", "udp"]
    assert question_words("Does glass flow?") == ["glass", "flow"]


def test_names_match_as_whole_words_with_plurals_and_hyphens_or_spaces():
    assert mentions("Retrieval-augmented generation helps.", "retrieval augmented generation")
    assert mentions("Transformers rely on attention.", "Transformer")
    assert not mentions("The WAP datagram layer looks a bit like UDP.", "User Datagram Protocol")
    assert not mentions("A transformational idea.", "Transformer")


# ----- main names with a bracketed qualifier -----

def test_a_bracketed_name_splits_into_its_bare_name_and_bracket_words():
    assert bare_name("BERT (language model)") == "BERT"
    assert bracket_words("BERT (language model)") == ["language", "model"]
    assert bracket_words("Transformer (machine learning architecture)") == \
        ["machine", "learning", "architecture"]
    assert bare_name("User Datagram Protocol") == "User Datagram Protocol"
    assert bracket_words("User Datagram Protocol") == []


def test_a_result_names_a_bracketed_topic_by_its_bare_name_and_one_bracket_word():
    main = "BERT (language model)"
    assert names_topic("BERT is a language model from Google.", main)
    assert not names_topic("Bert and Ernie are Muppets.", main)  # the bare name alone is not enough
    assert not names_topic("A language model predicts text.", main)  # nor a bracket word alone
    assert names_topic("The User Datagram Protocol is connectionless.", "User Datagram Protocol")


def test_the_name_search_keeps_the_right_article_for_a_bracketed_name_no_page_contains(tmp_path):
    electrical = RawDocument("Transformer", "https://example.org/transformer", "",
                             "A transformer transfers electrical energy between circuits.")
    machine = RawDocument("Transformer (deep learning)", "https://example.org/transformer_dl", "",
                          "The transformer is a deep learning architecture used in machine learning.")
    main = "Transformer (machine learning architecture)"
    wikipedia_search = FakeSource({main: [electrical, machine]})
    result, rows = search(tmp_path, wikipedia_search, article=FakeSource({}), buzz_words=[main, "Attention"])
    # the electrical page ranks first and names "transformer", but no bracket word
    assert [row.source_title for row in rows] == ["Transformer (deep learning)"]


def test_arxiv_keeps_a_paper_that_names_the_bare_name_and_a_bracket_word(tmp_path):
    papers = [RawDocument("Muppets", "https://arxiv.org/abs/1", "", "Bert and Ernie on television."),
              RawDocument("BERT", "https://arxiv.org/abs/2", "",
                          "We introduce BERT, a new language representation model.")]
    buzz = ["BERT (language model)", "Masked language model"]
    arxiv = FakeSource({" ".join(buzz): papers})
    result, rows = search(tmp_path, FakeSource({}), buzz_words=buzz, sources=("arxiv",), arxiv=arxiv)
    assert [row.source_title for row in rows] == ["BERT"]


def test_passages_score_the_bare_name_of_a_bracketed_main_name():
    sentences = ["Opening summary.", "Nothing relevant here.", "BERT reads text in both directions."]
    document = long_document("BERT (language model)", sentences)
    chosen = passages_of(document, SourceType.WIKIPEDIA, ["BERT (language model)"])
    assert chosen == [paragraph(sentences[0]), paragraph(sentences[2])]


def test_each_sub_question_logs_its_buzz_words(tmp_path, caplog):
    caplog.set_level("INFO", logger="search_agent")
    search(tmp_path, FakeSource({}), article=FakeSource({"User Datagram Protocol": [UDP]}))
    assert "sub-question 1: buzz words ['User Datagram Protocol', 'applications', 'DNS']" in caplog.text


# ----- the original paper, from the papers the main article cites -----

def atom_feed(entries):
    """An arXiv API answer holding the given (id, title, abstract, first posted) papers."""
    items = "".join(f"<entry><id>http://arxiv.org/abs/{paper_id}v2</id><title>{title}</title>"
                    f"<summary>{summary}</summary><published>{published}T00:00:00Z</published></entry>"
                    for paper_id, title, summary, published in entries)
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{items}</feed>'.encode()


def test_the_original_paper_is_the_oldest_cited_paper_that_names_the_topic():
    links = {"query": {"pages": [{"extlinks": [
        {"url": "https://arxiv.org/abs/2001.00001"}, {"url": "https://arxiv.org/pdf/1801.00002"},
        {"url": "https://example.org/not-a-paper"}, {"url": "https://arxiv.org/abs/1601.00003"}]}]}}
    feed = atom_feed([
        ("2001.00001", "A later follow-up", "We extend Low-Rank Adaptation to new settings.", "2020-01-05"),
        ("1801.00002", "LoRA: Low-Rank Adaptation of Large Language Models",
         "We propose Low-Rank Adaptation, which freezes the pre-trained weights.", "2018-01-03"),
        ("1601.00003", "An older paper on something else", "Matrices in general.", "2016-01-01")])
    asked = {}

    def wikipedia(url, params, limiter):
        asked["article"] = params["titles"]
        return links

    def arxiv(url, params, limiter):
        asked["ids"] = params["id_list"]
        return feed

    paper = original_paper("Fine-tuning (deep learning)", "Low-Rank Adaptation",
                           http_json=wikipedia, http=arxiv)
    assert asked == {"article": "Fine-tuning (deep learning)", "ids": "2001.00001,1801.00002,1601.00003"}
    # the 2016 paper is older but never names the topic
    assert paper.title == "LoRA: Low-Rank Adaptation of Large Language Models"
    assert paper.url == "http://arxiv.org/abs/1801.00002v2"


def test_a_paper_names_the_topic_by_the_phrase_or_by_every_word_of_it():
    assert names_paper("We present residual networks that ease training of deep neural networks.",
                       "Residual neural network")
    assert not names_paper("Residual connections help optimization.", "Residual neural network")
    assert names_paper("We introduce BERT, a language representation model.", "BERT (language model)")
    assert not names_paper("Bert and Ernie on television.", "BERT (language model)")


def test_arxiv_saves_the_original_paper_and_skips_its_keyword_search(tmp_path):
    paper = RawDocument("The original UDP paper", "http://arxiv.org/abs/1001.00001v1", "2010-01-01",
                        "We define the User Datagram Protocol for applications.")
    asked = []

    def finder(article_title, main):
        asked.append((article_title, main))
        return paper

    arxiv = FakeSource({BUZZ_QUERY: [WAP]})
    article = FakeSource({"User Datagram Protocol": [UDP]})
    # listed arxiv first, but Wikipedia is searched first so its article is known
    result, rows = search(tmp_path, FakeSource({}), article=article, sources=("arxiv", "wikipedia"),
                          arxiv=arxiv, paper_finder=finder)
    assert asked == [("User Datagram Protocol", ["User Datagram Protocol"])]
    assert arxiv.queries == []
    assert [row.source_title for row in rows] == ["User Datagram Protocol", "The original UDP paper"]


def test_without_an_original_paper_arxiv_runs_its_keyword_search(tmp_path):
    study = RawDocument("A UDP study", "https://arxiv.org/abs/2", "",
                        "The User Datagram Protocol under load.")
    arxiv = FakeSource({BUZZ_QUERY: [study]})
    article = FakeSource({"User Datagram Protocol": [UDP]})
    result, rows = search(tmp_path, FakeSource({}), article=article, sources=("wikipedia", "arxiv"),
                          arxiv=arxiv, paper_finder=lambda article_title, main: None)
    assert arxiv.queries == [BUZZ_QUERY]
    assert [row.source_title for row in rows] == ["User Datagram Protocol", "A UDP study"]


def test_the_original_paper_needs_a_wikipedia_article_to_start_from(tmp_path):
    asked = []
    search(tmp_path, FakeSource({}), sources=("arxiv",),
           paper_finder=lambda article_title, main: asked.append(article_title))
    assert asked == []


def test_a_failed_paper_lookup_falls_back_to_the_keyword_search(tmp_path):
    def finder(article_title, main):
        raise FetchError("HTTP 503 from export.arxiv.org")

    arxiv = FakeSource({})
    result, rows = search(tmp_path, FakeSource({}), article=FakeSource({"User Datagram Protocol": [UDP]}),
                          sources=("wikipedia", "arxiv"), arxiv=arxiv, paper_finder=finder)
    assert arxiv.queries == [BUZZ_QUERY, SUB_QUESTION]
    assert "arxiv original paper: HTTP 503" in result.error


def test_real_runs_find_original_papers_and_tests_with_fake_sources_do_not():
    assert SearchTool(store=object()).paper_finder is original_paper
    assert SearchTool(store=object(), fetchers={}).paper_finder is None
    assert SearchTool(store=object()).redirect_finder is section_redirects
    assert SearchTool(store=object(), fetchers={}).redirect_finder is None
    assert SearchTool(store=object()).page_reader is read_web_page
    assert SearchTool(store=object(), fetchers={}).page_reader is None


def test_the_paper_finder_tries_each_name_in_order_against_the_same_citations():
    links = {"query": {"pages": [{"extlinks": [{"url": "https://arxiv.org/abs/2201.00001"},
                                               {"url": "https://arxiv.org/abs/2211.00002"}]}]}}
    feed = atom_feed([("2201.00001", "Chain-of-Thought Prompting Elicits Reasoning",
                       "We show that chain of thought prompting improves reasoning.", "2022-01-28"),
                      ("2211.00002", "Large Language Models Are Human-Level Prompt Engineers",
                       "Automatic prompt engineering for large language models.", "2022-11-03")])
    requests = []

    def wikipedia(url, params, limiter):
        requests.append("wikipedia")
        return links

    def arxiv(url, params, limiter):
        requests.append("arxiv")
        return feed

    paper = original_paper("Prompt engineering", ["Chain-of-thought prompting", "Prompt engineering"],
                           http_json=wikipedia, http=arxiv)
    assert paper.title == "Chain-of-Thought Prompting Elicits Reasoning"
    assert requests == ["wikipedia", "arxiv"]  # one look at the citations, whatever the number of names


# ----- the focus topic, from the main article's section redirects -----

def test_only_redirects_to_a_section_of_the_article_are_kept():
    answer = {"query": {"pages": [{"title": "Prompt engineering", "redirects": [
        {"title": "Chain-of-thought prompting", "fragment": "Chain-of-thought"},
        {"title": "Prompt design"},
        {"title": "Few-shot prompting", "fragment": "In-context learning"}]}]}}
    titles = section_redirects("Prompt engineering", http=lambda url, params, limiter: answer)
    assert titles == ["Chain-of-thought prompting", "Few-shot prompting"]


def test_the_focus_is_the_longest_section_the_sub_question_names():
    sections = ["Chain-of-thought prompting", "Chain-of-thought", "CoT", "Few-shot prompting"]
    text = "What is chain-of-thought prompting, and which tasks does CoT improve?"
    assert focus_topic(sections, text) == "Chain-of-thought prompting"
    assert focus_topic(["CoT"], text) is None  # too short to trust
    assert focus_topic(sections, "What is prompt engineering?") is None


def test_a_bracketed_section_name_needs_a_word_from_its_bracket():
    assert focus_topic(["Seasons (China)"], "What causes the seasons on Earth?") is None
    assert focus_topic(["Seasons (China)"], "How are the seasons divided in China?") == "Seasons (China)"


def test_passages_about_the_focus_are_chosen_even_when_the_main_name_is_everywhere():
    sentences = ["Prompt engineering opening summary."] + \
                [f"Prompt engineering in general, part {n}." for n in range(1, 8)] + \
                ["Chain-of-thought prompting is a prompt engineering technique."]
    document = long_document("Prompt engineering", sentences)
    without = passages_of(document, SourceType.WIKIPEDIA, ["Prompt engineering"])
    with_focus = passages_of(document, SourceType.WIKIPEDIA, ["Prompt engineering"],
                             focus="Chain-of-thought prompting")
    assert paragraph(sentences[8]) not in without
    assert paragraph(sentences[8]) in with_focus


def test_run_search_steers_passages_and_the_paper_by_the_focus(tmp_path):
    text = "What is chain-of-thought prompting and which tasks does it improve?"
    sentences = ["Prompt engineering opening summary."] + \
                [f"Prompt engineering in general, part {n}." for n in range(1, 8)] + \
                ["Chain-of-thought prompting is a prompt engineering technique."]
    article = FakeSource({"Prompt engineering": [long_document("Prompt engineering", sentences)]})
    asked = []

    def finder(article_title, names):
        asked.append((article_title, names))
        return None

    store = EvidenceStore(tmp_path / "evidence.db")
    question_id = store.add_question("A question")
    fetchers = {SourceType.WIKIPEDIA: FakeSource({}), SourceType.ARXIV: FakeSource({}),
                SourceType.WEB: FakeSource({})}
    result = run_search(store, question_id, 1, text, ["wikipedia", "arxiv"], ["Prompt engineering"],
                        fetchers, {SourceType.WIKIPEDIA: article}, paper_finder=finder,
                        redirect_finder=lambda title: ["Chain-of-thought prompting"])
    passages = [row.passage for row in store.read_evidence(result.evidence_ids)]
    assert paragraph(sentences[8]) in passages
    assert asked == [("Prompt engineering", ["Chain-of-thought prompting", "Prompt engineering"])]


def test_a_failed_redirect_lookup_only_means_no_focus(tmp_path):
    def broken(title):
        raise FetchError("HTTP 503 from en.wikipedia.org")

    store = EvidenceStore(tmp_path / "evidence.db")
    question_id = store.add_question("A question")
    fetchers = {SourceType.WIKIPEDIA: FakeSource({}), SourceType.ARXIV: FakeSource({}),
                SourceType.WEB: FakeSource({})}
    result = run_search(store, question_id, 1, SUB_QUESTION, ["wikipedia"], BUZZ, fetchers,
                        {SourceType.WIKIPEDIA: FakeSource({"User Datagram Protocol": [UDP]})},
                        redirect_finder=broken)
    assert result.status.value == "found"
    assert "wikipedia redirects: HTTP 503" in result.error


# ----- the web: on-topic results, read as whole pages -----

ADK_QUESTION = "In Google's Agent Development Kit (ADK), how does a LoopAgent decide when to stop looping?"
ADK_BUZZ = ["Software development kit", "Intelligent agent"]
ADK_RESULT = RawDocument("Loop agents - Agent Development Kit (ADK)", "https://adk.example/loop-agents/", "",
                         "Crucially, the LoopAgent itself does not inherently decide when to stop looping.")
ADK_PAGE = "\n\n".join(paragraph(sentence) for sentence in [
    "Skip to content. Home. Docs. Search. Menu.",
    "The LoopAgent runs its sub-agents in a loop. It does not decide when to stop looping on its own.",
    "Set max_iterations on the LoopAgent to stop the loop after that many iterations.",
    "Copyright and cookie settings.",
    "A sub-agent can escalate to stop the LoopAgent early, when its condition is met.",
    "Copyright and cookie settings, again.",
])
ADK_PAGE_FACTS = [1, 2, 4]  # the paragraphs a reader needs; the rest is the site's frame


def web_search(tmp_path, web, page_reader=None, text=ADK_QUESTION, buzz_words=ADK_BUZZ):
    store = EvidenceStore(tmp_path / "evidence.db")
    question_id = store.add_question("A question")
    fetchers = {SourceType.WIKIPEDIA: FakeSource({}), SourceType.ARXIV: FakeSource({}), SourceType.WEB: web}
    result = run_search(store, question_id, 1, text, ["web"], buzz_words, fetchers, {},
                        page_reader=page_reader)
    return result, store.read_evidence(result.evidence_ids)


def test_web_searches_the_plain_sub_question_first_then_the_buzz_words(tmp_path):
    web = FakeSource({})
    web_search(tmp_path, web)
    assert web.queries == [ADK_QUESTION, "Software development kit Intelligent agent"]


def test_a_web_result_is_on_topic_if_it_names_the_topic_or_half_the_questions_words():
    words = question_words(ADK_QUESTION)
    assert about_sub_question(ADK_RESULT.title + " " + ADK_RESULT.content, "Software development kit", words)
    assert about_sub_question("A software development kit bundles tools.", "Software development kit", words)
    assert not about_sub_question("Google is a search company. Its agents sell ads.",
                                  "Software development kit", words)


def test_an_on_topic_web_result_is_read_and_its_best_passages_saved(tmp_path):
    read = []

    def reader(url):
        read.append(url)
        return ADK_PAGE

    web = FakeSource({ADK_QUESTION: [ADK_RESULT]})
    result, rows = web_search(tmp_path, web, page_reader=reader)
    assert read == ["https://adk.example/loop-agents/"]
    page = ADK_PAGE.split("\n\n")
    assert [row.passage for row in rows] == [page[n] for n in ADK_PAGE_FACTS]
    assert {row.source_url for row in rows} == {"https://adk.example/loop-agents/"}


def test_a_page_that_cannot_be_read_gives_its_snippet(tmp_path):
    def reader(url):
        raise FetchError("could not read adk.example: HTTP 403")

    result, rows = web_search(tmp_path, FakeSource({ADK_QUESTION: [ADK_RESULT]}), page_reader=reader)
    assert result.status.value == "found"
    assert [row.passage for row in rows] == [ADK_RESULT.content]
    assert result.error == ""  # the snippet stood in, so nothing failed


def test_a_page_of_menus_yields_no_passages_so_its_snippet_is_used(tmp_path):
    page = RawDocument("Menus", "https://adk.example/", "", "Skip to content. Home. Docs. Menu.")
    assert page_passages(page, ADK_BUZZ, question_words(ADK_QUESTION)) == []
    result, rows = web_search(tmp_path, FakeSource({ADK_QUESTION: [ADK_RESULT]}),
                              page_reader=lambda url: page.content)
    assert [row.passage for row in rows] == [ADK_RESULT.content]


def test_off_topic_web_results_are_skipped_and_at_most_two_pages_kept(tmp_path):
    ads = RawDocument("Google Ads", "https://ads.example/", "", "Advertise with Google.")
    others = [RawDocument(f"LoopAgent guide {n}", f"https://guide{n}.example/", "",
                          "How a LoopAgent in the Agent Development Kit decides to stop looping.")
              for n in range(3)]
    result, rows = web_search(tmp_path, FakeSource({ADK_QUESTION: [ads] + others}))
    assert [row.source_title for row in rows] == ["LoopAgent guide 0", "LoopAgent guide 1"]


def test_the_web_results_read_are_those_that_cover_most_of_the_question(tmp_path):
    # the engine's first result only names the Manager's generic buzz word
    generic = RawDocument("Google Antigravity", "https://wiki.example/antigravity", "",
                          "Google Antigravity is a software development kit where agents write code.")
    partial = RawDocument("ADK overview", "https://adk.example/", "",
                          "The Agent Development Kit (ADK) from Google builds agents.")
    best = RawDocument("Loop workflow", "https://adk.example/loop", "",
                       "In Google's Agent Development Kit (ADK) a LoopAgent decides to stop looping.")
    result, rows = web_search(tmp_path, FakeSource({ADK_QUESTION: [generic, partial, best]}))
    assert [row.source_title for row in rows] == ["Loop workflow", "ADK overview"]


def test_a_references_section_is_never_kept_from_a_page():
    references = paragraph("[12] A. Smith et al. LoopAgent stop looping in the Agent Development Kit. "
                           "Proceedings of ADK 2025, pp. 1 to 9. Retrieved 3 May 2025.")
    page = RawDocument("Loop agents", "https://adk.example/", "", references + "\n\n" + ADK_PAGE)
    chosen = page_passages(page, ADK_BUZZ, question_words(ADK_QUESTION))
    assert references not in chosen
    assert chosen == [ADK_PAGE.split("\n\n")[n] for n in ADK_PAGE_FACTS]


class FakeDDGS:
    """Stands in for ddgs.DDGS: answers text() and extract() from what it was given."""

    def __init__(self, results=None, error=None, page=None):
        self.results, self.error, self.page = results, error, page
        self.calls = []

    def __call__(self):
        return self

    def text(self, query, **kwargs):
        self.calls.append(("text", query, kwargs))
        if self.error is not None:
            error, self.error = self.error, None  # fail once, then answer
            raise error
        return self.results

    def extract(self, url, fmt):
        self.calls.append(("extract", url, fmt))
        if isinstance(self.page, Exception):
            raise self.page
        return {"url": url, "content": self.page}


NO_WAIT = RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0)


def test_fetch_web_maps_results_and_treats_no_results_as_nothing_found():
    from ddgs.exceptions import DDGSException

    fake = FakeDDGS(results=[
        {"title": "Loop agents", "href": "https://adk.example/", "body": "LoopAgent docs."},
        {"title": "No body", "href": "https://empty.example/", "body": ""}])
    documents = fetch_web("a query", ddgs_factory=fake, policy=NO_WAIT, sleep=lambda seconds: None)
    assert documents == [RawDocument("Loop agents", "https://adk.example/", "", "LoopAgent docs.")]
    assert fake.calls[0][2]["region"] == "us-en"
    assert "grokipedia" not in fake.calls[0][2]["backend"]  # AI-written, never evidence
    assert "auto" not in fake.calls[0][2]["backend"]
    nothing = FakeDDGS(results=None, error=DDGSException("No results found."))
    assert fetch_web("a query", ddgs_factory=nothing, policy=RetryPolicy(max_attempts=1),
                     sleep=lambda seconds: None) == []



# The glued titles of the 2026-11-13 runs, as saved (Q03's cut after its fourth piece)
GLUED_Q03 = ("LoRA: Low-Rank Adaptation of Large Language ModelsFine-Tuning using LoRA and QLoRA"
             " - GeeksforGeeksCE-LoRA: Computation-Efficient LoRA Fine-Tuning for Language ..."
             "LoRA (machine learning) - Wikipedia")
GLUED_Q19 = ("Control Flow in C: If-Else, Switch, Loops & Jump StatementsLoops in Programming"
             " - GeeksforGeeksControl Flow Structures in PythonControl Flow - Documentation"
             "JavaScript Control Flow - W3Schools")
GLUED_E07 = ("Antibiotic Stewardship | Georgia Department of Public HealthAntimicrobial "
             "resistance - World Health Organization (WHO)Antibiotic Resistance: A Global "
             "Problem and the Need to Do ...Antimicrobial Resistance Threats Report | CDC")


def test_first_title_cuts_a_glued_web_title_at_its_first_glue_point():
    assert first_title(GLUED_Q03) == "LoRA: Low-Rank Adaptation of Large Language Models"
    assert first_title(GLUED_Q19) == "Control Flow in C: If-Else, Switch, Loops & Jump Statements"
    assert first_title(GLUED_E07) == "Antibiotic Stewardship | Georgia Department of Public Health"


def test_first_title_leaves_real_titles_alone():
    for title in ["PyTorch vs TensorFlow - GeeksforGeeks", "LoRA (machine learning) - Wikipedia",
                  "LoopAgent - Agent Development Kit (ADK) | YouTube and LinkedIn",
                  # long but not glued: no capital follows a small letter directly
                  "Automated Detection of Patellofemoral Osteoarthritis from Knee Lateral View "
                  "Radiographs Using Deep Learning: Data from the Multicenter Osteoarthritis "
                  "Study (MOST)"]:
        assert first_title(title) == title


def test_fetch_web_cuts_a_glued_title_and_keeps_the_address_and_snippet():
    fake = FakeDDGS(results=[{"title": GLUED_Q19, "href": "https://codingzap.com/control-flow-in-c/",
                              "body": "Control flow decides the order statements run in."}])
    documents = fetch_web("a query", ddgs_factory=fake, policy=NO_WAIT, sleep=lambda seconds: None)
    assert documents == [RawDocument("Control Flow in C: If-Else, Switch, Loops & Jump Statements",
                                     "https://codingzap.com/control-flow-in-c/", "",
                                     "Control flow decides the order statements run in.")]


def test_fetch_web_retries_a_rate_limit_and_reports_other_errors():
    from ddgs.exceptions import DDGSException
    from ddgs.exceptions import RatelimitException

    limited = FakeDDGS(results=[{"title": "T", "href": "https://a.example/", "body": "B"}],
                       error=RatelimitException("202 Ratelimit"))
    assert len(fetch_web("q", ddgs_factory=limited, policy=NO_WAIT, sleep=lambda seconds: None)) == 1
    assert [call[0] for call in limited.calls] == ["text", "text"]
    broken = FakeDDGS(error=DDGSException("https://html.duckduckgo.com/html 403 Forbidden"))
    try:
        fetch_web("q", ddgs_factory=broken, policy=NO_WAIT, sleep=lambda seconds: None)
    except FetchError as error:
        assert "web search error" in str(error)
    else:
        raise AssertionError("a search engine error must be reported, not read as nothing found")


def test_a_pages_menus_are_dropped_and_its_wrapped_prose_kept_whole():
    page = "\n".join([
        "Skip to content", "Home", "Docs", "Python Python TypeScript JS Go Go Java Java Kotlin Kotlin",
        "Overview Studio Agents Models Notebooks API reference Pricing",
        "Crucially, the LoopAgent itself does not inherently decide when to stop looping. You must",
        "implement a termination mechanism.",
        "Get started", "Sign in",
        "Set max_iterations to stop the loop after that many iterations of the agent.",
    ])
    assert readable_text(page) == (
        "Crucially, the LoopAgent itself does not inherently decide when to stop looping. You must "
        "implement a termination mechanism.\n\n"
        "Set max_iterations to stop the loop after that many iterations of the agent.")


def test_only_one_short_line_may_end_a_wrapped_sentence():
    page = "\n".join(["The LoopAgent runs its sub-agents again and again in a loop until it", "stops",
                      "Home", "Docs"])
    assert readable_text(page) == "The LoopAgent runs its sub-agents again and again in a loop until it stops"


def test_read_web_page_returns_the_text_and_skips_what_it_cannot_read():
    fake = FakeDDGS(page="  Home\nThe LoopAgent page text is a sentence of plain prose.  ")
    assert read_web_page("https://adk.example/", ddgs_factory=fake, sleep=lambda seconds: None) == \
        "The LoopAgent page text is a sentence of plain prose."
    assert fake.calls == [("extract", "https://adk.example/", "text_plain")]
    skipped = FakeDDGS(page="unused")
    assert read_web_page("https://arxiv.example/paper.pdf", ddgs_factory=skipped) is None
    assert read_web_page("ftp://files.example/", ddgs_factory=skipped) is None
    assert skipped.calls == []
    try:
        read_web_page("https://blocked.example/", ddgs_factory=FakeDDGS(page=RuntimeError("HTTP 403")),
                      sleep=lambda seconds: None)
    except FetchError as error:
        assert "could not read blocked.example" in str(error)
    else:
        raise AssertionError("a page that cannot be fetched must raise FetchError")


# ----- what the Manager is told, and refusals -----

def test_the_manager_is_told_to_name_articles_and_split_comparisons():
    assert "Each buzz word is the name of a Wikipedia article" in SEARCH_INSTRUCTION
    assert "the most specific topic, not the broad subject" in SEARCH_INSTRUCTION
    assert "the first is the article about that belief" in SEARCH_INSTRUCTION
    assert "plan one sub-question for each thing" in SEARCH_INSTRUCTION
    # a topic Wikipedia covers only inside a broader article is still named first
    assert "covers only as a section of a broader article, the first buzz word is still" in SEARCH_INSTRUCTION
    assert "never a broader field" not in SEARCH_INSTRUCTION


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
