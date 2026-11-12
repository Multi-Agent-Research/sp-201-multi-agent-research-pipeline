# Tests for the pipeline assembly (SDD 3.6, 5.5). Run from the project root:
#     python -m pytest tests
# No test calls Gemini or a real source: every agent's model is replaced by a
# scripted one. Most tests leave Search out and put the search results and
# evidence in place themselves; the Search tests use fake sources and a
# throwaway database.
import asyncio
import json
from collections import namedtuple

import pytest
from google.adk.apps.app import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

import pipeline
from evidence_store import EvidenceStore
from manager_agent import build_manager_agent
from rate_limit import RateLimitedGemini
from schemas import STATE_DRAFT
from schemas import STATE_FEEDBACK
from schemas import STATE_PLAN
from schemas import STATE_SEARCH_RESULTS
from schemas import STATE_USER_QUESTION
from schemas import STATE_VERDICT
from schemas import SourceType
from search_agent import SEARCH_TOOL_NAME
from search_agent import STATE_QUESTION_ID
from search_agent import RawDocument
from search_agent import attach_search
from sp201 import cli
from synthesizer_agent import EvidenceError

QUESTION = "How do transformers handle long documents?"

PLAN = {
    "user_question": QUESTION,
    "sub_questions": [
        {"sub_question_id": 1, "text": "What is self-attention in a transformer?",
         "preferred_sources": ["wikipedia"], "priority": 1},
        {"sub_question_id": 2, "text": "Which methods extend transformers to long inputs?",
         "preferred_sources": ["arxiv"], "priority": 2},
    ],
}

SEARCH_RESULTS = {"results": [
    {"sub_question_id": 1, "status": "found", "evidence_ids": [1]},
    {"sub_question_id": 2, "status": "found", "evidence_ids": [2]},
]}

Row = namedtuple("Row", "evidence_id source_type source_title source_url published_date passage")

EVIDENCE = {
    1: Row(1, "wikipedia", "Transformer (deep learning)", "https://example.org/1", "",
           "Transformers rely on self-attention, whose cost grows quadratically with sequence length."),
    2: Row(2, "arxiv", "Longformer", "https://example.org/2", "2020-04-10",
           "Longformer combines local windowed attention with global attention to scale linearly."),
}


# Stands in for the evidence store
def fake_reader(evidence_ids):
    return [EVIDENCE[i] for i in evidence_ids if i in EVIDENCE]


def draft(second_sentence):
    return {
        "title": "Transformers and long documents",
        "summary": "How attention cost limits input length, and one way around it.",
        "sections": [{
            "heading": "Attention and length",
            "claims": [
                {"claim_number": 1, "sentence": "Transformers rely on self-attention.", "evidence_ids": [1]},
                {"claim_number": 2, "sentence": second_sentence, "evidence_ids": [2]},
            ],
        }],
    }


BAD_DRAFT = draft("Longformer removes attention entirely.")
FIXED_DRAFT = draft("Longformer combines local and global attention to scale linearly.")

PASS = {
    "checks": [
        {"claim_number": 1, "verdict": "supported", "reason": "Evidence 1 says so directly."},
        {"claim_number": 2, "verdict": "supported", "reason": "Evidence 2 says so directly."},
    ],
    "result": "pass",
}

FAIL = {
    "checks": [
        {"claim_number": 1, "verdict": "supported", "reason": "Evidence 1 says so directly."},
        {"claim_number": 2, "verdict": "contradicted", "reason": "Evidence 2 says it keeps attention.",
         "restatement": "Longformer combines local and global attention to scale linearly."},
    ],
    "result": "fail",
    "feedback": "Claim 2 contradicts evidence 2.",
}


# A model call the scripted Manager makes, such as a search
def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


# A model that returns its scripted replies in order and keeps every request.
# A reply is a function call made with call(), or anything else as JSON text.
class ScriptedLlm(BaseLlm):
    replies: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)

    async def generate_content_async(self, llm_request, stream=False):
        self.requests.append(llm_request)
        reply = self.replies.pop(0)
        part = reply if isinstance(reply, types.Part) else types.Part(text=json.dumps(reply))
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


# Puts a scripted model in each LLM agent of a built pipeline
def script(agent, manager_replies, synthesizer_replies, fact_checker_replies):
    manager, loop = agent.sub_agents
    synthesizer, fact_checker, _ = loop.sub_agents
    models = {
        "manager": ScriptedLlm(model="scripted", replies=list(manager_replies)),
        "synthesizer": ScriptedLlm(model="scripted", replies=list(synthesizer_replies)),
        "fact_checker": ScriptedLlm(model="scripted", replies=list(fact_checker_replies)),
    }
    manager.model = models["manager"]
    synthesizer.model = models["synthesizer"]
    fact_checker.model = models["fact_checker"]
    return models


# The real pipeline without Search, with a scripted model in each LLM agent
def scripted_pipeline(manager_replies, synthesizer_replies, fact_checker_replies):
    agent = pipeline.build_pipeline(evidence_reader=fake_reader, search=False)
    return agent, script(agent, manager_replies, synthesizer_replies, fact_checker_replies)


# Runs one question. By default the search results are already in state, as
# Search would leave them; the Search tests start from the question alone.
def run(agent, state=None):
    if state is None:
        state = {STATE_USER_QUESTION: QUESTION, STATE_SEARCH_RESULTS: SEARCH_RESULTS}

    async def go():
        counter = cli.ModelCallCounter()
        runner = InMemoryRunner(app=App(name="sp201-test", root_agent=agent, plugins=[counter]))
        session = await runner.session_service.create_session(
            app_name="sp201-test", user_id="test", state=state)
        message = types.Content(role="user", parts=[types.Part(text=QUESTION)])
        async for _ in runner.run_async(user_id="test", session_id=session.id, new_message=message):
            pass
        final = await runner.session_service.get_session(
            app_name="sp201-test", user_id="test", session_id=session.id)
        return final.state, counter.calls
    return asyncio.run(go())


# The text the model received on a given call
def request_text(model, call):
    return model.requests[call].contents[0].parts[0].text


def test_3_6_2_structure_matches_the_readme():
    agent = pipeline.build_pipeline(evidence_reader=fake_reader, search=False)
    assert [a.name for a in agent.sub_agents] == ["manager", "revision_loop"]
    loop = agent.sub_agents[1]
    assert [a.name for a in loop.sub_agents] == ["synthesizer", "fact_checker", "escalation_checker"]
    assert loop.max_iterations == 2


def test_3_6_2_no_revisions_means_one_pass_through_the_loop(monkeypatch):
    monkeypatch.setenv("SP201_MAX_REVISIONS", "0")
    assert pipeline.build_revision_loop(evidence_reader=fake_reader).max_iterations == 1


def test_escalation_checker_ends_the_loop_only_on_a_pass():
    assert pipeline.verdict_passed({STATE_VERDICT: PASS})
    assert not pipeline.verdict_passed({STATE_VERDICT: FAIL})
    assert not pipeline.verdict_passed({})


def test_a_passing_first_draft_is_checked_once_and_the_loop_stops():
    agent, models = scripted_pipeline([PLAN], [FIXED_DRAFT], [PASS])
    state, calls = run(agent)
    assert calls == 3
    assert len(models["synthesizer"].requests) == 1
    assert state[STATE_PLAN]["sub_questions"][0]["text"] == PLAN["sub_questions"][0]["text"]
    assert state[STATE_VERDICT]["result"] == "pass"
    assert not state.get(STATE_FEEDBACK)
    text = cli.render(state, calls, source_reader=fake_reader)
    assert "(2) Longformer combines local and global attention to scale linearly. [evidence 2]" in text
    assert "Flagged claims: none" in text
    assert "Support score: 100.0%" in text
    assert "WARNING" not in text


def test_3_6_1_a_failed_draft_is_revised_once_with_the_feedback():
    agent, models = scripted_pipeline([PLAN], [BAD_DRAFT, FIXED_DRAFT], [FAIL, PASS])
    state, calls = run(agent)
    assert calls == 5
    first, second = request_text(models["synthesizer"], 0), request_text(models["synthesizer"], 1)
    assert first.startswith("MODE: initial_draft")
    assert second.startswith("MODE: revision")
    assert "Restatement to use: Longformer combines local and global attention" in second
    assert state[STATE_DRAFT] == FIXED_DRAFT
    assert state[STATE_VERDICT]["result"] == "pass"
    assert not state.get(STATE_FEEDBACK)


def test_3_6_4_a_draft_that_fails_twice_still_ends_on_a_check():
    still_bad = draft("Longformer removes most of attention.")
    agent, models = scripted_pipeline([PLAN], [BAD_DRAFT, still_bad], [FAIL, FAIL])
    state, calls = run(agent)
    assert calls == 5
    assert len(models["fact_checker"].requests) == 2
    assert request_text(models["fact_checker"], 1).count("Longformer removes most of attention.") == 1
    assert state[STATE_DRAFT] == still_bad
    assert state[STATE_VERDICT]["result"] == "fail"
    text = cli.render(state, calls, source_reader=fake_reader)
    assert "WARNING: this report is incomplete" in text
    assert "(2) Longformer removes most of attention. [evidence 2] [FLAGGED]" in text
    assert "Support score: 50.0%" in text


# Search, wired in: fake sources that return the two passages above, so no
# network is touched, and a throwaway database per test

FAKE_FETCHERS = {
    SourceType.WIKIPEDIA: lambda query: [RawDocument(
        EVIDENCE[1].source_title, EVIDENCE[1].source_url, "", EVIDENCE[1].passage)],
    SourceType.ARXIV: lambda query: [RawDocument(
        EVIDENCE[2].source_title, EVIDENCE[2].source_url, EVIDENCE[2].published_date, EVIDENCE[2].passage)],
    SourceType.WEB: lambda query: [],
}


def search_calls():
    sub_questions = PLAN["sub_questions"]
    return [call(SEARCH_TOOL_NAME, sub_question_id=1, text=sub_questions[0]["text"],
                 preferred_sources=["wikipedia"], buzz_words=["Transformer", "self-attention"]),
            call(SEARCH_TOOL_NAME, sub_question_id=2, text=sub_questions[1]["text"],
                 preferred_sources=["arxiv"], buzz_words=["Longformer", "long inputs"])]


def search_pipeline(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.db")
    return pipeline.build_pipeline(store=store, fetchers=FAKE_FETCHERS), store


def tool_names(agent):
    return [getattr(tool, "name", None) for tool in agent.tools]


def test_search_is_attached_to_the_manager_and_every_model_is_rate_limited(tmp_path):
    agent, _ = search_pipeline(tmp_path)
    manager, loop = agent.sub_agents
    synthesizer, fact_checker, _ = loop.sub_agents
    assert tool_names(manager) == [SEARCH_TOOL_NAME]
    for llm_agent in (manager, synthesizer, fact_checker):
        assert isinstance(llm_agent.model, RateLimitedGemini)


# Gemini read the Manager's old "you do not search" literally and never called
# Search (live run, 2026-11-12), so the instruction must send the work to the tool
def test_the_manager_is_told_to_send_each_sub_question_to_search(tmp_path):
    agent, _ = search_pipeline(tmp_path)
    instruction = agent.sub_agents[0].instruction
    assert f"send each one to Search with the {SEARCH_TOOL_NAME} tool" in instruction
    assert "You plan; you do not search" not in instruction
    assert f"IMPORTANT: you must call {SEARCH_TOOL_NAME} once for every sub-question" in instruction
    assert f"call {SEARCH_TOOL_NAME} once for each sub-question" in instruction


def test_search_is_never_attached_twice(tmp_path):
    manager = build_manager_agent()
    attach_search(manager, store=EvidenceStore(tmp_path / "evidence.db"), fetchers=FAKE_FETCHERS)
    # A Manager that already holds Search keeps its own; the reader is given here
    # so the loop does not open the default database
    agent = pipeline.build_pipeline(evidence_reader=fake_reader, manager=manager)
    assert tool_names(agent.sub_agents[0]) == [SEARCH_TOOL_NAME]


def test_3_6_2_a_question_runs_end_to_end_through_search(tmp_path):
    agent, store = search_pipeline(tmp_path)
    models = script(
        agent,
        search_calls() + [call("set_model_response", **PLAN)],
        [FIXED_DRAFT],
        [PASS],
    )
    state, calls = run(agent, state={STATE_USER_QUESTION: QUESTION})

    # Search recorded the question, saved one passage per sub-question and
    # reported both as found; the Manager made 3 calls, the loop 2
    assert calls == 5
    assert state[STATE_QUESTION_ID] == 1
    assert state[STATE_SEARCH_RESULTS]["results"] == [
        {"sub_question_id": 1, "status": "found", "evidence_ids": [1], "error": ""},
        {"sub_question_id": 2, "status": "found", "evidence_ids": [2], "error": ""},
    ]
    assert state[STATE_PLAN]["sub_questions"][1]["text"] == PLAN["sub_questions"][1]["text"]

    # The Synthesizer drafted from the passages Search saved
    assert EVIDENCE[2].passage in request_text(models["synthesizer"], 0)
    assert state[STATE_VERDICT]["result"] == "pass"

    # The CLI reads the same store for the sources it prints
    text = cli.render(state, calls, source_reader=store.read_evidence)
    assert "[2] Longformer (arxiv) https://example.org/2" in text
    assert "Support score: 100.0%" in text


# The two locks that make the Manager search

def forced_to_search(request):
    config = request.config.tool_config if request.config else None
    if config is None or config.function_calling_config is None:
        return False
    calling = config.function_calling_config
    return calling.mode == types.FunctionCallingConfigMode.ANY and \
        calling.allowed_function_names == [SEARCH_TOOL_NAME]


def test_lock_1_the_first_manager_turn_may_only_call_search(tmp_path):
    agent, _ = search_pipeline(tmp_path)
    models = script(agent, search_calls() + [call("set_model_response", **PLAN)], [FIXED_DRAFT], [PASS])
    run(agent, state={STATE_USER_QUESTION: QUESTION})
    first, second, third = models["manager"].requests
    assert forced_to_search(first)
    # Forcing the search keeps the Manager's thinking level on the same request
    for request in (first, second, third):
        assert request.config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert not forced_to_search(second)
    assert not forced_to_search(third)


def test_lock_2_a_plan_with_unsearched_sub_questions_is_refused_until_they_are_searched(tmp_path):
    agent, _ = search_pipeline(tmp_path)
    # A Manager that ignores lock 1 and hands in its plan before searching
    hand_in = call("set_model_response", **PLAN)
    manager_replies = [hand_in] + search_calls() + [hand_in]
    models = script(agent, manager_replies, [FIXED_DRAFT], [PASS])
    state, calls = run(agent, state={STATE_USER_QUESTION: QUESTION})

    assert calls == 6
    assert state[pipeline.STATE_PLAN_REFUSALS] == 1
    refused_turn = models["manager"].requests[1]
    assert "Plan not accepted: sub-question 1, 2 has not been searched" in str(refused_turn.contents)
    assert forced_to_search(refused_turn)
    assert [r["status"] for r in state[STATE_SEARCH_RESULTS]["results"]] == ["found", "found"]
    assert state[STATE_VERDICT]["result"] == "pass"


def test_lock_2_lets_the_plan_through_after_two_refusals(tmp_path):
    agent, _ = search_pipeline(tmp_path)
    hand_in = call("set_model_response", **PLAN)
    models = script(agent, [hand_in, hand_in, hand_in], [FIXED_DRAFT], [PASS])
    # The third hand-in is accepted, so the run reaches the Synthesizer, which
    # stops because nothing was searched
    with pytest.raises(EvidenceError, match="no evidence was found"):
        run(agent, state={STATE_USER_QUESTION: QUESTION})
    assert len(models["manager"].requests) == 3
    assert models["synthesizer"].requests == []


def test_lock_2_a_search_refused_for_missing_buzz_words_does_not_count(tmp_path):
    agent, _ = search_pipeline(tmp_path)
    sub_questions = PLAN["sub_questions"]
    no_buzz = call(SEARCH_TOOL_NAME, sub_question_id=1, text=sub_questions[0]["text"],
                   preferred_sources=["wikipedia"], buzz_words=[])
    hand_in = call("set_model_response", **PLAN)
    # Search refuses the first call, so the plan is refused until both are searched
    models = script(agent, [no_buzz, hand_in] + search_calls() + [hand_in], [FIXED_DRAFT], [PASS])
    state, _ = run(agent, state={STATE_USER_QUESTION: QUESTION})
    assert "not searched: buzz_words is empty" in str(models["manager"].requests[1].contents)
    assert state[pipeline.STATE_PLAN_REFUSALS] == 1
    assert [r["status"] for r in state[STATE_SEARCH_RESULTS]["results"]] == ["found", "found"]
