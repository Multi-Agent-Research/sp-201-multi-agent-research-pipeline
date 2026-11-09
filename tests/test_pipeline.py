# Tests for the pipeline assembly (SDD 3.6, 5.5). Run from the project root:
#     python -m pytest tests
# No test calls Gemini: every agent's model is replaced by a scripted one, and
# the search results and evidence are put in place by the test, since Search
# is not wired into the Manager yet.
import asyncio
import json
from collections import namedtuple

from google.adk.apps.app import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

import pipeline
from schemas import STATE_DRAFT
from schemas import STATE_FEEDBACK
from schemas import STATE_PLAN
from schemas import STATE_SEARCH_RESULTS
from schemas import STATE_USER_QUESTION
from schemas import STATE_VERDICT
from sp201 import cli

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


# A model that returns its scripted replies in order and keeps every request
class ScriptedLlm(BaseLlm):
    replies: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)

    async def generate_content_async(self, llm_request, stream=False):
        self.requests.append(llm_request)
        reply = json.dumps(self.replies.pop(0))
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=reply)]))


# The real pipeline, with a scripted model in each LLM agent
def scripted_pipeline(manager_replies, synthesizer_replies, fact_checker_replies):
    agent = pipeline.build_pipeline(evidence_reader=fake_reader)
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
    return agent, models


# Runs one question with search results already in state, as Search would leave them
def run(agent):
    async def go():
        counter = cli.ModelCallCounter()
        runner = InMemoryRunner(app=App(name="sp201-test", root_agent=agent, plugins=[counter]))
        session = await runner.session_service.create_session(
            app_name="sp201-test", user_id="test",
            state={STATE_USER_QUESTION: QUESTION, STATE_SEARCH_RESULTS: SEARCH_RESULTS})
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
    agent = pipeline.build_pipeline(evidence_reader=fake_reader)
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
