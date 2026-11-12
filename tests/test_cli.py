# Tests for the command-line interface (SRS 3.8). Run from the project root:
#     python -m pytest tests
# No test calls Gemini: the model is replaced by a scripted one.
import asyncio
import json
from collections import namedtuple

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import errors as genai_errors
from google.genai import types

import pipeline
from manager_agent import build_manager_agent
from rate_limit import DailyQuotaExceeded
from schemas import STATE_DRAFT
from schemas import STATE_PLAN
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

DRAFT = {
    "title": "Transformers and long documents",
    "summary": "Transformers use attention, and long inputs need extra methods.",
    "sections": [{
        "heading": "Attention",
        "claims": [
            {"claim_number": 1, "sentence": "Transformers rely on self-attention.", "evidence_ids": [1]},
            {"claim_number": 2, "sentence": "Attention cost grows linearly with length.",
             "evidence_ids": [2]},
        ],
    }],
}

VERDICT = {
    "checks": [
        {"claim_number": 1, "verdict": "supported", "reason": "Evidence 1 says so directly."},
        {"claim_number": 2, "verdict": "contradicted", "reason": "Evidence 2 says the cost is quadratic.",
         "restatement": "Attention cost grows quadratically with length."},
    ],
    "result": "fail",
    "feedback": "Claim 2 contradicts evidence 2.",
}

Row = namedtuple("Row", "evidence_id source_type source_title source_url published_date passage")


# Stands in for the evidence store: returns a row for each requested ID
def fake_sources(evidence_ids):
    rows = []
    for evidence_id in evidence_ids:
        rows.append(Row(evidence_id, "wikipedia", f"Source {evidence_id}",
                        f"https://example.org/{evidence_id}", "", "text"))
    return rows


# A model that returns one fixed reply, or raises a fixed error, instead of calling Gemini
class ScriptedLlm(BaseLlm):
    reply: str = ""
    error_code: int = 0
    daily_quota_used: bool = False

    async def generate_content_async(self, llm_request, stream=False):
        if self.daily_quota_used:
            raise DailyQuotaExceeded("Gemini free-tier daily request quota is used up (scripted)")
        if self.error_code:
            status = {429: "RESOURCE_EXHAUSTED", 404: "NOT_FOUND", 503: "UNAVAILABLE"}[self.error_code]
            error_class = genai_errors.ServerError if self.error_code >= 500 else genai_errors.ClientError
            raise error_class(self.error_code, {"error": {
                "code": self.error_code, "message": "scripted error", "status": status}})
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.reply)]))


# Builds the real Manager agent with the scripted model in place of Gemini
def manager_with(model):
    agent = build_manager_agent()
    agent.model = model
    return agent


def test_plan_only_shows_sub_questions_and_no_report():
    text = cli.render({STATE_PLAN: PLAN}, calls=1, source_reader=fake_sources)
    assert "[1] What is self-attention in a transformer? (wikipedia)" in text
    assert "[2] Which methods extend transformers to long inputs? (arxiv)" in text
    assert "No report in this run" in text
    assert "Model calls: 1" in text


def test_3_8_2_checked_report_shows_evidence_sources_flags_and_score():
    state = {STATE_PLAN: PLAN, STATE_DRAFT: DRAFT, STATE_VERDICT: VERDICT}
    text = cli.render(state, calls=7, source_reader=fake_sources)
    assert "(1) Transformers rely on self-attention. [evidence 1]" in text
    assert "(2) Attention cost grows linearly with length. [evidence 2] [FLAGGED]" in text
    assert "[1] Source 1 (wikipedia) https://example.org/1" in text
    assert "(2) contradicted: Evidence 2 says the cost is quadratic." in text
    assert "supported version: Attention cost grows quadratically with length." in text
    assert "Support score: 50.0%" in text
    assert "Model calls: 7" in text


def test_3_6_4_draft_without_verdict_is_not_shown():
    text = cli.render({STATE_DRAFT: DRAFT}, calls=3, source_reader=fake_sources)
    assert "has not checked the draft" in text
    assert "Transformers rely on self-attention." not in text


def test_3_6_4_verdict_missing_a_claim_is_not_shown():
    partial = {"checks": [VERDICT["checks"][0]], "result": "pass"}
    text = cli.render({STATE_DRAFT: DRAFT, STATE_VERDICT: partial}, calls=3, source_reader=fake_sources)
    assert "No report shown" in text
    assert "Transformers rely on self-attention." not in text


def test_run_counts_model_calls_and_records_the_plan():
    counter = cli.ModelCallCounter()
    agent = manager_with(ScriptedLlm(model="scripted", reply=json.dumps(PLAN)))
    state = asyncio.run(cli.run_question(QUESTION, agent, counter))
    assert counter.calls == 1
    assert len(state[STATE_PLAN]["sub_questions"]) == 2


@pytest.mark.parametrize("code, expected", [
    (429, "free tier refused the request (429"),
    (404, "is not available to this API key (404)"),
    (503, "Gemini is overloaded right now (503). That's on Google's side"),
])
def test_model_errors_print_a_plain_message(monkeypatch, capsys, code, expected):
    agent = manager_with(ScriptedLlm(model="scripted", error_code=code))
    monkeypatch.setattr(cli, "build_root_agent", lambda: agent)
    with pytest.raises(SystemExit) as stopped:
        cli.main([QUESTION])
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    assert expected in captured.out
    assert "Model calls: 1" in captured.out
    assert "Traceback" not in captured.err


# The pipeline without Search and without search results, so the Synthesizer
# stops for lack of evidence, as a run where Search found nothing would
def test_a_run_with_no_evidence_prints_a_plain_message(monkeypatch, capsys):
    agent = pipeline.build_pipeline(evidence_reader=fake_sources, search=False)
    agent.sub_agents[0].model = ScriptedLlm(model="scripted", reply=json.dumps(PLAN))
    monkeypatch.setattr(cli, "build_root_agent", lambda: agent)
    with pytest.raises(SystemExit) as stopped:
        cli.main([QUESTION])
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    assert "The run stopped before writing a report: " in captured.out
    assert "Search has not run" in captured.out
    # The counter also counts the Synthesizer call its evidence check blocked
    assert "Model calls: 2" in captured.out
    assert "Traceback" not in captured.err


def test_a_used_up_daily_quota_prints_a_plain_message(monkeypatch, capsys):
    agent = manager_with(ScriptedLlm(model="scripted", daily_quota_used=True))
    monkeypatch.setattr(cli, "build_root_agent", lambda: agent)
    with pytest.raises(SystemExit) as stopped:
        cli.main([QUESTION])
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    assert "daily request quota is used up" in captured.out
    assert "Traceback" not in captured.err
