# Tests for what the Synthesizer is told (synthesizer_agent.build_instruction). Run from
# the project root:
#     python -m pytest tests
from google.genai import types

from schemas import STATE_DRAFT
from schemas import STATE_FEEDBACK
from schemas import STATE_VERDICT
from schemas import ManagerPlan
from schemas import SearchResults
from schemas import SearchStatus
from schemas import SubQuestion
from schemas import SubQuestionEvidence
from schemas import is_revision
from synthesizer_agent import build_data_message
from synthesizer_agent import build_instruction
from synthesizer_agent import build_synthesizer_agent
from synthesizer_agent import format_sub_question
from synthesizer_agent import read_revision_block


def test_3_4_2_the_report_answers_every_part_with_specific_facts():
    instruction = build_instruction()
    assert "Answer every part of each sub-question with the specific facts" in instruction
    assert "names, dates, numbers and places" in instruction
    assert "state each fact a passage gives that answers it, not only the first one" in instruction
    assert "Keep the report short" not in instruction


def test_3_4_2_evidence_under_one_sub_question_may_answer_another():
    # Q02 and Q05 on 2026-11-13: the answer was saved under the other sub-question
    instruction = build_instruction()
    assert "The evidence is one pool for the whole question" in instruction
    assert "A passage listed under one sub-question may answer another" in instruction
    empty = SubQuestionEvidence(sub_question_id=2, status=SearchStatus.EMPTY)
    sub_question = SubQuestion(sub_question_id=2, text="Does RAG prevent hallucinations?",
                               preferred_sources=["arxiv"], priority=1)
    block = format_sub_question(empty, sub_question, {})
    assert "If a passage under another sub-question answers it, answer it from that passage" in block
    assert "Do not answer it" not in block


def test_3_4_2_the_summary_names_the_parts_left_unanswered():
    instruction = build_instruction()
    assert "names each part they do not" in instruction
    assert "Never say that every part was answered when a part has no claim" in instruction



def test_3_4_2_a_part_answered_from_another_sub_questions_evidence_counts_as_answered():
    # Q05 on 2026-11-13: claim 4 answered a part that the summary called unanswered
    instruction = build_instruction()
    assert ("A part counts as answered when any claim answers it, wherever that claim's "
            "evidence was listed") in instruction


def test_the_synthesizer_thinks_at_medium():
    agent = build_synthesizer_agent(evidence_reader=lambda ids: [])
    assert agent.generate_content_config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM


def test_3_4_2_no_claim_may_be_about_the_evidence_itself():
    instruction = build_instruction()
    assert "Never write a claim about the evidence or the report itself" in instruction
    assert "say so in the summary only" in instruction


def test_3_7_4_the_only_state_slot_is_the_question():
    # ADK reads {name} as a state variable, so no other braces may appear
    instruction = build_instruction()
    assert instruction.count("{") == 1 and instruction.count("}") == 1
    assert "{user_question}" in instruction


# SRS 3.4.6: the Synthesizer knows a first draft from a revision by the feedback
# in state, and says which in the first line of what the model reads
def test_3_4_6_the_mode_is_initial_draft_without_feedback_and_revision_with_it():
    plan = ManagerPlan(user_question="What is attention?", sub_questions=[SubQuestion(
        sub_question_id=1, text="What is self-attention?", preferred_sources=["wikipedia"], priority=1)])
    results = SearchResults(results=[SubQuestionEvidence(sub_question_id=1, status=SearchStatus.EMPTY)])
    draft = {"title": "Attention", "summary": "A short answer.", "sections": [{"heading": "What it is",
             "claims": [{"claim_number": 1, "sentence": "Attention weighs tokens.", "evidence_ids": [1]}]}]}
    verdict = {"checks": [{"claim_number": 1, "verdict": "unsupported", "reason": "Evidence 1 does not say so."}],
               "result": "fail", "feedback": "Claim 1 is unsupported."}

    assert not is_revision({}) and not is_revision({STATE_FEEDBACK: ""})
    assert read_revision_block({STATE_FEEDBACK: ""}) is None
    first = build_data_message(plan, results, {}, None)
    assert first.startswith("MODE: initial_draft")

    state = {STATE_FEEDBACK: "Claim 1 is unsupported.", STATE_DRAFT: draft, STATE_VERDICT: verdict}
    assert is_revision(state)
    block = read_revision_block(state)
    second = build_data_message(plan, results, {}, block)
    assert second.startswith("MODE: revision")
    assert "Claim 1 is unsupported." in second and "Attention weighs tokens." in second
