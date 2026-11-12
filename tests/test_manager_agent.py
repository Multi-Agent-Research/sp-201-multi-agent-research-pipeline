# Run from the project root: python -m pytest tests
# Test names carry the requirement number, so `python -m pytest tests -k 3_1_6`
# runs every test for that requirement.

import copy

import pytest
from pydantic import ValidationError

from manager_agent import DEFAULT_MODEL_ID
from manager_agent import MANAGER_AGENT_NAME
from manager_agent import STATE_MODEL_CALLS
from manager_agent import build_instruction
from manager_agent import build_manager_agent
from manager_agent import count_model_call
from manager_agent import decompose_question
from manager_agent import enforce_sub_question_cap
from manager_agent import get_model_id
from manager_agent import read_user_question
from manager_agent import record_plan
from schemas import ABSOLUTE_MAX_SUB_QUESTIONS
from schemas import DEFAULT_MAX_SUB_QUESTIONS
from schemas import ManagerPlan
from schemas import STATE_PLAN
from schemas import STATE_USER_QUESTION
from schemas import max_sub_questions

QUESTION = "How do transformers handle long documents?"


# Stands in for the ADK CallbackContext; the callbacks only touch .state
class FakeCallbackContext:
    def __init__(self, state):
        self.state = state


def make_sub_question(sub_question_id, priority, sources=None):
    if sources is None:
        sources = ["wikipedia"]
    return {
        "sub_question_id": sub_question_id,
        "text": f"Sub-question number {sub_question_id} about long documents?",
        "preferred_sources": sources,
        "priority": priority,
    }


def make_plan(id_priority_pairs):
    sub_questions = []
    for sub_question_id, priority in id_priority_pairs:
        sub_questions.append(make_sub_question(sub_question_id, priority))
    return {"user_question": QUESTION, "sub_questions": sub_questions}


def ids_of(plan):
    ids = []
    for sub_question in plan.sub_questions:
        ids.append(sub_question.sub_question_id)
    return ids


@pytest.fixture
def default_cap(monkeypatch):
    monkeypatch.delenv("SP201_MAX_SUB_QUESTIONS", raising=False)


# Priority truncation (3.1.6)

def test_3_1_6_four_sub_questions_truncate_to_three(default_cap):
    plan = decompose_question(make_plan([(1, 2), (2, 1), (3, 4), (4, 3)]))
    assert len(plan.sub_questions) == 3

def test_3_1_6_keeps_lowest_priority_numbers(default_cap):
    # Priority 4 (id 3) is the one that should be dropped
    plan = decompose_question(make_plan([(1, 2), (2, 1), (3, 4), (4, 3)]))
    assert ids_of(plan) == [1, 2, 4]

def test_3_1_6_survivors_keep_id_order(default_cap):
    # Kept by priority, but listed back in id order
    plan = decompose_question(make_plan([(1, 3), (2, 2), (3, 1), (4, 5), (5, 4)]))
    assert ids_of(plan) == [1, 2, 3]

def test_3_1_6_priority_tie_breaks_on_lower_id(default_cap):
    plan = decompose_question(make_plan([(1, 1), (2, 1), (3, 1), (4, 1)]))
    assert ids_of(plan) == [1, 2, 3]

def test_3_1_6_plan_within_cap_is_unchanged(default_cap):
    plan = decompose_question(make_plan([(1, 1), (2, 2)]))
    assert ids_of(plan) == [1, 2]

def test_3_1_6_explicit_limit_overrides_environment(monkeypatch):
    monkeypatch.setenv("SP201_MAX_SUB_QUESTIONS", "5")
    plan = decompose_question(make_plan([(1, 1), (2, 2), (3, 3), (4, 4)]), limit=2)
    assert ids_of(plan) == [1, 2]

def test_3_1_6_explicit_limit_is_clamped_to_ceiling():
    plan = decompose_question(make_plan([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]), limit=10)
    assert len(plan.sub_questions) == ABSOLUTE_MAX_SUB_QUESTIONS

def test_3_1_6_explicit_limit_below_one_keeps_one():
    plan = decompose_question(make_plan([(1, 2), (2, 1)]), limit=0)
    assert ids_of(plan) == [2]

def test_3_1_6_user_question_survives_truncation(default_cap):
    plan = decompose_question(make_plan([(1, 1), (2, 2), (3, 3), (4, 4)]))
    assert plan.user_question == QUESTION


# Configuration (3.11.1, 3.11.2)

def test_3_11_2_default_cap_is_three(default_cap):
    assert max_sub_questions() == DEFAULT_MAX_SUB_QUESTIONS
    assert max_sub_questions() == 3

def test_3_11_2_cap_of_five_is_respected(monkeypatch):
    monkeypatch.setenv("SP201_MAX_SUB_QUESTIONS", "5")
    plan = decompose_question(make_plan([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]))
    assert len(plan.sub_questions) == 5

def test_3_11_2_cap_above_five_clamps_to_five(monkeypatch):
    monkeypatch.setenv("SP201_MAX_SUB_QUESTIONS", "10")
    assert max_sub_questions() == 5

def test_3_11_2_cap_below_one_clamps_to_one(monkeypatch):
    monkeypatch.setenv("SP201_MAX_SUB_QUESTIONS", "0")
    assert max_sub_questions() == 1
    monkeypatch.setenv("SP201_MAX_SUB_QUESTIONS", "-4")
    assert max_sub_questions() == 1

def test_3_11_2_cap_of_one_keeps_only_top_priority(monkeypatch):
    monkeypatch.setenv("SP201_MAX_SUB_QUESTIONS", "1")
    plan = decompose_question(make_plan([(1, 3), (2, 1), (3, 2)]))
    assert ids_of(plan) == [2]

def test_3_11_1_model_id_defaults_to_gemini_3_5_flash_lite(monkeypatch):
    monkeypatch.delenv("SP201_MODEL_ID", raising=False)
    assert get_model_id() == DEFAULT_MODEL_ID
    assert get_model_id() == "gemini-3.5-flash-lite"

def test_3_11_1_model_id_comes_from_environment(monkeypatch):
    monkeypatch.setenv("SP201_MODEL_ID", "gemini-9.9-test")
    assert get_model_id() == "gemini-9.9-test"
    assert build_manager_agent().model == "gemini-9.9-test"


# Schema enforcement (3.7.2, 3.1.2, 3.1.3)

def test_3_1_2_single_sub_question_is_valid(default_cap):
    plan = decompose_question(make_plan([(1, 1)]))
    assert ids_of(plan) == [1]

def test_3_7_2_rejects_empty_sub_question_list():
    with pytest.raises(ValidationError):
        decompose_question({"user_question": QUESTION, "sub_questions": []})

def test_3_7_2_rejects_duplicate_sub_question_ids():
    with pytest.raises(ValidationError):
        decompose_question(make_plan([(1, 1), (1, 2)]))

def test_3_1_3_rejects_empty_preferred_sources():
    raw_plan = {"user_question": QUESTION, "sub_questions": [make_sub_question(1, 1, sources=[])]}
    with pytest.raises(ValidationError):
        decompose_question(raw_plan)

def test_3_1_3_rejects_unknown_source():
    raw_plan = {"user_question": QUESTION, "sub_questions": [make_sub_question(1, 1, sources=["reddit"])]}
    with pytest.raises(ValidationError):
        decompose_question(raw_plan)

def test_3_1_3_accepts_every_known_source(default_cap):
    raw_plan = {
        "user_question": QUESTION,
        "sub_questions": [make_sub_question(1, 1, sources=["wikipedia", "arxiv", "web"])],
    }
    plan = decompose_question(raw_plan)
    assert len(plan.sub_questions[0].preferred_sources) == 3

def test_3_7_2_rejects_unknown_fields():
    raw_plan = make_plan([(1, 1)])
    raw_plan["notes"] = "extra field"
    with pytest.raises(ValidationError):
        decompose_question(raw_plan)

def test_3_7_2_rejects_malformed_json():
    with pytest.raises(ValidationError):
        decompose_question("{not json")

def test_3_7_2_accepts_json_string(default_cap):
    plan = decompose_question(ManagerPlan.model_validate(make_plan([(1, 1), (2, 2)])).model_dump_json())
    assert ids_of(plan) == [1, 2]

def test_3_7_2_accepts_manager_plan_instance(default_cap):
    plan = ManagerPlan.model_validate(make_plan([(1, 1), (2, 2), (3, 3), (4, 4)]))
    assert len(decompose_question(plan).sub_questions) == 3


# Session state (3.1.1, 3.1.4, 3.7.2)

def test_3_1_1_reads_user_question_from_state():
    state = {STATE_USER_QUESTION: QUESTION}
    assert read_user_question(state) == QUESTION

def test_3_1_4_record_plan_writes_state_plan(default_cap):
    state = {}
    plan = decompose_question(make_plan([(1, 1)]))
    record_plan(state, plan)
    assert STATE_PLAN in state
    assert ManagerPlan.model_validate(state[STATE_PLAN]) == plan

def test_3_1_4_recorded_plan_is_plain_json_data(default_cap):
    # Session state has to serialize, so enums must be stored as strings
    state = {}
    record_plan(state, decompose_question(make_plan([(1, 1)])))
    assert state[STATE_PLAN]["sub_questions"][0]["preferred_sources"] == ["wikipedia"]

def test_3_1_6_after_agent_callback_applies_cap_to_state(default_cap):
    state = {STATE_PLAN: make_plan([(1, 2), (2, 1), (3, 4), (4, 3)])}
    enforce_sub_question_cap(FakeCallbackContext(state))
    plan = ManagerPlan.model_validate(state[STATE_PLAN])
    assert ids_of(plan) == [1, 2, 4]

def test_3_7_2_after_agent_callback_fails_closed_without_mutating_state():
    bad_plan = make_plan([(1, 1), (1, 2)])
    state = {STATE_PLAN: copy.deepcopy(bad_plan)}
    with pytest.raises(ValidationError):
        enforce_sub_question_cap(FakeCallbackContext(state))
    assert state[STATE_PLAN] == bad_plan

def test_3_7_2_after_agent_callback_skips_when_no_plan():
    state = {}
    enforce_sub_question_cap(FakeCallbackContext(state))
    assert state == {}


# Model-call counter (3.9.5)

def test_3_9_5_count_model_call_starts_at_one():
    state = {}
    assert count_model_call(FakeCallbackContext(state), None) is None
    assert state[STATE_MODEL_CALLS] == 1

def test_3_9_5_count_model_call_accumulates():
    state = {STATE_MODEL_CALLS: 4}
    count_model_call(FakeCallbackContext(state), None)
    count_model_call(FakeCallbackContext(state), None)
    assert state[STATE_MODEL_CALLS] == 6


# Agent definition and constraints (3.1.5, 3.7.2)

def test_3_7_2_agent_output_schema_is_manager_plan(default_cap):
    agent = build_manager_agent()
    assert agent.output_schema is ManagerPlan
    assert agent.output_key == STATE_PLAN

def test_3_1_5_agent_has_no_retrieval_or_file_tools(default_cap):
    # Search is attached by pipeline.build_pipeline, never here
    agent = build_manager_agent()
    assert list(agent.tools) == []

def test_agent_has_no_sub_agents(default_cap):
    agent = build_manager_agent()
    assert list(agent.sub_agents) == []

def test_3_9_5_agent_wires_the_call_counter(default_cap):
    agent = build_manager_agent()
    assert agent.after_model_callback is count_model_call

def test_3_1_6_agent_wires_the_cap_callback(default_cap):
    agent = build_manager_agent()
    assert agent.after_agent_callback is enforce_sub_question_cap

def test_3_1_5_agent_name_and_description(default_cap):
    agent = build_manager_agent()
    assert agent.name == MANAGER_AGENT_NAME
    assert "Does not retrieve" in agent.description


# Instruction text (3.1.2, 3.1.3, 3.1.6)

def test_3_1_6_instruction_states_the_cap():
    assert "between 1 and 3 sub-questions" in build_instruction(3)
    assert "between 1 and 5 sub-questions" in build_instruction(5)

def test_3_1_6_agent_instruction_follows_environment_cap(monkeypatch):
    monkeypatch.setenv("SP201_MAX_SUB_QUESTIONS", "2")
    assert "between 1 and 2 sub-questions" in build_manager_agent().instruction

def test_3_1_3_instruction_names_every_source():
    instruction = build_instruction(3)
    assert "wikipedia, arxiv, web" in instruction

def test_3_1_1_instruction_reads_question_from_state_key():
    assert "{" + STATE_USER_QUESTION + "}" in build_instruction(3)

def test_3_1_5_instruction_forbids_retrieval_drafting_and_judging():
    # The Manager sends sub-questions to Search but never reads what it finds
    instruction = build_instruction(3)
    assert "never read what Search finds, write prose, or judge" in instruction
