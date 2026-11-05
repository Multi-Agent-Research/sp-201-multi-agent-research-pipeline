# Manager agent. Splits one research question into sub-questions,
# capped at max_sub_questions(), which reads SP201_MAX_SUB_QUESTIONS and
# clamps it to [1, ABSOLUTE_MAX_SUB_QUESTIONS]. No tools are attached yet;
# AgentTool(search_agent) is added in Step 2, once the output_schema vs
# AgentTool constraint on gemini-2.5-flash is settled in its own ADR.
# Step 2 must read the plan from state[STATE_PLAN], not from the Manager's
# last message. The cap is applied in enforce_sub_question_cap, which only
# rewrites state; the model's original uncapped reply stays in history.
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent

from schemas import ABSOLUTE_MAX_SUB_QUESTIONS
from schemas import ManagerPlan
from schemas import STATE_PLAN
from schemas import STATE_USER_QUESTION
from schemas import SourceType
from schemas import max_sub_questions

# Model ID comes from the environment, never from a call site
MODEL_ID_ENV = "SP201_MODEL_ID"
DEFAULT_MODEL_ID = "gemini-2.5-flash"

# Session-state key for the running model-call count
STATE_MODEL_CALLS = "model_call_count"

MANAGER_AGENT_NAME = "manager"
MANAGER_DESCRIPTION = (
    "Splits one research question into focused sub-questions and picks "
    "retrieval sources for each. Does not retrieve, draft, or verify."
)

# Reads the model ID from the environment
def get_model_id():
    return os.environ.get(MODEL_ID_ENV, DEFAULT_MODEL_ID)

# Lists the source names the model is allowed to pick from
def source_names():
    names = []
    for source in SourceType:
        names.append(source.value)
    return names

# Lists the source names chosen for one sub-question
def source_names_of(sub_question):
    names = []
    for source in sub_question.preferred_sources:
        names.append(source.value)
    return names

# Builds the system instruction for the given sub-question cap
def build_instruction(limit):
    sources = ", ".join(source_names())
    question_slot = "{" + STATE_USER_QUESTION + "}"
    return (
        "You are the Manager of a research pipeline. You plan; you do not "
        "search, write prose, or judge whether anything is true.\n"
        "\n"
        f"Research question: {question_slot}\n"
        "\n"
        "Split the question into focused sub-questions that together cover "
        "what a short, cited report would need to answer.\n"
        "Rules:\n"
        f"1. Produce between 1 and {limit} sub-questions. Fewer, broader "
        "sub-questions are better than many narrow ones; each extra "
        "sub-question costs a search run and narrow ones return overlapping "
        "sources.\n"
        "2. Number sub_question_id from 1 upward with no gaps or repeats.\n"
        "3. Give each sub-question a priority, where 1 is the most important. "
        "If the list must be cut, the lowest priority numbers are kept.\n"
        f"4. Pick preferred_sources for each sub-question from: {sources}. "
        "Use wikipedia for background and definitions, arxiv for scientific "
        "or technical research, and web only as a fallback when neither "
        "is likely to have it.\n"
        "5. Copy the research question into user_question unchanged.\n"
        "6. Return only the JSON object. No commentary."
    )

# Reads the question submitted by the user from session state
def read_user_question(state):
    return state[STATE_USER_QUESTION]

# Validates raw model output as a ManagerPlan and cuts it down to the
# configured cap by priority instead of rejecting it
def decompose_question(raw_plan, limit=None):
    if isinstance(raw_plan, ManagerPlan):
        plan = raw_plan
    elif isinstance(raw_plan, str):
        plan = ManagerPlan.model_validate_json(raw_plan)
    else:
        plan = ManagerPlan.model_validate(raw_plan)

    if limit is None:
        limit = max_sub_questions()
    limit = max(1, min(limit, ABSOLUTE_MAX_SUB_QUESTIONS))

    return plan.truncated(limit)

# Writes the plan into session state under STATE_PLAN
def record_plan(state, plan):
    state[STATE_PLAN] = plan.model_dump(mode="json")
    return plan

# ADK after_model_callback: counts every model response so calls per query can be
# measured instead of estimated. Returns None so the response is kept.
def count_model_call(callback_context, llm_response):
    calls = callback_context.state.get(STATE_MODEL_CALLS, 0)
    callback_context.state[STATE_MODEL_CALLS] = calls + 1
    return None

# ADK after_agent_callback: re-validates what ADK wrote under STATE_PLAN and applies
# the cap. A malformed plan raises here, so nothing invalid is written back.
def enforce_sub_question_cap(callback_context):
    raw_plan = callback_context.state.get(STATE_PLAN)
    if raw_plan is None:
        return None
    plan = decompose_question(raw_plan)
    record_plan(callback_context.state, plan)
    return None

# Builds the Manager LlmAgent. No tools yet; see the note at the top of the file.
def build_manager_agent():
    limit = max_sub_questions()
    return LlmAgent(
        name=MANAGER_AGENT_NAME,
        model=get_model_id(),
        description=MANAGER_DESCRIPTION,
        instruction=build_instruction(limit),
        output_schema=ManagerPlan,
        output_key=STATE_PLAN,
        after_model_callback=count_model_call,
        after_agent_callback=enforce_sub_question_cap,
    )

def main():
    # Read .env into the environment so the settings below come from the file,
    # not just the shell. Existing variables are not overridden.
    load_dotenv()

    # Show the configuration the agent will run with
    agent = build_manager_agent()
    print(f"Agent: {agent.name}")
    print(f"Model: {agent.model}")
    print(f"Sub-question cap: {max_sub_questions()}")
    print(f"Tools attached: {len(agent.tools)}")

    # Show the cap being applied to an over-decomposed plan
    sample_plan = {
        "user_question": "How do transformers handle long documents?",
        "sub_questions": [
            {"sub_question_id": 1, "text": "What is the attention mechanism in a transformer?",
             "preferred_sources": ["wikipedia"], "priority": 2},
            {"sub_question_id": 2, "text": "What are sparse attention methods for long inputs?",
             "preferred_sources": ["arxiv"], "priority": 1},
            {"sub_question_id": 3, "text": "What is the context window of current models?",
             "preferred_sources": ["web"], "priority": 4},
            {"sub_question_id": 4, "text": "How does retrieval augmentation extend context?",
             "preferred_sources": ["arxiv", "wikipedia"], "priority": 3},
        ],
    }
    plan = decompose_question(sample_plan)
    print(f"\nSample plan: {len(sample_plan['sub_questions'])} sub-questions in, "
          f"{len(plan.sub_questions)} out")
    for sub_question in plan.sub_questions:
        sources = ", ".join(source_names_of(sub_question))
        print(f"\t[{sub_question.sub_question_id}] priority {sub_question.priority}: "
              f"{sub_question.text} ({sources})")

if __name__ == "__main__":
    main()