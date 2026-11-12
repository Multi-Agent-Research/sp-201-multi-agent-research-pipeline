# Pipeline assembly (SDD 3.6, README "Architecture"). Wires the agents into the
# structure the README draws:
#
#     SequentialAgent
#       Manager
#       LoopAgent (max_iterations = max_revisions() + 1)
#         Synthesizer
#         Fact-Checker
#         EscalationChecker
#
# The Synthesizer is first inside the loop, so every iteration ends on a check:
# a run that uses up its iterations still ends on a checked draft, with its
# failing claims flagged (SRS 3.6.4, 3.6.5).
#
# Search is attached to the Manager as a tool (search_agent.attach_search). The
# Manager calls it once per sub-question; Search saves the passages through the
# EvidenceStore and leaves SearchResults in state[STATE_SEARCH_RESULTS], and the
# loop reads the passages back from the same store. Every model call goes
# through rate_limit.RateLimitedGemini, so runs stay under the free tier's
# per-minute limit and back off on a 429.
#
# Searching is a rule, not a request. On live runs gemini-3.5-flash-lite often
# handed in its plan without calling Search (2026-11-12), so two locks hold the
# Manager to it: its first turn, and the turn after a refused plan, may only
# call Search; and a plan with a sub-question it has not searched is refused,
# at most MAX_PLAN_REFUSALS times, so a run can never loop.
from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.agents import LoopAgent
from google.adk.agents import SequentialAgent
from google.adk.events import Event
from google.adk.events import EventActions
from google.genai import types

from fact_checker_agent import build_fact_checker_agent
from fact_checker_agent import parse_verdict
from manager_agent import build_manager_agent
from manager_agent import decompose_question
from manager_agent import get_model_id
from rate_limit import use_rate_limited_models
from schemas import STATE_VERDICT
from schemas import loop_max_iterations
from schemas import max_revisions
from schemas import should_escalate
from search_agent import NOT_SEARCHED_PREFIX
from search_agent import SEARCH_TOOL_NAME
from search_agent import as_callback_list
from search_agent import attach_search
from synthesizer_agent import build_synthesizer_agent

PIPELINE_NAME = "sp201_pipeline"
LOOP_NAME = "revision_loop"
ESCALATION_CHECKER_NAME = "escalation_checker"
ESCALATION_CHECKER_DESCRIPTION = (
    "Ends the revision loop when the Fact-Checker passes the draft. "
    "Makes no model call and no judgment of its own."
)


# True when the Fact-Checker's verdict in state is a pass. No verdict means
# the check did not happen, so the loop must not end as if it had passed.
def verdict_passed(state):
    raw_verdict = state.get(STATE_VERDICT)
    if raw_verdict is None:
        return False
    return should_escalate(parse_verdict(raw_verdict))


# EscalationChecker (SDD 5.5). The last step of each loop iteration: reads the
# verdict the Fact-Checker just wrote and escalates on a pass, which ends the
# LoopAgent. On a fail it does nothing, so the loop runs the Synthesizer again
# with the Fact-Checker's feedback in state, until max_iterations runs out.
class EscalationChecker(BaseAgent):
    async def _run_async_impl(self, ctx):
        passed = verdict_passed(ctx.session.state)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            actions=EventActions(escalate=passed),
        )


def build_escalation_checker():
    return EscalationChecker(name=ESCALATION_CHECKER_NAME, description=ESCALATION_CHECKER_DESCRIPTION)


# Builds the revision loop: draft (or revise), check, then stop on a pass.
# evidence_reader is passed to both agents that read passages; None means
# each one uses EvidenceStore().read_evidence.
def build_revision_loop(evidence_reader=None):
    return LoopAgent(
        name=LOOP_NAME,
        max_iterations=loop_max_iterations(),
        sub_agents=[
            build_synthesizer_agent(evidence_reader=evidence_reader),
            build_fact_checker_agent(evidence_reader=evidence_reader),
            build_escalation_checker(),
        ],
    )


# True when the Manager already holds the search tool, so it is never attached twice
def has_search(manager):
    return any(getattr(tool, "name", None) == SEARCH_TOOL_NAME for tool in manager.tools or [])


# ----- the two locks that make the Manager search -----

# ADK's tool for handing in a structured answer when an agent also has tools
# (on an AI Studio key the Manager's plan always comes back through it)
HAND_IN_TOOL_NAME = "set_model_response"
MAX_PLAN_REFUSALS = 2
STATE_SEARCHED_ANY = "search_lock_searched_any"
STATE_SEARCH_NEXT = "search_lock_search_next"
STATE_PLAN_REFUSALS = "search_lock_plan_refusals"


# One state key per sub-question, because parallel Search calls writing to
# one shared list would overwrite each other
def searched_key(sub_question_id):
    return f"search_lock_searched_{sub_question_id}"


# Lock 1 (before_model_callback): until the Manager has searched, and on the
# turn after a refused plan, Gemini may only call Search
def search_first(callback_context, llm_request):
    state = callback_context.state
    if state.get(STATE_SEARCHED_ANY) and not state.get(STATE_SEARCH_NEXT):
        return None
    state[STATE_SEARCH_NEXT] = False
    if llm_request.config is None:
        llm_request.config = types.GenerateContentConfig()
    llm_request.config.tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY,
            allowed_function_names=[SEARCH_TOOL_NAME],
        )
    )
    return None


# after_tool_callback: notes each sub-question the Manager sent to Search. A call
# Search refused before searching (no buzz words, a bad id) does not count.
def note_search(tool, args, tool_context, tool_response):
    if tool.name != SEARCH_TOOL_NAME:
        return None
    refused = isinstance(tool_response, dict) and \
        str(tool_response.get("error", "")).startswith(NOT_SEARCHED_PREFIX)
    if refused:
        return None
    try:
        sub_question_id = int(args.get("sub_question_id"))
    except (TypeError, ValueError):
        return None
    tool_context.state[STATE_SEARCHED_ANY] = True
    tool_context.state[searched_key(sub_question_id)] = True
    return None


# Lock 2 (before_tool_callback): refuses a plan while any sub-question in it,
# after the cap, has not been searched. Gemini gets the refusal as the tool's
# answer and the next turn may only call Search. After MAX_PLAN_REFUSALS the
# plan goes through, and Search reports the skipped ones as not searched.
def plan_needs_searches(tool, args, tool_context):
    if tool.name != HAND_IN_TOOL_NAME:
        return None
    try:
        plan = decompose_question(args)
    except ValueError:
        return None  # a malformed plan is set_model_response's to report
    state = tool_context.state
    missing = [sub_question.sub_question_id for sub_question in plan.sub_questions
               if not state.get(searched_key(sub_question.sub_question_id))]
    refusals = state.get(STATE_PLAN_REFUSALS, 0)
    if not missing or refusals >= MAX_PLAN_REFUSALS:
        return None
    state[STATE_PLAN_REFUSALS] = refusals + 1
    state[STATE_SEARCH_NEXT] = True
    numbers = ", ".join(str(number) for number in missing)
    return {"error": (
        f"Plan not accepted: sub-question {numbers} has not been searched. Call "
        f"{SEARCH_TOOL_NAME} for each one, then call {HAND_IN_TOOL_NAME} again with the same plan."
    )}


# Puts both locks on a Manager that holds Search; adding them twice does nothing
def require_search(manager):
    before_model = as_callback_list(manager.before_model_callback)
    if search_first in before_model:
        return manager
    manager.before_model_callback = before_model + [search_first]
    manager.before_tool_callback = as_callback_list(manager.before_tool_callback) + [plan_needs_searches]
    manager.after_tool_callback = as_callback_list(manager.after_tool_callback) + [note_search]
    return manager


# Builds the whole pipeline. ADK lets an agent have only one parent, so every
# call builds new agents. manager lets a caller supply its own Manager.
#
# With search on (the default), Search is attached to the Manager unless it
# already holds it, and the loop reads passages from Search's store unless
# evidence_reader says otherwise. store, fetchers and article_fetchers go to
# Search; None means the default database and the real sources. search=False leaves Search out,
# for tests that put search results in state themselves. With Search in, the
# Manager gets both locks. Every agent's model is wrapped in RateLimitedGemini.
def build_pipeline(evidence_reader=None, manager=None, search=True, store=None, fetchers=None,
                   article_fetchers=None):
    if manager is None:
        manager = build_manager_agent()
    if search and not has_search(manager):
        search_tool = attach_search(manager, store=store, fetchers=fetchers,
                                    article_fetchers=article_fetchers)
        if evidence_reader is None:
            evidence_reader = search_tool.store.read_evidence
    if search:
        require_search(manager)
    root = SequentialAgent(
        name=PIPELINE_NAME,
        sub_agents=[manager, build_revision_loop(evidence_reader=evidence_reader)],
    )
    return use_rate_limited_models(root)


# Prints the structure, without calling Gemini
def describe(agent, depth=0):
    lines = ["  " * depth + f"{agent.name} ({type(agent).__name__})"]
    for sub_agent in agent.sub_agents:
        lines.extend(describe(sub_agent, depth + 1))
    return lines


def main():
    load_dotenv()
    pipeline = build_pipeline()
    print(f"Model: {get_model_id()}")
    print(f"Revisions allowed: {max_revisions()} (loop runs at most {loop_max_iterations()} times)")
    print("")
    print("\n".join(describe(pipeline)))


if __name__ == "__main__":
    main()
