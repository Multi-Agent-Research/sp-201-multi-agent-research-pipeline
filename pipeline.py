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
# Search is not wired in yet. Per the README the Manager will hold
# AgentTool(Search), and Search leaves SearchResults in state[STATE_SEARCH_RESULTS]
# with the passages saved through EvidenceStore.add_evidence. Until then nothing
# writes search results, so a real run stops in the Synthesizer with "Search has
# not run". The tests put search results and evidence in place themselves.
from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.agents import LoopAgent
from google.adk.agents import SequentialAgent
from google.adk.events import Event
from google.adk.events import EventActions

from fact_checker_agent import build_fact_checker_agent
from fact_checker_agent import parse_verdict
from manager_agent import build_manager_agent
from manager_agent import get_model_id
from schemas import STATE_VERDICT
from schemas import loop_max_iterations
from schemas import max_revisions
from schemas import should_escalate
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


# Builds the whole pipeline. ADK lets an agent have only one parent, so every
# call builds new agents. manager lets a caller supply its own Manager, which
# is where AgentTool(Search) goes once Search is an agent.
def build_pipeline(evidence_reader=None, manager=None):
    if manager is None:
        manager = build_manager_agent()
    return SequentialAgent(
        name=PIPELINE_NAME,
        sub_agents=[manager, build_revision_loop(evidence_reader=evidence_reader)],
    )


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
