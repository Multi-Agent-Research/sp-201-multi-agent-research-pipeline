# Command-line interface (SRS 3.8). Runs the pipeline on one research question
# and prints what it produced: the sub-questions, the checked report with an
# evidence ID after every claim, the sources behind those IDs, the flagged
# claims, the support score and the number of model calls. It searches,
# writes and checks nothing itself.
#
#     python -m sp201 "How do transformers handle long documents?"
#     python -m sp201 --verbose "..."      # also prints each agent step
#
# The report is only shown through build_final_report, which refuses a draft
# the Fact-Checker did not fully check (SRS 3.6.4).
import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv
from google.adk.apps.app import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.genai import errors as genai_errors
from google.genai import types

from evidence_store import EvidenceStore
from manager_agent import build_manager_agent
from manager_agent import get_model_id
from schemas import STATE_DRAFT
from schemas import STATE_PLAN
from schemas import STATE_USER_QUESTION
from schemas import STATE_VERDICT
from schemas import DraftReport
from schemas import FactCheckVerdict
from schemas import ManagerPlan
from schemas import build_final_report

APP_NAME = "sp201-cli"
USER_ID = "cli"


# Counts every request sent to the model, from every agent, including any
# made inside an AgentTool, so the cost of a question is measured
class ModelCallCounter(BasePlugin):
    def __init__(self):
        super().__init__(name="model_call_counter")
        self.calls = 0

    async def before_model_callback(self, *, callback_context, llm_request):
        self.calls += 1
        return None


# Builds the agent the CLI runs. The full pipeline is pipeline.build_pipeline(),
# but Search is not wired into the Manager yet, so a full run would stop at the
# Synthesizer with no evidence. Until Search lands this stays the Manager on its
# own; then return pipeline.build_pipeline() here instead.
def build_root_agent():
    return build_manager_agent()


# Prints one line per agent step: always the agent's name when it starts
# working, and with verbose also the tools it calls and what it says
def print_progress(event, last_author, verbose):
    if event.author != last_author and event.author != "user":
        print(f"... {event.author} is working")
    if not verbose or event.content is None:
        return
    for part in event.content.parts or []:
        if part.function_call:
            print(f"    {event.author} calls {part.function_call.name}")
        elif part.function_response:
            print(f"    {part.function_response.name} returned")
        elif part.text and part.text.strip():
            print(f"    {event.author}: {part.text.strip()[:200]}")


# Raised when an agent's step ends in an error event, which is how ADK reports
# a failed model call (for example a 429) instead of raising it
class PipelineStopped(Exception):
    def __init__(self, author, code, message):
        super().__init__(f"{author} stopped with {code}: {message}")
        self.code = code
        self.message = message


# Turns a model error into a plain message. Only the two errors a new user
# hits are explained: the free-tier limit and a model the key can't use.
def explain_model_error(code, message):
    text = f"{code} {message}"
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return ("The Gemini free tier refused the request (429: too many requests). "
                "Wait a minute and try again, or set SP201_MODEL_ID to a model with a higher limit.")
    if "404" in text or "NOT_FOUND" in text:
        return (f"The model {get_model_id()} is not available to this API key (404). "
                "Set SP201_MODEL_ID in .env to a model your key can use.")
    return None


# Runs one question through the agent and returns the final session state
async def run_question(question, agent, counter, verbose=False):
    runner = InMemoryRunner(app=App(name=APP_NAME, root_agent=agent, plugins=[counter]))
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, state={STATE_USER_QUESTION: question})
    message = types.Content(role="user", parts=[types.Part(text=question)])
    last_author = None
    stopped = None
    # Errors are noted and raised after the run ends; leaving the loop early
    # cuts ADK's event stream off mid-step
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=message):
        print_progress(event, last_author, verbose)
        last_author = event.author
        if event.error_code and stopped is None:
            stopped = PipelineStopped(event.author, event.error_code, event.error_message)
    if stopped is not None:
        raise stopped
    final = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session.id)
    return final.state


# Reads the source rows behind a list of evidence IDs from the evidence store
def read_sources(evidence_ids):
    return EvidenceStore().read_evidence(evidence_ids)


# Builds the text the CLI prints from the final session state
def render(state, calls, source_reader=read_sources):
    lines = []

    raw_plan = state.get(STATE_PLAN)
    if raw_plan is not None:
        plan = ManagerPlan.model_validate(raw_plan)
        lines.append("Sub-questions:")
        for sub_question in plan.sub_questions:
            sources = ", ".join(source.value for source in sub_question.preferred_sources)
            lines.append(f"  [{sub_question.sub_question_id}] {sub_question.text} ({sources})")
        lines.append("")

    raw_draft = state.get(STATE_DRAFT)
    raw_verdict = state.get(STATE_VERDICT)
    if raw_draft is None:
        lines.append("No report in this run: the pipeline has not drafted one.")
    elif raw_verdict is None:
        lines.append("No report shown: the Fact-Checker has not checked the draft (SRS 3.6.4).")
    else:
        try:
            verdict = FactCheckVerdict.model_validate(raw_verdict)
            # The loop only ends on a fail once its revisions run out (SRS 3.10.2)
            report = build_final_report(DraftReport.model_validate(raw_draft), verdict,
                                        incomplete=verdict.result == "fail")
        except ValueError as error:
            lines.append(f"No report shown: {error}")
        else:
            lines.extend(render_report(report, source_reader))

    lines.append(f"Model calls: {calls}")
    return "\n".join(lines)


# Builds the report section: title, summary, every claim with its evidence IDs,
# the sources, the flagged claims and the support score
def render_report(report, source_reader):
    lines = []
    draft = report.draft
    checks = {check.claim_number: check for check in report.verdict.checks}

    if report.incomplete:
        lines.append("WARNING: this report is incomplete; the pipeline ran out of retries (SRS 3.10.2).")
    lines.append(draft.title)
    lines.append("")
    lines.append(draft.summary)

    cited = []
    for section in draft.sections:
        lines.append("")
        lines.append(section.heading)
        for claim in section.claims:
            ids = ", ".join(str(evidence_id) for evidence_id in claim.evidence_ids)
            mark = " [FLAGGED]" if claim.claim_number in report.flagged_claim_numbers else ""
            lines.append(f"  ({claim.claim_number}) {claim.sentence} [evidence {ids}]{mark}")
            for evidence_id in claim.evidence_ids:
                if evidence_id not in cited:
                    cited.append(evidence_id)

    lines.append("")
    lines.append("Sources:")
    for row in source_reader(cited):
        lines.append(f"  [{row.evidence_id}] {row.source_title} ({row.source_type}) {row.source_url}")

    lines.append("")
    if report.flagged_claim_numbers:
        lines.append("Flagged claims:")
        for number in report.flagged_claim_numbers:
            check = checks[number]
            lines.append(f"  ({number}) {check.verdict.value}: {check.reason}")
            if check.restatement:
                lines.append(f"      supported version: {check.restatement}")
    else:
        lines.append("Flagged claims: none")

    if report.support_score is None:
        lines.append("Support score: none (no claim could be checked)")
    else:
        lines.append(f"Support score: {report.support_score}%")
    return lines


# Prints why the run stopped and how many model calls it used, then exits
def stop(message, counter):
    print(message)
    print(f"Model calls: {counter.calls}")
    sys.exit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m sp201",
                                     description="Run one research question through the SP-201 pipeline.")
    parser.add_argument("question", help="the research question, in quotes")
    parser.add_argument("--verbose", action="store_true", help="print each agent step")
    args = parser.parse_args(argv)

    load_dotenv()
    if not args.verbose:
        # ADK logs a full traceback when a model call fails; the CLI explains the error itself
        logging.getLogger("google_adk").setLevel(logging.CRITICAL)
    counter = ModelCallCounter()
    print(f"Question: {args.question}")
    print(f"Model: {get_model_id()}")
    try:
        state = asyncio.run(run_question(args.question, build_root_agent(), counter, args.verbose))
    except genai_errors.APIError as error:
        message = explain_model_error(error.code, str(error))
        if message is None:
            raise
        stop(message, counter)
    except PipelineStopped as error:
        stop(explain_model_error(error.code, error.message) or str(error), counter)
    print("")
    print(render(state, counter.calls))
