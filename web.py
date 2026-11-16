# Web interface (SRS 3.8, 4.3, 5.1, 5.4). A small local web page for the same
# pipeline the CLI runs. The user types one question, watches the pipeline's
# progress, and reads the checked report with a source under every claim, the
# flagged claims, the support score and the number of model calls. Like the CLI
# it searches, writes and checks nothing itself.
#
#     python web.py                 # then open http://127.0.0.1:8000
#     python web.py --port 9000
#
# What it reuses from sp201/cli.py: build_root_agent, ModelCallCounter and
# explain_model_error, so the web page and the CLI run and explain failures the
# same way. cli.py is not modified. The event loop in execute() repeats the one in
# cli.run_question because run_question prints its progress instead of reporting
# it; if run_question gains a progress hook, execute() can be deleted.
#
# The report is only shown through build_final_report, which refuses a draft the
# Fact-Checker did not fully check (SRS 3.6.4).
#
# Safety. The page sits between the user and text that came from the open web, and
# it can start runs that use a limited free-tier quota, so:
#   * It binds to 127.0.0.1 only. The host is fixed and cannot be changed; only
#     the port can be chosen (--port) (SRS 5.4). There is no login, as the SRS
#     says.
#   * Every piece of text from a run is HTML-escaped before it is shown. Sentences,
#     titles and reasons derive from retrieved pages, so none of it is trusted.
#   * A source URL becomes a link only if it is http or https. Links carry
#     rel="noopener noreferrer".
#   * The page uses no JavaScript. Progress updates by a timed page refresh, so the
#     Content-Security-Policy allows no script at all.
#   * A web page open in the same browser must not be able to start a run. The
#     Host header is checked (DNS rebinding) and a cross-site POST is refused.
#   * Only one run at a time, so a stray click cannot burn the daily quota.
import argparse
import asyncio
import html
import logging
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs
from urllib.parse import urlparse

from dotenv import load_dotenv

from schemas import STATE_DRAFT
from schemas import STATE_PLAN
from schemas import STATE_USER_QUESTION
from schemas import STATE_VERDICT
from schemas import DraftReport
from schemas import FactCheckVerdict
from schemas import ManagerPlan
from schemas import build_final_report

logger = logging.getLogger("sp201.web")

APP_NAME = "sp201-web"
# The name the page shows (the project is SP-201; the tool is Sift)
APP_TITLE = "Sift"
USER_ID = "web"

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MIN_QUESTION_CHARS = 5
MAX_QUESTION_CHARS = 1000
MAX_BODY_BYTES = 8192
MAX_JOBS_KEPT = 20
REFRESH_SECONDS = 3

JOB_ID_PATTERN = re.compile(r"^/job/([0-9a-f]{16})$")

# The agents whose events move the waiting page's stage tracker
AGENT_AUTHORS = ("manager", "synthesizer", "fact_checker", "escalation_checker")
# The stages the waiting page shows, in order
STAGES = (("plan", "Plan"), ("search", "Search"), ("write", "Write"), ("check", "Check"),
          ("report", "Report"))
# How far the active stage has come, as a share of the progress bar
STAGE_PROGRESS = {"plan": 10, "search": 30, "write": 55, "check": 80, "report": 100}
# One cycle of the progress bar's moving stripes. Each refresh starts the
# animation at the phase the clock says, so it carries on through the reloads.
STRIPE_SECONDS = 1.2

# Headers sent with every page. default-src 'none' means no script, image, font
# or frame can load; only the page's own inline style applies.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    # same-origin, not no-referrer: under no-referrer a browser sends "Origin: null"
    # on the page's own form post, and origin_ok refuses it. Outside links still
    # carry rel="noreferrer", so nothing leaks to other sites.
    "Referrer-Policy": "same-origin",
    "Cache-Control": "no-store",
}


# ---------------------------------------------------------------------------
# Jobs: one pipeline run, tracked while it works
# ---------------------------------------------------------------------------

# One run of the pipeline. The worker thread writes to it and the request
# threads read it, so every access goes through the lock.
class Job:
    def __init__(self, question):
        self.job_id = secrets.token_hex(8)
        self.question = question
        self.status = "running"
        self.steps = []
        self.calls = 0
        # Progress for the waiting page, read from the pipeline's events
        self.author = None
        self.sub_questions = []
        self.searches_called = 0
        self.searches_done = 0
        self.write_rounds = 0
        self.notice = ""
        self.report_html = ""
        self.message = ""
        self.started = time.time()
        self.finished = None
        self.lock = threading.Lock()

    # Records that an agent has started working, once per consecutive run of events
    def add_step(self, step):
        with self.lock:
            if not self.steps or self.steps[-1] != step:
                self.steps.append(step)

    def set_calls(self, calls):
        with self.lock:
            self.calls = calls

    # Records one pipeline event: which agent is working, searches sent and
    # returned, and the Manager's sub-questions once it has planned. A new event
    # means any retry the notice described is over, so the notice is cleared.
    def record_event(self, author, search_calls=0, search_results=0, sub_questions=None):
        with self.lock:
            if author == "synthesizer" and self.author != "synthesizer":
                self.write_rounds += 1
            if author in AGENT_AUTHORS:
                self.author = author
            self.searches_called += search_calls
            self.searches_done += search_results
            if sub_questions:
                self.sub_questions = list(sub_questions)
            self.notice = ""

    # A retry or wait the page should explain while the run looks still
    def set_notice(self, notice):
        with self.lock:
            self.notice = notice

    def finish(self, report_html):
        with self.lock:
            self.report_html = report_html
            self.status = "done"
            self.finished = time.time()

    def fail(self, message):
        with self.lock:
            self.message = message
            self.status = "error"
            self.finished = time.time()

    # A consistent copy of the fields a page needs
    def snapshot(self):
        with self.lock:
            end = self.finished if self.finished is not None else time.time()
            return {
                "job_id": self.job_id, "question": self.question, "status": self.status,
                "steps": list(self.steps), "calls": self.calls, "report_html": self.report_html,
                "message": self.message, "elapsed": int(end - self.started),
                "author": self.author, "sub_questions": list(self.sub_questions),
                "searches_called": self.searches_called, "searches_done": self.searches_done,
                "write_rounds": self.write_rounds, "notice": self.notice,
            }


# Holds the recent jobs and allows only one run at a time
class JobStore:
    def __init__(self):
        self.jobs = {}
        self.order = []
        self.lock = threading.Lock()

    # Starts a job, or returns None if one is already running
    def start(self, question):
        with self.lock:
            for job in self.jobs.values():
                if job.status == "running":
                    return None
            job = Job(question)
            self.jobs[job.job_id] = job
            self.order.append(job.job_id)
            while len(self.order) > MAX_JOBS_KEPT:
                del self.jobs[self.order.pop(0)]
            return job

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    # The job that is running now, if any
    def running(self):
        with self.lock:
            for job in self.jobs.values():
                if job.status == "running":
                    return job
        return None


# ---------------------------------------------------------------------------
# Running the pipeline (reuses cli.py)
# ---------------------------------------------------------------------------

# Raised when an agent's step ends in an error event, which is how ADK reports a
# failed model call (for example a 429) instead of raising it
class RunStopped(Exception):
    def __init__(self, author, code, message):
        super().__init__(f"{author} stopped with {code}: {message}")
        self.author = author
        self.code = code
        self.message = message


# Runs one question through the agent, reporting each agent step and the running
# model-call count to the job, and returns the final session state
async def execute(question, job, agent, counter):
    from google.adk.apps.app import App
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from search_agent import SEARCH_TOOL_NAME

    runner = InMemoryRunner(app=App(name=APP_NAME, root_agent=agent, plugins=[counter]))
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, state={STATE_USER_QUESTION: question})
    message = types.Content(role="user", parts=[types.Part(text=question)])
    last_author = None
    stopped = None
    # Errors are noted and raised after the run ends; leaving the loop early cuts
    # ADK's event stream off mid-step (the CLI does the same)
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=message):
        if event.author != last_author and event.author != "user":
            job.add_step(f"{event.author} is working")
        last_author = event.author
        job.record_event(
            event.author,
            search_calls=sum(1 for c in event.get_function_calls() if c.name == SEARCH_TOOL_NAME),
            search_results=sum(1 for r in event.get_function_responses() if r.name == SEARCH_TOOL_NAME),
            sub_questions=plan_texts(event.actions.state_delta if event.actions else None),
        )
        job.set_calls(counter.calls)
        if event.error_code and stopped is None:
            stopped = RunStopped(event.author, event.error_code, event.error_message)
    job.set_calls(counter.calls)
    if stopped is not None:
        raise stopped
    final = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session.id)
    return final.state


# The sub-questions in an event's state change, once the Manager has planned
def plan_texts(state_delta):
    if not state_delta or STATE_PLAN not in state_delta:
        return []
    raw = state_delta[STATE_PLAN]
    try:
        if isinstance(raw, str):
            plan = ManagerPlan.model_validate_json(raw)
        else:
            plan = ManagerPlan.model_validate(raw)
    except ValueError:
        return []
    return [sub_question.text for sub_question in plan.sub_questions]


RETRY_PATTERN = re.compile(r"retry (\d+) of (\d+) in (\d+)")


# Turns a retry warning from rate_limit into a plain line for the waiting page.
# Only fixed wording and numbers reach the page, never the error text itself.
def retry_notice(message):
    match = RETRY_PATTERN.search(message)
    detail = f" (retry {match.group(1)} of {match.group(2)}, in {match.group(3)} s)" if match else ""
    if message.startswith("Gemini is overloaded"):
        return ("Google's Gemini servers are busy right now. "
                f"The pipeline waits and retries on its own{detail}.")
    if message.startswith("Gemini"):
        return f"Gemini asked us to slow down. The pipeline waits and retries on its own{detail}."
    return "A source (Wikipedia, arXiv or the web) is slow to answer. Search retries on its own."


# Passes rate_limit's retry warnings to the job while it runs
class RetryNoticeHandler(logging.Handler):
    def __init__(self, job):
        super().__init__(level=logging.WARNING)
        self.job = job

    def emit(self, record):
        try:
            self.job.set_notice(retry_notice(record.getMessage()))
        except Exception:
            self.handleError(record)


# The default runner: builds the same pipeline the CLI runs and returns the final
# state. cli is imported here so that web.py loads without the pipeline. Only one
# run goes at a time, so the retry handler is this run's alone.
def pipeline_runner(question, job):
    from sp201 import cli
    handler = RetryNoticeHandler(job)
    rate_logger = logging.getLogger("rate_limit")
    rate_logger.addHandler(handler)
    try:
        return asyncio.run(execute(question, job, cli.build_root_agent(), cli.ModelCallCounter()))
    finally:
        rate_logger.removeHandler(handler)


# Turns a failed run into a message for the page, using the same explanations as
# the CLI. Nothing from an unexpected error is shown beyond its type: its text
# could hold file paths or retrieved content.
def explain_failure(error):
    from sp201 import cli
    from google.genai import errors as genai_errors
    from rate_limit import DailyQuotaExceeded
    from synthesizer_agent import EvidenceError

    if isinstance(error, genai_errors.APIError):
        message = cli.explain_model_error(error.code, str(error))
        if message is not None:
            return message
    elif isinstance(error, (RunStopped, cli.PipelineStopped)):
        message = cli.explain_model_error(error.code, error.message)
        return message or str(error)
    elif isinstance(error, EvidenceError):
        return f"The run stopped before writing a report: {error}"
    elif isinstance(error, DailyQuotaExceeded):
        return str(error)
    logger.error("Unexpected error in a run", exc_info=error)
    return f"Unexpected error ({type(error).__name__}). The details are in the terminal running web.py."


# Reads the source rows behind a list of evidence IDs from the evidence store
def read_sources(evidence_ids):
    from evidence_store import EvidenceStore
    return EvidenceStore().read_evidence(evidence_ids)


# Everything a running server needs, kept on the server object
class WebApp:
    def __init__(self, runner=None, source_reader=None, explain=None):
        self.store = JobStore()
        self.runner = runner or pipeline_runner
        self.source_reader = source_reader or read_sources
        self.explain = explain or explain_failure


# Worker thread body: runs the job and records either the rendered report or the
# plain-language reason it stopped
def work(app, job):
    try:
        state = app.runner(job.question, job)
        snapshot = job.snapshot()
        job.finish(render_report_html(state, snapshot["calls"], app.source_reader, snapshot["elapsed"]))
    except Exception as error:
        try:
            message = app.explain(error)
        except Exception:
            logger.exception("Could not explain a failed run")
            message = f"Unexpected error ({type(error).__name__})."
        job.fail(message)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

STYLE = """
:root { color-scheme: light; --gold: #fdbb30; --ink: #1a1a1a; --muted: #5f6368; --line: #d9d9d9;
  --ok: #1a7f37; --flag: #b45309; --bad: #b42318; --link: #1a4f8b; }
body { margin: 0; background: #fff; color: var(--ink);
  font: 16px/1.55 "Helvetica Neue", Helvetica, Arial, sans-serif; }
a { color: var(--link); }
.top { background: #111; color: #fff; border-bottom: 4px solid var(--gold); }
.top div { max-width: 46rem; margin: 0 auto; padding: .75rem 1rem; display: flex;
  justify-content: space-between;
  align-items: baseline; gap: 1rem; flex-wrap: wrap; }
.top a { color: #fff; text-decoration: none; font-weight: 700; font-size: 1.3rem; letter-spacing: .02em; }
.top span { color: var(--gold); font-size: .85rem; }
main { max-width: 46rem; margin: 1.5rem auto 3rem; padding: 0 1rem; }
h1 { font-size: 1.45rem; margin: 0 0 .4rem; }
h2 { font-size: 1.1rem; margin-top: 1.75rem; padding-bottom: .2rem; border-bottom: 2px solid var(--gold); }
h3 { font-size: 1rem; margin-bottom: .2rem; }
label { font-weight: 600; }
textarea { width: 100%; box-sizing: border-box; font: inherit; padding: .5rem; border: 1px solid #b5b5b5;
  border-radius: 3px; margin-top: .3rem; }
button { font: inherit; font-weight: 600; padding: .45rem 1.2rem; border: 1px solid #111; border-radius: 3px;
  background: #111; color: #fff; cursor: pointer; }
button:hover { background: #333; }
.lead { margin-top: 0; }
.muted { color: var(--muted); font-size: .9rem; }
.question { border-left: 3px solid #111; padding-left: .8rem; }
.notice { border: 1px solid var(--line); background: #fafafa; padding: .6rem .8rem; margin: 1rem 0; }
.busy { border: 1px solid #e3c26a; background: #fff8e1; padding: .6rem .8rem; margin: 1rem 0; }
.warning { border: 1px solid var(--flag); background: #fff4e5; padding: .6rem .8rem; margin: 1rem 0;
  font-weight: 600; }
.error { border: 1px solid var(--bad); background: #fdecea; padding: .6rem .8rem; margin: 1rem 0; }
.step { font-weight: 600; margin: 1.5rem 0 .4rem; }
.bar { height: .7rem; background: #ececec; border: 1px solid var(--line); }
.bar div { height: 100%; background-color: var(--gold);
  background-image: linear-gradient(45deg, rgba(255,255,255,.35) 25%, transparent 25%, transparent 50%,
    rgba(255,255,255,.35) 50%, rgba(255,255,255,.35) 75%, transparent 75%, transparent);
  background-size: 1rem 1rem; animation: stripes 1.2s linear infinite; }
@keyframes stripes { from { background-position: 1rem 0; } to { background-position: 0 0; } }
.steps { color: var(--muted); font-size: .9rem; margin: .5rem 0 1rem; }
.steps b { color: var(--ink); }
.now { margin: .5rem 0 1rem; }
table.stats { border-collapse: collapse; width: 100%; margin: 1rem 0 .5rem; }
table.stats th, table.stats td { border: 1px solid var(--line); padding: .35rem .6rem; text-align: left; }
table.stats th { background: #f5f5f5; font-size: .8rem; font-weight: 600; color: var(--muted); }
table.stats td { font-size: 1.15rem; font-weight: 700; }
ul.claims { padding-left: 1.2rem; }
li.claim { margin: .8rem 0; padding-left: .6rem; border-left: 3px solid transparent; }
li.flagged { border-left-color: var(--flag); }
.tag { font-size: .8rem; font-weight: 700; margin-left: .3rem; white-space: nowrap; }
.tag.ok { color: var(--ok); } .tag.flag { color: var(--flag); } .tag.na { color: var(--muted); }
.src { color: var(--muted); font-size: .9rem; }
"""


# Escapes any text from a run for placing in a page
def esc(text):
    return html.escape(str(text), quote=True)


# Returns the URL if it is safe to link to (http or https, no spaces or control
# characters), otherwise None
def safe_url(url):
    url = str(url)
    if re.search(r"[\s\x00-\x1f\x7f]", url):
        return None
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return url
    return None


# A link to an outside page, or plain text when the URL is not safe to link
def external_link(url, label):
    safe = safe_url(url)
    if safe is None:
        return esc(label)
    return f'<a href="{esc(safe)}" rel="noopener noreferrer">{esc(label)}</a>'


# Wraps a body in a full page
def page(title, body, refresh=False):
    refresh_tag = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">' if refresh else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{refresh_tag}<title>{esc(title)} | {APP_TITLE}</title><style>{STYLE}</style></head>"
        f'<body><header class="top"><div><a href="/">{APP_TITLE}</a>'
        "<span>Research answers, every claim checked.</span></div></header>"
        f"<main>{body}</main></body></html>"
    )


# The home page: the question form, with a link to a run already in progress
def render_index(running_job_id=None):
    banner = ""
    if running_job_id:
        banner = (f'<div class="notice">A run is in progress. '
                  f'<a href="/job/{esc(running_job_id)}">Watch it</a>. '
                  "Only one run can go at a time.</div>")
    body = (
        "<h1>Ask a research question</h1>"
        '<p class="lead">A Manager splits your question into sub-questions, Search reads Wikipedia, '
        "arXiv and the web, a Synthesizer writes claims that each cite a source, and a Fact-Checker "
        "checks every claim before you see it.</p>"
        f"{banner}"
        '<form method="post" action="/run">'
        '<label for="question">Research question</label>'
        f'<textarea id="question" name="question" rows="4" required minlength="{MIN_QUESTION_CHARS}" '
        f'maxlength="{MAX_QUESTION_CHARS}"></textarea>'
        '<p><button type="submit">Research it</button></p></form>'
        '<p class="muted">A run takes a few minutes on the free tier. Every claim in the report is '
        "checked against its cited source before you see it.</p>"
        '<p class="muted">On the free tier, Google may use prompts and responses to improve its '
        "products. Do not enter confidential or personal information (SRS 3.7.8).</p>"
    )
    return page("Ask a research question", body)


# A plain page for a refused request
def render_message(title, message):
    body = (f"<h1>{esc(title)}</h1><div class=\"error\">{esc(message)}</div>"
            f'<p><a href="/">Back</a></p>')
    return page(title, body)


# Seconds as m:ss
def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


# Which stage the run is in, from the last agent that worked
def current_stage(snapshot):
    author = snapshot["author"]
    if author == "synthesizer":
        return "write"
    if author in ("fact_checker", "escalation_checker"):
        return "check"
    return "search" if snapshot["searches_called"] else "plan"


# One plain sentence on what is happening now
def now_line(snapshot, stage):
    revising = snapshot["write_rounds"] >= 2
    if stage == "plan":
        return "The Manager is splitting your question into sub-questions."
    if stage == "search":
        return (f"Search is reading Wikipedia, arXiv and the web: {snapshot['searches_done']} of "
                f"{snapshot['searches_called']} searches back.")
    if stage == "write":
        if revising:
            return "Round 2: the Synthesizer is revising the claims the Fact-Checker sent back."
        return "The Synthesizer is writing the report, and every claim cites the passage it came from."
    if revising:
        return "The Fact-Checker is checking the revised claims against their sources."
    return "The Fact-Checker is checking every claim against the passage it cites."


# "Step 2 of 4", a progress bar, and the stages with the done ones checked
def render_stages(stage):
    names = [name for name, _ in STAGES]
    position = names.index(stage)
    working = len(STAGES) - 1
    delay = -(time.time() % STRIPE_SECONDS)
    labels = []
    for index, (_name, label) in enumerate(STAGES):
        if index < position:
            labels.append(f"{label} &#10003;")
        elif index == position:
            labels.append(f"<b>{label}</b>")
        else:
            labels.append(label)
    return (f'<p class="step">Step {min(position + 1, working)} of {working}</p>'
            f'<div class="bar"><div style="width:{STAGE_PROGRESS[stage]}%;animation-delay:{delay:.2f}s">'
            f'</div></div><p class="steps">{" &rarr; ".join(labels)}</p>')


# The page for one job: progress while it runs, the report when it is done, or the
# reason it stopped
def render_job(snapshot):
    status = snapshot["status"]
    head = f'<p class="question"><strong>Question:</strong> {esc(snapshot["question"])}</p>'
    if status == "running":
        stage = current_stage(snapshot)
        parts = [head, render_stages(stage), f'<p class="now">{esc(now_line(snapshot, stage))}</p>']
        if snapshot["notice"]:
            parts.append(f'<div class="busy">{esc(snapshot["notice"])}</div>')
        if snapshot["sub_questions"]:
            items = "".join(f"<li>{esc(text)}</li>" for text in snapshot["sub_questions"])
            parts.append(f"<h2>Sub-questions</h2><ol>{items}</ol>")
        calls = snapshot["calls"]
        parts.append(f'<p class="muted">{clock(snapshot["elapsed"])} elapsed. {calls} model '
                     f'call{"" if calls == 1 else "s"} so far. This page updates by itself.</p>')
        return page("Working", "".join(parts), refresh=True)
    if status == "error":
        body = (f"{head}<div class=\"error\">{esc(snapshot['message'])}</div>"
                f'<p class="muted">Model calls used: {snapshot["calls"]}.</p>'
                '<p><a href="/">Ask another question</a></p>')
        return page("Run stopped", body)
    body = f'{head}{snapshot["report_html"]}<p><a href="/">Ask another question</a></p>'
    return page("Report", body)


# Builds the report section for a finished run from its final session state. Only
# a draft the Fact-Checker fully checked is ever shown (SRS 3.6.4).
def render_report_html(state, calls, source_reader, elapsed=None):
    parts = []

    raw_plan = state.get(STATE_PLAN)
    if raw_plan is not None:
        plan = ManagerPlan.model_validate(raw_plan)
        items = []
        for sub_question in plan.sub_questions:
            sources = ", ".join(source.value for source in sub_question.preferred_sources)
            items.append(f"<li>{esc(sub_question.text)} <span class=\"muted\">({esc(sources)})</span></li>")
        parts.append(f"<h2>Sub-questions</h2><ol>{''.join(items)}</ol>")

    raw_draft = state.get(STATE_DRAFT)
    raw_verdict = state.get(STATE_VERDICT)
    if raw_draft is None:
        parts.append('<div class="error">No report in this run: the pipeline has not drafted one.</div>')
    elif raw_verdict is None:
        parts.append('<div class="error">No report shown: the Fact-Checker has not checked the draft '
                     "(SRS 3.6.4).</div>")
    else:
        try:
            verdict = FactCheckVerdict.model_validate(raw_verdict)
            # The loop only ends on a fail once its revisions run out (SRS 3.10.2)
            report = build_final_report(DraftReport.model_validate(raw_draft), verdict,
                                        incomplete=verdict.result == "fail")
        except ValueError as error:
            parts.append(f'<div class="error">No report shown: {esc(error)}</div>')
        else:
            parts.append(render_report_body(report, source_reader, calls, elapsed))

    parts.append(f'<p class="muted">Model calls: {int(calls)}</p>')
    return "".join(parts)


# The report itself: title, summary, each claim with its sources, the source list,
# the flagged claims and the support score
def render_report_body(report, source_reader, calls=None, elapsed=None):
    draft = report.draft
    checks = {}
    for check in report.verdict.checks:
        checks[check.claim_number] = check

    cited = []
    for claim in draft.all_claims():
        for evidence_id in claim.evidence_ids:
            if evidence_id not in cited:
                cited.append(evidence_id)
    sources = {}
    for row in source_reader(cited):
        sources[row.evidence_id] = row

    stats = [(f"{report.support_score}%" if report.support_score is not None else "none", "supported"),
             (len(draft.all_claims()), "claims"),
             (len({row.source_url for row in sources.values()}), "sources")]
    if calls is not None:
        stats.append((int(calls), "model calls"))
    if elapsed is not None:
        stats.append((clock(elapsed), "time taken"))
    parts = ['<table class="stats"><tr>'
             + "".join(f"<th>{esc(label)}</th>" for _, label in stats) + "</tr><tr>"
             + "".join(f"<td>{esc(value)}</td>" for value, _ in stats) + "</tr></table>"]
    if report.incomplete:
        parts.append('<div class="warning">This report is incomplete: the pipeline ran out of '
                     "revisions or retries, and the claims marked FLAGGED could not be supported "
                     "(SRS 3.10.2).</div>")
    parts.append(f"<h2>{esc(draft.title)}</h2><p>{esc(draft.summary)}</p>")

    for section in draft.sections:
        items = []
        for claim in section.claims:
            flagged = claim.claim_number in report.flagged_claim_numbers
            verdict = checks[claim.claim_number].verdict.value
            if flagged:
                badge = f' <span class="tag flag">FLAGGED: {esc(verdict)}</span>'
            elif verdict == "supported":
                badge = ' <span class="tag ok">&#10003; Supported</span>'
            else:
                badge = ' <span class="tag na">Not verifiable</span>'
            cites = []
            for evidence_id in claim.evidence_ids:
                row = sources.get(evidence_id)
                if row is None:
                    cites.append(f"[{evidence_id}] not found in the evidence store")
                else:
                    cites.append(f'<a href="#src-{evidence_id}">[{evidence_id}]</a> '
                                 f"{external_link(row.source_url, row.source_title)} ({esc(row.source_type)})")
            css = "claim flagged" if flagged else "claim"
            items.append(
                f'<li class="{css}" id="claim-{claim.claim_number}">({claim.claim_number}) '
                f'{esc(claim.sentence)}{badge}<div class="src">Source: {"; ".join(cites)}</div></li>'
            )
        parts.append(f'<h3>{esc(section.heading)}</h3><ul class="claims">{"".join(items)}</ul>')

    items = []
    for evidence_id in cited:
        row = sources.get(evidence_id)
        if row is not None:
            items.append(f'<li id="src-{evidence_id}">[{evidence_id}] '
                         f"{external_link(row.source_url, row.source_title)} "
                         f'<span class="muted">({esc(row.source_type)}) {esc(row.source_url)}</span></li>')
    parts.append(f"<h2>Sources</h2><ul>{''.join(items)}</ul>")

    if report.flagged_claim_numbers:
        items = []
        for number in report.flagged_claim_numbers:
            check = checks[number]
            restatement = ""
            if check.restatement:
                restatement = f"<br>Supported version: {esc(check.restatement)}"
            items.append(f'<li><a href="#claim-{number}">({number})</a> {esc(check.verdict.value)}: '
                         f"{esc(check.reason)}{restatement}</li>")
        parts.append(f"<h2>Flagged claims</h2><ul>{''.join(items)}</ul>")
    else:
        parts.append("<h2>Flagged claims</h2><p>None.</p>")

    if report.support_score is None:
        score = "none (no claim could be checked)"
    else:
        score = f"{report.support_score}%"
    parts.append(f"<h2>Support score</h2><p><strong>{esc(score)}</strong> "
                 '<span class="muted">(supported claims out of claims that could be checked)</span></p>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "sp201-web"

    # Sends a page with the security headers
    def send_page(self, status, body, location=None):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        if location is not None:
            self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(data)

    # The origins a request may legitimately come from: this server, by either name
    def allowed_hosts(self):
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    # Refuses a request whose Host header is not this server (DNS rebinding)
    def host_ok(self):
        return self.headers.get("Host", "") in self.allowed_hosts()

    # Refuses a POST that came from another site
    def origin_ok(self):
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{host}" for host in self.allowed_hosts()}:
            return False
        fetch_site = self.headers.get("Sec-Fetch-Site")
        if fetch_site is not None and fetch_site not in ("same-origin", "none"):
            return False
        return True

    def do_GET(self):
        if not self.host_ok():
            self.send_page(400, render_message("Bad request", "This address is not allowed."))
            return
        app = self.server.web_app
        path = self.path.split("?", 1)[0]
        if path == "/":
            running = app.store.running()
            self.send_page(200, render_index(running.job_id if running else None))
            return
        match = JOB_ID_PATTERN.match(path)
        job = app.store.get(match.group(1)) if match else None
        if job is None:
            self.send_page(404, render_message("Not found", "There is no such page or run."))
            return
        self.send_page(200, render_job(job.snapshot()))

    def do_POST(self):
        if not self.host_ok():
            self.send_page(400, render_message("Bad request", "This address is not allowed."))
            return
        if not self.origin_ok():
            self.send_page(403, render_message("Refused", "Requests from other sites are not accepted."))
            return
        if self.path.split("?", 1)[0] != "/run":
            self.send_page(404, render_message("Not found", "There is no such page."))
            return
        length_header = self.headers.get("Content-Length")
        if length_header is None or not length_header.isdigit():
            self.send_page(411, render_message("Refused", "The request needs a Content-Length."))
            return
        if int(length_header) > MAX_BODY_BYTES:
            self.send_page(413, render_message("Too large", "That request is too large."))
            return
        body = self.rfile.read(int(length_header)).decode("utf-8", errors="replace")
        question = " ".join(parse_qs(body).get("question", [""])[0].split())
        if len(question) < MIN_QUESTION_CHARS or len(question) > MAX_QUESTION_CHARS:
            self.send_page(400, render_message(
                "Question not accepted",
                f"Write a question of {MIN_QUESTION_CHARS} to {MAX_QUESTION_CHARS} characters."))
            return
        app = self.server.web_app
        job = app.store.start(question)
        if job is None:
            self.send_page(409, render_message(
                "Busy", "A run is already in progress. Only one run can go at a time, "
                        "to protect the free-tier quota."))
            return
        threading.Thread(target=work, args=(app, job), daemon=True).start()
        self.send_page(303, "", location=f"/job/{job.job_id}")


# Builds the server, bound to the loopback address only (SRS 5.4). Port 0 picks a
# free port.
def make_server(port=DEFAULT_PORT, runner=None, source_reader=None, explain=None):
    server = ThreadingHTTPServer((HOST, port), Handler)
    server.daemon_threads = True
    server.web_app = WebApp(runner=runner, source_reader=source_reader, explain=explain)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the SP-201 web interface on this computer only.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    try:
        server = make_server(args.port)
    except OSError as error:
        print(f"Could not start on port {args.port}: {error}")
        raise SystemExit(1)
    print(f"SP-201 web interface: http://{HOST}:{server.server_address[1]}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
