# Web interface (SRS 3.8, 4.3, 5.1, 5.4). A small local web page for the same
# pipeline the CLI runs. The user types one question, watches the pipeline's
# progress, and reads the checked report with a source under every claim, the
# flagged claims, the support score and the number of model calls. Like the CLI
# it searches, writes and checks nothing itself.
#
#     python web.py                 # then open http://127.0.0.1:8000
#     python web.py --port 9000
#
# What it reuses from cli.py: build_root_agent, ModelCallCounter and
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
#   * It binds to 127.0.0.1 only and the port is not configurable beyond that
#     (SRS 5.4). There is no login, as the SRS says.
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
USER_ID = "web"

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MIN_QUESTION_CHARS = 5
MAX_QUESTION_CHARS = 1000
MAX_BODY_BYTES = 8192
MAX_JOBS_KEPT = 20
REFRESH_SECONDS = 3

JOB_ID_PATTERN = re.compile(r"^/job/([0-9a-f]{16})$")

# Headers sent with every page. default-src 'none' means no script, image, font
# or frame can load; only the page's own inline style applies.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
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
        job.set_calls(counter.calls)
        if event.error_code and stopped is None:
            stopped = RunStopped(event.author, event.error_code, event.error_message)
    job.set_calls(counter.calls)
    if stopped is not None:
        raise stopped
    final = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session.id)
    return final.state


# The default runner: builds the same pipeline the CLI runs and returns the final
# state. cli is imported here so that web.py loads without the pipeline.
def pipeline_runner(question, job):
    import cli
    return asyncio.run(execute(question, job, cli.build_root_agent(), cli.ModelCallCounter()))


# Turns a failed run into a message for the page, using the same explanations as
# the CLI. Nothing from an unexpected error is shown beyond its type: its text
# could hold file paths or retrieved content.
def explain_failure(error):
    import cli
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
        job.finish(render_report_html(state, job.snapshot()["calls"], app.source_reader))
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
:root { color-scheme: light dark; --muted: #667; --line: #bbc; --flag: #b45309; --bad: #b91c1c; }
@media (prefers-color-scheme: dark) { :root { --muted: #aab; --line: #445; --flag: #fbbf24; --bad: #f87171; } }
body { font: 16px/1.5 system-ui, sans-serif; max-width: 46rem; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 1.75rem; border-bottom: 1px solid var(--line); }
textarea { width: 100%; box-sizing: border-box; font: inherit; padding: .5rem; }
button { font: inherit; padding: .4rem 1rem; }
.muted { color: var(--muted); font-size: .9rem; }
.notice { border: 1px solid var(--line); padding: .6rem .8rem; margin: 1rem 0; }
.warning { border: 2px solid var(--flag); padding: .6rem .8rem; margin: 1rem 0; font-weight: 600; }
.error { border: 2px solid var(--bad); padding: .6rem .8rem; margin: 1rem 0; }
ul.claims { padding-left: 1.2rem; } li.claim { margin: .8rem 0; padding-left: .6rem; border-left: 4px solid transparent; }
li.flagged { border-left-color: var(--flag); }
.badge { font-weight: 700; color: var(--flag); font-size: .85rem; }
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
        f"{refresh_tag}<title>{esc(title)}</title><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )


# The home page: the question form, with a link to a run already in progress
def render_index(running_job_id=None):
    banner = ""
    if running_job_id:
        banner = (f'<div class="notice">A run is in progress. '
                  f'<a href="/job/{esc(running_job_id)}">Watch it</a>. '
                  "Only one run can go at a time.</div>")
    body = (
        "<h1>SP-201 Research &amp; Fact-Checking</h1>"
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
    return page("SP-201 Research & Fact-Checking", body)


# A plain page for a refused request
def render_message(title, message):
    body = (f"<h1>{esc(title)}</h1><div class=\"error\">{esc(message)}</div>"
            f'<p><a href="/">Back</a></p>')
    return page(title, body)


# The page for one job: progress while it runs, the report when it is done, or the
# reason it stopped
def render_job(snapshot):
    status = snapshot["status"]
    head = (f"<h1>SP-201 Research &amp; Fact-Checking</h1>"
            f'<p><strong>Question:</strong> {esc(snapshot["question"])}</p>')
    if status == "running":
        steps = "".join(f"<li>{esc(step)}</li>" for step in snapshot["steps"]) or "<li>Starting</li>"
        body = (
            f"{head}<h2>Working</h2><ol>{steps}</ol>"
            f'<p class="muted">{snapshot["elapsed"]} seconds so far. '
            f'Model calls used: {snapshot["calls"]}. This page refreshes by itself.</p>'
        )
        return page("Working", body, refresh=True)
    if status == "error":
        body = (f"{head}<div class=\"error\">{esc(snapshot['message'])}</div>"
                f'<p class="muted">Model calls used: {snapshot["calls"]}.</p>'
                '<p><a href="/">Ask another question</a></p>')
        return page("Run stopped", body)
    body = f'{head}{snapshot["report_html"]}<p><a href="/">Ask another question</a></p>'
    return page("Report", body)


# Builds the report section for a finished run from its final session state. Only
# a draft the Fact-Checker fully checked is ever shown (SRS 3.6.4).
def render_report_html(state, calls, source_reader):
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
            parts.append(render_report_body(report, source_reader))

    parts.append(f'<p class="muted">Model calls: {int(calls)}</p>')
    return "".join(parts)


# The report itself: title, summary, each claim with its sources, the source list,
# the flagged claims and the support score
def render_report_body(report, source_reader):
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

    parts = []
    if report.incomplete:
        parts.append('<div class="warning">This report is incomplete: the pipeline ran out of '
                     "revisions or retries, and the claims marked FLAGGED could not be supported "
                     "(SRS 3.10.2).</div>")
    parts.append(f"<h2>{esc(draft.title)}</h2><p>{esc(draft.summary)}</p>")

    for section in draft.sections:
        items = []
        for claim in section.claims:
            flagged = claim.claim_number in report.flagged_claim_numbers
            badge = ""
            if flagged:
                badge = f' <span class="badge">FLAGGED: {esc(checks[claim.claim_number].verdict.value)}</span>'
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
# free port, which the tests use.
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
