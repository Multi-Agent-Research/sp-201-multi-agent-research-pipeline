# Evaluation harness (SRS 3.9). Runs every question of an evaluation set through
# the same pipeline the CLI runs, then scores each report from what the run saved:
#
#     python -m sp201.evaluate                    # the twenty in evaluation/eval_set.json
#     python -m sp201.evaluate --only Q01,Q13     # some of them
#     python -m sp201.evaluate --set other_set.json --out evaluation/runs/other
#
# Per question and in aggregate (SRS 3.9.1):
#   claim-level support rate    supported over verifiable claims, read back from the
#                               database (SRS 3.5.6, 3.9.4); the aggregate pools
#                               every claim of every report
#   report-level support score  the Fact-Checker's verdict in state (SRS 3.6.7); the
#                               aggregate is the mean of the reports
#   citation precision          the share of the evidence a report cites that backs
#                               at least one claim judged supported or partially
#                               supported; a made-up id counts as cited, never backing
#   citation recall             the share of the question's expected source documents
#                               (the eval set's expected_facts sources) that the report
#                               cites, matched by normalized URL
# plus uncited sentences (SRS 3.4.3, target zero): claims saved with no evidence
# link. The summary's sentences are counted apart, because the summary carries no
# citations by design. Also model calls (3.9.5), time and errors per question. The
# run records the model id, date, temperature and thinking levels (3.9.3), the git
# commit and the caps.
#
# Each run gets its own folder: evaluation.db (one database for the whole run),
# reports/<id>.txt as the CLI prints it, reports/<id>.json with the plan, search
# results, draft and verdict, logs/<id>.log, results.json and summary.txt.
#
# Not covered here: SRS 3.9.2's ADK eval-set format, and accuracy against the
# expected facts, which is judged by reading the reports, not by code.
import argparse
import asyncio
import contextlib
import json
import logging
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from datetime import timezone
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlparse

from dotenv import load_dotenv
from google.genai import errors as genai_errors

from evidence_store import EvidenceStore
from manager_agent import get_model_id
from pipeline import build_pipeline
from rate_limit import DailyQuotaExceeded
from rate_limit import MalformedReply
from rate_limit import model_retry_policy
from rate_limit import model_rpm
from schemas import STATE_DRAFT
from schemas import STATE_PLAN
from schemas import STATE_QUESTION_ID
from schemas import STATE_SEARCH_RESULTS
from schemas import STATE_VERDICT
from schemas import DraftReport
from schemas import FactCheckVerdict
from schemas import Verdict
from schemas import build_final_report
from schemas import max_revisions
from schemas import max_sub_questions
from sp201 import cli
from synthesizer_agent import EvidenceError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SET = PROJECT_ROOT / "evaluation" / "eval_set.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "evaluation" / "runs"
DB_NAME = "evaluation.db"

# No agent sets a temperature; only thinking levels are set (checked 2026-11-22)
TEMPERATURE = "model default (no temperature is set in the code)"
BACKING_VERDICTS = {Verdict.SUPPORTED.value, Verdict.PARTIALLY_SUPPORTED.value}
SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
QUOTA_STOP = "not run: the daily quota ran out earlier in this run"


# ----- matching sources -----

# One form for every way a source's address is written: no scheme, no "www.",
# no trailing slash or fragment, decoded, lower case; an arXiv paper by its id
# alone (abs or pdf, any version), a Wikipedia title with underscores
def normalize_url(url):
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    host = host.removeprefix("www.")
    path = unquote(parsed.path).rstrip("/")
    if host.endswith("arxiv.org"):
        match = ARXIV_ID.search(path)
        if match:
            return "arxiv.org/abs/" + match.group(1)
    if host.endswith("wikipedia.org"):
        path = path.replace(" ", "_")
    query = "?" + parsed.query if parsed.query else ""
    return (host + path + query).lower()


# The question's expected source documents, normalized, without repeats, in order
def expected_sources(item):
    sources = []
    for fact in item.get("expected_facts", []):
        url = normalize_url(fact.get("source", ""))
        if url and url not in sources:
            sources.append(url)
    return sources


# Sentences in a block of text, counted by their ends (a full stop, ? or ! before
# a space or the end); text with no end counts as one
def count_sentences(text):
    text = (text or "").strip()
    if not text:
        return 0
    return max(1, len(SENTENCE_END.findall(text)))


def percent(part, whole):
    if not whole:
        return None
    return round(100.0 * part / whole, 1)


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 1)


# ----- scoring one question -----

# The scores for one question from its final state and the run's database.
# state is None when the run stopped before it ended; error says why.
def score_question(item, state, store, calls, seconds, error=None):
    result = {
        "id": item.get("id"), "type": item.get("type"), "question": item.get("question"),
        "report": False, "result": None, "error": error,
        "model_calls": calls, "seconds": round(seconds, 1),
        "expected_sources": expected_sources(item),
    }
    state = state or {}
    raw_results = state.get(STATE_SEARCH_RESULTS)
    if raw_results is not None:
        result["search_status"] = [entry.get("status") for entry in raw_results.get("results", [])]
    raw_draft, raw_verdict = state.get(STATE_DRAFT), state.get(STATE_VERDICT)
    if raw_draft is None or raw_verdict is None:
        if result["error"] is None:
            result["error"] = "the run ended without a checked report"
        return result
    try:
        draft = DraftReport.model_validate(raw_draft)
        verdict = FactCheckVerdict.model_validate(raw_verdict)
        report = build_final_report(draft, verdict, incomplete=verdict.result == "fail")
    except ValueError as problem:
        result["error"] = f"no report shown: {problem}"
        return result

    claims = draft.all_claims()
    checks = {check.claim_number: check for check in verdict.checks}
    result.update({
        "report": True, "result": verdict.result, "claims": len(claims),
        "flagged": len(report.flagged_claim_numbers),
        "report_support_score": report.support_score,
        "summary_sentences": count_sentences(draft.summary),
    })

    # From the database alone (SRS 3.5.6): the last saved iteration is the report
    question_id = state.get(STATE_QUESTION_ID)
    iteration = store.latest_iteration(question_id) if question_id is not None else None
    saved = store.get_claims(question_id, iteration) if iteration is not None else []
    verifiable = [claim for claim in saved if claim["verdict"] != Verdict.NOT_VERIFIABLE.value]
    supported = [claim for claim in verifiable if claim["verdict"] == Verdict.SUPPORTED.value]
    result.update({
        "iterations_saved": iteration or 0,
        "saved_claims": len(saved),
        "verifiable": len(verifiable),
        "supported": len(supported),
        "claim_support_rate": percent(len(supported), len(verifiable)),
        "uncited_claims": sum(1 for claim in saved if not claim["evidence_ids"]) if saved else None,
    })
    result["saved_matches_report"] = (
        len(saved) == len(claims) and result["claim_support_rate"] == report.support_score
    )

    # Citations: every id the draft cites, whether or not it exists
    owned = set()
    if question_id is not None:
        for evidence_ids in store.evidence_ids_by_plan_number(question_id).values():
            owned.update(evidence_ids)
    cited = draft.cited_evidence_ids()
    backing = set()
    for claim in claims:
        if checks[claim.claim_number].verdict.value in BACKING_VERDICTS:
            backing.update(evidence_id for evidence_id in claim.evidence_ids if evidence_id in owned)
    result["cited_evidence"] = len(cited)
    result["citation_precision"] = percent(len(set(cited) & backing), len(cited))

    cited_urls = []
    for row in store.read_evidence([evidence_id for evidence_id in cited if evidence_id in owned]):
        url = normalize_url(row.source_url)
        if url not in cited_urls:
            cited_urls.append(url)
    found = [url for url in result["expected_sources"] if url in cited_urls]
    result["sources_cited"] = len(cited_urls)
    result["expected_sources_cited"] = found
    result["citation_recall"] = percent(len(found), len(result["expected_sources"]))
    return result


# Totals over every question (SRS 3.9.1, 3.9.5)
def aggregate(results):
    reports = [result for result in results if result["report"]]
    attempted = [result for result in results if result.get("error") != QUOTA_STOP]
    verifiable = sum(result["verifiable"] for result in reports)
    supported = sum(result["supported"] for result in reports)
    calls = sum(result["model_calls"] for result in attempted)
    return {
        "questions": len(results),
        "questions_run": len(attempted),
        "reports": len(reports),
        "passed": sum(1 for result in reports if result["result"] == "pass"),
        "errors": sum(1 for result in results if result.get("error")),
        "claims": sum(result["claims"] for result in reports),
        "verifiable": verifiable,
        "supported": supported,
        "claim_support_rate": percent(supported, verifiable),
        "report_support_score": mean(result["report_support_score"] for result in reports),
        "citation_precision": mean(result["citation_precision"] for result in reports),
        "citation_recall": mean(result["citation_recall"] for result in reports),
        "uncited_claims": sum(result["uncited_claims"] or 0 for result in reports),
        "summary_sentences": sum(result["summary_sentences"] for result in reports),
        "saved_matches_report": sum(1 for result in reports if result["saved_matches_report"]),
        "model_calls": calls,
        "model_calls_per_question": round(calls / len(attempted), 1) if attempted else None,
        "seconds": round(sum(result["seconds"] for result in attempted), 1),
    }


# ----- the run's record (SRS 3.9.3) -----

# Each Gemini agent's thinking level, or the model's default when none is set
def thinking_levels(agent, levels=None):
    if levels is None:
        levels = {}
    config = getattr(agent, "generate_content_config", None)
    if hasattr(agent, "instruction"):
        thinking = getattr(config, "thinking_config", None) if config is not None else None
        level = getattr(thinking, "thinking_level", None) if thinking is not None else None
        levels[agent.name] = getattr(level, "value", str(level)) if level is not None else "model default"
    for sub_agent in getattr(agent, "sub_agents", []) or []:
        thinking_levels(sub_agent, levels)
    return levels


def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + (" (with uncommitted changes)" if changes else "")


def run_metadata(agent, set_path):
    return {
        "model_id": get_model_id(),
        "run_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temperature": TEMPERATURE,
        "thinking_levels": thinking_levels(agent),
        "git_commit": git_commit(),
        "eval_set": str(set_path),
        "max_sub_questions": max_sub_questions(),
        "max_revisions": max_revisions(),
        "model_rpm": model_rpm(),
        "model_max_attempts": model_retry_policy().max_attempts,
    }


# ----- running -----

def default_agent_factory(store):
    return build_pipeline(store=store)


# The plain reason a run stopped, worded the way the CLI words it
def explain(error):
    if isinstance(error, genai_errors.APIError):
        return cli.explain_model_error(error.code, str(error)) or f"{type(error).__name__}: {error}"
    if isinstance(error, cli.PipelineStopped):
        return cli.explain_model_error(error.code, error.message) or str(error)
    if isinstance(error, EvidenceError):
        return f"The run stopped before writing a report: {error}"
    if isinstance(error, (DailyQuotaExceeded, MalformedReply)):
        return str(error)
    return f"Unexpected error ({type(error).__name__}: {error})"


# Runs one question with its progress lines and log records written to log_path.
# Returns (final state or None, model calls, seconds, error or None, quota used up).
def run_one(item, store, agent_factory, log_path):
    counter = cli.ModelCallCounter()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    old_level = root.level
    root.setLevel(logging.INFO)
    state, error, quota = None, None, False
    start = time.monotonic()
    try:
        with open(log_path, "a", encoding="utf-8") as out, contextlib.redirect_stdout(out):
            print(f"Question {item['id']}: {item['question']}")
            try:
                state = asyncio.run(cli.run_question(item["question"], agent_factory(store), counter))
            except Exception as problem:  # noqa: BLE001  one question failing must not end the run
                error = explain(problem)
                quota = isinstance(problem, DailyQuotaExceeded)
                traceback.print_exc(file=out)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()
    return state, counter.calls, time.monotonic() - start, error, quota


def report_record(state):
    keys = (STATE_QUESTION_ID, STATE_PLAN, STATE_SEARCH_RESULTS, STATE_DRAFT, STATE_VERDICT)
    return {key: state.get(key) for key in keys}


# Runs every item and writes the run folder. Returns {"run": ..., "aggregate": ...,
# "questions": [...]}. agent_factory(store) builds the agent for one question.
def run_evaluation(items, out_dir, agent_factory=default_agent_factory, set_path=DEFAULT_SET,
                   echo=print):
    out_dir = Path(out_dir)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(out_dir / DB_NAME)
    record = {"run": run_metadata(agent_factory(store), set_path), "questions": []}
    quota_gone = False
    for number, item in enumerate(items, 1):
        if quota_gone:
            record["questions"].append(score_question(item, None, store, 0, 0.0, QUOTA_STOP))
            continue
        echo(f"[{number}/{len(items)}] {item['id']}: {item['question']}")
        state, calls, seconds, error, quota = run_one(item, store, agent_factory,
                                                      out_dir / "logs" / f"{item['id']}.log")
        quota_gone = quota
        result = score_question(item, state, store, calls, seconds, error)
        record["questions"].append(result)
        if state is not None:
            (out_dir / "reports" / f"{item['id']}.txt").write_text(
                cli.render(state, calls, source_reader=store.read_evidence) + "\n", encoding="utf-8")
            (out_dir / "reports" / f"{item['id']}.json").write_text(
                json.dumps(report_record(state), indent=1, default=str) + "\n", encoding="utf-8")
        echo("    " + one_line(result))
    record["aggregate"] = aggregate(record["questions"])
    (out_dir / "results.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    (out_dir / "summary.txt").write_text(summary_text(record) + "\n", encoding="utf-8")
    return record


# ----- what a person reads -----

def show(value, suffix=""):
    return "n/a" if value is None else f"{value}{suffix}"


def one_line(result):
    if not result["report"]:
        return f"no report: {result['error']} ({result['model_calls']} calls, {result['seconds']} s)"
    return (f"{result['result']}, {result['claims']} claims, support {show(result['claim_support_rate'], '%')}, "
            f"precision {show(result['citation_precision'], '%')}, recall {show(result['citation_recall'], '%')}, "
            f"{result['model_calls']} calls, {result['seconds']} s")


def summary_text(record):
    run, total = record["run"], record["aggregate"]
    lines = [
        "SP-201 evaluation run",
        (f"Model: {run['model_id']}   Date (UTC): {run['run_date_utc']}   "
         f"Temperature: {run['temperature']}"),
        "Thinking: " + ", ".join(f"{name} {level}" for name, level in run["thinking_levels"].items()),
        f"Commit: {run['git_commit']}",
        (f"Caps: {run['max_sub_questions']} sub-questions, {run['max_revisions']} revision, "
         f"{run['model_rpm']:g} requests a minute, {run['model_max_attempts']} attempts a call"),
        "",
        (f"{'ID':<5}{'Result':<8}{'Claims':>7}{'Support':>9}{'Report':>8}{'Uncited':>8}"
         f"{'Prec.':>7}{'Recall':>8}{'Calls':>7}{'Secs':>7}  Error"),
    ]
    for result in record["questions"]:
        if result["report"]:
            lines.append(
                f"{result['id']:<5}{result['result']:<8}{result['claims']:>7}"
                f"{show(result['claim_support_rate']):>9}{show(result['report_support_score']):>8}"
                f"{show(result['uncited_claims']):>8}{show(result['citation_precision']):>7}"
                f"{show(result['citation_recall']):>8}{result['model_calls']:>7}{result['seconds']:>7.0f}"
                f"  {result['error'] or ''}")
        else:
            lines.append(f"{result['id']:<5}{'none':<8}{'':>7}{'':>9}{'':>8}{'':>8}{'':>7}{'':>8}"
                         f"{result['model_calls']:>7}{result['seconds']:>7.0f}  {result['error']}")
    lines += [
        "",
        (f"Reports: {total['reports']} of {total['questions']} ({total['passed']} passed, "
         f"{total['errors']} with an error)"),
        (f"Claim-level support rate (pooled, from the database): {show(total['claim_support_rate'], '%')} "
         f"({total['supported']} of {total['verifiable']} verifiable claims)"),
        f"Report-level support score (mean): {show(total['report_support_score'], '%')}",
        f"Database matches the report on: {total['saved_matches_report']} of {total['reports']}",
        f"Citation precision (mean): {show(total['citation_precision'], '%')}",
        f"Citation recall (mean): {show(total['citation_recall'], '%')}",
        (f"Uncited claims: {total['uncited_claims']} (target 0); summary sentences, uncited by design: "
         f"{total['summary_sentences']}"),
        (f"Model calls: {total['model_calls']} ({show(total['model_calls_per_question'])} a question); "
         f"time {total['seconds']:.0f} s"),
    ]
    return "\n".join(lines)


def load_items(set_path, only=None):
    data = json.loads(Path(set_path).read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    if only:
        wanted = [name.strip().upper() for name in only.split(",") if name.strip()]
        unknown = sorted(set(wanted) - {item["id"].upper() for item in items})
        if unknown:
            raise SystemExit(f"Not in {set_path}: {', '.join(unknown)}")
        items = [item for item in items if item["id"].upper() in wanted]
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m sp201.evaluate",
                                     description="Run an evaluation set through the pipeline and score it.")
    parser.add_argument("--set", default=str(DEFAULT_SET), help="the evaluation set (default: %(default)s)")
    parser.add_argument("--only", help="comma-separated question ids, for example Q01,Q13")
    parser.add_argument("--out", help="the run folder (default: evaluation/runs/<UTC date and time>)")
    args = parser.parse_args(argv)

    load_dotenv()
    # ADK logs a full traceback when a model call fails; each question's log keeps it
    logging.getLogger("google_adk").setLevel(logging.WARNING)
    items = load_items(args.set, args.only)
    out_dir = Path(args.out) if args.out else \
        DEFAULT_RUNS_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    print(f"Running {len(items)} questions into {out_dir}")
    record = run_evaluation(items, out_dir, set_path=args.set)
    print()
    print(summary_text(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
