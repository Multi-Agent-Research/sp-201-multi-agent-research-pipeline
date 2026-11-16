# Tests for the web page's request checks (SRS 5.4). Run from the project root:
#     python -m pytest tests
# No test calls Gemini: the runner is replaced by one that stops at once. Each
# test starts its own server on a free port, bound to the loopback address.
import http.client
import logging
import threading
import urllib.parse
from collections import namedtuple

import pytest

import web


def stop_at_once(question, job):
    raise RuntimeError("stopped by the test")


@pytest.fixture
def server():
    srv = web.make_server(port=0, runner=stop_at_once, explain=lambda error: "stopped")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def request(srv, method, path, headers=None, body=None):
    port = srv.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    sent = {"Host": f"127.0.0.1:{port}"}
    sent.update(headers or {})
    connection.request(method, path, body=body, headers=sent)
    response = connection.getresponse()
    response.read()
    connection.close()
    return response


def post_question(srv, origin, fetch_site="same-origin"):
    body = urllib.parse.urlencode({"question": "How does attention work in a transformer?"})
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(len(body))}
    if origin is not None:
        headers["Origin"] = origin
    if fetch_site is not None:
        headers["Sec-Fetch-Site"] = fetch_site
    return request(srv, "POST", "/run", headers, body)


def test_5_4_page_policy_lets_the_browser_send_its_real_origin(server):
    # Under no-referrer a browser sends "Origin: null" on the page's own form,
    # which the origin check refuses, so the Research button never worked
    response = request(server, "GET", "/")
    assert response.status == 200
    assert response.getheader("Referrer-Policy") == "same-origin"


def test_5_4_the_pages_own_form_post_is_accepted(server):
    port = server.server_address[1]
    response = post_question(server, f"http://127.0.0.1:{port}")
    assert response.status == 303
    assert response.getheader("Location").startswith("/job/")


def test_5_4_a_post_from_another_site_or_an_opaque_origin_is_refused(server):
    assert post_question(server, "http://evil.example", fetch_site="cross-site").status == 403
    assert post_question(server, "null").status == 403


# The waiting page (SRS 3.8): a stage tracker, a plain line on what is happening,
# the sub-questions once planned, and a notice while Gemini or a source is retried

def waiting_page(job):
    return web.render_job(job.snapshot())


def test_3_8_a_new_run_shows_the_plan_stage_as_active():
    text = waiting_page(web.Job("How does attention work in a transformer?"))
    assert "Step 1 of 4" in text and "<b>Plan</b>" in text
    assert "The Manager is splitting your question into sub-questions." in text
    assert '<a href="/">Sift</a>' in text and "<title>Working | Sift</title>" in text
    assert 'http-equiv="refresh"' in text


def test_3_8_search_progress_and_sub_questions_are_shown_escaped():
    job = web.Job("How does attention work?")
    job.record_event("manager", search_calls=2)
    job.record_event("manager", search_results=1,
                     sub_questions=["What is <b>attention</b>?", "Why it scales"])
    text = waiting_page(job)
    assert "Step 2 of 4" in text and "Plan &#10003;" in text and "<b>Search</b>" in text
    assert "1 of 2 searches back" in text
    assert "What is &lt;b&gt;attention&lt;/b&gt;?" in text and "<b>attention</b>" not in text


def test_3_8_a_second_writing_round_is_named_as_a_revision():
    job = web.Job("How does attention work?")
    for author in ("manager", "synthesizer", "fact_checker", "escalation_checker", "synthesizer"):
        job.record_event(author)
    assert "Round 2: the Synthesizer is revising" in waiting_page(job)


def test_3_8_events_from_other_authors_do_not_move_the_tracker():
    job = web.Job("How does attention work?")
    job.record_event("synthesizer")
    job.record_event("revision_loop")
    assert job.snapshot()["author"] == "synthesizer"


def test_3_8_a_gemini_retry_shows_a_notice_until_the_next_event():
    job = web.Job("How does attention work?")
    handler = web.RetryNoticeHandler(job)
    rate_logger = logging.getLogger("rate_limit")
    rate_logger.addHandler(handler)
    try:
        rate_logger.warning("Gemini is overloaded right now (503); retry %d of %d in %.0fs", 2, 3, 20)
    finally:
        rate_logger.removeHandler(handler)
    text = waiting_page(job)
    assert "Google&#x27;s Gemini servers are busy right now" in text
    assert "(retry 2 of 3, in 20 s)" in text
    job.record_event("synthesizer")
    assert job.snapshot()["notice"] == ""


def test_3_8_a_retry_notice_never_carries_the_error_text():
    notice = web.retry_notice("Wikipedia search failed (<script>https://x.example</script>); retry 1 in 2.0s")
    assert notice == "A source (Wikipedia, arXiv or the web) is slow to answer. Search retries on its own."


def test_3_8_sub_questions_are_read_from_a_state_change():
    plan = {"user_question": "Q?", "sub_questions": [
        {"sub_question_id": 1, "text": "How was AlphaGo trained?", "preferred_sources": ["wikipedia"],
         "priority": 1}]}
    assert web.plan_texts({web.STATE_PLAN: plan}) == ["How was AlphaGo trained?"]
    assert web.plan_texts({web.STATE_PLAN: "not a plan"}) == []
    assert web.plan_texts({}) == []
    assert web.plan_texts(None) == []


# The report (SRS 3.8): a stats row and a tag on every claim

Row = namedtuple("Row", "evidence_id source_type source_title source_url published_date passage")


def test_3_8_report_shows_stats_and_a_tag_on_every_claim():
    draft = {"title": "Attention", "summary": "What attention does.",
             "sections": [{"heading": "Basics", "claims": [
        {"claim_number": 1, "sentence": "Transformers rely on self-attention.", "evidence_ids": [1]},
        {"claim_number": 2, "sentence": "Attention cost grows with length.", "evidence_ids": [1, 2]}]}]}
    verdict = {"checks": [
        {"claim_number": 1, "verdict": "supported", "reason": "Evidence 1 says so."},
        {"claim_number": 2, "verdict": "supported", "reason": "Evidence 2 says so."}], "result": "pass"}
    rows = {1: Row(1, "wikipedia", "Transformer", "https://en.wikipedia.org/wiki/Transformer", "", "p1"),
            2: Row(2, "wikipedia", "Transformer", "https://en.wikipedia.org/wiki/Transformer", "", "p2")}
    text = web.render_report_html({web.STATE_DRAFT: draft, web.STATE_VERDICT: verdict}, 4,
                                  lambda ids: [rows[i] for i in ids if i in rows], elapsed=142)
    for value, label in (("100.0%", "supported"), ("2", "claims"), ("1", "sources"), ("4", "model calls"),
                         ("2:22", "time taken")):
        assert f"<th>{label}</th>" in text and f"<td>{value}</td>" in text
    assert text.count("&#10003; Supported") == 2
