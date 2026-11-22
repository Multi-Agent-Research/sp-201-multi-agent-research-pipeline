# Tests for the evaluation harness (SRS 3.9.1, 3.4.3, 3.9.3, 3.9.5). Run from
# the project root:
#     python -m pytest tests
# No test calls Gemini or a real source: the harness runs the real pipeline with
# scripted models and fake sources (from test_pipeline), into a throwaway folder.
import json

import pytest

import pipeline
from evidence_store import EvidenceStore
from rate_limit import MalformedReply
from sp201 import evaluate
from test_pipeline import BAD_DRAFT
from test_pipeline import FAIL
from test_pipeline import FAKE_FETCHERS
from test_pipeline import FIXED_DRAFT
from test_pipeline import PASS
from test_pipeline import PLAN
from test_pipeline import QUESTION
from test_pipeline import call
from test_pipeline import draft
from test_pipeline import script
from test_pipeline import search_calls

# Expects evidence 1's page and a page the fake sources never return
ITEM = {"id": "T01", "type": "technical", "question": QUESTION, "expected_facts": [
    {"fact": "Transformers rely on self-attention.", "source": "https://example.org/1/"},
    {"fact": "Something no source here holds.", "source": "https://example.org/3"},
]}


def factory(synthesizer_replies, fact_checker_replies):
    def build(store):
        agent = pipeline.build_pipeline(store=store, fetchers=FAKE_FETCHERS)
        script(agent, search_calls() + [call("set_model_response", **PLAN)],
               synthesizer_replies, fact_checker_replies)
        return agent
    return build


def run(tmp_path, synthesizer_replies, fact_checker_replies, items=(ITEM,)):
    return evaluate.run_evaluation(list(items), tmp_path / "run",
                                   agent_factory=factory(synthesizer_replies, fact_checker_replies),
                                   echo=lambda line: None)


def test_3_9_1_the_harness_scores_each_report_from_the_database(tmp_path):
    record = run(tmp_path, [BAD_DRAFT, FIXED_DRAFT], [FAIL, PASS])
    result = record["questions"][0]
    assert result["report"] and result["result"] == "pass" and result["error"] is None
    assert result["claims"] == 2
    assert result["iterations_saved"] == 2
    assert result["claim_support_rate"] == 100.0
    assert result["report_support_score"] == 100.0
    assert result["saved_matches_report"]
    assert result["citation_precision"] == 100.0
    # example.org/1 is cited; example.org/3 never came back from a source
    assert result["expected_sources_cited"] == ["example.org/1"]
    assert result["citation_recall"] == 50.0
    # SRS 3.9.5: the Manager's 3 calls, then 2 drafts and 2 checks
    assert result["model_calls"] == 7


def scored(report_score, verifiable, supported, precision, recall, calls, uncited=0):
    return {"report": True, "result": "pass", "error": None, "claims": verifiable,
            "verifiable": verifiable, "supported": supported, "report_support_score": report_score,
            "citation_precision": precision, "citation_recall": recall, "uncited_claims": uncited,
            "summary_sentences": 2, "saved_matches_report": True, "model_calls": calls, "seconds": 10.0}


def test_3_9_1_the_aggregate_pools_claims_and_averages_reports():
    total = evaluate.aggregate([
        scored(100.0, 2, 2, 100.0, 50.0, 6),
        scored(50.0, 8, 4, 50.0, 100.0, 8, uncited=1),
        {"report": False, "error": "Gemini is overloaded right now (503).", "model_calls": 4, "seconds": 5.0},
    ])
    # Pooled: 6 of 10 claims, not the mean of 100% and 50%
    assert total["claim_support_rate"] == 60.0
    assert total["report_support_score"] == 75.0
    assert total["citation_precision"] == 75.0 and total["citation_recall"] == 75.0
    assert total["uncited_claims"] == 1
    assert total["reports"] == 2 and total["errors"] == 1
    # SRS 3.9.5: calls per question count the question that stopped too
    assert total["model_calls"] == 18 and total["model_calls_per_question"] == 6.0


def test_3_9_1_the_run_folder_holds_the_database_reports_and_summary(tmp_path):
    run(tmp_path, [BAD_DRAFT, FIXED_DRAFT], [FAIL, PASS])
    folder = tmp_path / "run"
    saved = json.loads((folder / "results.json").read_text())
    assert saved["aggregate"]["reports"] == 1
    assert (folder / "evaluation.db").exists()
    assert "Support score: 100.0%" in (folder / "reports" / "T01.txt").read_text()
    assert json.loads((folder / "reports" / "T01.json").read_text())["question_id"] == 1
    assert "manager is working" in (folder / "logs" / "T01.log").read_text()
    summary = (folder / "summary.txt").read_text()
    assert "Claim-level support rate (pooled, from the database): 100.0%" in summary
    assert "Uncited claims: 0 (target 0)" in summary
    # One database for the run: both iterations of T01 are in it
    assert EvidenceStore(folder / "evaluation.db").latest_iteration(1) == 2


def test_3_9_3_the_run_records_model_date_temperature_and_thinking(tmp_path):
    run_info = run(tmp_path, [FIXED_DRAFT], [PASS])["run"]
    assert run_info["model_id"] == "gemini-3.5-flash-lite"
    assert run_info["run_date_utc"].endswith("+00:00")
    assert run_info["temperature"].startswith("model default")
    assert run_info["thinking_levels"] == {"manager": "MEDIUM", "synthesizer": "MEDIUM",
                                           "fact_checker": "model default"}
    assert run_info["git_commit"]
    assert run_info["max_sub_questions"] == 3 and run_info["max_revisions"] == 1


def test_3_4_3_a_claim_with_no_stored_evidence_is_counted_as_uncited(tmp_path, monkeypatch):
    monkeypatch.setenv("SP201_MAX_REVISIONS", "0")
    made_up = draft("Longformer scales linearly.")
    made_up["sections"][0]["claims"][1]["evidence_ids"] = [9]
    result = run(tmp_path, [made_up], [PASS])["questions"][0]
    assert result["uncited_claims"] == 1
    assert result["claim_support_rate"] == 50.0 and result["report_support_score"] == 50.0
    # Evidence 1 backs a supported claim; the made-up 9 is cited and backs nothing
    assert result["cited_evidence"] == 2
    assert result["citation_precision"] == 50.0
    assert result["result"] == "fail"


def test_3_4_3_a_fully_cited_report_has_no_uncited_claims_and_its_summary_counted_apart(tmp_path):
    result = run(tmp_path, [FIXED_DRAFT], [PASS])["questions"][0]
    assert result["uncited_claims"] == 0
    assert result["summary_sentences"] == 1


def test_3_9_1_a_reply_cut_off_twice_is_recorded_in_plain_words():
    assert evaluate.explain(MalformedReply()).startswith("Gemini's reply came back cut off or malformed twice")


def test_3_9_1_a_question_that_stops_is_recorded_with_its_reason(tmp_path):
    # The Synthesizer's script is empty, so its model call fails
    result = run(tmp_path, [], [PASS])["questions"][0]
    assert not result["report"]
    assert result["error"].startswith("Unexpected error (IndexError")
    assert "Traceback" in (tmp_path / "run" / "logs" / "T01.log").read_text()


@pytest.mark.parametrize("url, expected", [
    ("https://arxiv.org/abs/1706.03762", "arxiv.org/abs/1706.03762"),
    ("http://arxiv.org/abs/1706.03762v7", "arxiv.org/abs/1706.03762"),
    ("https://arxiv.org/pdf/1706.03762v2", "arxiv.org/abs/1706.03762"),
    ("https://en.wikipedia.org/wiki/Transformer_(deep_learning)",
     "en.wikipedia.org/wiki/transformer_(deep_learning)"),
    ("https://en.wikipedia.org/wiki/Transformer_%28deep_learning%29#History",
     "en.wikipedia.org/wiki/transformer_(deep_learning)"),
    ("https://www.Example.org/page/", "example.org/page"),
])
def test_3_9_1_citation_recall_matches_sources_however_their_address_is_written(url, expected):
    assert evaluate.normalize_url(url) == expected


def test_3_9_1_only_picks_questions_by_id(tmp_path):
    path = tmp_path / "set.json"
    path.write_text(json.dumps({"questions": [ITEM, dict(ITEM, id="T02")]}))
    assert [item["id"] for item in evaluate.load_items(path, "t02")] == ["T02"]
    with pytest.raises(SystemExit, match="T09"):
        evaluate.load_items(path, "T09")
