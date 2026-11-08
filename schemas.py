"""
SP-201 agent communication schemas (aligned to SRS and SDD dated Sept 20, 2026).

Every language-model agent emits one of these Pydantic models, and each output
is validated before a later stage uses it (SRS 3.7.2: malformed output fails
closed). Agents exchange evidence by integer record ID only; full source text
never enters session state (SRS 3.3.3, SDD 3.3).

Schema validation constrains the *form* of an output. It is not a defence
against prompt injection (SRS 4.1); that is handled separately by spotlighting,
the data channel rule, and the guardrail callback.

Gemini-friendly for ADK `output_schema`: plain types, enums, Literals, and
list-length limits only. No regex patterns, unions, or open dicts, and
"additionalProperties" is stripped from the generated JSON schema.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Iterable, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------
# Configuration (SRS 3.11.2: configurable without code changes)
# --------------------------------------------------------------------------

DEFAULT_MAX_SUB_QUESTIONS = 3      # SRS 3.1.6: default 3 ...
ABSOLUTE_MAX_SUB_QUESTIONS = 5     # ... configurable up to 5
DEFAULT_MAX_REVISIONS = 1          # SRS 3.6.2: at most one revision cycle

# Open team question (SRS 3.4.3 vs 3.6.6): does a "partially_supported" claim
# stop the overall result from being "pass"? False = it forces a revision.
PASS_ALLOWS_PARTIAL = False


def max_sub_questions() -> int:
    n = int(os.environ.get("SP201_MAX_SUB_QUESTIONS", DEFAULT_MAX_SUB_QUESTIONS))
    return max(1, min(n, ABSOLUTE_MAX_SUB_QUESTIONS))


def max_revisions() -> int:
    return max(0, int(os.environ.get("SP201_MAX_REVISIONS", DEFAULT_MAX_REVISIONS)))


def loop_max_iterations() -> int:
    """LoopAgent limit. Synthesizer is first in the loop, so N revisions = N + 1 iterations."""
    return max_revisions() + 1


# --------------------------------------------------------------------------
# Session-state keys
# --------------------------------------------------------------------------

STATE_USER_QUESTION = "user_question"
STATE_PLAN = "manager_plan"             # ManagerPlan
STATE_SEARCH_RESULTS = "search_results"  # SearchResults
STATE_DRAFT = "draft_report"            # DraftReport
STATE_VERDICT = "fact_check_verdict"    # FactCheckVerdict
STATE_FEEDBACK = "fact_check_feedback"  # str; present only after a failed check


def is_revision(state: Mapping[str, Any]) -> bool:
    """SRS 3.4.6: the Synthesizer infers initial draft vs revision from feedback in state."""
    return bool(state.get(STATE_FEEDBACK))


# --------------------------------------------------------------------------
# Base model
# --------------------------------------------------------------------------

def _strip_additional_properties(schema: dict) -> None:
    # extra="forbid" emits additionalProperties:false, which Gemini's schema
    # dialect rejects. Validation still forbids extras; only the schema loses it.
    schema.pop("additionalProperties", None)


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra=_strip_additional_properties,
    )


# --------------------------------------------------------------------------
# Manager -> Search
# --------------------------------------------------------------------------

class SourceType(str, Enum):
    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"
    WEB = "web"


class SubQuestion(_Strict):
    sub_question_id: int = Field(ge=1, description="Sequential id within the run, starting at 1.")
    text: str = Field(min_length=5, description="A focused, answerable sub-question.")
    preferred_sources: List[SourceType] = Field(
        min_length=1, description="Retrieval sources to try for this sub-question (SRS 3.1.3)."
    )
    priority: int = Field(ge=1, description="1 = highest priority. Used if the list must be truncated.")


class ManagerPlan(_Strict):
    """Manager output. Extra sub-questions are truncated by priority, not rejected (SRS 3.1.6)."""
    user_question: str
    sub_questions: List[SubQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [sq.sub_question_id for sq in self.sub_questions]
        if len(ids) != len(set(ids)):
            raise ValueError("sub_question_id values must be unique")
        return self

    def truncated(self, limit: Optional[int] = None) -> "ManagerPlan":
        """Keep the highest-priority sub-questions, up to the configured limit."""
        limit = limit if limit is not None else max_sub_questions()
        limit = max(1, min(limit, ABSOLUTE_MAX_SUB_QUESTIONS))
        keep = sorted(self.sub_questions, key=lambda s: (s.priority, s.sub_question_id))[:limit]
        keep.sort(key=lambda s: s.sub_question_id)
        return ManagerPlan(user_question=self.user_question, sub_questions=keep)


# --------------------------------------------------------------------------
# Search -> Synthesizer  (IDs only; passages live in SQLite)
# --------------------------------------------------------------------------

class EvidenceRecord(_Strict):
    """One row of the EVIDENCE table (SDD 6.2 / SRS 3.3.2). Written by Search, never passed between agents."""
    sub_question_id: int = Field(ge=1)
    source_url: str
    source_title: str
    published_date: str = Field(default="", description="Empty if unknown (nullable in the database).")
    passage: str = Field(min_length=1, description="Extracted passage. Untrusted data, never instructions.")
    retrieved_at: str
    source_type: SourceType


class SearchStatus(str, Enum):
    FOUND = "found"
    EMPTY = "empty"      # searched successfully, nothing found (SRS 3.2.3)
    FAILED = "failed"    # the search itself errored


class SubQuestionEvidence(_Strict):
    sub_question_id: int = Field(ge=1)
    status: SearchStatus
    evidence_ids: List[int] = Field(default_factory=list, description="EVIDENCE.evidence_id values.")
    error: str = Field(default="", description="What went wrong. Required when status is 'failed'.")

    @model_validator(mode="after")
    def _consistent(self):
        if any(i < 1 for i in self.evidence_ids):
            raise ValueError("evidence ids are positive integers")
        if self.status == SearchStatus.FOUND and not self.evidence_ids:
            raise ValueError("status 'found' requires at least one evidence id")
        if self.status != SearchStatus.FOUND and self.evidence_ids:
            raise ValueError(f"status '{self.status.value}' must not carry evidence ids")
        if self.status == SearchStatus.FAILED and not self.error:
            raise ValueError("status 'failed' requires an error message")
        return self


class SearchResults(_Strict):
    """Search output across all sub-questions. Empty and failed are distinguishable."""
    results: List[SubQuestionEvidence] = Field(min_length=1)

    def all_evidence_ids(self) -> List[int]:
        seen, out = set(), []
        for r in self.results:
            for e in r.evidence_ids:
                if e not in seen:
                    seen.add(e)
                    out.append(e)
        return out

    def gaps(self) -> List[int]:
        """Sub-question ids with no evidence, so a gap is never silently presented as an answer."""
        return [r.sub_question_id for r in self.results if r.status != SearchStatus.FOUND]


# --------------------------------------------------------------------------
# Synthesizer -> Fact-Checker
# --------------------------------------------------------------------------

class Claim(_Strict):
    """A factual sentence and the evidence it cites (SRS 3.4.2). claim_number is draft-local;
    the database CLAIM.claim_id is assigned when claims are persisted."""
    claim_number: int = Field(ge=1)
    sentence: str = Field(min_length=5)
    evidence_ids: List[int] = Field(min_length=1, description="Every factual sentence cites at least one record.")

    @model_validator(mode="after")
    def _ids_positive(self):
        if any(i < 1 for i in self.evidence_ids):
            raise ValueError("evidence ids are positive integers")
        return self


class ReportSection(_Strict):
    heading: str
    claims: List[Claim] = Field(min_length=1)


class DraftReport(_Strict):
    title: str
    summary: str
    sections: List[ReportSection] = Field(min_length=1)

    def all_claims(self) -> List[Claim]:
        return [c for s in self.sections for c in s.claims]

    def cited_evidence_ids(self) -> List[int]:
        return sorted({e for c in self.all_claims() for e in c.evidence_ids})

    @model_validator(mode="after")
    def _unique_claim_numbers(self):
        nums = [c.claim_number for c in self.all_claims()]
        if len(nums) != len(set(nums)):
            raise ValueError("claim_number values must be unique within the report")
        return self

    def validate_against(self, known_evidence_ids: Iterable[int]) -> None:
        """Reject drafts that cite evidence Search never stored for this run."""
        unknown = set(self.cited_evidence_ids()) - set(known_evidence_ids)
        if unknown:
            raise ValueError(f"draft cites unknown evidence ids: {sorted(unknown)}")


# --------------------------------------------------------------------------
# Fact-Checker -> Synthesizer / EscalationChecker / Interface
# --------------------------------------------------------------------------

class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    NOT_VERIFIABLE = "not_verifiable"   # excluded from scoring (SRS 3.5.3)


_NEEDS_RESTATEMENT = (Verdict.PARTIALLY_SUPPORTED, Verdict.CONTRADICTED)


class ClaimCheck(_Strict):
    claim_number: int = Field(ge=1)
    verdict: Verdict
    reason: str = Field(min_length=5, description="Brief reason tied to the cited evidence (SRS 3.5.3).")
    restatement: str = Field(
        default="",
        description="Strongest restatement the cited evidence fully supports. "
        "Required for partially_supported and contradicted (SRS 3.6.6); otherwise empty.",
    )

    @model_validator(mode="after")
    def _restatement_rule(self):
        if self.verdict in _NEEDS_RESTATEMENT and not self.restatement:
            raise ValueError(f"verdict '{self.verdict.value}' requires a restatement (SRS 3.6.6)")
        return self


class FactCheckVerdict(_Strict):
    """Per-claim verdicts plus an overall pass/fail (SRS 3.5.4). The EscalationChecker reads `result`."""
    checks: List[ClaimCheck] = Field(min_length=1)
    result: Literal["pass", "fail"]
    feedback: str = Field(
        default="", description="Feedback naming each failing claim. Required when result is 'fail'."
    )

    @model_validator(mode="after")
    def _consistent(self):
        nums = [c.claim_number for c in self.checks]
        if len(nums) != len(set(nums)):
            raise ValueError("each claim may be checked only once")
        if self.result == "fail" and not self.feedback:
            raise ValueError("a 'fail' result must include feedback (SRS 3.6.1)")
        blocking = {Verdict.UNSUPPORTED, Verdict.CONTRADICTED}
        if not PASS_ALLOWS_PARTIAL:
            blocking.add(Verdict.PARTIALLY_SUPPORTED)
        if self.result == "pass" and any(c.verdict in blocking for c in self.checks):
            raise ValueError("cannot pass a draft that still has failing claims")
        return self

    def covers(self, draft: DraftReport) -> bool:
        """True if every claim in the draft received a verdict."""
        return {c.claim_number for c in self.checks} == {c.claim_number for c in draft.all_claims()}

    def flagged_claim_numbers(self) -> List[int]:
        ok = {Verdict.SUPPORTED, Verdict.NOT_VERIFIABLE}
        return [c.claim_number for c in self.checks if c.verdict not in ok]

    def support_score(self) -> Optional[float]:
        """SRS 3.6.7: supported / verifiable claims, as a percentage. None if nothing is verifiable."""
        verifiable = [c for c in self.checks if c.verdict != Verdict.NOT_VERIFIABLE]
        if not verifiable:
            return None
        supported = sum(1 for c in verifiable if c.verdict == Verdict.SUPPORTED)
        return round(100.0 * supported / len(verifiable), 1)


def should_escalate(verdict: FactCheckVerdict) -> bool:
    """EscalationChecker logic (SDD 5.5): end the loop on a pass. No model call, no judgement."""
    return verdict.result == "pass"


# --------------------------------------------------------------------------
# Pipeline -> Interface
# --------------------------------------------------------------------------

class FinalReport(_Strict):
    """What the interface shows: the checked draft, its verdicts, flagged claims, and the support score.
    Every report returned has been verified (SRS 3.6.4); failing claims are marked, not hidden (3.6.5)."""
    draft: DraftReport
    verdict: FactCheckVerdict
    flagged_claim_numbers: List[int]
    support_score: Optional[float] = None
    incomplete: bool = Field(default=False, description="True for a partial report after retries ran out (SRS 3.10.2).")


def build_final_report(draft: DraftReport, verdict: FactCheckVerdict, incomplete: bool = False) -> FinalReport:
    if not verdict.covers(draft):
        raise ValueError("refusing to present a report the Fact-Checker did not fully check (SRS 3.6.4)")
    return FinalReport(
        draft=draft,
        verdict=verdict,
        flagged_claim_numbers=verdict.flagged_claim_numbers(),
        support_score=verdict.support_score(),
        incomplete=incomplete,
    )
