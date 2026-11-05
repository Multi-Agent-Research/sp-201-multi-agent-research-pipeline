# SP-201 Multi-Agent Research and Fact-Checking Pipeline

CS 4850 Senior Project, Section 01, Fall 2026, Kennesaw State University.
Advisor: Professor Sharon Perry.

Four language-model agents, orchestrated with the Google Agent Development Kit
(ADK), take one research question, gather source material, draft a cited
report, and check every claim in that report against the evidence it cites.
The pipeline runs on a Gemini Flash model through the Google AI Studio free
tier.

This is coursework in development. The agent data contract, the Manager agent
and the search fetchers are in this repository and tested. The Synthesizer,
the Fact-Checker, the revision loop and the interfaces are not. The Manager is
not connected to Search, so the pipeline cannot answer a question end to end
today.

## Components

| Component | What it does | Owner |
|---|---|---|
| Manager | Splits the question into sub-questions and picks sources for each | Jackson |
| Search | Retrieves from Wikipedia, arXiv and web search; writes passages to SQLite | Raul |
| Synthesizer | Drafts a structured report from stored evidence only | Samit |
| Fact-Checker | Judges each claim against the evidence it cites | Jackson |
| EscalationChecker | Ends the revision loop when the Fact-Checker passes the draft | Raul |
| Interface | Takes the question; shows the report, flagged claims and support score | Samit (web), Vansh (CLI) |

The first four are LLM agents. The EscalationChecker reads the Fact-Checker's
pass/fail result and makes no model call and no judgment of its own. The
Interface displays what the pipeline produced and verifies nothing itself.

## Architecture

```
SequentialAgent
  Manager                        # holds AgentTool(Search)
  LoopAgent (max_iterations=2)
    Synthesizer                  # first in the loop, not before it
    Fact-Checker
    EscalationChecker            # control only, no model call
```

One question goes to the Manager, which splits it into at most three
sub-questions and calls Search once for each. Search writes passages to SQLite
and returns record IDs; full source text never enters session state. The loop
then drafts, checks and escalates, at most twice, which is exactly one
revision. Every iteration ends on a verification, so no path through the
system returns a report the Fact-Checker has not checked.

**The Synthesizer is first inside the loop on purpose.** The obvious layout
puts it before the loop, with the loop running Fact-Checker then Synthesizer.
That ends a failing run on a revision no Fact-Checker has seen, and that
unverified draft is what the user gets. Drafting first means every iteration
ends on a check, so a run that exhausts the limit still returns a verified
draft with its failing claims marked. It is also one call cheaper on the worst
path.

## Setup

Python 3.10 or later, developed and tested on 3.12. The macOS system Python is
3.9 and will not work. A Google AI Studio API key is needed for live runs but
not for the tests.

```
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` in the project root and run every command from there. Git
ignores it. Variables already set in the shell win over the file.

```
GOOGLE_API_KEY=your-key-here
SP201_MODEL_ID=your-model-id     # not every model is available to every key
SP201_MAX_SUB_QUESTIONS=3        # default 3, ceiling 5
SP201_MAX_REVISIONS=1
```

Verify with `python -m pytest tests`, which needs no key and reports
`45 passed, 1 warning`. The warning is an ADK deprecation notice we expect
from choosing the classic API. Then `python manager_agent.py` prints the
loaded configuration and `Sample plan: 4 sub-questions in, 3 out`, without
calling Gemini.

## Design decisions

Reversing one of these is a team decision, not a quiet refactor.

| Decision | Reason |
|---|---|
| Classic ADK API (`SequentialAgent`, `LoopAgent`) at a pinned version | The graph API cannot nest a loop inside a sequence |
| Workers exposed as `AgentTool`, not sub-agents | Control returns to the Manager after every call instead of the worker keeping it |
| Evidence stored in SQLite and passed between agents by integer ID | Keeps full text out of the context window and keeps every claim traceable to the retrieval that produced it |
| Model ID read from the environment, never written at a call site | Models get retired and access to them changes; re-pointing should not need a code change |
| One revision cycle, then stop; unsupported claims are flagged, not dropped | Bounded call count per question |
| Wikipedia and academic APIs primary, web search as fallback | Free web-search libraries are unofficial and get rate-limited or blocked |
| Fact-Checker is an LLM agent; hosted model by default, local entailment classifier held in reserve | Keeps the option of moving the most call-hungry step off the daily cap |
| Synthesizer runs first inside the revision loop | See above |

## Constraints

**Call budget.** The free tier caps requests per minute, input tokens per
minute, and requests per day, per Google Cloud project, resetting at midnight
Pacific. A question with three sub-questions is estimated at 12 to 14 model
calls rather than the six a per-agent count suggests, because Search runs once
per sub-question and each tool exchange costs the Manager extra calls. Treat
that as an estimate until a full run measures it. A 20-query evaluation is
roughly 250 calls, so the daily cap binds before the per-minute one.

**Prompt injection.** Retrieved web content is the main untrusted input.
Passages are wrapped in explicit delimiters, supplied through the data channel
and never the system instruction, and inspected by a guardrail callback before
reaching a model. These measures reduce the risk; they do not eliminate it,
and they are designed but not yet implemented.

**Improper output handling.** Every agent output is validated against a
Pydantic schema before the next stage uses it, so bad output fails closed.
Schema validation constrains the form of an output. It is not a defense
against prompt injection, and nothing in the code or docs should claim it is.

Credentials come from the environment and are never committed. On the free
tier, prompts and responses may be used by Google to improve its products, so
no confidential or personal data goes into evaluation queries.

## Conventions

- `requirements.txt` pins `google-adk`. Do not float the version.
- No credentials in the repository. Use environment variables and keep `.env`
  in `.gitignore`.
- Import session-state keys from `schemas.py`. Do not type `"manager_plan"`
  at a call site.
- Test names carry the requirement number with underscores, so
  `python -m pytest tests -k 3_1_6` runs every test for that requirement.
- Changes reach `main` through a reviewed pull request, never a direct push.

`schemas.py` is authoritative for the agent I/O contract. Where this file
disagrees with it, `schemas.py` is right and this file needs fixing.
