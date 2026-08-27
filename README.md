# Agent Reliability and Regression Testing Framework
# Agent Reliability & Regression Testing Framework

A framework that answers one question: **did my AI agent get worse after I changed something?**

When you tweak a prompt, swap a model, or change a tool, an LLM-powered feature can silently get worse in ways that are easy to miss just by eyeballing a few outputs. This framework automates the check run a fixed set of test questions through two versions of an agent, score every answer with an ensemble of LLM judges, and use real statistics (not vibes) to say whether one version is genuinely worse.

## The headline finding

While building the evaluation itself, the framework caught a bug in its own judges: when the agent honestly said *"I don't know, let me get a human"* to a question it had no data for, the judges scored it **1.3/10**. When the agent confidently invented a plausible-sounding but false answer to the same kind of question, the judges scored it **9.3/10**.

The judges were rewarding hallucination over honesty. A dedicated **groundedness judge** was added specifically to catch this checking not just "is the answer plausible" but "is the answer actually supported by what the agent's tools returned."

## How it works

```
scenarios/*.yaml  →  agent under test  →  traces/*.json
                                              ↓
                            4 LLM judges (parallel, LangGraph)
                     correctness · tool-efficiency · completeness · groundedness
                                              ↓
                                    confidence-weighted score
                                              ↓
                          Fisher's exact test (strong vs weak pass rates)
                                              ↓
                    significant regression? → Slack alert + CI failure
```

1. **Scenarios**: a fixed set of test questions (some hand-written, some LLM-generated) run against both a baseline ("strong") and a modified ("weak") version of the agent
2. **Judges**: 4 LLM judges score every answer in parallel, each returning a score, a reason, and a confidence level
3. **Statistics**: scores are combined into a confidence-weighted average; Fisher's exact test compares pass rates between the two versions to determine if a difference is statistically real or just noise
4. **Automation**: the full pipeline runs in CI on every push, posts a Slack alert if a real regression is detected, and fails the build

## Key features

- **Any agent, not just this one.** `targets.yaml` maps a "strong" and "weak" variant to any Python function  the framework doesn't care if it's a LangGraph agent, a one-line summarizer, or anything else that takes a question and returns an answer.
- **Confidence-weighted scoring.** Judges report how confident they are in each verdict; a shaky "score: 2, confidence: 0.2" counts less than a firm "score: 9, confidence: 0.98"  instead of averaging every judge equally regardless of certainty.
- **Auto-generated test scenarios.** `generate_scenarios.py` uses an LLM to write new test questions from a description of the agent's tools, growing the test suite without every question being hand-authored by the same person who built the agent.
- **CI-integrated with real alerts.** A GitHub Actions workflow runs the full pipeline on every push and posts to Slack if a regression is found not just a red X nobody checks.
- **Live investigator dashboard.** A small Flask app with an LLM-powered Q&A interface ask plain-English questions like *"why did the weak agent regress?"* and get an answer grounded in the actual trace and judge data, not a canned response.

## Tech stack

**Core:** Python, LangGraph, LangChain, Groq API (structured LLM outputs)
**Data:** SQLite (agent's backend), Pydantic (schema validation)
**Statistics:** SciPy (Fisher's exact test)
**Web:** Flask, vanilla HTML/CSS/JS
**Automation:** GitHub Actions, Slack webhooks

## Project structure

```
test_agent.py / agent_v2_weak.py   agent under test (baseline vs modified prompt)
store.py                            SQLite-backed fake backend (orders, FAQs)
scenarios/*.yaml                    fixed test questions
targets.yaml                        maps variants to any agent/function to test
run_suite.py                        runs a variant through all scenarios
judges/                             4 LLM judges (correctness, completeness, tool-efficiency, groundedness)
run_judges.py                       parallel judge orchestration (LangGraph)
detect_regressions.py               Fisher's exact test + Slack alert on regression
generate_scenarios.py               LLM-generated test scenarios
pipeline.py                         runs the full flow end to end
alerts.py                           Slack webhook integration
app.py / templates/index.html       live investigator dashboard
.github/workflows/                  CI automation
```

## Setup

```bash
pip install -r requirements.txt
python store.py                     # seed the database

# Add to .env:
# GROQ_API_KEY=your_key
# SLACK_WEBHOOK_URL=your_webhook (optional)

python pipeline.py                  # run the full pipeline
python app.py                       # start the live dashboard at localhost:5000
```

## Results

Initial validation run (10 hand-written scenarios): the framework detected a statistically significant regression between the baseline and degraded-prompt agent (**9-10/10 vs 4/10 pass rate, p = 0.0108**), driven almost entirely by groundedness failures, the weak agent hallucinated specific, plausible-sounding facts on 6 of 10 questions when its tools returned no data.

Results shift somewhat run-to-run given LLM output is non-deterministic and the current scenario set is still small this is a known, honestly-documented limitation, not a hidden one. Running the pipeline multiple times and aggregating, or growing the scenario count further, are the direct fixes (see below).

## Known limitations / future work

- **Sample size.** 10-25 scenarios is enough to demonstrate the framework works, not enough for high statistical confidence on every run. Next: aggregate across multiple runs, or grow the scenario set substantially.
- **Non-determinism.** LLM agents don't give identical answers twice. A single pipeline run can vary; averaging across N runs would give a more stable signal.
- **Toy backend.** The agent's data (orders, FAQs) is a small seeded SQLite database, not a real production system — intentional, since the framework is what's being demonstrated, not the agent.
- **Single alert channel.** Slack only, for now  email/PagerDuty integration would be a natural extension for a real team setting.

