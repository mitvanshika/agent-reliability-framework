# Agent Reliability & Regression Testing Framework

A framework that answers one question: **did my AI agent get worse after I changed something?** And, for any system you plug in, *what* a specific change did and *why*.

When you tweak a prompt, swap a model, or change a tool, an LLM-powered feature can silently get worse in ways that are easy to miss just by eyeballing a few outputs. This framework automates the check run a fixed set of test questions through two versions of an agent, score every answer with an ensemble of LLM judges, and use real statistics (not vibes) to say whether one version is genuinely worse.

## The headline finding

While building the evaluation itself, the framework caught a bug in its own judges: when the agent honestly said *"I don't know, let me get a human"* to a question it had no data for, the judges scored it **1.3/10**. When the agent confidently invented a plausible-sounding but false answer to the same kind of question, the judges scored it **9.3/10**.

The judges were rewarding hallucination over honesty. A dedicated **groundedness judge** was added specifically to catch this checking not just "is the answer plausible" but "is the answer actually supported by what the agent's tools returned."

## Plug in your system in 4 steps

The support bot in this repo is only the first system tested here. Anything that turns an input into an output can be
plugged in: a chatbot, a RAG pipeline, a summarizer, a classifier, an agent with or without tools.
`examples/summarizer/` is a complete example that needed **zero framework edits** (`tests/test_summarizer_example.py`
checks that from git).

1. **Expose one function**: `answer(question) -> str`, or a dict with `"answer"` and optionally `"tools_used"`,
   `"tool_outputs"`, `"usage"`. Give it a `settings=` parameter if experiments should vary it.
2. **Add it to `targets.yaml`** (copy the `summarizer` entry):
   ```yaml
   summarizer:
     module: examples.summarizer.summarizer
     function: summarize
     description: "summarises a short news article in at most two sentences"
     judge_set: generic_v1                                  # domain-neutral judge wording
     judges: [correctness_gt, completeness, groundedness]   # no tools: tool judges are N/A
     pass_threshold: 7
     scenarios: examples/summarizer/scenarios
   ```
3. **Write scenarios**: one YAML file per test with `id`, `question` and `expected_answer` (what a correct output
   must contain, and what it must not claim).
4. **Check, then run**:
   ```
   python check_target.py summarizer               # conformance checks, no API calls (--call: one test call)
   python experiment_pipeline.py examples/summarizer/summarizer_baseline.yaml
   python promote_baseline.py runs/<that run>      # once; every later run is compared with it
   ```

## Architecture: two tracks

```
                       targets.yaml   (what to test: module + function [+ accepts_settings])
                             │
        ┌────────────────────┴─────────────────────┐
  TRACK 1: demo regression gate              TRACK 2: experiments (any system)
  two fixed agents: strong vs weak            experiments/<name>.yaml  (extends + settings)

  pipeline.py                                 experiment_pipeline.py
   run_suite.py   → traces/                    run_experiment.py  → runs/<exp>_<ts>/traces
   run_judges.py  (4 classic judges)           judge_run.py       (5 judges + tool_match, answer key)
   detect_regressions.py                       compare.py         vs baselines/<target>.json
     Fisher + Mann-Whitney, strong vs weak       McNemar + Wilcoxon, per-question noise bands
                                               diagnose.py        cause + next experiment (--draft)
                                               promote_baseline.py   (you decide, never automatic)
                                               export_finetune.py    (train split only)
  CI: regression-check.yml                    CI: experiment-check.yml  (PRs, call budget)
  dashboard: /                                dashboard: /experiments
```

**Track 1** is the original demo: two hard-coded agents, one deliberately degraded, and a statistical gate that proves the regression is caught. **Track 2** is the general tool: describe any version of any system as settings, run it, compare it question by question against a promoted baseline, and get a diagnosis of what went wrong.

## How it works (Track 1)

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

## The experiment workflow (Track 2)

One command runs the whole loop:

```bash
python experiment_pipeline.py experiments/exp_05_embeddings.yaml [--scenarios id ...] [--yes] [--alert]
#   run_experiment -> judge_run -> compare -> diagnose; exit 0 ok, 1 DECLINED, 2 a step failed
```

Or step by step:

```bash
python run_experiment.py experiments/baseline.yaml          # 1. run the scenarios -> runs/baseline_<ts>/
python judge_run.py runs/baseline_<ts>                      # 2. judge every answer
python promote_baseline.py runs/baseline_<ts> --note "..."  # 3. make it the baseline (asks first, never automatic)
python calibrate_noise.py --experiment experiments/baseline.yaml --runs 3   # 4. measure natural wobble (once)
python run_experiment.py experiments/exp_01_weak_prompt.yaml
python judge_run.py runs/exp_01_weak_prompt_<ts>
python compare.py runs/exp_01_weak_prompt_<ts>              # 5. question-by-question vs the baseline
python diagnose.py runs/exp_01_weak_prompt_<ts> --draft     # 6. why it got worse + a draft next experiment
```

| Command | Reads | Writes |
|---|---|---|
| `run_experiment.py` | `experiments/*.yaml`, `scenarios/` | `runs/<exp>_<ts>/traces/`, `run_meta.json` |
| `judge_run.py` | the run's traces, the answer key in `scenarios/` | `judged/`, `summary.json` |
| `promote_baseline.py` | a judged run | `baselines/<target>.json` (+ `baselines/history/`) |
| `calibrate_noise.py --experiment` | an experiment | `noise/<target>_noise.json` |
| `compare.py` | a judged run + its baseline + noise file | `comparison.json` |
| `diagnose.py` | `comparison.json`, `runs/` | `diagnosis.json`, optional `experiments/draft_*.yaml` |
| `experiment_pipeline.py` | an experiment | all of the above for one run + `pipeline.json` |

Every command that calls an API first prints an estimate of the number of calls and asks for confirmation (`--yes` skips it).

## Experiments

The strong/weak demo shows *that* a regression is caught. Experiments measure *what a specific change did* to any system plugged in via `targets.yaml`.

An experiment is a small YAML file describing one version of the system:

```yaml
# experiments/exp_01_weak_prompt.yaml
name: exp_01_weak_prompt
description: "Same as baseline but with the weak system prompt"
extends: baseline            # start from baseline, override only what's below
settings:
  prompt: prompts/weak.txt
```

`settings` is passed straight to the target (`fn(question, settings=...)` when the target has `accepts_settings: true` in `targets.yaml`); the harness never interprets it, so your own system can use whatever keys it needs (`chunk_size`, `top_k`, ...). The built-in `configurable` target (`agent_configurable.py`) is the demo agent built from settings: `experiments/baseline.yaml` is provably identical to the strong agent (`python tests/test_prompt_equivalence.py`).

The demo agent's settings: `model`, `temperature`, `prompt` (a text file), `docstrings` (tool descriptions), `recursion_limit`, and `retrieval: {method: substring | bm25 | embeddings, top_k, min_score}` (`retrieval.py`; embeddings need `pip install -r requirements-rag.txt`). The examples each change exactly one thing vs baseline: `exp_01_weak_prompt`, `exp_02_vague_docstrings`, `exp_04_bm25`, `exp_05_embeddings`.

Each run gets its own folder `runs/<experiment>_<timestamp>/` with per-scenario traces (answer, tools, latency, token usage, error type, full conversation), `run_meta.json` (resolved settings, what changed vs the parent, git commit) and, after judging, `judged/` and `summary.json` (pass rate, per-judge means, tokens, p95 latency, errors). Add an optional `pricing.yaml` (USD per 1M tokens per model, filled in by you) to get an estimated cost.

## Judges and the answer key

Experiment runs are judged with the **experiment judge set** (`judge_experiment.py`): five LLM judges in parallel plus a code check.

- **correctness_gt**: the answer vs the scenario's `expected_answer` (falls back to the classic correctness judge if a scenario has none)
- **completeness**, **groundedness**, **tool_efficiency**: as in Track 1
- **context_relevance**: did the tools / retrieval return what was needed? N/A (left out of the score, not counted as 0) when nothing was retrieved
- **tool_match**: plain code comparing the tools called with `expected_tools`; shown, never part of `final_score`

Scenarios carry the answer key as optional fields (`expected_answer`, `expected_tools`; old code ignores them). Scores are only comparable when made with the same judge set and the same answer key, so `compare.py` refuses otherwise: re-judge or re-promote the baseline.

## Baselines and comparison

A judged run can be promoted to the **baseline** for its target; every later run is compared against it, question by question.

```bash
python promote_baseline.py runs/<run> --note "why this is the baseline"   # shows summary + diff vs current, asks first
python compare.py runs/<run> [--alpha 0.05] [--alert]                      # exit 1 if DECLINED
```

`baselines/<target>.json` (commit it) stores the config, git commit, note and per-scenario scores; the previous one is kept in `baselines/history/`. Promotion is refused for runs that are not fully judged or have judge errors (`--force` to override).

`compare.py` pairs scenarios by id and marks each one ▲ / ▼ / ═ depending on whether the score moved more than that scenario's **noise band** (from `noise/<target>_noise.json` produced by `calibrate_noise.py`, default 1.0). Two paired tests decide the verdict:

- **McNemar exact** on pass/fail flips (did questions that passed start failing?)
- **Wilcoxon signed-rank** on the paired scores (did scores shift, even without crossing 7?)

| Verdict | Meaning |
|---|---|
| DECLINED | a test shows a significant decline (exit code 1, `--alert` posts to Slack) |
| IMPROVED | significant improvement and no question went ▼ |
| IMPROVED WITH REGRESSIONS | significant improvement, but some questions went ▼ |
| NO REAL DIFFERENCE | nothing significant (any ▼ questions are still listed) |
| NOT ENOUGH DATA | fewer than 5 paired questions |

When the two tests disagree, the more cautious verdict wins. A crashed answer counts as a fail with score 0; a judge error is excluded (not the agent's fault). The report also shows token, latency, cost and error deltas, and is saved as `runs/<run>/comparison.json`. If the noise file was calibrated on a different config or judge set than the baseline, compare warns that it is **stale** and falls back to the default band. Exit codes: 0 ok, 1 DECLINED, 2 refused (targets, judge sets or answer key differ).

## Diagnosis

`diagnose.py runs/<run>` explains every ▼ question with the most likely cause, its evidence (the judge deltas) and a suggested next experiment, checking in this order:

| Signal | Cause |
|---|---|
| rate_limit / timeout error | INFRA: not agent quality, re-run |
| auth error | CONFIG: check the key / model |
| loop_limit error | TOOLS/PROMPT: stuck calling tools |
| context_relevance dropped | RAG: retrieval fetched the wrong information |
| context fine, correctness_gt / groundedness dropped | PROMPT: had the information, used it badly |
| tool_efficiency / tool_match dropped | TOOL DESCRIPTIONS |
| completeness dropped | PROMPT: missed part of the question |
| judges disagreed / low confidence | MEASUREMENT UNSURE: check by hand |
| several judges dropped about equally | UNCLEAR: lists them |

Thresholds are constants at the top of the file. `--draft` writes `experiments/draft_<cause>_<timestamp>.yaml` extending the run's experiment with a TODO on the setting to change; it is never run automatically. Diagnosis also scans `runs/`: a scenario that failed under 3 or more different prompt settings is flagged as a **fine-tuning candidate** (prompting isn't fixing it).

## Plug in your own system

The demo support agent is only an example. To measure changes to your own system (say a Financial Due Diligence RAG pipeline):

1. **Expose one function** that takes the question and your settings:
   ```python
   # my_rag/entry.py
   def answer(question: str, settings: dict) -> dict:
       ...
       return {"answer": text,                                # required
               "tools_used": [...], "tool_outputs": [...],    # optional: retrieved chunks etc.
               "error": None, "error_type": None,             # optional: "rate_limit", "timeout", ...
               "usage": {"input_tokens": n, "output_tokens": m, "total_tokens": n + m},  # optional
               "messages": [...]}                             # optional: full chat, for fine-tune export
   ```
   A plain string return also works (answer only).
2. **Register it** in `targets.yaml`:
   ```yaml
   financial_dd:
     module: my_rag.entry
     function: answer
     accepts_settings: true     # the harness calls answer(question, settings=...)
   ```
3. **Optional: fail fast on bad settings.** Define `validate_settings(settings)` in the same module; `run_experiment.py` calls it before any API call.
4. **Describe versions as experiments.** `settings` is passed through untouched, so use whatever keys your system needs:
   ```yaml
   # experiments/fdd_baseline.yaml
   name: fdd_baseline
   target: financial_dd
   settings: {chunk_size: 512, top_k: 5, prompt: prompts/fdd_v1.txt, model: openai/gpt-oss-120b}
   ```
   ```yaml
   # experiments/fdd_bigger_chunks.yaml
   name: fdd_bigger_chunks
   extends: fdd_baseline
   settings: {chunk_size: 1024}
   ```
5. **Write scenarios** with an answer key (`expected_answer`, optionally `expected_tools`), then run `python experiment_pipeline.py experiments/fdd_baseline.yaml`, promote it, and compare every change against it.

If your system reports `tool_outputs`, the context_relevance judge scores your retrieval. If it reports `usage` and you add a `pricing.yaml`, you get cost per run.

**Per-target settings in `targets.yaml`** (all optional; left out = the behaviour of the support bot):

| key | what it does | default |
|---|---|---|
| `description` | one line about the system; given to the judges (generic_v1, safety), the `scope` addon and `generate_adversarial.py` | none |
| `judge_set` | judge wording: `generic_v1` (domain-neutral, for most systems) or `experiment_v2` (the support-bot wording) | `experiment_v2` |
| `judges` | which judges apply; the others are recorded N/A and left out of final_score | all of the set |
| `pass_threshold` | final_score at or above this counts as a pass | 7 |
| `scenarios` | the target's own scenario folder (its main suite, baseline `baselines/<target>.json`) | `scenarios/` |
| `non_retrieval_tools` | tools whose output is not retrieved information (ignored by context_relevance) | none |
| `tools` | `module:function` returning the tools, opts in to `fuzz_tools.py` | not fuzzed |
| `action_tools` | tools that act: must be `approval.safe_action`, checked by `check_actions.py`, never fuzzed | none |
| `max_steps` | `step_checks.py`: more LLM turns or tool calls per answer than this is a finding | 8 |
| `actions_sandbox` | `module:function(tmpdir)` that points side effects at a temp folder during `check_actions.py` | none |

`python check_target.py <target>` checks all of this before you spend a token.

**Sample-agent only.** These files belong to the demo support bot and are support-specific on purpose; your system does
not need them: `test_agent.py`, `agent_v2_weak.py`, `agent_configurable.py` (with `prompts/`, `docstrings/`,
`scenarios/`), `store.py`, `retrieval.py`, `support_actions.py`, `generate.py`, `export_finetune.py`, and the Track 1 scripts (`pipeline.py`,
`run_judges.py`, `find_failed.py`, `export_for_dashboard.py`, `rejudge_specific.py`, the Track 1 dashboard).

## Making any agent's actions safe

Reading tools are harmless if they run twice. Tools that ACT (send an email, refund, pay, delete) are not: agents
re-run graphs after API errors, models call the same tool twice, users double-click. Two reusable guards, for any
plain function or LangChain tool, nothing tool-specific:

- **`@idempotent`** (`idempotent.py`): acts at most once per request. Key = tool name + normalized arguments +
  request id, claimed in SQLite before acting, so even simultaneous callers can't both act; a repeat gets the stored
  result.
- **`requires_approval` / `safe_action`** (`approval.py`): ask an approver before acting. `safe_action` = approval +
  idempotency in the only safe order:
  1. already done for this request? return the stored result (nobody is asked twice)
  2. ask the approver; a reject is passed to the model ("not approved ... Nothing was done") and NOT stored
  3. claim the key, act once, store the result

```python
from approval import safe_action
refund = safe_action(issue_refund_tool, approver="dashboard",          # or "cli", "deny", a function, "langgraph"
                     describe=lambda a: f"Refund order {a['order_id']}")
```

| approver | how it decides |
|---|---|
| `deny` (default) | always no: an unattended run never acts |
| `cli` | asks on the terminal; only `y` / `yes` approves |
| `dashboard` | shows it on `/approvals` (Approve / Reject); no decision within the timeout = reject. Deciding needs `APPROVALS_TOKEN` (header `X-Approval-Token`) or a request from the same machine |
| a function | `fn(request) -> bool` or `Decision`: a chat bot, a policy check |
| `langgraph` | pauses a LangGraph run at a checkpoint (`interrupt`); `run_with_approvals(app, input, approver, config)` asks the approver and resumes |

Anything unclear fails closed: an approver that raises or answers "maybe" is a reject. Every decision is logged
(`approval_log`), including replays that needed none. The request id comes from `request_context(...)`,
`run_experiment.py` sets one per scenario, and inside LangGraph the run's `request_id` / `thread_id` is used.

**LangGraph detail that matters:** LangGraph hands resume values to `interrupt()` by position within a step, and an
action that is already done no longer interrupts, so positions shift: in testing, a rejected refund received another
action's approval and was executed. Fixed: every decision carries its action's key, is filed by key, and each action
looks up its own; resume values without a key are refused. Stress-tested on all 8 approve/reject patterns of 3
parallel actions.

**Declare and check them per target:**
```yaml
my_target:
  tools: my_module:get_tools
  action_tools:
    - name: send_email
      test_args: {to: "test@example.com", body: "hi"}
  actions_sandbox: my_module:sandbox     # fn(tmpdir) -> restore(): side effects go to a temp folder during checks
```
```
python check_actions.py my_target       # no LLM: protected, blocked by deny, reject = nothing, approve = once,
                                        # retry = still once, new request = again, 5 concurrent = once, LangGraph
```
`fuzz_tools.py` never calls an action tool: junk input must not trigger a real action.

**First user: the support bot.** `experiments/exp_09_actions.yaml` (`actions: enabled`, one change from the baseline)
adds `issue_refund(order_id)` from `support_actions.py`: it refuses orders older than 30 days and writes to its own
`refunds.db` ledger, never to `store.db`. Its approver is the `approver` setting (`deny` by default, `cli` or
`dashboard`). The baseline has no action tool and is unchanged. `python check_actions.py configurable` passes every
check. Limits: a LangGraph pause lasts while the process runs (MemorySaver; install `langgraph-checkpoint-sqlite` to
survive restarts), and an action interrupted by a crash stays "in progress" and is refused until cleared by hand.

## Fine-tuning data

`python export_finetune.py [--dry-run] [--yes]` turns clearly good answers into chat-format training data (`finetune/train_<ts>.jsonl`, with tool calls and tool schemas). Only scenarios marked `split: train` are ever exported (missing = test), and only answers with final_score ≥ 9, no errors, no judge disagreement or low confidence, judged against the current answer key. Test scenarios are refused: they are the exam `compare.py` grades, and training on them would let the model memorise the answers. A review file (`finetune/review_<ts>.md`) is always written first; the JSONL only after you confirm.

## CI

- **`regression-check.yml`** (Track 1): runs `pipeline.py` on pushes to master, PRs, nightly and on demand; fails on a significant strong-vs-weak regression.
- **`experiment-check.yml`** (Track 2): on PRs that touch prompts, docstrings, experiments, scenarios, baselines, the agent, judges or retrieval, runs `experiment_pipeline.py` on `experiments/baseline.yaml` and compares it with the committed `baselines/<target>.json`. Fails if DECLINED; writes the report to the job summary and uploads `comparison.json` as an artifact. Without a `GROQ_API_KEY` secret (e.g. PRs from forks) it skips with a notice. The repo variable `EXPERIMENT_CI_SCENARIOS` sets the call budget (space-separated scenario ids, or `all`; the default is 10 scenarios).

## Dashboards

`python app.py`, then:

- **`/`**: the Track 1 investigator, with plain-English questions answered from the strong/weak trace and judge data (the system prompt is built on the server; rate-limited).
- **`/experiments`**: every run with its pass rate, score, tokens, cost and verdict; a quality-vs-cost scatter (one dot per run); click a run for its ▲/▼/═ comparison and diagnosis. Read-only from `runs/` and `baselines/`, no API calls.

## Key features

- **Any agent, not just this one.** `targets.yaml` maps a "strong" and "weak" variant to any Python function  the framework doesn't care if it's a LangGraph agent, a one-line summarizer, or anything else that takes a question and returns an answer.
- **Experiments for any system.** Describe a version as settings, inherit from a baseline, and the framework records exactly what changed, runs it, and compares it question by question.
- **Confidence-weighted scoring.** Judges report how confident they are in each verdict; a shaky "score: 2, confidence: 0.2" counts less than a firm "score: 9, confidence: 0.98"  instead of averaging every judge equally regardless of certainty.
- **Answer key + retrieval judging.** Correctness is checked against the store's own answer key, and context relevance scores the retrieval step separately from the answer.
- **Measured noise, not guessed.** `calibrate_noise.py` measures how much scores wobble when nothing changes, so a drop is only called real if it is bigger than that.
- **Diagnosis.** Every regression gets a likely cause and a suggested next experiment.
- **Auto-generated test scenarios.** `generate.py` uses an LLM to write new test questions from a description of the agent's tools, growing the test suite without every question being hand-authored by the same person who built the agent.
- **CI-integrated with real alerts.** A GitHub Actions workflow runs the full pipeline on every push and posts to Slack if a regression is found not just a red X nobody checks.
- **Live investigator dashboard.** A small Flask app with an LLM-powered Q&A interface ask plain-English questions like *"why did the weak agent regress?"* and get an answer grounded in the actual trace and judge data, not a canned response.

## Tech stack

**Core:** Python, LangGraph, LangChain, Groq API (structured LLM outputs)
**Data:** SQLite (agent's backend), Pydantic (schema validation)
**Retrieval:** BM25 (in-house), fastembed + bge-small-en-v1.5 (optional, ONNX)
**Statistics:** SciPy (Fisher's exact, Mann-Whitney U, McNemar exact, Wilcoxon signed-rank)
**Web:** Flask, vanilla HTML/CSS/JS
**Automation:** GitHub Actions, Slack webhooks

## Project structure

```
test_agent.py / agent_v2_weak.py   Track 1 agents (baseline vs modified prompt)
store.py                            SQLite-backed fake backend (orders, FAQs)
scenarios/*.yaml                    fixed test questions (+ optional answer key, split)
targets.yaml                        maps variants / targets to any function to test
run_suite.py                        Track 1: runs a variant through all scenarios
judges/                             LLM judges (classic 4 + correctness_gt, context_relevance) + tool_match
run_judges.py                       parallel judge orchestration (LangGraph)
detect_regressions.py               Track 1: Fisher's exact test + Slack alert on regression
generate.py                         LLM-generated test scenarios
pipeline.py                         Track 1: runs the full flow end to end
alerts.py                           Slack webhook integration
calibrate_noise.py                  measures score noise (variants or --experiment)

experiments/*.yaml                  Track 2: versions of a system as settings (extends)
prompts/, docstrings/               prompt and tool-description files used by experiments
agent_configurable.py               the demo agent built from settings
retrieval.py                        substring / bm25 / embeddings FAQ search
experiment_config.py                loads experiments, resolves extends, "what changed"
run_experiment.py                   runs one experiment -> runs/<exp>_<ts>/
judge_experiment.py / judge_sets.py experiment judge set + judge-set identities
judge_run.py                        judges one run + summary.json
promote_baseline.py                 promotes a judged run to baselines/<target>.json
compare.py                          paired comparison vs the baseline (McNemar + Wilcoxon)
diagnose.py                         likely cause per regression, fine-tune signal, drafts
experiment_pipeline.py              run -> judge -> compare -> diagnose in one command
export_finetune.py                  train-split fine-tuning export with leakage guard
app.py / templates/                 dashboards: / (investigator) and /experiments
tests/                              offline tests (python tests/test_<name>.py)
.github/workflows/                  CI: regression-check.yml (Track 1), experiment-check.yml (Track 2)
```

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-rag.txt   # optional: only for retrieval.method: embeddings
python store.py                       # seed the database

# Add to .env:
# GROQ_API_KEY=your_key
# SLACK_WEBHOOK_URL=your_webhook (optional)

python pipeline.py                    # Track 1: run the full pipeline
python experiment_pipeline.py experiments/baseline.yaml   # Track 2: one experiment end to end
python app.py                         # dashboards at localhost:5000 and localhost:5000/experiments
```

## Results

Initial validation run (10 hand-written scenarios): the framework detected a statistically significant regression between the baseline and degraded-prompt agent (**9-10/10 vs 4/10 pass rate, p = 0.0108**), driven almost entirely by groundedness failures, the weak agent hallucinated specific, plausible-sounding facts on 6 of 10 questions when its tools returned no data.

Results shift somewhat run-to-run given LLM output is non-deterministic and the current scenario set is still small this is a known, honestly-documented limitation, not a hidden one. Running the pipeline multiple times and aggregating, or growing the scenario count further, are the direct fixes (see below).

## Known limitations / future work

- **Sample size.** 10-25 scenarios is enough to demonstrate the framework works, not enough for high statistical confidence on every run. Next: aggregate across multiple runs, or grow the scenario set substantially.
- **Statistical power for targeted changes.** A change that only affects a handful of questions (e.g. 5 paraphrase scenarios for retrieval) cannot reach significance on its own: McNemar needs at least 6 one-way flips. The per-question ▲/▼ list is the evidence there; more targeted scenarios would make it testable.
- **Non-determinism.** LLM agents don't give identical answers twice. A single pipeline run can vary; averaging across N runs would give a more stable signal. `calibrate_noise.py` measures this, but costs about (agent + 5 judge) calls per scenario per run.
- **Judge identity.** A judge set is identified by its judges' names and model; editing a judge's prompt text does not change that identity, so re-judge the baseline after such edits.
- **Cost covers the system under test only.** Judge tokens are not counted in the estimated cost.
- **Toy backend.** The agent's data (orders, FAQs) is a small seeded SQLite database, not a real production system — intentional, since the framework is what's being demonstrated, not the agent.
- **Single alert channel.** Slack only, for now  email/PagerDuty integration would be a natural extension for a real team setting.
