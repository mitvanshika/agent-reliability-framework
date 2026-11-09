"""
The "Ask" box on /experiments: the instructions and the data context, both
built on the server from the run files (comparison.json, diagnosis.json,
summary.json, judged/, baselines/, noise/). The browser sends only the
question. Plain text, one line per item, sorted, so the same files always
give the same context (and so, at temperature 0, much the same answer).
No LLM calls here and no heavy imports (also used in demo mode).
"""

import hashlib
import json
import os

import overview
from trace_utils import load_latest_traces

# The 4 suggestion cards. make_demo_data.py --ask pre-computes their answers
# into demo_data/ask_cache.json; demo mode serves only those.
SUGGESTIONS = [
    {"id": "why_declined", "label": "Why did exp_01 decline?",
     "question": "Why did exp_01 decline?"},
    {"id": "invented", "label": "Which answers were invented?",
     "question": "Which questions were invented answers?"},
    {"id": "next", "label": "What should I try next?",
     "question": "What should I try next?"},
    {"id": "noise", "label": "How noisy are the judges?",
     "question": "How noisy are the judges, and can I trust a single run?"},
]

INSTRUCTIONS = """You are explaining results from an AI agent reliability testing framework to someone who is not an ML expert.

How the framework works: each experiment changes ONE setting of a system (e.g. its prompt) from the baseline. Both versions answer the same test questions (scenarios). LLM judges score each answer 0-10 (correctness_gt = matches the answer key, completeness, groundedness = supported by what the tools returned, tool_efficiency, context_relevance); their mean is final_score, and final_score >= 7 is a pass. A run is compared with the baseline using paired tests: McNemar on pass/fail flips and Wilcoxon on the scores. Verdicts: DECLINED, NO REAL DIFFERENCE, IMPROVED, IMPROVED WITH REGRESSIONS. Groundedness <= 3 usually means the answer was made up (not supported by the tool outputs). "Jumpy" scenarios move a lot between two identical runs (noise).

How to answer:
- Plain, simple language ("made it up", not "hallucinated"). Start with one clear sentence that answers the question.
- Then 2-4 sentences of specific evidence: cite real scenario ids and their scores from the DATA (e.g. "rag_06 fell from 8.2 to 3.0").
- End with one short paragraph that starts with "Try this:" and gives one concrete next step (an experiment to run, a setting to change, a scenario to re-check). Prefer experiments listed under NOT RUN YET when they fit.
- When you name why a scenario failed, use its "diagnosis" exactly as given (PROMPT, RAG, TOOL DESCRIPTIONS, ...).
- Some experiments deliberately make the system WORSE (a weak prompt, vague tool descriptions) to check that the framework catches it. Never suggest those as a fix; read each experiment's description literally.
- Under 150 words in total. Use only the DATA below; if it does not support an answer, say so.

DATA:
"""


def _fmt(x, d=2):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def _pct(x):
    return "-" if x is None else f"{round(x * 100)}%"


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _reason(t, judge, n=90):
    r = (t or {}).get(f"{judge}_result") or {}
    s = " ".join(str(r.get("reason") or "").split())
    return s if len(s) <= n else s[:n] + "..."


def _run_block(runs_dir, run_id, title):
    path = os.path.join(runs_dir, run_id)
    s = _read_json(os.path.join(path, "summary.json")) or {}
    c = _read_json(os.path.join(path, "comparison.json")) or {}
    d = _read_json(os.path.join(path, "diagnosis.json")) or {}
    judged = load_latest_traces(os.path.join(path, "judged")) if os.path.isdir(os.path.join(path, "judged")) else {}
    lines = [f"## {title}: run {run_id}"]
    lines.append(f"pass {s.get('passed')}/{s.get('scenarios')} ({_pct(s.get('pass_rate'))}), "
                 f"mean final_score {_fmt(s.get('mean_final_score'))}, judge set {(s.get('judge_set') or {}).get('name')}")
    if not c:
        lines.append("no comparison: this target has no promoted baseline yet")
    else:
        f, m, w = c.get("flips") or {}, c.get("mcnemar") or {}, c.get("wilcoxon") or {}
        cnt = c.get("counts") or {}
        lines.append(f"vs baseline {os.path.basename(str(c.get('baseline_run_dir') or ''))}: verdict {c.get('verdict')} "
                     f"({c.get('verdict_reason')}); {c.get('pairs')} paired; pass->fail {f.get('pass_to_fail')}, "
                     f"fail->pass {f.get('fail_to_pass')}; McNemar p={_fmt(m.get('p'), 3)}, Wilcoxon p={_fmt(w.get('p'), 3)}; "
                     f"up/down/same (outside noise band) {cnt.get('up')}/{cnt.get('down')}/{cnt.get('same')}")
        pj = [f"{k} {v.get('baseline_mean')}->{v.get('run_mean')} ({v.get('delta'):+.2f})"
              for k, v in sorted((c.get("per_judge") or {}).items())
              if isinstance(v, dict) and v.get("delta") is not None and v.get("in_final_score")]
        if pj:
            lines.append("per-judge means (baseline->run): " + "; ".join(pj))
        if c.get("excluded"):
            ex = [f"{x.get('scenario_id')} ({x.get('reason')})" if isinstance(x, dict) else str(x) for x in c["excluded"]]
            lines.append(f"excluded from the comparison (not scored, not counted): {', '.join(ex)}")
        diag = {x.get("scenario_id"): x for x in d.get("diagnoses") or []}
        moved = [r for r in c.get("rows") or [] if r.get("direction") != "same" or r.get("flip")]
        if moved:
            lines.append("scenarios that moved (baseline -> run):")
        for r in sorted(moved, key=lambda r: r["scenario_id"]):
            x = diag.get(r["scenario_id"]) or {}
            lines.append(f"- {r['scenario_id']}: {_fmt(r.get('baseline_score'))} -> {_fmt(r.get('run_score'))}"
                         f"{' (' + r['flip'] + ')' if r.get('flip') else ''}; judges moved: "
                         f"{', '.join(r.get('judges_moved') or []) or 'none'}"
                         + (f"; diagnosis: {x.get('cause')}, suggestion: {x.get('suggestion')}" if x else ""))
    low = sorted((sid, t["groundedness_result"]["score"]) for sid, t in judged.items()
                 if isinstance(t.get("groundedness_result"), dict)
                 and isinstance(t["groundedness_result"].get("score"), (int, float))
                 and t["groundedness_result"]["score"] <= 3)
    if low:
        lines.append("answers with groundedness <= 3 (likely made up) in this run:")
        for sid, g in low:
            lines.append(f"- {sid}: groundedness {g}, completeness {(judged[sid].get('completeness_result') or {}).get('score')}, "
                         f"final {_fmt(judged[sid].get('final_score'))}: {_reason(judged[sid], 'groundedness')}")
    return lines


def build_context(runs_dir="runs", baselines_dir="baselines", noise_dir="noise") -> str:
    lines = ["# BASELINES (the reference run each target is compared with)"]
    for f in sorted(os.listdir(baselines_dir)) if os.path.isdir(baselines_dir) else []:
        b = _read_json(os.path.join(baselines_dir, f)) if f.endswith(".json") else None
        if b:
            s = b.get("summary") or {}
            lines.append(f"- target {b.get('target')}: run {os.path.basename(str(b.get('run_dir') or '').replace(chr(92), '/'))}, "
                         f"pass {_pct(s.get('pass_rate'))}, mean {_fmt(s.get('mean_final_score'))}, "
                         f"judge set {(b.get('judge_set') or {}).get('name')}")
    verdicts = overview.latest_verdicts(runs_dir)
    for v in verdicts:
        if v.get("run_id"):
            title = (f"EXPERIMENT {v['experiment']} ({v['description']})" if v["experiment"] != "baseline"
                     else "A/A: the baseline config run again (should be NO REAL DIFFERENCE)")
            lines.append("")
            lines += _run_block(runs_dir, v["run_id"], title)
    noise = _read_json(os.path.join(noise_dir, "configurable_noise.json")) or {}
    if noise:
        suite = noise.get("suite") or {}
        lines += ["", "# NOISE: the SAME baseline config run twice, nothing changed (only this section and the A/A "
                      "section measure noise; an experiment's changes are NOT noise)"]
        for r in suite.get("per_run") or []:
            lines.append(f"- identical run {r.get('run')} ({os.path.basename(str(r.get('run_dir')))}): pass "
                         f"{r.get('passed')}/{r.get('total')} ({_pct(r.get('pass_rate'))}), mean {_fmt(r.get('mean_score'))}")
        ppr = suite.get("pass_rate_range")
        lines.append(f"suite pass rate moved {'-' if ppr is None else round(ppr * 100)} percentage points and the mean score "
                     f"{_fmt(suite.get('mean_score_range'))} between these identical runs")
        per = noise.get("scenarios") or {}
        jumpy = [f"{sid} {' / '.join(_fmt(x) for x in per[sid].get('scores') or [])} ({per[sid].get('pass_fail')})"
                 for sid in noise.get("jumpy") or [] if sid in per]
        lines.append(f"jumpy scenarios (final_score moved > {noise.get('jumpy_range')} points between the identical "
                     f"runs; scores run1 / run2, P=pass F=fail): {'; '.join(jumpy) or 'none'}")
        calm = sum(1 for v in per.values() if not v.get("jumpy"))
        lines.append(f"{calm} of {len(per)} scenarios moved {noise.get('jumpy_range')} points or less")
        j1 = load_latest_traces(os.path.join(runs_dir, "baseline_20261106_101856", "judged")) \
            if os.path.isdir(os.path.join(runs_dir, "baseline_20261106_101856", "judged")) else {}
        j2 = load_latest_traces(os.path.join(runs_dir, "baseline_20261106_111506", "judged")) \
            if os.path.isdir(os.path.join(runs_dir, "baseline_20261106_111506", "judged")) else {}
        t1, t2 = j1.get("edge_03") or {}, j2.get("edge_03") or {}
        a = (t1.get("tool_efficiency_result") or {}).get("score")
        b = (t2.get("tool_efficiency_result") or {}).get("score")
        if a is not None and b is not None:
            lines.append(f"judge noise example: edge_03's answer was word-for-word identical in both identical runs, "
                         f"yet the tool_efficiency judge scored it {a} and then {b}")
    todo = [v for v in verdicts if v.get("status") == "not run yet"]
    if todo:
        lines += ["", "# NOT RUN YET (defined experiments, one setting changed each)"]
        lines += [f"- {v['experiment']}: {v['description']}" for v in todo]
    return "\n".join(lines)


def system_prompt(runs_dir="runs", baselines_dir="baselines", noise_dir="noise") -> str:
    return INSTRUCTIONS + build_context(runs_dir, baselines_dir, noise_dir)


def context_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize(q: str) -> str:
    return " ".join(str(q or "").lower().replace("?", " ").split())


def find_suggestion(question: str):
    n = normalize(question)
    return next((s for s in SUGGESTIONS if normalize(s["question"]) == n or normalize(s["label"]) == n), None)
