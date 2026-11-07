"""
Small backend that holds the Groq API key privately (via environment
variable / Replit Secret) and proxies requests from the frontend.
The browser never sees the key.

Also serves /api/data, which reads judged_traces/ and
judged_traces_weak/ directly off disk on every request — no more
manually running export_for_dashboard.py and re-uploading a JSON
file every time the results change. Reload the page, get current
data.

Run locally:
    python store.py               (creates + seeds the database, once)
    set GROQ_API_KEY in .env
    python app.py

On Replit:
    1. Create a Python Repl, upload the project
    2. Go to the "Secrets" tab (lock icon) and add: GROQ_API_KEY = your key
    3. Click Run — Replit gives you a public URL
"""

import json
import os
import re
import threading
import time
from collections import defaultdict, deque

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from experiment_config import format_changes
from trace_utils import load_latest_traces

load_dotenv()

app = Flask(__name__)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

STRONG_FOLDER = "judged_traces"
WEAK_FOLDER = "judged_traces_weak"

# /api/ask protection: the server's Groq key must not become a free,
# unlimited LLM for anyone who can reach this page.
ASK_LIMIT_PER_MINUTE = 10
MAX_QUESTION_CHARS = 1_000
_ask_log = defaultdict(deque)   # client ip -> timestamps of recent requests
_ask_lock = threading.Lock()

# Fixed instructions for the "ask" box. Built here on the server (moved
# from index.html) so a visitor can't swap in their own system prompt.
# This is the Track 1 demo dashboard: it only shows the sample support bot
# (strong vs weak), so the wording below is about that bot on purpose. The
# experiments page (/experiments) is target-neutral.
ASK_INSTRUCTIONS = """You are analyzing results from an AI agent reliability testing framework. Two versions of a customer-support agent (a "strong" baseline and a "weak" degraded-prompt variant) were each run against the same set of test questions and scored 0-10 by 4 LLM judges (correctness, tool_efficiency, completeness, groundedness), averaged into final_score. A score >= 7 counts as a pass.

Explain findings in plain, simple language — the way you'd explain it to a friend who isn't deep into ML, not like an academic report. Avoid jargon-heavy phrasing; say "makes stuff up" instead of "hallucinates," "guesses confidently" instead of "fabricates unsupported claims," etc. Keep it to a couple of short, easy sentences up front stating the core finding in one clear line, then 2-4 sentences of specific evidence (cite real scenario_ids and scores, short quotes ok).

After the explanation, always add one short paragraph starting with "Try this:" that suggests one concrete, actionable next step someone could take to investigate further or fix the issue — a specific prompt tweak, a scenario to re-check, or a follow-up question to ask. Keep the whole answer under 150 words total.

If the data doesn't support a claim, say so rather than speculating.

Each line below is one scenario: its final scores for strong and weak, plus the single lowest-scoring judge and a short reason for that side (not all 4 judges — just the weakest one, to keep this compact).

DATA:
"""


def _rate_limited(client_ip: str, now: float | None = None) -> int:
    """Sliding 60s window. Returns 0 if allowed (and records the request),
    else the number of seconds until the client may retry."""
    now = time.monotonic() if now is None else now
    with _ask_lock:
        log = _ask_log[client_ip]
        while log and now - log[0] >= 60:
            log.popleft()
        if len(log) >= ASK_LIMIT_PER_MINUTE:
            return max(1, int(60 - (now - log[0])) + 1)
        log.append(now)
        return 0


def load_judged_traces(folder: str) -> dict:
    if not os.path.isdir(folder):
        return {}
    # Latest run per scenario only (see trace_utils.py).
    return load_latest_traces(folder)


def _truncate(s, max_len):
    if not s:
        return s or ""
    return s[:max_len] + "..." if len(s) > max_len else s


def _lowest_judge_note(t):
    if not t:
        return "ERR"
    judges = [(name, t.get(key)) for name, key in [
        ("correctness", "correctness_result"), ("tool_eff", "tool_efficiency_result"),
        ("completeness", "completeness_result"), ("grounded", "groundedness_result")]]
    judges = [(n, v) for n, v in judges if isinstance(v, dict) and "score" in v]
    if not judges:
        return ""
    name, lowest = min(judges, key=lambda j: j[1]["score"])   # first lowest wins ties
    return f"{name}={lowest['score']}: {_truncate(lowest.get('reason'), 55)}"


# Plain-text, one line per scenario instead of full nested JSON (moved
# here from index.html): repeated JSON field names across 50 objects
# burn far more tokens than the content does, so a compact table keeps
# this under Groq's per-request token limit. Sorted so the same data
# always produces the same context (and so the same answer).
def build_ask_context(strong: dict, weak: dict) -> str:
    lines = []
    for sid in sorted(set(strong) | set(weak)):
        s, w = strong.get(sid), weak.get(sid)
        q = _truncate((s or w).get("question"), 70)
        s_score = f"{s['final_score']:.1f}" if s and s.get("final_score") is not None else "ERR"
        w_score = f"{w['final_score']:.1f}" if w and w.get("final_score") is not None else "ERR"
        lines.append(f"{sid} [{q}]\n  strong={s_score} (lowest judge: {_lowest_judge_note(s)})\n"
                     f"  weak={w_score} (lowest judge: {_lowest_judge_note(w)})")
    return "\n".join(lines)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    strong = load_judged_traces(STRONG_FOLDER)
    weak = load_judged_traces(WEAK_FOLDER)
    return jsonify({"strong": strong, "weak": weak})


@app.route("/api/ask", methods=["POST"])
def ask():
    if not GROQ_API_KEY:
        return jsonify({"error": "Server is missing GROQ_API_KEY. Set it as a Secret/env var."}), 500

    retry_after = _rate_limited(request.remote_addr or "unknown")
    if retry_after:
        resp = jsonify({"error": f"Too many questions. Limit is {ASK_LIMIT_PER_MINUTE} per minute; "
                                 f"try again in {retry_after}s."})
        resp.headers["Retry-After"] = str(retry_after)
        return resp, 429

    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    question = payload.get("question", "")
    if not isinstance(question, str):
        return jsonify({"error": "'question' must be a string."}), 400
    if not question.strip():
        return jsonify({"error": "Question is empty."}), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({"error": f"Question too long ({len(question)} chars, max {MAX_QUESTION_CHARS})."}), 400

    # Built server-side from the judged results on disk; anything the
    # browser sends as "system" is ignored.
    system_prompt = ASK_INSTRUCTIONS + build_ask_context(
        load_judged_traces(STRONG_FOLDER), load_judged_traces(WEAK_FOLDER))

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-120b",
                "max_tokens": 600,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            },
            timeout=30,
        )
        return jsonify(resp.json()), resp.status_code
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


# ---------------------------------------------------------------- /experiments dashboard
# Read-only views of runs/ and baselines/ on disk. No LLM calls, no keys sent.
RUNS_DIR = "runs"
BASELINES_DIR = "baselines"
RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _run_path(run_id):
    """Only a direct sub-folder of runs/ with a run_meta.json: blocks ../ path tricks."""
    if not RUN_ID.match(run_id or "") or run_id in (".", ".."):
        return None
    path = os.path.join(RUNS_DIR, run_id)
    return path if os.path.isfile(os.path.join(path, "run_meta.json")) else None


def _current_baselines():
    """{(target, suite): baseline}: one baseline per target and suite."""
    out = {}
    for f in sorted(os.listdir(BASELINES_DIR)) if os.path.isdir(BASELINES_DIR) else []:
        b = _read_json(os.path.join(BASELINES_DIR, f)) if f.endswith(".json") else None
        if b:
            out[(b.get("target"), b.get("suite") or "main")] = b
    return out


def _run_row(run_id, baselines):
    path = os.path.join(RUNS_DIR, run_id)
    meta = _read_json(os.path.join(path, "run_meta.json")) or {}
    summary = _read_json(os.path.join(path, "summary.json")) or {}
    comparison = _read_json(os.path.join(path, "comparison.json")) or {}
    totals = meta.get("totals") or {}
    target = (meta.get("resolved_config") or {}).get("target")
    base = baselines.get((target, meta.get("suite") or "main")) or {}
    return {
        "run_id": run_id, "experiment": meta.get("experiment"), "timestamp": meta.get("timestamp"),
        "target": target,
        "what_changed": format_changes({"parent": meta.get("parent"),
                                        "changes_vs_parent": meta.get("changes_vs_parent") or []}),
        "judged": bool(summary), "judge_set": (summary.get("judge_set") or {}).get("name"),
        "pass_rate": summary.get("pass_rate"), "mean_final_score": summary.get("mean_final_score"),
        "scenarios": summary.get("scenarios") or len(meta.get("scenario_ids") or []),
        "total_tokens": totals.get("total_tokens"), "estimated_cost_usd": totals.get("estimated_cost_usd"),
        "verdict": comparison.get("verdict"),
        "suite": meta.get("suite") or "main",
        "is_baseline": (base.get("run_dir") or "").replace("\\", "/").rstrip("/").split("/")[-1] == run_id,
    }


@app.route("/experiments")
def experiments_page():
    return render_template("experiments.html")


@app.route("/api/experiments/runs")
def api_experiment_runs():
    baselines = _current_baselines()
    ids = [d for d in os.listdir(RUNS_DIR) if _run_path(d)] if os.path.isdir(RUNS_DIR) else []
    rows = [_run_row(d, baselines) for d in ids]
    return jsonify(sorted(rows, key=lambda r: (r["timestamp"] or "", r["run_id"]), reverse=True))


@app.route("/api/experiments/runs/<run_id>")
def api_experiment_run(run_id):
    path = _run_path(run_id)
    if path is None:
        return jsonify({"error": "unknown run"}), 404
    row = _run_row(run_id, _current_baselines())
    summary = _read_json(os.path.join(path, "summary.json")) or {}
    comparison = _read_json(os.path.join(path, "comparison.json"))
    diagnosis = _read_json(os.path.join(path, "diagnosis.json")) or {}
    return jsonify({**row, "per_judge_mean": summary.get("per_judge_mean"),
                    "tool_match_mean": summary.get("tool_match_mean"),
                    "errors_by_type": summary.get("errors_by_type"), "comparison": comparison,
                    "diagnosis": (comparison or {}).get("diagnosis") or diagnosis.get("diagnoses") or [],
                    "fine_tune_candidates": diagnosis.get("fine_tune_candidates") or {}})


@app.route("/api/experiments/baselines")
def api_experiment_baselines():
    return jsonify([{**{k: b.get(k) for k in ("target", "experiment", "run_dir", "promoted_at", "note")},
                     "suite": b.get("suite") or "main"} for b in _current_baselines().values()])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # The Werkzeug debugger is an in-browser Python console: never on by
    # default on a public host. Opt in locally with FLASK_DEBUG=1.
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")