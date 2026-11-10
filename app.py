

import json
import os
import re
import threading
import time
from collections import defaultdict, deque

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request

from experiment_config import format_changes
from trace_utils import load_latest_traces

load_dotenv()

app = Flask(__name__)

# DEMO_MODE=1: a read-only public demo (Render). Everything is read from the
# snapshot in demo_data/ (make_demo_data.py), the Ask box serves only the
# recorded answers in demo_data/ask_cache.json, approval decisions are off,
# and no LLM client or LangChain module is ever imported (fits in 512 MB).
DEMO_MODE = os.environ.get("DEMO_MODE") == "1"
DEMO_DIR = "demo_data"
GROQ_API_KEY = None if DEMO_MODE else os.environ.get("GROQ_API_KEY")

if os.environ.get("TRUST_PROXY") == "1":
    # Behind one reverse proxy (Render): take the client IP from its X-Forwarded-For,
    # so the per-IP rate limit sees visitors, not the proxy.
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

STRONG_FOLDER = os.path.join(DEMO_DIR, "judged_traces") if DEMO_MODE else "judged_traces"
WEAK_FOLDER = os.path.join(DEMO_DIR, "judged_traces_weak") if DEMO_MODE else "judged_traces_weak"
DEMO_ASK_MESSAGE = ("This is the read-only demo: it shows recorded answers to the suggested questions only. "
                    "Run the dashboard locally with a GROQ_API_KEY to ask anything.")

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


# Top nav on every page (templates/_nav.html). A page is "active" when the
# request path starts with its link; /runs/... (step timeline) counts as Experiments.
NAV_ITEMS = [("/", "Overview"), ("/experiments", "Experiments"), ("/evidence", "Evidence"),
             ("/systems", "Systems"), ("/approvals", "Approvals"), ("/track1", "Track 1")]
if DEMO_MODE:   # Track 1's judged traces aren't in demo_data/: hide the tab rather than show an empty page
    NAV_ITEMS = [n for n in NAV_ITEMS if n[0] != "/track1"]


@app.context_processor
def _nav_context():
    path = request.path
    active = "/experiments" if path.startswith("/runs/") else next(
        (p for p, _ in reversed(NAV_ITEMS) if p != "/" and path.startswith(p)), "/" if path == "/" else None)
    return {"nav_items": NAV_ITEMS, "nav_active": active, "demo_mode": DEMO_MODE}


@app.route("/")
def overview_page():
    import overview
    return render_template("overview.html", data=overview.overview(RUNS_DIR, NOISE_DIR))


@app.route("/evidence")
def evidence_index():
    import overview
    return render_template("evidence_list.html", notes=overview.evidence_list(EVIDENCE_DIR))


@app.route("/evidence/<name>")
def evidence_page(name):
    """One evidence/*.md note as a page. Only names that evidence_list() found are served (no paths)."""
    import overview
    page = overview.evidence_html(name, EVIDENCE_DIR)
    if page is None:
        return render_template("evidence_list.html", notes=overview.evidence_list(EVIDENCE_DIR)), 404
    return render_template("evidence.html", name=name, title=page[0], html=page[1])


@app.route("/systems")
def systems_page():
    import overview
    return render_template("systems.html", systems=overview.systems(TARGETS_FILE, BASELINES_DIR, RUNS_DIR))


@app.route("/track1")
def index():
    """The original strong-vs-weak investigator (Track 1), unchanged. Not in the demo."""
    if DEMO_MODE:
        return redirect("/")
    return render_template("track1.html")


@app.route("/api/data")
def api_data():
    if DEMO_MODE:
        return jsonify({"strong": [], "weak": []})
    strong = load_judged_traces(STRONG_FOLDER)
    weak = load_judged_traces(WEAK_FOLDER)
    return jsonify({"strong": strong, "weak": weak})


def _check_question():
    """(question, None) or (None, error response): rate limit + validation, shared by both Ask boxes."""
    retry_after = _rate_limited(request.remote_addr or "unknown")
    if retry_after:
        resp = jsonify({"error": f"Too many questions. Limit is {ASK_LIMIT_PER_MINUTE} per minute; "
                                 f"try again in {retry_after}s."})
        resp.headers["Retry-After"] = str(retry_after)
        return None, (resp, 429)
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return None, (jsonify({"error": "Request body must be a JSON object."}), 400)
    question = payload.get("question", "")
    if not isinstance(question, str):
        return None, (jsonify({"error": "'question' must be a string."}), 400)
    if not question.strip():
        return None, (jsonify({"error": "Question is empty."}), 400)
    if len(question) > MAX_QUESTION_CHARS:
        return None, (jsonify({"error": f"Question too long ({len(question)} chars, max {MAX_QUESTION_CHARS})."}), 400)
    return question, None


ASK_MODEL = "openai/gpt-oss-120b"


def groq_chat(system_prompt, question, api_key=None, max_tokens=1500):
    """One chat call to Groq; returns (response json, status). Used by the experiments Ask box
    and by make_demo_data.py --ask, so the cached answers come from the same call."""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key or GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": ASK_MODEL, "max_tokens": max_tokens, "temperature": 0,
              "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]},
        timeout=60,
    )
    return resp.json(), resp.status_code


def answer_text(groq_json):
    try:
        return (groq_json["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


@app.route("/api/ask", methods=["POST"])
def ask():
    if DEMO_MODE:
        return jsonify({"error": DEMO_ASK_MESSAGE}), 403
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
RUNS_DIR = os.path.join(DEMO_DIR, "runs") if DEMO_MODE else "runs"
BASELINES_DIR = os.path.join(DEMO_DIR, "baselines") if DEMO_MODE else "baselines"
NOISE_DIR = os.path.join(DEMO_DIR, "noise") if DEMO_MODE else "noise"
EVIDENCE_DIR = os.path.join(DEMO_DIR, "evidence") if DEMO_MODE else "evidence"
TARGETS_FILE = "targets.yaml"
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
    import overview
    is_baseline = (base.get("run_dir") or "").replace("\\", "/").rstrip("/").split("/")[-1] == run_id
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
        "is_baseline": is_baseline,
        "role": overview.run_role(run_id, is_baseline),
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
                    "fine_tune_candidates": diagnosis.get("fine_tune_candidates") or {},
                    "scenarios": _scenario_steps(path)})


SCENARIO_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _scenario_steps(path):
    """[{scenario_id, steps, tool_calls, findings, source}] for the run's step-timeline links."""
    import step_checks
    import trajectory_read as trajectory
    out = []
    for sid, t in sorted(load_latest_traces(os.path.join(path, "traces")).items()):
        steps, source = trajectory.of_trace(t)
        out.append({"scenario_id": sid, "steps": len(steps), "source": source,
                    "tool_calls": sum(1 for s in steps if s.get("type") == "tool"),
                    "findings": len(step_checks.check_steps(steps)) if steps else 0, "error": t.get("error")})
    return out


@app.route("/runs/<run_id>/<scenario_id>")
def timeline_page(run_id, scenario_id):
    return render_template("timeline.html")


@app.route("/api/experiments/runs/<run_id>/steps/<scenario_id>")
def api_steps(run_id, scenario_id):
    """One answer, step by step: the steps (recorded or rebuilt), step-check findings,
    the answer's judge scores and, if judge_steps.py ran, the per-step scores."""
    import step_checks
    import trajectory_read as trajectory
    path = _run_path(run_id)
    if path is None or not SCENARIO_ID.match(scenario_id or ""):
        return jsonify({"error": "unknown run or scenario"}), 404
    trace = load_latest_traces(os.path.join(path, "traces")).get(scenario_id)
    if trace is None:
        return jsonify({"error": "unknown scenario"}), 404
    steps, source = trajectory.of_trace(trace)
    judged = load_latest_traces(os.path.join(path, "judged")).get(scenario_id) if os.path.isdir(
        os.path.join(path, "judged")) else None
    judges = {}
    for k, v in (judged or {}).items():
        if k.endswith("_result") and isinstance(v, dict):
            judges[k[:-7]] = {"score": v.get("score"), "na": bool(v.get("na")), "reason": v.get("reason")}
    step_scores = None
    js_dir = os.path.join(path, "judged_steps")
    if os.path.isdir(js_dir):
        rec = load_latest_traces(js_dir).get(scenario_id)
        if rec and not rec.get("na"):
            step_scores = {str(s["index"]): {"score": s["score"], "reason": s["reason"]} for s in rec.get("steps") or []}
    return jsonify({
        "run_id": run_id, "scenario_id": scenario_id, "question": trace.get("question"),
        "answer": trace.get("answer"), "error": trace.get("error"), "latency_s": trace.get("latency_s"),
        "usage": trace.get("usage"), "steps": steps, "source": source,
        "findings": step_checks.check_steps(steps) if steps else [],
        "final_score": (judged or {}).get("final_score"), "judges": judges, "step_scores": step_scores})


@app.route("/api/experiments/baselines")
def api_experiment_baselines():
    return jsonify([{**{k: b.get(k) for k in ("target", "experiment", "run_dir", "promoted_at", "note")},
                     "suite": b.get("suite") or "main"} for b in _current_baselines().values()])


# ---------------------------------------------------------------- Ask box on /experiments
# The server builds the whole context from the run files (ask_context.py); the
# browser sends only {"question": ...}. Same per-IP rate limit as /api/ask.

def _ask_cache():
    with open(os.path.join(DEMO_DIR, "ask_cache.json"), encoding="utf-8") as f:
        return json.load(f)


@app.route("/api/experiments/ask/suggestions")
def api_ask_suggestions():
    import ask_context
    cached = set()
    if DEMO_MODE:
        try:
            cached = {a["id"] for a in _ask_cache().get("answers") or [] if a.get("answer")}
        except (OSError, ValueError):
            pass
    return jsonify({"demo": DEMO_MODE, "live": bool(GROQ_API_KEY),
                    "suggestions": [s for s in ask_context.SUGGESTIONS if not DEMO_MODE or s["id"] in cached]})


@app.route("/api/experiments/ask", methods=["POST"])
def api_experiments_ask():
    import ask_context
    question, err = _check_question()
    if err:
        return err
    if DEMO_MODE:
        card = ask_context.find_suggestion(question)
        try:
            cache = _ask_cache()
        except (OSError, ValueError):
            cache = {}
        hit = next((a for a in cache.get("answers") or [] if card and a.get("id") == card["id"]), None)
        if not hit or not hit.get("answer"):
            return jsonify({"error": DEMO_ASK_MESSAGE, "demo": True}), 403
        return jsonify({"answer": hit["answer"], "source": "recorded", "question": card["question"],
                        "recorded_at": cache.get("generated_at"), "model": cache.get("model")})
    if not GROQ_API_KEY:
        return jsonify({"error": "Server is missing GROQ_API_KEY. Set it as a Secret/env var."}), 500
    prompt = ask_context.system_prompt(RUNS_DIR, BASELINES_DIR, NOISE_DIR)
    try:
        data, status = groq_chat(prompt, question)
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502
    if status != 200:
        msg = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
        return jsonify({"error": msg or f"Groq returned HTTP {status}"}), status
    text = answer_text(data)
    if not text:
        return jsonify({"error": "The model returned an empty answer; try again."}), 502
    return jsonify({"answer": text, "source": "live", "model": ASK_MODEL})


# ---------------------------------------------------------------- approvals (approval.py)
# Pending actions waiting for a human (DashboardApprover). Deciding is protected:
# with APPROVALS_TOKEN set, the request must send it (X-Approval-Token header);
# without it, decisions are accepted only from this machine, because the app
# listens on 0.0.0.0 and approving an action is not something any visitor may do.

def _approvals_store():
    import idempotent
    return idempotent.default_store()


def _may_decide():
    token = os.environ.get("APPROVALS_TOKEN")
    if token:
        import hmac
        return hmac.compare_digest(request.headers.get("X-Approval-Token") or "", token)
    return request.remote_addr in ("127.0.0.1", "::1")


@app.route("/approvals")
def approvals_page():
    return render_template("approvals.html")


@app.route("/api/approvals")
def api_approvals():
    if DEMO_MODE:
        return jsonify({"pending": [], "recent": [], "token_required": False, "demo": True})
    import approval
    st = _approvals_store()
    return jsonify({"pending": approval.queue(st, "pending"), "recent": approval.queue(st, limit=50),
                    "token_required": bool(os.environ.get("APPROVALS_TOKEN"))})


@app.route("/api/approvals/<qid>", methods=["POST"])
def api_decide(qid):
    if DEMO_MODE:
        return jsonify({"error": "approvals are read-only in the demo"}), 403
    import approval
    if not _may_decide():
        return jsonify({"error": "not allowed to decide: set APPROVALS_TOKEN and send it as X-Approval-Token, "
                                 "or decide from this machine"}), 403
    body = request.get_json(silent=True) or {}
    by = str(body.get("by") or "").strip()
    if not isinstance(body.get("approved"), bool) or not by:
        return jsonify({"error": "send {\"approved\": true|false, \"by\": \"your name\", \"reason\": \"...\"}"}), 400
    st = _approvals_store()
    if approval.queue_item(st, qid) is None:
        return jsonify({"error": "no such request"}), 404
    if not approval.decide_pending(st, qid, body["approved"], by, str(body.get("reason") or "")[:500]):
        return jsonify({"error": "already decided or expired", "item": approval.queue_item(st, qid)}), 409
    return jsonify({"ok": True, "item": approval.queue_item(st, qid)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # The Werkzeug debugger is an in-browser Python console: never on by
    # default on a public host. Opt in locally with FLASK_DEBUG=1.
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")