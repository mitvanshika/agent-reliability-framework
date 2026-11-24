"""
Runs INSIDE the Financial Due Diligence environment (its own venv: torch,
groq 1.x, qdrant ...), never in this project's Python. adapters/financial_dd.py
starts it and talks to it over stdin/stdout, one JSON object per line:

    {"op": "validate", "settings": {...}}               -> {"ok": true} or {"error": "..."}
    {"op": "answer", "question": "...", "settings": {...}}
        -> {"answer", "tools_used", "tool_outputs", "error", "error_type", "usage", "llm_calls"}
    {"op": "describe", "settings": {...}}
        -> {"system", "git", "config_hash", "effective_config"}   (which version ran)

Uses only the standard library, the dd package and token_meter.py (next to this file,
standard library only): nothing else from this repo.
"""

import hashlib
import json
import subprocess
import sys

PROTOCOL = sys.stdout
sys.stdout = sys.stderr            # anything dd / torch prints must not break the protocol
for stream in (sys.stdin, PROTOCOL):
    stream.reconfigure(encoding="utf-8")   # Windows defaults to cp1252

from dd.config import PROJECT_ROOT, Config, get_config  # noqa: E402
from dd.generation import answer  # noqa: E402
from dd.structured import describe_fact  # noqa: E402

import token_meter  # noqa: E402  (this file's folder is on sys.path: dd's Groq calls get counted)

MODEL_FAILED = "Model call failed"   # how dd.generation reports a failed Groq call


def config_for(settings):
    """config.yaml with the experiment's overrides on top. Unknown keys are refused:
    dd's Config ignores them, so a typo like max_attemps would silently change nothing."""
    base = get_config().model_dump()
    for section, values in settings.items():
        unknown = sorted(set(values) - set(base.get(section, {})))
        if unknown:
            raise ValueError(f"settings.{section}: unknown key(s) {unknown}; known: {sorted(base[section])}")
        base[section] = {**base[section], **values}
    return Config.model_validate(base)


def git_info():
    """The Financial DD repo's commit; dirty = tracked files edited (code, config: things that can
    change answers), listed in changed_files; untracked_files = how many new files that git does
    not track (eval outputs and the like) - counted, but they don't make the repo dirty."""
    def git(*args):
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout
    try:
        commit = git("rev-parse", "HEAD").strip()
        lines = [line for line in git("status", "--porcelain").splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None, "changed_files": None, "untracked_files": None}
    changed = [line[3:] for line in lines if not line.startswith("??")]
    return {"commit": commit, "dirty": bool(changed), "changed_files": changed[:10],
            "untracked_files": sum(1 for line in lines if line.startswith("??"))}


def describe(settings):
    """Which version of Financial DD runs: its own config.yaml (before this experiment's overrides,
    which run_meta already records) as a short hash, its git commit, and the full effective config."""
    base = get_config().model_dump(mode="json")
    effective = config_for(settings).model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:12]
    return {"system": "financial_dd", "git": git_info(), "config_hash": digest, "effective_config": effective}


def error_type(text):
    t = text.lower()
    if "ratelimit" in t or "429" in t:
        return "rate_limit"
    if "timeout" in t:
        return "timeout"
    if "authentication" in t or "401" in t:
        return "auth"
    return "crash"


def handle(req):
    cfg = config_for(req.get("settings") or {})
    if req.get("op") == "validate":
        return {"ok": True}
    if req.get("op") == "describe":
        return describe(req.get("settings") or {})
    with token_meter.measure() as meter:      # dd reports no tokens; count its Groq calls here
        a = answer(req["question"], cfg)
    o = a.outcome
    evidence = []
    if o is not None:
        evidence = [g.retrieved.chunk.text for g in o.relevant_chunks]
        evidence += [describe_fact(f) for f in o.structured_facts if "value" in f]
    # A failed API call is the infrastructure's fault, not an honest refusal: report it as an error
    failed = not a.answered and a.text.startswith(MODEL_FAILED)
    if failed:
        return {"answer": None, "tools_used": [], "tool_outputs": evidence,
                "error": a.text, "error_type": error_type(a.text), "usage": meter.usage(),
                "llm_calls": meter.calls()}
    # dd marks a refusal with answered=False and puts only the REASON in text; its own CLI
    # shows it as "no answer: <reason>". Keep that label, or a judge reads a bare reason
    # ("only 0 relevant chunk(s) found...") with no sign that the system declined to answer.
    text = a.text if a.answered else f"No answer: {a.text}"
    return {"answer": text, "tools_used": [], "tool_outputs": evidence, "error": None, "error_type": None,
            "usage": meter.usage(), "llm_calls": meter.calls()}


for line in sys.stdin:
    try:
        reply = handle(json.loads(line))
    except Exception as e:  # noqa: BLE001 - reported to the harness, never crashes the worker
        reply = {"answer": None, "tool_outputs": [], "error": f"{type(e).__name__}: {e}", "error_type": "crash"}
    PROTOCOL.write(json.dumps(reply) + "\n")
    PROTOCOL.flush()