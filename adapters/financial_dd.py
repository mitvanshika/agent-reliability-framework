"""
Target adapter for the Financial Due Diligence RAG
(github.com/samarth6/Financial-Due-Diligence).

It needs its OWN Python environment (torch, groq 1.x, qdrant: versions that
clash with this project's), so this file starts adapters/financial_dd_worker.py
with that environment's python and passes questions to it, one JSON line each
way (the same pattern as judges/deepeval_bridge.py). The worker stays alive for
the whole run, so the indexes and reranker load once, not once per question.

    .env:  FDD_PYTHON=<path to the Financial DD venv's python>

settings = overrides of its config.yaml, by section:
    settings: {self_healing: {max_attempts: 1}}
"""

import atexit
import json
import os
import subprocess

from dotenv import load_dotenv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO, "adapters", "financial_dd_worker.py")
load_dotenv(os.path.join(REPO, ".env"))

# Read on every question. chunking / embedding / vector_store / company are only used
# when the index is BUILT, so changing them here would silently do nothing: refused.
RUNTIME_SECTIONS = ("retrieval", "reranker", "self_healing", "generation", "confidence")

_proc = None


class SettingsError(ValueError):
    """Invalid settings for this target, with a message meant for a human."""


def available():
    """Is the Financial DD environment set up on this machine? (Offline tests skip what needs it.)"""
    return os.path.isfile(os.environ.get("FDD_PYTHON", ""))


def _worker():
    global _proc
    if _proc is None or _proc.poll() is not None:
        py = os.environ.get("FDD_PYTHON", "")
        if not os.path.isfile(py):
            raise SettingsError("FDD_PYTHON in .env must point to the Financial DD environment's python")
        _proc = subprocess.Popen([py, WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 text=True, encoding="utf-8")
        atexit.register(_proc.terminate)
    return _proc


def _ask(request):
    p = _worker()
    p.stdin.write(json.dumps(request) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    if not line:
        return {"answer": None, "error": "Financial DD worker exited (its error is printed above)",
                "error_type": "crash"}
    return json.loads(line)


def validate_settings(settings):
    """Called by run_experiment.py before any API call."""
    if not isinstance(settings, dict):
        raise SettingsError("settings must map config.yaml sections to overrides")
    for section, values in settings.items():
        if section not in RUNTIME_SECTIONS:
            raise SettingsError(f"settings.{section}: not allowed here (allowed: {', '.join(RUNTIME_SECTIONS)}); "
                                "index-time sections need `dd chunk` + `dd index` instead")
        if not isinstance(values, dict):
            raise SettingsError(f"settings.{section} must be a mapping")
    reply = _ask({"op": "validate", "settings": settings})
    if not reply.get("ok"):
        raise SettingsError(reply.get("error") or "the Financial DD worker rejected the settings")


def describe(settings=None):
    """Called by run_experiment.py once per run: which version of Financial DD this is
    (git commit, config.yaml hash, effective settings). Saved in run_meta.json as "system"."""
    return _ask({"op": "describe", "settings": settings or {}})


def run(question, settings=None):
    return _ask({"op": "answer", "question": question, "settings": settings or {}})
