"""
Target adapter for the Financial Due Diligence RAG
(github.com/samarth6/Financial-Due-Diligence).

It needs its OWN Python environment (torch, groq 1.x, qdrant: versions that
clash with this project's), so this file starts adapters/financial_dd_worker.py
with that environment's python and passes questions to it, one JSON line each
way (the same pattern as judges/deepeval_bridge.py). The worker stays alive for
the whole run, so the indexes and reranker load once, not once per question.

    .env:  FDD_PYTHON=<path to the Financial DD venv's python>

Financial DD uses ITS OWN API key (Financial-Due-Diligence/.env): the worker starts
without the variables from this framework's .env (worker_env).

settings = overrides of its config.yaml, by section:
    settings: {self_healing: {max_attempts: 1}}
"""

import atexit
import json
import os
import subprocess

from dotenv import dotenv_values, load_dotenv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO, "adapters", "financial_dd_worker.py")
ENV_FILE = os.path.join(REPO, ".env")
load_dotenv(ENV_FILE)

# Read on every question. chunking / embedding / vector_store / company are only used
# when the index is BUILT, so changing them here would silently do nothing: refused.
RUNTIME_SECTIONS = ("retrieval", "reranker", "self_healing", "generation", "confidence")

_proc = None


class SettingsError(ValueError):
    """Invalid settings for this target, with a message meant for a human."""


def available():
    """Is the Financial DD environment set up on this machine? (Offline tests skip what needs it.)"""
    return os.path.isfile(os.environ.get("FDD_PYTHON", ""))


def source_dirs():
    """Where Financial DD's own code lives, for suggest_fixes.py (read-only): the project folder
    that holds the venv FDD_PYTHON points at (the first parent with pyproject.toml, setup.py or .git)."""
    py = os.environ.get("FDD_PYTHON", "")
    d = os.path.dirname(os.path.abspath(py)) if py else ""
    while d and os.path.dirname(d) != d:
        if any(os.path.exists(os.path.join(d, m)) for m in ("pyproject.toml", "setup.py", ".git")):
            return [d]
        d = os.path.dirname(d)
    return []


def worker_env(env=None, env_file=None):
    """The environment the worker starts with: this one MINUS everything that came from this
    framework's .env (its GROQ_API_KEY and other secrets). Financial DD then reads its own .env
    and uses its own key, as when it runs on its own; python-dotenv never overrides a variable
    that is already set, so without this it would silently use the framework's key."""
    env = dict(os.environ if env is None else env)
    ours = dotenv_values(env_file or ENV_FILE) if os.path.isfile(env_file or ENV_FILE) else {}
    for name, value in ours.items():
        if env.get(name) == value:
            env.pop(name)
    return env


def _worker():
    global _proc
    if _proc is None or _proc.poll() is not None:
        py = os.environ.get("FDD_PYTHON", "")
        if not os.path.isfile(py):
            raise SettingsError("FDD_PYTHON in .env must point to the Financial DD environment's python")
        _proc = subprocess.Popen([py, WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 text=True, encoding="utf-8", env=worker_env())
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


def cancel():
    """Called by run_experiment.py when a question passes max_seconds (targets.yaml): kill the
    worker so its work really stops (the waiting call then returns an error). The next question
    starts a fresh worker, which reloads the models (about a minute)."""
    global _proc
    p, _proc = _proc, None
    if p is not None and p.poll() is None:
        p.kill()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


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
