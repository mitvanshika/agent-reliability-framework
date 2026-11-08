"""
DeepEval metrics as judges (judge set deepeval_v1), WITHOUT importing DeepEval
into the main project's Python: DeepEval lives in its own environment
(.venv-deepeval, see requirements-deepeval.txt) and runs in deepeval_worker.py;
this module starts that worker once and talks to it over stdin/stdout.

    de_faithfulness      DeepEval FaithfulnessMetric: are the answer's claims supported by
                         the retrieved context (the tool outputs)? N/A without tool outputs.
    de_answer_relevancy  DeepEval AnswerRelevancyMetric: do the answer's statements address
                         the question?

DeepEval scores 0-1; they are scaled to 0-10 like every other judge. DeepEval
gives no confidence, so a fixed CONFIDENCE is used for the weighted final_score.

Where the environment is: $DEEPEVAL_PYTHON, else .venv-deepeval/Scripts/python.exe
(Windows) or .venv-deepeval/bin/python. Missing -> a clear error telling you how
to create it (a judge error, like any other: the scenario is left out, not failed).
"""

import json
import os
import subprocess
import sys
import threading

from judges.schema import JudgeVerdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(REPO, ".venv-deepeval")
CONFIDENCE = 0.8
SETUP = ("python -m venv .venv-deepeval && .venv-deepeval/Scripts/python -m pip install -r requirements-deepeval.txt"
         "   (bin/python on Linux/macOS)")


class DeepEvalUnavailable(RuntimeError):
    pass


def python_path():
    if os.environ.get("DEEPEVAL_PYTHON"):
        p = os.environ["DEEPEVAL_PYTHON"]
        return p if os.path.isfile(p) else None
    for p in (os.path.join(ENV_DIR, "Scripts", "python.exe"), os.path.join(ENV_DIR, "bin", "python")):
        if os.path.isfile(p):
            return p
    return None


def worker_path():
    return os.environ.get("DEEPEVAL_WORKER") or os.path.join(REPO, "deepeval_worker.py")


class _Worker:
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()

    def _start(self):
        py = python_path()
        if py is None:
            where = (f"$DEEPEVAL_PYTHON={os.environ['DEEPEVAL_PYTHON']} does not exist"
                     if os.environ.get("DEEPEVAL_PYTHON") else "DeepEval environment not found (.venv-deepeval)")
            raise DeepEvalUnavailable(f"{where}. Create it once: {SETUP}")
        self.proc = subprocess.Popen([py, worker_path()], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", cwd=REPO)

    def request(self, obj):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.poll()
                self.proc = None
                raise DeepEvalUnavailable(f"the DeepEval worker exited (code {code}); is {SETUP!r} done?")
            return json.loads(line)

    def close(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            self.proc = None


WORKER = _Worker()


def ping():
    return WORKER.request({"op": "ping"})


def _measure(metric, question, answer, context=None, model=None):
    out = WORKER.request({"op": "measure", "metric": metric, "input": question, "actual_output": answer,
                          "retrieval_context": context, "model": model})
    if "error" in out:
        raise RuntimeError(f"DeepEval {metric}: {out['error']}")
    score = out.get("score")
    if not isinstance(score, (int, float)):
        raise RuntimeError(f"DeepEval {metric} returned no score: {out}")
    return JudgeVerdict(score=max(0, min(10, round(10 * float(score)))), confidence=CONFIDENCE,
                        reason=f"DeepEval {metric} {float(score):.2f}: {out.get('reason') or ''}".strip())


def judge_faithfulness(question, answer, tool_outputs, model=None):
    """None = N/A (no retrieved context to be faithful to)."""
    context = [str(o) for o in (tool_outputs or []) if str(o).strip()]
    if not context:
        return None
    return _measure("faithfulness", question, answer, context, model)


def judge_answer_relevancy(question, answer, model=None):
    return _measure("answer_relevancy", question, answer, None, model)
