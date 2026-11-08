"""
DeepEval metrics as judges (judge set deepeval_v1) from the MAIN Python: the
bridge, the judge set and the separate environment. Fake model, no API calls.
If .venv-deepeval does not exist, the parts that need it are skipped (with a note).
    python tests/test_deepeval.py
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")
os.environ["DEEPEVAL_FAKE_MODEL"] = "fake_model:FakeModel"          # read by the worker
os.environ["PYTHONPATH"] = os.path.join(REPO, "tests", "deepeval_env")  # so the worker finds fake_model

import judge_experiment  # noqa: E402
import judge_sets  # noqa: E402
import run_suite  # noqa: E402
import target_config  # noqa: E402
from judges import deepeval_bridge as de  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


print("the main Python stays clean:")
import importlib.util  # noqa: E402
ok("deepeval is NOT importable in the main Python", importlib.util.find_spec("deepeval") is None)
ok("...and the framework never imported it", "deepeval" not in sys.modules)

print("the judge set:")
D = judge_sets.DEEPEVAL_V1
ok("deepeval_v1 = de_faithfulness + de_answer_relevancy, opt-in", D["llm_judges"] == ["de_faithfulness",
   "de_answer_relevancy"] and "deepeval_v1" in judge_sets.MAIN_SETS
   and judge_sets.for_scenarios(run_suite.load_scenarios())["name"] == "experiment_v2")
ok("a target may pick it", target_config.validate("x", {"module": "m", "function": "f", "judge_set": "deepeval_v1",
                                                         "judges": ["de_faithfulness"]}, judge_sets.ALL_NAMES) == [])

print("environment missing -> a clear error, not a crash:")
saved = os.environ.get("DEEPEVAL_PYTHON")
os.environ["DEEPEVAL_PYTHON"] = os.path.join(REPO, "no-such-env", "python.exe")
de.WORKER.close()
try:
    de.judge_answer_relevancy("q", "a")
    ok("missing environment raises", False)
except Exception as e:  # noqa: BLE001
    ok("missing environment -> DeepEvalUnavailable that says what is wrong and how to create it",
       isinstance(e, de.DeepEvalUnavailable) and "does not exist" in str(e)
       and "requirements-deepeval.txt" in str(e), f"{type(e).__name__}: {e}")
if saved is None:
    del os.environ["DEEPEVAL_PYTHON"]
else:
    os.environ["DEEPEVAL_PYTHON"] = saved
de.WORKER.close()
ok("no tool outputs -> faithfulness N/A without starting anything", de.judge_faithfulness("q", "a", []) is None)

if de.python_path() is None:
    print("  SKIP  .venv-deepeval not found: the remaining checks need it (see requirements-deepeval.txt)")
else:
    print("through the real worker in .venv-deepeval (fake model):")
    p = de.ping()
    ok("worker runs in the separate environment, telemetry off", p["ok"] and ".venv-deepeval" in p["python"]
       and p["telemetry_off"] is True, p)
    v = de.judge_faithfulness("Where is A1001?", "Delivered. INVENTED: a free gift.", ["Order A1001: delivered"])
    ok("faithfulness 0.5 -> 5/10 with DeepEval's reason", v.score == 5 and v.confidence == de.CONFIDENCE
       and v.reason.startswith("DeepEval faithfulness 0.50"), v)
    v = de.judge_answer_relevancy("Where is A1001?", "It was delivered.")
    ok("answer relevancy 1.0 -> 10/10", v.score == 10, v)

    print("as a judge set through judge_experiment.evaluate:")
    with_tools = {"scenario_id": "s", "question": "Where is A1001?", "answer": "It was delivered after 5 days.",
                  "tools_used": ["order_lookup"], "tool_outputs": ["Order A1001: delivered, delivery took 5 days"]}
    f = judge_experiment.evaluate(with_tools, {}, D)
    ok("both judges scored, final_score from them", f["de_faithfulness_result"]["score"] == 10
       and f["de_answer_relevancy_result"]["score"] == 10 and f["final_score"] == 10, f)
    no_tools = {**with_tools, "tools_used": [], "tool_outputs": []}
    f = judge_experiment.evaluate(no_tools, {}, D)
    ok("no tool outputs -> de_faithfulness N/A, final_score from relevancy alone",
       f["de_faithfulness_result"].get("na") and "de_faithfulness" in f["na_judges"] and f["final_score"] == 10, f)
    f = judge_experiment.evaluate({**with_tools, "answer": "Delivered. OFFTOPIC: nice weather."}, {}, D)
    ok("an off-topic statement lowers relevancy to 5", f["de_answer_relevancy_result"]["score"] == 5)

    print("inside the environment (its own checks):")
    py = de.python_path()
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    r = subprocess.run([py, os.path.join("tests", "deepeval_env", "check_deepeval_env.py")], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env, cwd=REPO, timeout=600)
    ok("tests/deepeval_env/check_deepeval_env.py passes (real metrics, fake model, network blocked)",
       r.returncode == 0 and r.stdout.strip().endswith("PASS"), (r.stdout[-400:] + r.stderr[-300:]))
    de.WORKER.close()

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
