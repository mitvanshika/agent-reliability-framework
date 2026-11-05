"""
Checks every experiments/*.yaml (no API calls):
  - it resolves (extends chain, target exists in targets.yaml)
  - its settings pass the target's own validate_settings()
  - every non-baseline experiment changes exactly ONE thing vs its parent
  - baseline matches the strong agent's model / temperature / recursion limit
    (prompt + tool descriptions are covered by test_prompt_equivalence.py)
    python tests/test_experiments.py
"""

import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import test_agent  # noqa: E402
from experiment_config import format_changes, load_experiment  # noqa: E402
from run_experiment import resolve_target  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


drafts = sorted(glob.glob(os.path.join("experiments", "draft_*.yaml")))
files = sorted(f for f in glob.glob(os.path.join("experiments", "*.yaml")) if f not in drafts)
for d in drafts:
    print(f"  SKIP  {d}: diagnose.py draft with TODOs (fill it in and rename it first)")
print(f"{len(files)} experiment files")
for path in files:
    exp = load_experiment(path)
    fn, accepts, validate = resolve_target(exp["target"])
    print(f"{exp['name']}:")
    ok("resolves; target accepts settings", accepts and fn.__module__ == "agent_configurable", exp["target"])
    import retrieval
    if exp["settings"].get("retrieval", {}).get("method") == "embeddings" and not retrieval.embeddings_available():
        print("  SKIP  settings validation: fastembed not installed (pip install -r requirements-rag.txt)")
    else:
        try:
            validate(exp["settings"])
            ok("settings valid for the target", True)
        except Exception as e:
            ok("settings valid for the target", False, str(e))
    if exp["parent"] is None:
        ok("baseline: no parent, no changes", exp["changes_vs_parent"] == [], format_changes(exp))
    else:
        ok("exactly ONE change vs parent", len(exp["changes_vs_parent"]) == 1, format_changes(exp))

base = load_experiment(os.path.join("experiments", "baseline.yaml"))["settings"]
print("baseline vs strong agent (test_agent.py):")
ok("model", base["model"] == test_agent.AGENT_MODEL, f"{base['model']} vs {test_agent.AGENT_MODEL}")
ok("temperature 0", base["temperature"] == 0 and test_agent.llm.bound.temperature == 1e-08)
ok("recursion_limit 25 (as in test_agent.run_agent)", base["recursion_limit"] == 25)
ok("prompt + docstrings point at the strong agent's copies",
   (base["prompt"], base["docstrings"]) == ("prompts/v1.txt", "docstrings/detailed.yaml"))
ok("retrieval substring (= test_agent.faq_search)", base["retrieval"] == {"method": "substring"})

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
