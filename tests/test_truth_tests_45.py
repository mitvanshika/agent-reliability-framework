"""
Prep for truth tests 4 (planted refund bug) and 5 (paraphrases / retrieval): the
setting, the temporary experiment, both scenario sets and source_hits.py.
No LLM, no API calls.
    python tests/test_truth_tests_45.py
"""

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import agent_configurable as ac  # noqa: E402
import check_target  # noqa: E402
import retrieval  # noqa: E402
import run_experiment  # noqa: E402
import source_hits  # noqa: E402
import test_agent  # noqa: E402
from experiment_config import config_hash, load_experiment  # noqa: E402
from store import get_order  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


BASE = load_experiment(os.path.join("experiments", "baseline.yaml"))
BUG = load_experiment(os.path.join("experiments", "tmp_test4_refund_10_days.yaml"))

print("test 4: the planted bug (no edit to test_agent.py):")
ok("one change vs baseline: refund_window_days 10", BUG["changes_vs_parent"] == ["settings.refund_window_days: (unset) -> 10"])
ok("baseline untouched: no new key, same config_hash, test_agent's own function",
   "refund_window_days" not in BASE["settings"] and config_hash(BASE) == config_hash(
       {"target": "configurable", "settings": BASE["settings"]})
   and {t.name: t for t in ac.build_tools(BASE["settings"])}["refund_calculator"].func is test_agent.refund_calculator.func)
base_tool = {t.name: t for t in ac.build_tools(BASE["settings"])}["refund_calculator"]
bug_tool = {t.name: t for t in ac.build_tools(BUG["settings"])}["refund_calculator"]
ok("the model sees the same tool (name, description, schema)", (bug_tool.name, bug_tool.description, bug_tool.args)
   == (base_tool.name, base_tool.description, base_tool.args))
ok("only the comparison changed: an ineligible order gets byte-identical output",
   bug_tool.invoke({"order_total": 45.5, "days_since_purchase": 45})
   == base_tool.invoke({"order_total": 45.5, "days_since_purchase": 45}))
try:
    ac.validate_settings({**BASE["settings"], "refund_window_days": 0})
    ok("refund_window_days must be >= 1", False)
except ac.SettingsError:
    ok("refund_window_days must be >= 1", True)


def numbers(s):
    m = re.search(r"order (A\d{4})", s["question"])
    if m:
        o = get_order(m.group(1))
        return o["total"], o["days_since_purchase"]
    if re.search(r"\$\d", s["question"]):
        return float(re.search(r"\$(\d+\.\d\d)", s["question"]).group(1)), int(re.search(r"(\d+) days", s["question"]).group(1))
    return None


T4 = run_experiment.load_scenarios("scenarios_test4")
det = [s for s in T4 if s["category"] == "detector"]
ctl = [s for s in T4 if s["category"] == "control"]
ok("14 scenarios: 8 detectors + 6 controls", (len(T4), len(det), len(ctl)) == (14, 8, 6))
changed = {s["id"] for s in T4 if numbers(s) and base_tool.invoke(dict(zip(("order_total", "days_since_purchase"),
          numbers(s)))) != bug_tool.invoke(dict(zip(("order_total", "days_since_purchase"), numbers(s))))}
ok("every detector's tool output changes with the bug", {s["id"] for s in det} <= changed, sorted(changed))
ok("no control's tool output changes", not ({s["id"] for s in ctl} & changed), sorted(changed))
ok("every detector is in (10, 30] days, so the policy says eligible", all(10 < numbers(s)[1] <= 30 for s in det))
ok("answer keys follow the real 30-day policy", all("30-day" in s["expected_answer"] or "30 days" in s["expected_answer"]
   for s in det))
ok("8 one-way flips are enough for McNemar (2 x 0.5^8 = 0.0078; even 6 of 8 -> 0.031)", 2 * 0.5 ** 6 < 0.05)

print("test 5: paraphrases:")
T5 = run_experiment.load_scenarios("scenarios_test5")
faqs = dict(retrieval.load_faqs())
with_faq = [s for s in T5 if s["faq_topic"] != "none"]
ok("12 scenarios: 10 with an FAQ (2 per topic), 2 with none", len(T5) == 12 and len(with_faq) == 10
   and sorted({s["faq_topic"] for s in with_faq}) == sorted(faqs))
ok("no question contains its FAQ's topic words (the substring search misses all 10 raw)",
   all(not retrieval.search(s["question"], "substring", 1, None) for s in with_faq))
ok("each expected_source phrase is in its own FAQ and in no other", all(
   s["expected_source"].lower() in faqs[s["faq_topic"]].lower()
   and not any(s["expected_source"].lower() in a.lower() for t, a in faqs.items() if t != s["faq_topic"]) for s in with_faq))
ok("the 'none' scenarios: no expected tools, must not invent", all(s["expected_tools"] is None and "Must not invent"
   in s["expected_answer"] for s in T5 if s["faq_topic"] == "none"))

print("both folders pass check_target's scenario checks:")
for folder in ("scenarios_test4", "scenarios_test5"):
    rep = check_target.Report()
    check_target.check_scenarios(folder, rep)
    ok(f"{folder}: valid, unique ids, ids = file names", not rep.failed and not any(s == "WARN" for s, *_ in rep.rows),
       rep.rows)
ok("suite names: separate baselines per test", (run_experiment.suite_name("scenarios_test4", "configurable"),
   run_experiment.suite_name("scenarios_test5", "configurable")) == ("scenarios_test4", "scenarios_test5"))

print("source_hits.py (plain-code retrieval check):")
root = tempfile.mkdtemp()
d = os.path.join(root, "run")
os.makedirs(os.path.join(d, "traces"))
json.dump({"experiment": "x", "scenario_dir": "scenarios_test5", "resolved_config": {"settings": {
    "retrieval": {"method": "bm25"}}}}, open(os.path.join(d, "run_meta.json"), "w"))
fake = {"t5_ship_01": [faqs["shipping"]], "t5_ret_01": [faqs["customs"]], "t5_none_01": ["No matching FAQ found"],
        "t5_none_02": [faqs["customs"]]}
for sid, outs in fake.items():
    json.dump({"scenario_id": sid, "timestamp": "t", "tool_outputs": outs, "tools_used": ["faq_search"] * len(outs)},
              open(os.path.join(d, "traces", f"{sid}.json"), "w"))
with contextlib.redirect_stdout(io.StringIO()):
    r = source_hits.main([d])[0]
by = {x["scenario_id"]: x for x in r["rows"]}
ok("right FAQ -> hit; wrong FAQ -> miss", by["t5_ship_01"]["hit"] and not by["t5_ret_01"]["hit"])
ok("'none' + nothing found -> correct; 'none' + another FAQ fetched -> WRONG source",
   by["t5_none_01"]["hit"] and not by["t5_none_02"]["hit"] and "WRONG source" in by["t5_none_02"]["result"])
ok("hit rate over the answered scenarios, saved", r["hits"] == 2 and r["scenarios"] == 4
   and os.path.isfile(os.path.join(d, "source_hits.json")))
shutil.rmtree(root, ignore_errors=True)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
