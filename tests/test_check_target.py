"""
Tests for check_target.py (no LLM, no API calls).
    python tests/test_check_target.py
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import judges.base  # noqa: E402


def _no_llm(*a, **k):
    raise AssertionError("check_target reached an LLM judge")


judges.base.call_judge = _no_llm
import check_target  # noqa: E402
import run_suite  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
sys.path.insert(0, root)
open(os.path.join(root, "ct_targets.py"), "w").write(textwrap.dedent('''
    import time
    def good(question):
        return "an answer"
    def good_dict(question):
        return {"answer": "x", "tools_used": []}
    def returns_none(question):
        return None
    def returns_int(question):
        return 42
    def raises(question):
        raise RuntimeError("boom")
    def no_args():
        return "x"
    def two_required(question, mode):
        return "x"
    def reports_error(question):
        return {"answer": None, "error": "upstream 500"}
    def with_settings(question, settings=None):
        return "x"
    def no_settings_param(question):
        return "x"
    not_callable = 5
'''))


def scen(folder, files):
    d = os.path.join(root, folder)
    os.makedirs(d, exist_ok=True)
    for name, text in files.items():
        open(os.path.join(d, name), "w", encoding="utf-8").write(text)
    return d


GOOD = scen("sc_good", {"a.yaml": 'id: a\nquestion: "q1?"\nexpected_answer: "x"\n',
                        "b.yaml": 'id: b\nquestion: ""\nexpected_answer: "y"\n'})


def T(**kw):
    return {"t": {"module": "ct_targets", "function": "good", "scenarios": GOOD, **kw}}


def run(targets, call=False, experiment=None, name="t"):
    rep = check_target.check(name, call, experiment, targets)
    return rep, {c: (s, d) for s, c, d in rep.rows}, rep.rows


print("a good plain target:")
rep, by, rows = run(T())
ok("passes", not rep.failed, rows)
ok("no call by default (it may use a paid API)", by["test call"][0] == "SKIP")
rep, by, _ = run(T(), call=True)
ok("--call: string return OK", by["test call"][0] == "OK" and "returned a string" in by["test call"][1])
rep, by, _ = run(T(function="good_dict"), call=True)
ok("--call: dict with answer OK", by["test call"][0] == "OK" and "dict" in by["test call"][1])
ok("empty question \"\" allowed in a scenario", by["scenarios"][0] == "OK")

print("targets.yaml problems:")
rep, by, _ = run(T(), name="nope")
ok("unknown target", rep.failed and "no target 'nope'" in by["targets.yaml"][1])
for kw, words in (({"pass_threshold": 12}, "0 to 10"), ({"judges": ["corectness"]}, "unknown ['corectness']"),
                  ({"judge_set": "v9"}, "judge_set"), ({"scenarios": ""}, "scenarios"),
                  ({"descripton": "typo"}, "unknown keys"), ({"function": None}, "function: required")):
    rep, by, rows = run(T(**kw))
    ok(f"{list(kw)[0]} -> {words}", rep.failed and any(words in d for s, c, d in rows if s == "FAIL"), rows)

print("import / signature:")
for fn, words in (("missing_fn", "AttributeError"), ("not_callable", "not callable"), ("no_args", "takes no question"),
                  ("two_required", "must have defaults")):
    rep, by, rows = run(T(function=fn))
    ok(f"{fn} -> {words}", rep.failed and any(words in d for s, c, d in rows if s == "FAIL"), rows)
rep, by, rows = run(T(module="no_such_module_xyz"))
ok("module missing -> ModuleNotFoundError", rep.failed and "ModuleNotFoundError" in by["import"][1])
rep, by, rows = run(T(function="no_settings_param", accepts_settings=True))
ok("accepts_settings but no settings= parameter", any("must take settings=" in d for s, c, d in rows if s == "FAIL"), rows)
rep, by, rows = run(T(function="with_settings", accepts_settings=True))
ok("accepts_settings + no experiment for it -> warning, not failure", not rep.failed and by["settings"][0] == "WARN")

print("the one call:")
for fn, words in (("returns_none", "expected a string or a dict"), ("returns_int", "expected a string or a dict"),
                  ("raises", "raised RuntimeError: boom"), ("reports_error", "returned an error: upstream 500")):
    rep, by, _ = run(T(function=fn), call=True)
    ok(f"{fn} -> {words}", rep.failed and words in by["test call"][1], by["test call"])
check_target.CALL_TIMEOUT_S = 0.2
open(os.path.join(root, "ct_slow.py"), "w").write("import time\ndef slow(q):\n    time.sleep(2)\n    return 'x'\n")
rep, by, _ = run(T(module="ct_slow", function="slow"), call=True)
ok("hangs -> timeout failure", rep.failed and "no answer after" in by["test call"][1])
check_target.CALL_TIMEOUT_S = 30

print("scenario files:")
cases = {
    "parse": ({"a.yaml": "id: a\nquestion: [unclosed\n"}, "does not parse"),
    "no id": ({"a.yaml": 'question: "q"\n'}, "no id"),
    "no question": ({"a.yaml": "id: a\n"}, "no question"),
    "duplicate": ({"a.yaml": 'id: a\nquestion: "q"\n', "b.yaml": 'id: a\nquestion: "q"\n'}, "duplicate id 'a'"),
    "not a mapping": ({"a.yaml": "- a\n- b\n"}, "must be a mapping"),
}
for label, (files, words) in cases.items():
    d = scen(f"sc_{label.replace(' ', '_')}", files)
    rep, by, rows = run(T(scenarios=d))
    ok(f"{label} -> {words}", rep.failed and any(words in dd for s, c, dd in rows if s == "FAIL"), rows)
rep, by, _ = run(T(scenarios=os.path.join(root, "nowhere")))
ok("missing folder", rep.failed and "folder not found" in by["scenarios"][1])
d = scen("sc_empty", {})
rep, by, _ = run(T(scenarios=d))
ok("empty folder", rep.failed and "no .yaml files" in by["scenarios"][1])
d = scen("sc_nokey", {"a.yaml": 'id: a\nquestion: "q"\n'})
rep, by, _ = run(T(scenarios=d))
ok("no expected_answer -> warning (correctness falls back), not failure", not rep.failed and by["answer key"][0] == "WARN")
rep, by, _ = run(T(scenarios=d, judges=["groundedness"]))
ok("...not even asked when correctness_gt doesn't apply", "answer key" not in by)
d = scen("sc_name", {"x.yaml": 'id: other\nquestion: "q"\nexpected_answer: "a"\n'})
rep, by, rows = run(T(scenarios=d))
ok("id != file name -> warning only", not rep.failed and any("differs from the file name" in dd for s, c, dd in rows))

print("description + judges:")
rep, by, _ = run(T(judge_set="generic_v1"))
ok("generic_v1 without description -> warning that says why", by["description"][0] == "WARN"
   and "generic_v1" in by["description"][1])
rep, by, _ = run(T(judges=["correctness_gt", "groundedness"]))
ok("judges line shows what applies and what is N/A", "N/A: completeness, tool_efficiency" in by["judges"][1])
rep, by, _ = run(T(pass_threshold=6.5))
ok("threshold shown", "6.5" in by["pass threshold"][1])

print("tools opt-in:")
rep, by, rows = run(T(tools="ct_targets:missing"))
ok("bad tools: entry -> failure", rep.failed and "tools" in by and by["tools"][0] == "FAIL", rows)

print("the real targets (no call):")
real = run_suite.load_targets()
for name in ("configurable", "strong", "weak"):
    rep, by, rows = run(real, name=name)
    ok(f"{name}: passes", not rep.failed, [r for r in rows if r[0] == "FAIL"])
rep, by, _ = run(real, name="configurable")
ok("configurable: settings checked with its own validate_settings + 4 tools found",
   "validate_settings" in by["settings"][1] and by["tools"][1].startswith("4 tools"))
out = io.StringIO()
with contextlib.redirect_stdout(out):
    code = check_target.main(["configurable"])
ok("CLI: exit 0 and a PASS line", code == 0 and "PASS: 0 failed" in out.getvalue())
with contextlib.redirect_stdout(io.StringIO()):
    code = check_target.main(["not-a-target"])
ok("CLI: exit 1 on failure", code == 1)

shutil.rmtree(root)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
