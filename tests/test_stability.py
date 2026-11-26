"""
Tests for stability.py: why a question is unstable across runs where nothing changed
(SYSTEM / JUDGE / WORDING / borderline), and its report in calibrate_noise.py --from-runs.
Built from the real fdd_q20 / fdd_q25 / fdd_q09 cases of the Financial DD A/A test. No API calls.
    python tests/test_stability.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import stability  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def trace(score, answer, evidence, judges, steps=None, error=None):
    t = {"final_score": score, "answer": answer, "tool_outputs": evidence, "trajectory": steps, "error": error}
    t.update({f"{n}_result": {"score": s} for n, s in judges.items()})
    return t


def retrieve(query, note):
    return {"type": "tool", "name": "retrieve", "args": {"query": query, "strategy": "decompose"},
            "output": "chunks: ...", "note": note}


print("the four kinds, on the real A/A cases:")
q20 = stability.explain_one([
    trace(9.1, "A one-time $10.2 billion tax charge (EU State Aid) caused the dip.",
          ["tax charge chunk", "income statement", "eu state aid"], {"correctness_gt": 6, "completeness": 10},
          [retrieve("Apple net income FY2024", "1 of 6 relevant"),
           retrieve("Provision for income taxes; one-time items or charges (legal settlement, tax charge)",
                    "5 of 10 relevant; split into 4 parts")]),
    trace(4.0, "Taxes rose to $29,749 million, more than offsetting higher operating income.",
          ["income statement", "operating income table", "other income"], {"correctness_gt": 2, "completeness": 0},
          [retrieve("Apple net income FY2024", "1 of 6 relevant"),
           retrieve("Provision for income taxes; one-time items (legal settlements) disclosed",
                    "1 of 10 relevant; split into 4 parts")])])
ok("fdd_q20: different evidence -> SYSTEM", q20["kind"] == "SYSTEM" and q20["evidence_overlap"] < 0.5, q20)
text = "\n".join(q20["detail"])
ok("...shows the search step that differed and the words only one run used ('tax charge')",
   "step 2 (retrieve) searched differently" in text and "only in run 1:" in text and "tax charge" in text
   and "run 1 found: 5 of 10 relevant" in text and "run 2 found: 1 of 10 relevant" in text, text)
ok("...and which judges moved", q20["judges_moved"] == {"completeness": [10, 0], "correctness_gt": [6, 2]},
   q20["judges_moved"])
ok("...a large drop is not 'borderline'", q20["borderline"] is False)

same = "No answer: only 0 relevant chunk(s) found after 1 attempt(s) [S1]."
q25 = stability.explain_one([trace(8.6, same, ["x"], {"correctness_gt": 5}),
                             trace(10.0, same.replace("[S1]", "[S2]"), ["x"], {"correctness_gt": 10})])
ok("fdd_q25: the same answer word for word (citation markers ignored) -> JUDGE", q25["kind"] == "JUDGE", q25)

q09 = stability.explain_one([
    trace(8.0, "Tariffs could raise costs and disrupt the supply chain and component availability.",
          ["tariff chunk", "trade chunk"], {"correctness_gt": 4, "completeness": 10}),
    trace(7.0 - 0.02, "Trade restrictions and tariffs may increase Apple's costs and limit parts supply.",
          ["tariff chunk", "trade chunk"], {"correctness_gt": 2, "completeness": 8})])
ok("fdd_q09: same evidence, reworded answer -> WORDING", q09["kind"] == "WORDING", q09)
ok("...a small move across the pass mark -> borderline", q09["borderline"] is True)

crash = stability.explain_one([trace(9.0, "ok", ["x"], {}), trace(None, None, [], {}, error="Model call failed")])
ok("a run that crashed -> SYSTEM, says which run", crash["kind"] == "SYSTEM" and "run 2 crashed" in crash["why"])
noev = stability.explain_one([trace(9.0, "The council voted 7-2 for a library.", [], {}),
                              trace(5.0, "A new library opens in 2027 on Elm Street.", [], {})])
ok("no evidence reported (e.g. a summarizer): different answers -> SYSTEM, says why",
   noev["kind"] == "SYSTEM" and "no evidence reported" in noev["why"], noev["why"])

print("printing:")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    stability.print_explanations({"fdd_q20": q20, "fdd_q25": q25, "fdd_q09": q09})
out = buf.getvalue()
ok("one line per question with kind and scores, plus the key", "fdd_q20" in out and "SYSTEM" in out
   and "JUDGE" in out and "WORDING + borderline" in out and "9.1 / 4.0" in out
   and "SYSTEM = the system behaved differently" in out, out)

print("calibrate_noise --from-runs explains its jumpy questions:")
import calibrate_noise  # noqa: E402
import compare  # noqa: E402
from fake_runs import make_run  # noqa: E402

root = tempfile.mkdtemp()
calibrate_noise.NOISE_DIR = compare.NOISE_DIR = os.path.join(root, "noise")
runs = os.path.join(root, "runs")
S = {"model": "m1"}
a = make_run(runs, "baseline", {"s1": 9.0, "s2": {"score": 9.0, "tool_outputs": ["chunk A", "chunk B"]}},
             settings=S, timestamp="20261106_100000")
b = make_run(runs, "baseline", {"s1": 9.0, "s2": {"score": 4.0, "tool_outputs": ["chunk C"]}},
             settings=S, timestamp="20261106_110000")
out = io.StringIO()
with contextlib.redirect_stdout(out):
    rep = calibrate_noise.main(["--from-runs", a, b])
ok("the jumpy question is explained (SYSTEM: different evidence)", rep["jumpy"] == ["s2"]
   and rep["jumpy_explained"]["s2"]["kind"] == "SYSTEM", rep.get("jumpy_explained"))
ok("printed after the table", "Why these questions are unstable" in out.getvalue())
with open(os.path.join(root, "noise", "configurable_noise.json"), encoding="utf-8") as f:
    ok("saved in the noise file", json.load(f)["jumpy_explained"]["s2"]["kind"] == "SYSTEM")
ok("the header no longer calls from-runs bands 'judge noise only'",
   "from_runs mode: agent + judge noise" in out.getvalue(), out.getvalue()[:200])

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
