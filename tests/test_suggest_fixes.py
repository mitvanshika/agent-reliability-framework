"""
Tests for the fix suggester (suggest/, suggest_fixes.py) with a fake model: 0 tokens, no network.
Problems found from run files, the read-only sandbox (secrets never readable or sent), the patch
check (never applies), the agent's budgets, and the command (asks first, saves after each problem,
continues where it stopped).
    python tests/test_suggest_fixes.py
"""

import builtins
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import suggest_fixes  # noqa: E402
from suggest import problems as P  # noqa: E402
from suggest.agent import FakeClient, suggest_one  # noqa: E402
from suggest.tools import Refused, Sandbox, check_patch, redact  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail[:300]}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data if isinstance(data, str) else json.dumps(data))


root = tempfile.mkdtemp()
runs = os.path.join(root, "runs")


def trace(sid, **kw):
    return {"scenario_id": sid, "question": f"Question {sid}?", "answer": f"answer {sid}", "error": None,
            "timestamp": "2026-11-26T10:00:00", **kw}


# a baseline run, an experiment run and an A/A run of target "toy"
meta = {"resolved_config": {"target": "toy", "settings": {"top_k": 3}}, "experiment": "toy_exp"}
write(os.path.join(runs, "base", "run_meta.json"), {**meta, "experiment": "toy_base"})
write(os.path.join(runs, "base", "judged", "q3.json"), trace("q3", answer="Revenue was 10.", final_score=10,
      correctness_gt_result={"score": 10, "reason": "matches"}))
write(os.path.join(runs, "exp", "run_meta.json"), meta)
write(os.path.join(runs, "exp", "traces", "q1.json"), trace("q1", error="ValueError: bad json", error_type="crash"))
write(os.path.join(runs, "exp", "traces", "q2.json"), trace("q2", error="429 too many", error_type="rate_limit"))
for sid in ("q3", "q4"):
    write(os.path.join(runs, "exp", "traces", f"{sid}.json"), trace(sid, trajectory=[
        {"index": 0, "type": "tool", "name": "retrieve", "args": {"query": "x"}, "note": "2 of 6 relevant"},
        {"index": 1, "type": "llm", "name": "m", "label": "chunk_grades", "error": "BadRequestError", "tokens": None},
        {"index": 2, "type": "llm", "name": "m", "label": "grounded_answer", "tokens": {"total": 900}}]))
write(os.path.join(runs, "exp", "judged", "q3.json"), trace("q3", answer="Revenue was 7.", final_score=6,
      correctness_gt_result={"score": 4, "reason": "wrong number"}))
write(os.path.join(runs, "exp", "comparison.json"), {
    "target": "toy", "config_diff": ["settings.top_k: (unset) -> 3"], "baseline_run_dir": "runs\\base",
    "rows": [{"scenario_id": "q3", "baseline_score": 10, "run_score": 6, "band": 0.5, "delta": -4}],
    "diagnosis": [{"scenario_id": "q3", "cause": "PROMPT (used info badly)", "evidence": "correctness_gt -6",
                   "suggestion": "try a prompt change"},
                  {"scenario_id": "q5", "cause": "INFRA", "evidence": "rate_limit"},
                  {"scenario_id": "q6", "cause": "UNCLEAR", "evidence": "several"}]})
write(os.path.join(runs, "aa", "run_meta.json"), {**meta, "experiment": "toy_base"})
write(os.path.join(runs, "aa", "comparison.json"), {"target": "toy", "config_diff": ["settings: {} -> (unset)"],
                                                    "rows": [{"scenario_id": "q3", "delta": 1.0}]})

print("problems (from the run files, no calls):")
found, skipped = P.problems(os.path.join(runs, "exp"), runs)
ids = {p["id"]: p for p in found}
ok("a crash is a problem", "crash:crash" in ids and ids["crash:crash"]["scenario_ids"] == ["q1"], list(ids))
ok("the same failed call in 2 questions is ONE problem", ids.get("llm_error:chunk_grades:BadRequestError", {})
   .get("scenario_ids") == ["q3", "q4"], list(ids))
ok("a diagnosed drop is a problem, with both answers and the judge reasons",
   ids.get("drop:q3", {}).get("evidence", {}).get("baseline_answer") == "Revenue was 10."
   and ids["drop:q3"]["evidence"]["run_judges"]["correctness_gt"]["reason"] == "wrong number", ids.get("drop:q3"))
ok("...marked maybe noise: q3 also moved in the A/A test", ids["drop:q3"].get("maybe_noise") is True)
ok("rate limit / INFRA / UNCLEAR skipped with a reason",
   {s["scenario_id"] for s in skipped} == {"q2", "q5", "q6"} and all(s["why"] for s in skipped), skipped)

print("\nthe sandbox (read-only, secrets never shown):")
code = os.path.join(root, "code")
write(os.path.join(code, "app.py"), "KEY = 'gsk_" + "A" * 30 + "'\n" + "\n".join(f"line {i}" for i in range(2, 301)))
write(os.path.join(code, ".env"), "GROQ_API_KEY=gsk_secret_value_1234567890")
write(os.path.join(code, "sub", "prod.env"), "X=1")
write(os.path.join(code, ".venv", "lib", "x.py"), "venv code")
write(os.path.join(code, "runs", "r", "t.json"), "{}")
write(os.path.join(root, "outside.py"), "secret outside")
box = Sandbox([code])
for path, why in ((".env", "the .env file"), ("sub/prod.env", "another .env"), (".venv/lib/x.py", "a venv"),
                  ("runs/r/t.json", "runs/"), ("../outside.py", "outside the folder"),
                  (os.path.join(root, "outside.py"), "an absolute path outside")):
    try:
        box.read_file(path)
        ok(f"refuses {why}", False)
    except Refused:
        ok(f"refuses {why}", True)
if hasattr(os, "symlink"):
    try:
        os.symlink(os.path.join(root, "outside.py"), os.path.join(code, "link.py"))
        try:
            box.read_file("link.py")
            ok("refuses a link pointing outside", False)
        except Refused:
            ok("refuses a link pointing outside", True)
    except OSError:
        print("  SKIP  links not allowed here")
out = box.read_file("app.py", 1, 3)
ok("a key inside code is redacted", "gsk_A" not in out and "[REDACTED]" in out, out)
ok("lines are numbered", "    2  line 2" in out, out)
ok("list_files hides .env, venvs, runs", ".env" not in box.list_files(".") and ".venv" not in box.list_files(".")
   and "runs" not in box.list_files("."), box.list_files("."))
ok("search skips .env and venvs", "secret" not in box.search_code("gsk_") and "venv code" not in box.search_code("venv"))
ok("files the model read are recorded", "app.py:1-3" in box.reads, box.reads)
ok("redact() catches OpenAI/Anthropic/GitHub keys too", "[REDACTED]" in redact("sk-ant-abcdefghijklmnop")
   and "[REDACTED]" in redact("ghp_" + "x" * 30))

sysdir = os.path.join(root, "Their-System")
write(os.path.join(sysdir, "src", "answer.py"), "def grounded_answer():\n    return 1\n")
write(os.path.join(sysdir, ".env"), "GROQ_API_KEY=gsk_their_key_1234567890")
two = Sandbox([code, sysdir])
hit = two.search_code("grounded_answer")
ok("the system's files are shown as Their-System/<path>", "Their-System/src/answer.py:1" in hit, hit)
write(os.path.join(code, "fw_grounded.py"), "# grounded_answer is recorded here\n")
both = two.search_code("grounded_answer").splitlines()
ok("a search with no folder covers the system's folder FIRST, then the framework",
   both[0].startswith("Their-System/") and any(l.startswith("fw_grounded.py") for l in both), both)
ok("...and can be opened with exactly that path", "def grounded_answer" in two.read_file("Their-System/src/answer.py"))
ok("listing the system's folder by its name works", "src/" in two.list_files("Their-System"), two.list_files("Their-System"))
try:
    two.read_file("Their-System/.env")
    ok("the system's own .env is refused too", False)
except Refused:
    ok("the system's own .env is refused too", True)
ok("a patch on the system's file, with the path as shown, fits",
   check_patch(two, "--- a/Their-System/src/answer.py\n+++ b/Their-System/src/answer.py\n@@ -2 +2 @@\n-    return 1\n+    return 2\n")[0])

print("\nthe patch check (never applies):")
good = "--- a/app.py\n+++ b/app.py\n@@ -3,2 +3,2 @@\n line 3\n-line 4\n+line four\n"
before = open(os.path.join(code, "app.py"), encoding="utf-8").read()
ok("a patch that matches the code fits", check_patch(box, good)[0], check_patch(box, good))
ok("...and the file is unchanged", open(os.path.join(code, "app.py"), encoding="utf-8").read() == before)
ok("a patch whose old lines aren't there doesn't fit", not check_patch(box, good.replace("-line 4", "-line 44"))[0])
ok("a patch on .env doesn't fit (not readable)", not check_patch(box, "--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-a\n+b\n")[0])
ok("not a diff -> doesn't fit", not check_patch(box, "just change line 4")[0])

print("\nthe agent (fake model):")
fc = FakeClient([("tool", "search_code", {"text": "line 4"}), ("tool", "read_file", {"path": ".env"}),
                 ("tool", "read_file", {"path": "app.py", "start": 1, "end": 10}),
                 ("submit", {"title": "fix", "what_is_wrong": "w", "evidence": ["app.py:4"], "change_type": "code",
                             "file": "app.py", "patch": good, "confidence": "medium", "how_to_check": "re-run q3"})])
r = suggest_one(ids["drop:q3"], box, fc, system_info={"target": "toy"})
ok("suggested, patch fits, 4 calls", r["status"] == "suggested" and r["patch_check"]["fits"] and r["steps"] == 4, r)
tool_msgs = [m["content"] for m in fc.requests[-1]["messages"] if m["role"] == "tool"]
ok("asking for .env gets 'refused' and its content is never sent",
   any(m.startswith("refused:") for m in tool_msgs)
   and not any("gsk_secret" in json.dumps(q["messages"]) for q in fc.requests), tool_msgs)
ok("temperature 0, one fixed model", all(q["temperature"] == 0 and q["model"] == fc.requests[0]["model"] for q in fc.requests))
fc = FakeClient([("tool", "list_files", {"path": "."})] * 30)
r = suggest_one(ids["drop:q3"], box, fc, max_steps=5)
final_call = fc.requests[-1]
ok("budget: on the last call only submit_suggestion is offered (and required)",
   [t["function"]["name"] for t in final_call["tools"]] == ["submit_suggestion"] and final_call["tool_choice"] == "required"
   and r["steps"] == 5, r)
ok("...and the model is told why", "budget for this problem is almost used" in json.dumps(final_call["messages"]))
ok("the first message says the budget", "BUDGET: at most 5 calls" in fc.requests[0]["messages"][1]["content"])
fc = FakeClient([("tool", "list_files", {"path": "."})] * 30, tokens_per_call=30_000)
r = suggest_one(ids["drop:q3"], box, fc, max_tokens=60_000)
ok("token budget: made to submit at 80%", r["steps"] == 3 and len(fc.requests[-1]["tools"]) == 1, r)

# what Groq did on the first real run: on the final call the model tried another tool and the API refused it
refused = RuntimeError("Error code: 400 - {'error': {'message': \"Tool call validation failed: attempted to call tool "
                       "'search_code' which does not match request.tool_choice\", 'code': 'tool_use_failed'}}")
fc = FakeClient([("tool", "list_files", {"path": "."}), ("raise", refused),
                 ("submit", {"title": "late", "what_is_wrong": "w", "evidence": [], "change_type": "setting",
                             "setting_change": "s", "confidence": "low", "how_to_check": "-"})], tokens_per_call=30_000)
r = suggest_one(ids["drop:q3"], box, fc, max_tokens=60_000)
ok("a refused tool call is retried with a note, then the suggestion arrives", r["status"] == "suggested"
   and "rejected by the API" in json.dumps(fc.requests[-1]["messages"]), r)
fc = FakeClient([("raise", refused), ("raise", refused), ("raise", refused)])
r = suggest_one(ids["drop:q3"], box, fc)
ok("refused 3 times -> stopped, keeping the real step count and the reason",
   r["status"] == "stopped" and r["steps"] == 3 and "tool_use_failed" in r["reason"], r)
fc = FakeClient([("tool", "list_files", {"path": "."}), ("raise", ValueError("boom"))])
r = suggest_one(ids["drop:q3"], box, fc)
ok("any other error -> stopped, tokens and files so far kept", r["status"] == "stopped" and r["tokens"] == 1000
   and "boom" in r["reason"], r)
print("\nkeeping calls small (the tokens-per-minute limit makes big calls slow):")
fc = FakeClient([("tool", "read_file", {"path": "app.py", "start": s0, "end": s0 + 79}) for s0 in (1, 81, 161, 241)]
                + [submit_turn := ("submit", {"title": "t", "what_is_wrong": "w", "evidence": [], "change_type": "none",
                                              "confidence": "low", "how_to_check": "-"})])
seen = []
r = suggest_one(ids["drop:q3"], box, fc, on_step=lambda n, what, tok: seen.append((n, what)))
last = [m for m in fc.requests[-1]["messages"] if m["role"] == "tool"]
ok("4 reads: the 3 newest stay in full, the oldest keeps only which file/lines it was",
   len(last) == 4 and last[0]["content"].startswith("[earlier result shortened") and "app.py lines 1-80" in last[0]["content"]
   and all("line" in m["content"] and not m["content"].startswith("[earlier") for m in last[1:]), [m["content"][:60] for m in last])
ok("no internal markers are sent to the API", all(not any(k.startswith("_") for k in m) for q in fc.requests for m in q["messages"]))
ok("a read is 80 lines at most", box.read_file("app.py", 1, 500).count("\n") <= 82)
ok("progress is reported for every call and tool", any("read_file app.py:81-160" in w for _, w in seen)
   and sum(1 for _, w in seen if w == "thinking") == 5, seen)

r = suggest_one(ids["drop:q3"], box, FakeClient([("text", "hm"), ("text", "hm")]))
ok("a model that never submits -> stopped, with the reason", r["status"] == "stopped" and r["reason"], r)
r = suggest_one(ids["drop:q3"], box, FakeClient([("submit", {"title": "t", "what_is_wrong": "noise", "evidence": [],
                                                             "change_type": "none", "confidence": "low", "how_to_check": "-"})]))
ok("change_type none -> no_fix", r["status"] == "no_fix", r)
r = suggest_one(ids["drop:q3"], box, FakeClient([("submit", {"title": "t", "what_is_wrong": "w", "evidence": [],
                                                             "change_type": "code", "confidence": "low", "how_to_check": "-"})]))
ok("a code change without a patch is flagged", r["patch_check"] == {"fits": False, "why": "no patch given"}, r)

print("\nthe command:")


class NoCalls:
    def __getattr__(self, name):
        raise AssertionError("no API call expected")


def cli(argv, client=None, answer=None):
    real = builtins.input
    if answer is not None:
        builtins.input = lambda prompt="": answer
    try:
        with contextlib.redirect_stdout(io.StringIO()) as o:
            code = suggest_fixes.main(argv, client=client)
    finally:
        builtins.input = real
    return code, o.getvalue()


exp = os.path.join(runs, "exp")
suggest_fixes.source_dirs = lambda target: [code]
code_, out = cli([exp, "--list"], client=NoCalls())
ok("--list: the problems and the skipped ones, 0 calls", code_ == 0 and "got worse" in out and "skip" in out, out)
code_, out = cli([exp], client=NoCalls(), answer="n")
ok("answering no sends nothing", code_ == 0 and "Nothing sent" in out and not os.path.exists(os.path.join(exp, "suggestions.json")))
ok("says which folders it will read before asking", code in out, out)


def submit(title):
    return ("submit", {"title": title, "what_is_wrong": "w", "evidence": ["e"], "change_type": "setting",
                       "setting_change": "top_k back to 6", "confidence": "medium", "how_to_check": "re-run"})


fc = FakeClient([submit("one"), submit("two")])
code_, out = cli([exp, "--yes", "--max", "2"], client=fc)
saved = json.load(open(os.path.join(exp, "suggestions.json"), encoding="utf-8"))
ok("--max 2: two problems done and saved", code_ == 0 and len(saved["items"]) == 2, out)
md = open(os.path.join(exp, "suggestions.md"), encoding="utf-8").read()
ok("suggestions.md says nothing was applied, lists skipped", "Nothing here was applied" in md and "Skipped" in md, md[:300])
fc = FakeClient([submit("three")])
code_, out = cli([exp, "--yes"], client=fc)
saved = json.load(open(os.path.join(exp, "suggestions.json"), encoding="utf-8"))
ok("a re-run continues with the rest only", len(fc.requests) == 1 and len(saved["items"]) == 3, out)
code_, out = cli([exp, "--yes"], client=NoCalls())
ok("all done -> nothing new, 0 calls", "Nothing new" in out, out)
fc = FakeClient([submit("again")])
code_, out = cli([exp, "--yes", "--only", "q1", "--redo"], client=fc)
ok("--only q1 --redo: just that problem again", len(fc.requests) == 1 and code_ == 0, out)


class DailyLimit:
    def __init__(self):
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kw):
        raise RuntimeError("Rate limit reached: tokens per day (TPD)")


os.remove(os.path.join(exp, "suggestions.json"))
code_, out = cli([exp, "--yes"], client=DailyLimit())
ok("daily limit: stops, says to continue tomorrow", code_ == 1 and "tomorrow" in out, out)

print("\nwhere the system's code is (source_dirs):")
import importlib  # noqa: E402
importlib.reload(suggest_fixes)
os.environ["TOY_SRC"] = code
import target_config  # noqa: E402
real_targets = target_config._targets
target_config._targets = lambda: {"toy": {"module": "m", "function": "f", "source_dirs": ["$TOY_SRC"]},
                                  "fdd_like": {"module": "adapters.financial_dd", "function": "run"}}
try:
    ok("targets.yaml source_dirs, $VARS expanded", suggest_fixes.source_dirs("toy") == [os.path.abspath(code)],
       suggest_fixes.source_dirs("toy"))
    fdd = os.path.join(root, "Financial-Due-Diligence")
    write(os.path.join(fdd, "pyproject.toml"), "")
    write(os.path.join(fdd, ".venv", "Scripts", "python.exe"), "")
    os.environ["FDD_PYTHON"] = os.path.join(fdd, ".venv", "Scripts", "python.exe")
    ok("the FDD adapter finds its project from FDD_PYTHON", suggest_fixes.source_dirs("fdd_like") == [fdd],
       suggest_fixes.source_dirs("fdd_like"))
    ok("an unknown target: framework folder only", suggest_fixes.source_dirs("nope") == [])
finally:
    target_config._targets = real_targets

print("\ndashboard: the run's suggestions are in its API")
import app  # noqa: E402
app.RUNS_DIR = runs
write(os.path.join(exp, "suggestions.json"), {"model": "m", "items": [{"problem": {"title": "t"}, "status": "suggested"}]})
d = app.app.test_client().get("/api/experiments/runs/exp").get_json()
ok("/api/experiments/runs/<run> includes suggestions", (d.get("suggestions") or {}).get("items"), d.get("suggestions"))

shutil.rmtree(root, ignore_errors=True)
print("\nPASS" if not failures else f"\nFAIL: {failures}")
sys.exit(1 if failures else 0)
