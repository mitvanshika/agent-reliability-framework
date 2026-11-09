"""
Tests for the /experiments dashboard routes in app.py (Flask test client,
fake runs, no API calls).
    python tests/test_dashboard.py
"""

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
SECRET = "gsk_THIS_KEY_MUST_NEVER_REACH_THE_BROWSER"
os.environ["GROQ_API_KEY"] = SECRET

import app as appmod  # noqa: E402
import compare  # noqa: E402
import judge_run  # noqa: E402
import promote_baseline  # noqa: E402
import run_experiment  # noqa: E402
from fake_runs import make_run  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
runs = os.path.join(root, "runs")
appmod.RUNS_DIR, appmod.BASELINES_DIR = runs, os.path.join(root, "baselines")
compare.BASELINES_DIR, compare.NOISE_DIR = appmod.BASELINES_DIR, os.path.join(root, "noise")
run_experiment.PRICING_FILE = os.path.join(root, "none.yaml")
IDS = [f"s{i:02d}" for i in range(1, 9)]


def finish(run_dir, tokens):
    meta = json.load(open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8"))
    meta["totals"] = {"total_tokens": tokens, "estimated_cost_usd": None}
    json.dump(meta, open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        judge_run.main([run_dir, "--summary-only"])


base = make_run(runs, "baseline", {s: 9.0 for s in IDS}, timestamp="20261101_090000")
finish(base, 12000)
with contextlib.redirect_stdout(io.StringIO()):
    promote_baseline.main([base, "--yes", "--note", "<b>first</b>"])
bad = make_run(runs, "exp_01_weak_prompt", {s: (3.0 if i < 6 else 9.0) for i, s in enumerate(IDS)},
               settings={"model": "m1", "prompt": "prompts/weak.txt"}, timestamp="20261102_090000")
finish(bad, 9000)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        compare.main([bad])
    except SystemExit:
        pass
xss = make_run(runs, "x_script", {"s01": 9.0}, judged=False, timestamp="20261103_090000")
meta = json.load(open(os.path.join(xss, "run_meta.json"), encoding="utf-8"))
meta["experiment"] = "<script>alert(1)</script>"
json.dump(meta, open(os.path.join(xss, "run_meta.json"), "w", encoding="utf-8"))
os.makedirs(os.path.join(runs, "not_a_run"))                        # no run_meta.json -> ignored

client = appmod.app.test_client()
bodies = []


def get(url):
    r = client.get(url)
    bodies.append(r.get_data(as_text=True))
    return r


print("page:")
r = get("/experiments")
html = r.get_data(as_text=True)
ok("GET /experiments renders", r.status_code == 200 and "Quality vs cost" in html and "/api/experiments/runs" in html)
node = shutil.which("node")
if node:
    js = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
    path = os.path.join(root, "page.js")
    open(path, "w", encoding="utf-8").write(js)
    ok("page JavaScript parses (node --check)", subprocess.run([node, "--check", path], capture_output=True).returncode == 0)
if node:
    # Run the page's own renderRuns() in node with a tiny fake DOM and a malicious run name.
    harness = """
const els = {};
function el(id) { return els[id] || (els[id] = { innerHTML: '', textContent: '', hidden: true,
  querySelectorAll: () => [], addEventListener: () => {} }); }
global.document = {
  getElementById: el,
  querySelectorAll: () => [],
  createElement: () => { let t = ''; return {
    set textContent(v) { t = v; },
    get innerHTML() { return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); } }; },
};
global.fetch = async () => ({ ok: true, json: async () => [] });
""" + js.replace("load();", "") + """
RUNS = [{ run_id: 'x', experiment: '<script>alert(1)</script>', what_changed: '<img src=x onerror=alert(1)>',
          timestamp: '20261106_120000', judged: false, verdict: null, is_baseline: false }];
renderRuns();
process.stdout.write(el('runs').innerHTML);
"""
    path2 = os.path.join(root, "xss.js")
    open(path2, "w", encoding="utf-8").write(harness)
    out = subprocess.run([node, path2], capture_output=True, text=True, encoding="utf-8")
    rendered = out.stdout
    ok("malicious run name / change text rendered as text, not HTML",
       out.returncode == 0 and "<script>" not in rendered and "<img" not in rendered
       and "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered, (out.stderr or rendered)[:120])

print("runs list:")
r = get("/api/experiments/runs")
data = r.get_json()
ok("3 runs, newest first, folder without run_meta ignored", [d["experiment"] for d in data]
   == ["<script>alert(1)</script>", "exp_01_weak_prompt", "baseline"], [d["run_id"] for d in data])
b = next(d for d in data if d["experiment"] == "baseline")
w = next(d for d in data if d["experiment"] == "exp_01_weak_prompt")
ok("baseline row: star, pass rate, score, tokens", b["is_baseline"] and b["pass_rate"] == 1.0 and b["mean_final_score"] == 9.0
   and b["total_tokens"] == 12000 and b["what_changed"] == "(baseline: no parent)")
ok("experiment row: what changed, verdict DECLINED, not the baseline", not w["is_baseline"] and w["verdict"] == "DECLINED"
   and w["pass_rate"] == 0.25 and w["what_changed"].startswith("settings.prompt: prompts/v1.txt -> ")
   and w["what_changed"].endswith("(vs baseline)"), w["what_changed"])
x = next(d for d in data if d["run_id"].startswith("x_script"))
ok("unjudged run listed as not judged", x["judged"] is False and x["pass_rate"] is None and x["verdict"] is None)

print("run detail:")
r = get(f"/api/experiments/runs/{os.path.basename(bad)}")
d = r.get_json()
ok("comparison rows with directions", r.status_code == 200 and sum(1 for row in d["comparison"]["rows"]
   if row["direction"] == "down") == 6 and d["comparison"]["verdict"] == "DECLINED")
ok("diagnosis included", len(d["diagnosis"]) == 6 and all("cause" in x and "suggestion" in x for x in d["diagnosis"]))
ok("per-judge means included", set(d["per_judge_mean"]) == {"correctness", "tool_efficiency", "completeness", "groundedness"})
r = get(f"/api/experiments/runs/{os.path.basename(base)}")
ok("baseline detail: no comparison yet", r.status_code == 200 and r.get_json()["comparison"] is None)

print("unknown / traversal -> 404:")
for bad_id in ["nope", "not_a_run", "..", "..%2F..%2Fapp.py", "%2E%2E", "runs%5C..%5Capp.py"]:
    r = get(f"/api/experiments/runs/{bad_id}")
    ok(f"{bad_id!r} -> 404", r.status_code == 404, r.status_code)
ok("app._run_path rejects '../app.py'", appmod._run_path("../app.py") is None and appmod._run_path("..") is None)

print("baselines:")
r = get("/api/experiments/baselines")
bl = r.get_json()
ok("one baseline per target, note returned as data", len(bl) == 1 and bl[0]["target"] == "configurable"
   and bl[0]["note"] == "<b>first</b>")

print("no secrets, existing routes untouched:")
ok("the API key never appears in any response", all(SECRET not in body for body in bodies))
ok("GET /track1 is the investigator page", client.get("/track1").status_code == 200
   and "Agent Reliability Investigator" in client.get("/track1").get_data(as_text=True))
appmod.STRONG_FOLDER, appmod.WEAK_FOLDER = os.path.join(root, "nope1"), os.path.join(root, "nope2")
ok("GET /api/data still works", client.get("/api/data").get_json() == {"strong": {}, "weak": {}})

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
