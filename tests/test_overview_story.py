"""
The guided story on the Overview page (story.py + templates/overview.html), against the
demo_data/ snapshot. Expected values are read from the same run/config files the page
uses, so the test checks that the page shows the data rather than numbers typed in.
No API calls.  python tests/test_overview_story.py
"""

import html as htmllib
import json
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("ACTIONS_DB", os.path.join(tempfile.mkdtemp(), "actions.db"))

import app as appmod  # noqa: E402
import story  # noqa: E402
from trace_utils import load_latest_traces  # noqa: E402

D = "demo_data"
failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def section(h, sid):
    i = h.index(f'id="{sid}"')
    return h[i:h.index("</section>", i)]


def text(s):
    return " ".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", s)).split())


def jread(*p):
    with open(os.path.join(*p), encoding="utf-8") as f:
        return json.load(f)


appmod.RUNS_DIR, appmod.BASELINES_DIR, appmod.NOISE_DIR = f"{D}/runs", f"{D}/baselines", f"{D}/noise"
appmod.EVIDENCE_DIR, appmod.CONFIG_DIR = f"{D}/evidence", f"{D}/config"
c = appmod.app.test_client()
r = c.get("/")
h = r.get_data(as_text=True)
ok("/ renders", r.status_code == 200, r.status_code)

BASE = story.baseline_run_id(f"{D}/baselines")
RUN = story.known_bad_run()
R = f"{D}/runs"
ok("baseline and known-bad runs found", BASE == "baseline_20261106_101856"
   and RUN == "exp_01_weak_prompt_20261107_164943", (BASE, RUN))

print("\n[structure]")
ids = ["agent", "tests", "evaluation", "change", "result", "up-close", "diagnosis", "trust", "explore"]
ok("9 story sections, in order", all(f'id="{i}"' in h for i in ids)
   and [h.index(f'id="{i}"') for i in ids] == sorted(h.index(f'id="{i}"') for i in ids))
ok("table of contents links every section", all(f'href="#{i}"' in h for i in ids))
ok("truth-test table kept (compact)", "Truth tests: does the framework tell the truth?" in h and "compact-panel" in h)
used = set(re.findall(r'class="term" tabindex="0" data-tip="([^"]*)"', h))
ok("tooltips for all 7 terms", all(htmllib.escape(v) in used or v in {htmllib.unescape(u) for u in used}
                                   for v in story.GLOSSARY.values()), len(used))

print("\n[1. the agent]")
s = section(h, "agent")
with open(f"{D}/config/prompts/v1.txt", encoding="utf-8") as f:
    v1 = " ".join(f.read().split())
ok("prompt shown word for word (= prompts/v1.txt and the trace)", v1 in text(s))
ok("4 tools with their descriptions", all(t in s for t in ("order_lookup", "refund_calculator", "faq_search",
                                                          "escalate_to_human")) and "Searches FAQ topics" in s)
t = jread(R, BASE, "traces", "edge_01.json")
calls = [m["tool_calls"][0]["function"]["name"] for m in t["messages"] if m.get("tool_calls")]
pos = [s.find(f'<span class="tool">{n}</span>(') for n in calls]
ok("example edge_01: its tool calls, in order", "edge_01" in s and all(p > 0 for p in pos) and pos == sorted(pos), calls)
ok("example: each tool result shown", all(htmllib.escape(o) in s or o in text(s) for o in t["tool_outputs"]))
ok("link to its step timeline (200)", f'href="/runs/{BASE}/edge_01"' in s
   and c.get(f"/runs/{BASE}/edge_01").status_code == 200)

print("\n[2. the tests]")
s = text(section(h, "tests"))
meta = jread(R, BASE, "run_meta.json")
ok("31 questions with answer keys", f"{len(meta['scenario_ids'])} questions with answer keys" in s)
import yaml  # noqa: E402
from markupsafe import escape  # noqa: E402
sc = [yaml.safe_load(open(f"{D}/config/scenarios/{i}.yaml", encoding="utf-8")) for i in meta["scenario_ids"]]
cats = {}
for y in sc:
    cats[y["category"]] = cats.get(y["category"], 0) + 1
ok("category counts from the scenario files", all(f"{n} {k} questions" in s for k, n in cats.items()), cats)
samples = [y for y in sc if y["id"] in ("normal_01", "rag_01", "gen_001")]
ok("3 sample questions with their answer keys", len(samples) == 3
   and all(y["question"] in s and " ".join(y["expected_answer"].split()) in s for y in samples))

print("\n[3. evaluation]")
s = section(h, "evaluation")
js = jread(R, RUN, "summary.json")["judge_set"]
ok("diagram: answer -> judges -> score -> baseline -> tests -> verdict -> diagnosis",
   all(x in text(s) for x in ("Agent's answer", "LLM judges in parallel", "Final score", "Compare with the",
                              "McNemar", "Wilcoxon", "Verdict", "Diagnosis")))
ok("one plain line per judge", all(f'<td class="mono tool">{j}</td>' in s for j in js["llm_judges"] + js["checks"]))
ok("pass >= 7 and noise band explained", "7 or more out of 10" in text(s) and "noise band" in text(s).lower())

print("\n[4. the change]")
s = section(h, "change")
ok("v1 vs weak side by side, with the difference highlighted", "prompts/v1.txt" in s and "prompts/weak.txt" in s
   and "<del>" in s and "<ins>" in s and "Help" in s)
ok("only the prompt changed (from run_meta)", "settings.prompt: prompts/v1.txt -&gt; prompts/weak.txt" in s)

print("\n[5. the result]")
s = text(section(h, "result"))
cmp = jread(R, RUN, "comparison.json")
sb, sr = jread(R, BASE, "summary.json"), jread(R, RUN, "summary.json")
ok("verdict DECLINED", cmp["verdict"] == "DECLINED" and "The result: DECLINED" in s)
ok("pass rate before -> after", f"{round(sb['pass_rate'] * 100)}% → {round(sr['pass_rate'] * 100)}%" in s
   and f"{sb['passed']}/31 → {sr['passed']}/31" in s, (sb["pass_rate"], sr["pass_rate"]))
ok("flips: 10 worse, 0 better", "10 worse, 0 better" in s and cmp["flips"] == {"pass_to_fail": 10, "fail_to_pass": 0})
ok("p-values each with a plain sentence", f"McNemar p={cmp['mcnemar']['p']:.3f}" in s and "about 2 in 1,000" in s
   and f"Wilcoxon p={cmp['wilcoxon']['p']:.3f}" in s and "about 2 in 100" in s)
pct = cmp["resources"]["total_tokens"]["pct"]
ok("tokens: 26% cheaper but worse", f"{round(-pct)}% fewer tokens" in s and "cheaper, but worse" in s, pct)

print("\n[6. gen_004 up close]")
s = section(h, "up-close")
ok("both runs' tool calls and answers", s.count('class="c-tool"') == len(jread(R, BASE, "traces", "gen_004.json")["tools_used"])
   + len(jread(R, RUN, "traces", "gen_004.json")["tools_used"]) and s.count("md-answer") == 2)
jb = load_latest_traces(f"{R}/{BASE}/judged")["gen_004"]
jr = load_latest_traces(f"{R}/{RUN}/judged")["gen_004"]
reasons = [j[f"{n}_result"]["reason"] for j in (jb, jr) for n in story.JUDGE_ORDER if j.get(f"{n}_result")]
ok("every judge's score WITH its written reason (both sides)", len(reasons) >= 10
   and all(str(escape(x)) in s for x in reasons), len(reasons))
ok("step-timeline links (200)", all(f'href="/runs/{x}/gen_004"' in s and c.get(f"/runs/{x}/gen_004").status_code == 200
                                    for x in (BASE, RUN)))
ok("no raw Markdown in the answers", "**" not in s)

print("\n[7. diagnosis]")
s = section(h, "diagnosis")
dg = jread(R, RUN, "diagnosis.json")["diagnoses"]
ok("every ▼ question with its cause and suggestion", s.count('<span class="badge cause">') == len(dg)
   and all(str(escape(x["suggestion"])) in s for x in dg), len(dg))
ok("causes summary", all(k in text(s) for k in ("PROMPT", "RAG", "TOOL DESCRIPTIONS")))
ok("honest note: gen_004 labelled RAG, but the agent stopped searching", "symptoms, not always root causes" in s
   and "gen_004" in text(s.split('class="note"')[1]) and "stopped searching" in s)

print("\n[8. why trust it]")
s = section(h, "trust")
ok("A/A result", "NO REAL DIFFERENCE" in s and "McNemar p=0.250" in s and 'href="/evidence/aa_test"' in s)
ok("judge noise: identical answer, 9 vs 2", "same answer, 9 vs 2" in s
   and jread(R, BASE, "traces", "edge_03.json")["answer"] == jread(R, "baseline_20261106_111506", "traces", "edge_03.json")["answer"])
ok("judge fix, linked", 'href="/evidence/judge_fix_completeness"' in s and "gen_005 (invented): 0" in s)

print("\n[9. explore]")
s = section(h, "explore")
ok("links to Experiments, Evidence, Systems", all(f'href="{p}"' in s for p in ("/experiments", "/evidence", "/systems")))

print("\n[robustness]")
empty = tempfile.mkdtemp()
appmod.RUNS_DIR = appmod.BASELINES_DIR = appmod.NOISE_DIR = appmod.CONFIG_DIR = empty
ok("no data at all: / still renders, story sections left out", c.get("/").status_code == 200
   and 'id="result"' not in c.get("/").get_data(as_text=True))

print(f"\n{'FAIL: ' + ', '.join(failures) if failures else 'PASS'}")
sys.exit(1 if failures else 0)
