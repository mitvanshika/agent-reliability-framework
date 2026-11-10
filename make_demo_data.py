"""
Build demo_data/: the read-only snapshot the dashboard serves with DEMO_MODE=1
(the public Render demo). No LLM calls unless --ask.

    python make_demo_data.py            # copy the runs, baselines, noise, evidence (0 API calls)
    python make_demo_data.py --ask      # also pre-compute the 4 Ask-card answers (4 Groq calls, asks first)

What is copied:
  runs/<run>/   the runs the dashboard points at: every truth-test run (overview.TRUTH_TESTS),
                each experiment's latest compared run (overview.latest_verdicts) and the
                runs the current baselines were promoted from. Per run: run_meta, summary,
                comparison, diagnosis, step_checks, pipeline + traces/, judged/, judged_steps/
                (judged_experiment_v1/ and other old judge folders are left out).
  baselines/*.json, noise/*.json, evidence/*.md, judged_traces(_weak)/ (Track 1) if present.
  config/       the prompt, tool-description and scenario files those runs point at
                (story.config_files), for the Overview story: what the runs actually used.
demo_data/ask_cache.json is kept as it is unless --ask is given. Its answers are computed
from the context built from demo_data/ itself, so they match what the demo shows.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import time
import sys
from datetime import datetime

import overview
import story

OUT = "demo_data"
RUN_FILES = ("run_meta.json", "summary.json", "comparison.json", "diagnosis.json", "step_checks.json", "pipeline.json")
RUN_DIRS = ("traces", "judged", "judged_steps")


def demo_runs(runs_dir="runs", baselines_dir="baselines"):
    ids = set()
    for t in overview.TRUTH_TESTS:
        if t.get("run"):
            ids.add(t["run"])
        ids.update(run for run, _, _ in t.get("checks") or [])
    ids.update(v["run_id"] for v in overview.latest_verdicts(runs_dir) if v.get("run_id"))
    for f in glob.glob(os.path.join(baselines_dir, "*.json")):
        with open(f, encoding="utf-8") as fh:
            b = json.load(fh)
        if b.get("run_dir"):
            ids.add(os.path.basename(str(b["run_dir"]).replace("\\", "/").rstrip("/")))
    return sorted(i for i in ids if os.path.isfile(os.path.join(runs_dir, i, "run_meta.json")))


def _copytree(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)


def snapshot(runs_dir="runs", out=OUT):
    keep_cache = None
    cache_path = os.path.join(out, "ask_cache.json")
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            keep_cache = f.read()
    for sub in ("runs", "baselines", "noise", "evidence", "judged_traces", "judged_traces_weak", "config"):
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
    runs = demo_runs(runs_dir)
    for run in runs:
        src, dst = os.path.join(runs_dir, run), os.path.join(out, "runs", run)
        os.makedirs(dst, exist_ok=True)
        for f in RUN_FILES:
            if os.path.isfile(os.path.join(src, f)):
                shutil.copy2(os.path.join(src, f), dst)
        for d in RUN_DIRS:
            _copytree(os.path.join(src, d), os.path.join(dst, d))
    for sub, pattern in (("baselines", "baselines/*.json"), ("noise", "noise/*.json"), ("evidence", "evidence/*.md")):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
        for f in sorted(glob.glob(pattern)):
            shutil.copy2(f, os.path.join(out, sub))
    for rel in story.config_files(runs_dir, runs):
        if os.path.isfile(rel):
            os.makedirs(os.path.join(out, "config", os.path.dirname(rel)), exist_ok=True)
            shutil.copy2(rel, os.path.join(out, "config", rel))
    for d in ("judged_traces", "judged_traces_weak"):
        if glob.glob(os.path.join(d, "*.json")):
            _copytree(d, os.path.join(out, d))
    if keep_cache is not None:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(keep_cache)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "git_commit": commit, "runs": runs,
                "note": "snapshot for DEMO_MODE=1; rebuild with python make_demo_data.py"}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return runs


def precompute_answers(out=OUT, yes=False, api_key=None, redo=()):
    """One Groq call per suggestion card, with the same prompt and call the live Ask box uses.
    Each answer records the hash of the context it was made from. An answer is kept (no call) when
    it was made from the current context, or when it exists and only other ids are in `redo`."""
    import ask_context
    import app as appmod   # not in demo mode here: groq_chat + the model name
    from dotenv import load_dotenv
    load_dotenv()
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY is not set (.env or environment).")
    prompt = ask_context.system_prompt(os.path.join(out, "runs"), os.path.join(out, "baselines"),
                                       os.path.join(out, "noise"))
    chash = ask_context.context_hash(prompt)
    old, prev = {}, {}
    try:
        with open(os.path.join(out, "ask_cache.json"), encoding="utf-8") as f:
            prev = json.load(f)
        for a in prev.get("answers") or []:
            a.setdefault("context_hash", prev.get("context_hash"))
            if a.get("answer") and a["id"] not in redo and (redo or a["context_hash"] == chash):
                old[a["id"]] = a
    except (OSError, ValueError):
        pass
    todo = [s for s in ask_context.SUGGESTIONS if s["id"] not in old]
    if not todo:
        print("All suggestion answers are already recorded for this context: no calls needed.")
        return prev
    print(f"About to make {len(todo)} Groq call(s) ({appmod.ASK_MODEL}), prompt ~{len(prompt) // 4:,} tokens each"
          + (f"; keeping {len(old)} recorded answer(s)" if old else "") + ".")
    if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit("Stopped: no calls made.")
    answers, calls = [], 0
    for s in ask_context.SUGGESTIONS:
        if s["id"] in old:
            answers.append(old[s["id"]])
            continue
        if calls:   # ~3.7k tokens per call vs Groq's free 8k tokens/minute: at most 2 calls a minute
            time.sleep(35)
        calls += 1
        data, status = appmod.groq_chat(prompt, s["question"], api_key=key)
        if status == 429:   # tokens-per-minute limit: wait once, then retry the same call
            print("  rate limited (tokens per minute): waiting 65 s, then one retry")
            time.sleep(65)
            data, status = appmod.groq_chat(prompt, s["question"], api_key=key)
        text = appmod.answer_text(data) if status == 200 else ""
        usage = (data or {}).get("usage") if isinstance(data, dict) else None
        print(f"  {s['id']}: HTTP {status}, {len(text.split())} words, usage {usage}")
        if not text:
            print(f"    no answer: {json.dumps(data)[:300]}")
        answers.append({"id": s["id"], "question": s["question"], "answer": text or None,
                        "context_hash": chash, "recorded_at": datetime.now().isoformat(timespec="seconds")})
    for a in answers:
        a.setdefault("recorded_at", prev.get("generated_at"))
    cache = {"generated_at": datetime.now().isoformat(timespec="seconds"), "model": appmod.ASK_MODEL,
             "answers": answers}
    with open(os.path.join(out, "ask_cache.json"), "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    return cache


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ask", action="store_true", help="also pre-compute the Ask-card answers (Groq calls)")
    ap.add_argument("--yes", action="store_true", help="don't ask before the Groq calls")
    ap.add_argument("--redo", nargs="+", default=(), metavar="ID",
                    help="with --ask: re-record only these card ids (why_declined, invented, next, noise)")
    args = ap.parse_args(argv)
    runs = snapshot()
    print(f"demo_data/: {len(runs)} runs: {', '.join(runs)}")
    if args.ask:
        precompute_answers(yes=args.yes, redo=tuple(args.redo))
    elif not os.path.isfile(os.path.join(OUT, "ask_cache.json")):
        print("no demo_data/ask_cache.json yet: the demo Ask box will show no cards (run with --ask)")


if __name__ == "__main__":
    main()
