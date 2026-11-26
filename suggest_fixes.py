"""
Suggest fixes for the problems a run found. Never edits, runs or applies anything:
it writes suggestions for a human to read and decide on.

    python suggest_fixes.py runs/<run> --list          # the problems it would look at (0 API calls)
    python suggest_fixes.py runs/<run>                 # investigate them (asks first; framework key)
    python suggest_fixes.py runs/<run> --only fdd_q20  # just the problems on these questions
    python suggest_fixes.py runs/<run> --redo          # look again at problems already done

Problems (suggest/problems.py): crashes, failed LLM/tool calls inside an answer, and drops that
diagnose.py explained with a fixable cause. INFRA / CONFIG / UNCLEAR drops are listed as skipped,
with the reason. A drop on a question that also moved in an A/A test is marked "maybe noise".

For each problem a model (one fixed model, temperature 0) reads the evidence and the code with
read-only tools, then submits: what's wrong in plain words, the evidence (file:line), the change
(a patch, a prompt change or a setting), confidence, how to check it and the risks. Each patch is
checked against the current files (fits / doesn't fit); nothing is applied.

What the model can read: the framework's folder plus the system's own source folders
(targets.yaml `source_dirs`, or the target module's source_dirs() function). Never .env files,
keys, virtual environments, .git, runs/ or data files. Parts of that code and the problem's
evidence are sent to the model's provider, so the command says which folders before asking.

Budgets: --max problems per run (default 5), --max-steps model calls and --max-tokens tokens per
problem. Results: runs/<run>/suggestions.json and suggestions.md, saved after each problem, so a
stop (daily limit, Ctrl+C) loses nothing and a re-run continues.
"""

import argparse
import json
import os
import sys
from datetime import datetime

from suggest import problems as problems_mod
from suggest.agent import DEFAULT_MODEL, suggest_one
from suggest.tools import Sandbox

REPO = os.path.dirname(os.path.abspath(__file__))


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def source_dirs(target):
    """The system's own code folders: targets.yaml source_dirs, else the target module's source_dirs()."""
    import importlib

    import target_config
    try:
        cfg, entry = target_config.get(target), target_config._targets().get(target) or {}
    except Exception:  # noqa: BLE001  (an unknown target: framework folder only)
        return []
    dirs = [os.path.expandvars(os.path.expanduser(d)) for d in (cfg.get("source_dirs") or [])]
    if not dirs and entry.get("module"):
        try:
            mod = importlib.import_module(entry["module"])
            dirs = list(mod.source_dirs()) if hasattr(mod, "source_dirs") else []
        except Exception:  # noqa: BLE001
            dirs = []
    return [os.path.abspath(d) for d in dirs if d and os.path.isdir(d)]


def system_info(target, run_dir):
    import target_config
    meta = _read_json(os.path.join(run_dir, "run_meta.json")) or {}
    try:
        cfg, entry = target_config.get(target), target_config._targets().get(target) or {}
    except Exception:  # noqa: BLE001
        cfg, entry = {}, {}
    return {"target": target,
            "entry_point": f"{entry.get('module')}.{entry.get('function')}" if entry.get("module") else None,
            "description": cfg.get("about") or cfg.get("description"),
            "experiment": meta.get("experiment"), "settings": (meta.get("resolved_config") or {}).get("settings")}


def load_report(run_dir):
    return _read_json(os.path.join(run_dir, "suggestions.json")) or {"items": []}


def save_report(run_dir, report):
    report["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = os.path.join(run_dir, "suggestions.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(run_dir, "suggestions.json"))
    with open(os.path.join(run_dir, "suggestions.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(report))


def to_markdown(report):
    out = [f"# Suggested fixes: {report.get('run')}", "",
           f"Model {report.get('model')} · {len(report['items'])} problem(s) looked at · updated {report.get('updated_at')}.",
           "Nothing here was applied. Each patch was only checked against the current code.", ""]
    for it in report["items"]:
        s = it.get("suggestion") or {}
        out += [f"## {it['problem']['title']}", "",
                f"Questions: {', '.join(it['problem']['scenario_ids'])} · status: **{it['status']}**"
                + (f" · confidence: {s.get('confidence')}" if s.get("confidence") else "")
                + (" · maybe noise (also moved in the A/A test)" if it["problem"].get("maybe_noise") else ""), ""]
        if it["status"] == "stopped":
            out += [f"Stopped: {it.get('reason')}", ""]
            continue
        if s.get("what_is_wrong"):
            out += [s["what_is_wrong"], ""]
        for e in s.get("evidence") or []:
            out.append(f"- {e}")
        out.append("")
        if s.get("setting_change"):
            out += [f"**Setting change:** {s['setting_change']}", ""]
        if s.get("patch"):
            pc = it.get("patch_check") or {}
            out += [f"**Patch** ({'fits the current code' if pc.get('fits') else 'does NOT fit: ' + str(pc.get('why'))}):", "",
                    "```diff", s["patch"].rstrip(), "```", ""]
        for k, label in (("how_to_check", "How to check"), ("risks", "Risks")):
            if s.get(k):
                out += [f"**{label}:** {s[k]}", ""]
        out += [f"_{it['steps']} model calls, {it['tokens']:,} tokens; read: {', '.join(it.get('files_read') or []) or 'nothing'}_", ""]
    if report.get("skipped"):
        out += ["## Skipped (not something code can fix)", ""] + [
            f"- {s.get('scenario_id')}: {s.get('cause') or s.get('kind')}: {s['why']}" for s in report["skipped"]]
    return "\n".join(out) + "\n"


def make_client():
    import groq
    from dotenv import load_dotenv

    import rate_limit
    load_dotenv(os.path.join(REPO, ".env"))
    return groq.Groq(http_client=rate_limit.http_client("groq"), max_retries=0)


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description="Suggest fixes for a run's problems (never applies them).")
    ap.add_argument("run_dir")
    ap.add_argument("--list", action="store_true", help="show the problems only (0 API calls)")
    ap.add_argument("--only", default="", help="comma-separated question ids")
    ap.add_argument("--max", type=int, default=5, help="problems to look at in this run (default 5)")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=30_000)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--redo", action="store_true", help="look again at problems already done")
    ap.add_argument("--yes", action="store_true", help="don't ask before the API calls")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.rstrip("/\\")
    if not os.path.isfile(os.path.join(run_dir, "run_meta.json")):
        print(f"{run_dir}: not a run folder (no run_meta.json)")
        return 1
    found, skipped = problems_mod.problems(run_dir)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only:
        found = [p for p in found if only & set(p["scenario_ids"])]
    report = load_report(run_dir)
    done = {it["problem"]["id"] for it in report["items"] if it["status"] in ("suggested", "no_fix")}
    todo = [p for p in found if args.redo or p["id"] not in done]

    print(f"{len(found)} problem(s) in {os.path.basename(run_dir)}"
          + (f" ({len(found) - len(todo)} already have a suggestion)" if len(found) != len(todo) else "") + ":")
    for p in found:
        mark = "done" if p["id"] in done and not args.redo else "    "
        print(f"  {mark}  {p['title']}  [{', '.join(p['scenario_ids'])}]" + ("  (maybe noise)" if p.get("maybe_noise") else ""))
    for s in skipped:
        print(f"  skip  {s.get('scenario_id')}: {s.get('cause') or s.get('kind')}: {s['why']}")
    if args.list or not todo:
        if not todo and not args.list:
            print("Nothing new to look at (use --redo to look again).")
        return 0

    todo = todo[:args.max]
    target = found[0]["target"] if found else None
    box = Sandbox([REPO] + source_dirs(target))
    print(f"\nWill investigate {len(todo)} problem(s) with {args.model}, at most {args.max_steps} calls and "
          f"{args.max_tokens:,} tokens each (at most {len(todo) * args.max_tokens:,} tokens, usually far fewer).")
    print("On a free Groq account (about 8,000 tokens a minute) expect about 1 to 2 minutes per call; "
          "progress is printed as it goes.")
    print("It reads (never .env, keys, venvs, .git, runs/ or data files) and sends parts of this code to the model:")
    for d in box.describe():
        print(f"  {d['path']}")
    if not args.yes and input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing sent.")
        return 0

    import rate_limit
    client = client or make_client()
    report.update({"run": os.path.basename(run_dir), "target": target, "model": args.model,
                   "skipped": skipped, "folders": [d["path"] for d in box.describe()]})
    info = system_info(target, run_dir)
    for i, p in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {p['title']} ...")
        box.reads = []
        try:
            res = suggest_one(p, box, client, args.model, info, args.max_steps, args.max_tokens,
                              on_step=lambda n, what, tok: print(f"    call {n}: {what}  ({tok:,} tokens so far)", flush=True))
        except KeyboardInterrupt:
            print("Stopped. What's done is saved; run the same command to continue.")
            return 1
        except Exception as e:  # noqa: BLE001
            if rate_limit.is_daily_limit_error(e):
                print("Daily token limit reached. What's done is saved; run the same command tomorrow to continue.")
                return 1
            res = {"status": "stopped", "suggestion": None, "reason": f"{type(e).__name__}: {str(e)[:300]}",
                   "steps": 0, "tokens": 0, "files_read": [], "patch_check": None}
        report["items"] = [it for it in report["items"] if it["problem"]["id"] != p["id"]] + [{"problem": p, **res}]
        save_report(run_dir, report)
        s = res.get("suggestion") or {}
        pc = res.get("patch_check")
        print(f"  {res['status']}: {s.get('title') or res.get('reason')}"
              + (f" [{s.get('change_type')}, confidence {s.get('confidence')}]" if s else "")
              + (f"\n  patch: {'fits the current code' if pc['fits'] else 'does NOT fit: ' + pc['why']}" if pc else "")
              + f"\n  {res['steps']} calls, {res['tokens']:,} tokens")
    print(f"\nSaved: {os.path.join(run_dir, 'suggestions.md')} (and on the dashboard: Experiments -> this run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
