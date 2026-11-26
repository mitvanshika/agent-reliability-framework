"""
Update the public demo site with your latest results, in one command.

The site on Render can't run evaluations (the systems under test, your API keys and
hours-long runs all live on your laptop), so it shows a copy of your real run files
in demo_data/. This refreshes that copy and sends it to GitHub:

    python publish.py          # refresh demo_data/, show what changed, ask, then commit + push
    python publish.py --yes    # same, without asking

1. make_demo_data.snapshot(): copies the runs the dashboard shows (baselines, the latest
   run of each experiment, the A/A runs), baselines/, noise/, evidence/ and the question
   files into demo_data/. No API calls.
2. Shows what changed in demo_data/ (git status), and nothing else: your other
   uncommitted changes are never added.
3. Commits only demo_data/ and pushes the current branch.
   On master, Render redeploys the site in a few minutes. On another branch, open a pull
   request into master (GitHub runs the Tests check first), merge, and Render redeploys.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import make_demo_data

REPO = os.path.dirname(os.path.abspath(__file__))
DEMO = "demo_data"


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {(r.stderr or r.stdout).strip()}")
    return r.stdout


def demo_changes():
    """[(status, path)] for demo_data/ only, new files listed one by one."""
    out = git("status", "--porcelain", "--untracked-files=all", "--", DEMO)
    return [(line[:2].strip() or "?", line[3:].strip().strip('"')) for line in out.splitlines() if line.strip()]


def summarize(changes):
    words = {"M": "changed", "A": "new", "?": "new", "??": "new", "D": "removed"}
    counts, runs = {}, set()
    for st, path in changes:
        counts[words.get(st, st)] = counts.get(words.get(st, st), 0) + 1
        parts = path.replace("\\", "/").split("/")
        if len(parts) > 2 and parts[1] == "runs":
            runs.add(parts[2])
    return counts, sorted(runs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--yes", action="store_true", help="don't ask before committing and pushing")
    args = ap.parse_args(argv)

    try:
        branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    except RuntimeError as e:
        print(f"Not a git repository here? {e}")
        return 1

    print("1. Refreshing demo_data/ from your runs (no API calls) ...")
    runs = make_demo_data.snapshot()
    print(f"   {len(runs)} runs copied: {', '.join(runs)}")

    changes = demo_changes()
    if not changes:
        print("\n2. demo_data/ is already up to date: nothing to publish.")
        return 0
    counts, touched = summarize(changes)
    print("\n2. What changed in demo_data/: " + ", ".join(f"{n} {w}" for w, n in sorted(counts.items())))
    if touched:
        print("   runs: " + ", ".join(touched))
    other = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip() and DEMO not in ln]
    if other:
        print(f"   ({len(other)} other uncommitted change(s) in your repo are left alone)")

    where = ("master: Render redeploys the site in a few minutes" if branch in ("master", "main")
             else f"{branch}: then open a pull request into master and merge it to update the site")
    if not args.yes:
        reply = input(f"\n3. Commit demo_data/ and push to {where}? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Nothing committed. demo_data/ is refreshed on disk; run again when you're ready.")
            return 0

    git("add", "--", DEMO)
    msg = f"Update demo data ({datetime.now():%Y-%m-%d %H:%M}): {len(runs)} runs"
    git("commit", "-m", msg, "--", DEMO)
    print(f"\n3. Committed: {msg}")
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False).strip()
    try:
        git("push") if upstream else git("push", "--set-upstream", "origin", branch)
    except RuntimeError as e:
        print(f"   Push failed (the commit is saved; fix the problem, then run: git push)\n   {e}")
        return 1
    print(f"   Pushed to {branch}.")
    print("   " + ("Render will redeploy from master in a few minutes." if branch in ("master", "main")
                   else "Next: on GitHub, open a pull request into master, wait for the Tests check to pass, merge."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
