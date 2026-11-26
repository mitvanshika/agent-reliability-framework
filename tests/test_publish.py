"""
Tests for publish.py in a throwaway git repo with a local bare repo as "GitHub": only
demo_data/ is committed, other changes are left alone, "no" commits nothing, an unchanged
snapshot publishes nothing, and the push really reaches the remote. No API calls, no network.
    python tests/test_publish.py
"""

import builtins
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]

import make_demo_data  # noqa: E402
import publish  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def sh(cwd, *args):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout


root = tempfile.mkdtemp()
remote, work = os.path.join(root, "remote.git"), os.path.join(root, "work")
sh(root, "git", "init", "-q", "--bare", remote)
sh(root, "git", "clone", "-q", remote, work)
for k, v in (("user.email", "t@t"), ("user.name", "t"), ("commit.gpgsign", "false")):
    sh(work, "git", "config", k, v)
open(os.path.join(work, "app.py"), "w").write("print('app')\n")
sh(work, "git", "add", "-A")
sh(work, "git", "commit", "-q", "-m", "start")
sh(work, "git", "branch", "-M", "master")
sh(work, "git", "push", "-q", "-u", "origin", "master")

version = {"n": 1}


def fake_snapshot(runs_dir="runs", out="demo_data"):
    os.makedirs(os.path.join(work, "demo_data", "runs", "r1"), exist_ok=True)
    open(os.path.join(work, "demo_data", "runs", "r1", "summary.json"), "w").write(f'{{"v": {version["n"]}}}')
    return ["r1"]


make_demo_data.snapshot = fake_snapshot
publish.REPO = work
real_input = builtins.input


def run(argv, answer=None):
    if answer is not None:
        builtins.input = lambda prompt="": answer
    try:
        with contextlib.redirect_stdout(io.StringIO()) as o:
            code = publish.main(argv)
    finally:
        builtins.input = real_input
    return code, o.getvalue()


try:
    print("answering no:")
    open(os.path.join(work, "app.py"), "a").write("# my unfinished edit\n")
    code, out = run([], answer="n")
    ok("exit 0, nothing committed", code == 0 and "Nothing committed" in out
       and sh(work, "git", "log", "--oneline").count("\n") == 1, out[-200:])
    ok("shows what changed and that other changes are left alone", "1 new" in out and "runs: r1" in out
       and "other uncommitted change" in out, out)

    print("answering yes on master:")
    code, out = run([], answer="y")
    ok("exit 0, committed and pushed", code == 0 and "Pushed to master" in out and "Render will redeploy" in out, out[-300:])
    ok("the commit has demo_data/ only", sh(work, "git", "show", "--name-only", "--format=", "HEAD").split()
       == ["demo_data/runs/r1/summary.json"])
    ok("your unfinished edit to app.py is still uncommitted", " M app.py" in sh(work, "git", "status", "--porcelain"))
    ok("the remote (GitHub) has it", "Update demo data" in sh(remote, "git", "log", "--oneline", "master"))

    print("nothing changed:")
    code, out = run(["--yes"])
    ok("publishes nothing", code == 0 and "already up to date" in out, out[-200:])

    print("--yes on a new branch:")
    sh(work, "git", "checkout", "-q", "-b", "feature/x")
    version["n"] = 2
    code, out = run(["--yes"])
    ok("pushed with upstream set, tells you to open a pull request",
       code == 0 and "Pushed to feature/x" in out and "pull request" in out, out[-300:])
    ok("branch exists on the remote", "feature/x" in sh(remote, "git", "branch"))
finally:
    shutil.rmtree(root, ignore_errors=True)

print("\nPASS" if not failures else f"\nFAIL: {failures}")
sys.exit(1 if failures else 0)
