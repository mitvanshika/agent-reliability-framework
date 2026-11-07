"""
Is a target plugged in correctly? Conformance checks before any paid run.
No LLM judges, no API calls (unless the target's own function makes them
in the one optional test call).

    python check_target.py <target> [--call | --no-call] [--experiment FILE]

Checks: the targets.yaml entry, that the function imports and has the right
signature, the experiment settings (for accepts_settings targets), every
scenario file, the answer key and description the judges will need, the
tools opt-in, and - only with --call - ONE call with a test question to see
that it returns a string or a dict with "answer".

Default: no call, because a target function may call a paid API (the demo
agents do, even the plain ones). --call makes exactly one call.

Exit code 0 if nothing FAILed (warnings are fine), 1 otherwise.
"""

import argparse
import glob
import inspect
import os
import sys
import threading
import time

import yaml

import judge_sets
import run_suite
import target_config

TEST_QUESTION = "check_target: this is a connectivity test, please reply with any short answer."
CALL_TIMEOUT_S = 30


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, check, detail=""):
        self.rows.append((status, check, detail))

    def ok(self, check, detail=""):
        self.add("OK", check, detail)

    def warn(self, check, detail):
        self.add("WARN", check, detail)

    def fail(self, check, detail):
        self.add("FAIL", check, detail)

    @property
    def failed(self):
        return any(s == "FAIL" for s, _, _ in self.rows)


def _default_experiment(target):
    base = os.path.join("experiments", "baseline.yaml")
    candidates = [base] + sorted(p for p in glob.glob(os.path.join("experiments", "*.yaml"))
                                 if p != base and not os.path.basename(p).startswith("draft_"))
    from experiment_config import load_experiment
    for path in candidates:
        try:
            if os.path.isfile(path) and load_experiment(path)["target"] == target:
                return path
        except Exception:
            continue
    return None


def check_scenarios(folder, rep):
    if not os.path.isdir(folder):
        rep.fail("scenarios", f"folder not found: {folder} (set `scenarios:` in targets.yaml)")
        return []
    files = sorted(f for f in os.listdir(folder) if f.endswith(".yaml"))
    if not files:
        rep.fail("scenarios", f"no .yaml files in {folder}")
        return []
    scenarios, problems, ids = [], [], {}
    for f in files:
        try:
            with open(os.path.join(folder, f), encoding="utf-8") as fh:
                s = yaml.safe_load(fh)
        except Exception as e:
            problems.append(f"{f}: does not parse ({type(e).__name__}: {str(e).splitlines()[0]})")
            continue
        if not isinstance(s, dict):
            problems.append(f"{f}: must be a mapping with id + question")
            continue
        if not s.get("id"):
            problems.append(f"{f}: no id")
            continue
        if not isinstance(s.get("question"), str):
            problems.append(f"{f}: no question (a string; \"\" is allowed)")
            continue
        if s["id"] in ids:
            problems.append(f"{f}: duplicate id '{s['id']}' (also in {ids[s['id']]})")
            continue
        ids[s["id"]] = f
        if s["id"] != os.path.splitext(f)[0]:
            rep.warn("scenarios", f"{f}: id '{s['id']}' differs from the file name (allowed, but confusing)")
        scenarios.append(s)
    for p in problems:
        rep.fail("scenarios", p)
    if not problems:
        rep.ok("scenarios", f"{len(scenarios)} scenarios in {folder}, ids unique")
    return scenarios


def _signature_ok(fn, accepts):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True, ""          # builtins / C functions: can't tell, the call will show it
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    varkw = any(p.kind == p.VAR_KEYWORD for p in params)
    if not positional and not varargs:
        return False, f"{fn.__name__}{sig} takes no question argument"
    required = [p for p in positional[1:] if p.default is p.empty]
    if required and not (accepts and [p.name for p in required] == ["settings"]):
        return False, f"{fn.__name__}{sig}: arguments after the question must have defaults ({[p.name for p in required]})"
    if accepts and "settings" not in sig.parameters and not varkw:
        return False, f"{fn.__name__}{sig}: accepts_settings is true, so it must take settings="
    return True, f"{fn.__module__}.{fn.__name__}{sig}"


def _call(fn, accepts, settings):
    box = {}

    def run():
        try:
            box["value"] = fn(TEST_QUESTION, settings=settings) if accepts else fn(TEST_QUESTION)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    start = time.perf_counter()
    t.start()
    t.join(CALL_TIMEOUT_S)
    return box, time.perf_counter() - start, t.is_alive()


def check(target, call=False, experiment=None, targets=None) -> Report:
    rep = Report()
    targets = targets if targets is not None else run_suite.load_targets()
    if target not in targets:
        rep.fail("targets.yaml", f"no target '{target}'. Known: {', '.join(sorted(targets))}")
        return rep
    entry = targets[target] or {}
    problems = target_config.validate(target, entry, judge_sets.ALL_NAMES)
    for key in ("module", "function"):
        if not entry.get(key):
            problems.append(f"{target}.{key}: required")
    for p in problems:
        rep.fail("targets.yaml", p)
    if problems:
        return rep
    rep.ok("targets.yaml", "entry valid")
    cfg = target_config.get(target, targets)
    accepts = entry.get("accepts_settings") is True

    # import + signature
    try:
        import importlib
        fn = getattr(importlib.import_module(entry["module"]), entry["function"])
    except Exception as e:
        rep.fail("import", f"{entry['module']}.{entry['function']}: {type(e).__name__}: {e}")
        return rep
    if not callable(fn):
        rep.fail("import", f"{entry['module']}.{entry['function']} is not callable")
        return rep
    good, detail = _signature_ok(fn, accepts)
    (rep.ok if good else rep.fail)("signature", detail)

    # settings
    settings = None
    if accepts:
        path = experiment or _default_experiment(target)
        if not path:
            rep.warn("settings", f"accepts_settings is true but no experiments/*.yaml targets '{target}': "
                                 "write one before run_experiment.py")
        else:
            from experiment_config import load_experiment
            try:
                exp = load_experiment(path)
                if exp["target"] != target:
                    rep.fail("settings", f"{path} is for target '{exp['target']}'")
                else:
                    settings = exp["settings"]
                    hook = getattr(sys.modules[fn.__module__], "validate_settings", None)
                    if callable(hook):
                        hook(settings)
                    if settings.get("prompt_addons"):
                        import prompt_addons
                        prompt_addons.validate(settings["prompt_addons"], cfg["description"])
                    rep.ok("settings", f"{path} valid" + (" (target's validate_settings)" if callable(hook) else ""))
            except Exception as e:
                rep.fail("settings", f"{path}: {str(e).splitlines()[0] if str(e) else type(e).__name__}")

    # judges + scenarios
    scenarios = check_scenarios(cfg["scenarios"], rep)
    try:
        js = judge_sets.for_run(scenarios or [{}], cfg)
        rep.ok("judges", judge_sets.describe(js) + (f"; N/A: {', '.join(js['not_applicable'])}"
                                                    if js.get("not_applicable") else ""))
        if scenarios and "correctness_gt" in js["llm_judges"]:
            check_answer_key(scenarios, rep)
    except ValueError as e:
        rep.fail("judges", str(e))
    rep.ok("pass threshold", f"final_score >= {cfg['pass_threshold']:g}")

    # description
    needs = [why for why, cond in (
        ("judge_set generic_v1 gives it to every judge", cfg.get("judge_set") == "generic_v1"),
        ("the scope prompt addon uses it", "scope" in ((settings or {}).get("prompt_addons") or [])),
        ("generate_adversarial.py builds attacks from it", True),
    ) if cond]
    if cfg["description"]:
        rep.ok("description", cfg["description"])
    else:
        rep.warn("description", "none set. It is used by: " + "; ".join(needs))

    # tools opt-in
    if entry.get("tools"):
        try:
            import fuzz_tools
            tools, _ = fuzz_tools.load_tools(target, experiment, targets)
            if not tools:
                rep.fail("tools", f"{entry['tools']} returned no tools")
            else:
                rep.ok("tools", f"{len(tools)} tools for fuzz_tools.py: "
                                + ", ".join(fuzz_tools.tool_name(t) for t in tools))
        except Exception as e:
            rep.fail("tools", f"{entry['tools']}: {type(e).__name__}: {e}")

    # one call
    if not call:
        rep.add("SKIP", "test call", "not called (it may use a paid API): --call makes one call")
        return rep
    if accepts and settings is None:
        rep.add("SKIP", "test call", "no valid settings to call it with")
        return rep
    box, took, hung = _call(fn, accepts, settings)
    if hung:
        rep.fail("test call", f"no answer after {CALL_TIMEOUT_S}s")
    elif "error" in box:
        e = box["error"]
        rep.fail("test call", f"raised {type(e).__name__}: {e}")
    else:
        raw = box.get("value")
        try:
            norm = run_suite.normalize_result(raw)
        except TypeError as e:
            rep.fail("test call", str(e))
        else:
            if norm.get("error"):
                rep.fail("test call", f"returned an error: {norm['error']}")
            elif not isinstance(norm.get("answer"), str):
                rep.fail("test call", f"no answer string (got {type(norm.get('answer')).__name__}); "
                                      "return a string or a dict with \"answer\"")
            else:
                shape = "string" if isinstance(raw, str) else "dict"
                rep.ok("test call", f"returned a {shape} in {took:.2f}s: {norm['answer'][:60]!r}")
    return rep


def check_answer_key(scenarios, rep):
    """Only asked when the correctness_gt judge applies to the target."""
    keyed = sum(1 for s in scenarios if s.get("expected_answer"))
    if keyed == 0:
        rep.warn("answer key", "no scenario has expected_answer: correctness falls back to judging without a "
                               "reference (weaker). Add expected_answer to your scenarios.")
    elif keyed < len(scenarios):
        rep.warn("answer key", f"{keyed}/{len(scenarios)} scenarios have expected_answer; the rest fall back")
    else:
        rep.ok("answer key", "every scenario has expected_answer")


def main(argv=None):
    p = argparse.ArgumentParser(description="Conformance checks for a target (no LLM judges, no API calls).")
    p.add_argument("target")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--call", dest="call", action="store_true", default=False,
                   help="call the function once with a test question")
    g.add_argument("--no-call", dest="call", action="store_false", help="don't call it (the default)")
    p.add_argument("--experiment", help="settings to check (accepts_settings targets)")
    args = p.parse_args(argv)
    rep = check(args.target, args.call, args.experiment)
    print(f"check_target {args.target}")
    for status, name, detail in rep.rows:
        print(f"  {status:<4}  {name:<15} {detail}".encode("ascii", "backslashreplace").decode())
    n_fail = sum(1 for s, *_ in rep.rows if s == "FAIL")
    n_warn = sum(1 for s, *_ in rep.rows if s == "WARN")
    print(f"\n{'FAIL' if n_fail else 'PASS'}: {n_fail} failed, {n_warn} warnings")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
