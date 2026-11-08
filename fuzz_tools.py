"""
Universal tool fuzz test: call every tool of a target with bad inputs and
check that each one comes back as a clean error, not a crash.
No LLM, no API calls, deterministic.

    python fuzz_tools.py configurable                                   # baseline settings
    python fuzz_tools.py configurable --experiment experiments/exp_08_tool_validation.yaml

The target opts in through targets.yaml:

    my_target:
      tools: my_module:get_tools     # get_tools(settings) if accepts_settings, else get_tools()

It may return LangChain tools or plain Python functions. Bad inputs are
generated from each argument's TYPE, so nothing here knows what the tools do.

Each call is classified as:
  CRASH    the tool raised an exception                       -> exit code 1
  CLEAN    it returned an error message (text starting "Error")
  ACCEPTED it returned normal output (a warning only: whether e.g. 0 is a
           valid amount is a business question the fuzzer can't decide)
"""

import argparse
import importlib
import inspect
import math
import os
import sys

from experiment_config import ExperimentConfigError, load_experiment
import run_suite

DEFAULT_EXPERIMENT = os.path.join("experiments", "baseline.yaml")

# Bad values per JSON-schema type. A label is shown instead of the value itself.
BAD = {
    "string": [("empty", ""), ("whitespace", "   "), ("10k chars", "x" * 10_000),
               ("unicode/emoji", "\u00fc\u00f1\u00ef\u00e7\u00f8d\u00e9 \U0001F642 \u0000"),
               ("sql-ish", "'; DROP TABLE x; --"), ("None", None), ("int 123", 123)],
    "number": [("-1", -1), ("0", 0), ("1e12", 1e12), ("'abc'", "abc"), ("None", None), ("NaN", math.nan)],
    "integer": [("-1", -1), ("0", 0), ("1e12", 10**12), ("1.5", 1.5), ("'abc'", "abc"), ("None", None)],
    "boolean": [("'maybe'", "maybe"), ("None", None), ("int 2", 2)],
}
# A harmless value for the OTHER arguments while one argument gets a bad value.
NEUTRAL = {"string": "test", "number": 1.0, "integer": 1, "boolean": True}
_PY_TYPES = {str: "string", float: "number", int: "integer", bool: "boolean"}


class FuzzError(Exception):
    """A setup problem (target not opted in, bad experiment) meant for a human."""


def _schema(tool):
    """{arg: type}, [required args] for a LangChain tool or a plain function."""
    if hasattr(tool, "get_input_schema"):
        js = tool.get_input_schema().model_json_schema()
        props = {k: v.get("type") or next((o.get("type") for o in v.get("anyOf", []) if o.get("type") != "null"), None)
                 for k, v in js.get("properties", {}).items()}
        return props, list(js.get("required", []))
    sig = inspect.signature(tool)
    props = {n: _PY_TYPES.get(p.annotation) for n, p in sig.parameters.items()}
    return props, [n for n, p in sig.parameters.items() if p.default is inspect.Parameter.empty]


def tool_name(tool):
    return getattr(tool, "name", None) or getattr(tool, "__name__", "tool")


def cases(tool):
    """[(label, kwargs)]: one bad value per argument, then missing / extra arguments."""
    props, required = _schema(tool)
    neutral = {a: NEUTRAL.get(t, "test") for a, t in props.items()}
    out = []
    for arg, typ in props.items():
        for label, value in BAD.get(typ, BAD["string"]):
            out.append((f"{arg}={label}", {**neutral, arg: value}))
    for arg in required:
        out.append((f"missing {arg}", {k: v for k, v in neutral.items() if k != arg}))
    if len(required) > 1:
        out.append(("all args missing", {}))
    out.append(("unknown extra arg", {**neutral, "zzz_unknown_arg": 1}))
    return out


def call(tool, kwargs):
    """('CRASH'|'CLEAN'|'ACCEPTED', short output)."""
    try:
        result = tool.invoke(kwargs) if hasattr(tool, "invoke") else tool(**kwargs)
    except Exception as e:  # noqa: BLE001 - the point is to see what escapes
        return "CRASH", f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    text = str(result)
    return ("CLEAN" if text.strip().lower().startswith("error") else "ACCEPTED"), text


def load_tools(target, experiment=None, targets=None):
    targets = targets if targets is not None else run_suite.load_targets()
    if target not in targets:
        raise FuzzError(f"Unknown target '{target}'. Known: {', '.join(sorted(targets))}")
    entry = targets[target] or {}
    ref = entry.get("tools")
    if not ref:
        raise FuzzError(f"Target '{target}' has no `tools:` entry in targets.yaml, so it hasn't opted in to fuzzing.\n"
                        f"Add e.g.  tools: my_module:get_tools  (only list tools that are safe to call with junk).")
    module_name, _, attr = ref.partition(":")
    getter = getattr(importlib.import_module(module_name), attr)
    if not entry.get("accepts_settings"):
        return list(getter()), None
    path = experiment or DEFAULT_EXPERIMENT
    try:
        exp = load_experiment(path)
    except (ExperimentConfigError, OSError) as e:
        raise FuzzError(str(e)) from None
    if exp["target"] != target:
        raise FuzzError(f"{path} is for target '{exp['target']}', not '{target}'.")
    try:
        return list(getter(exp["settings"])), path
    except ValueError as e:              # e.g. agent_configurable.SettingsError
        raise FuzzError(f"{path}: {e}") from None


def fuzz(tools):
    """{tool name: [(label, status, output)]}"""
    return {tool_name(t): [(label, *call(t, kw)) for label, kw in cases(t)] for t in tools}


def _safe(s, n=70):
    s = s.replace("\n", " ")
    s = s if len(s) <= n else s[:n] + "..."
    return s.encode("ascii", "backslashreplace").decode()


def report(results):
    lines, totals = [], {"CRASH": 0, "CLEAN": 0, "ACCEPTED": 0}
    for name, rows in results.items():
        counts = {k: sum(1 for _, s, _ in rows if s == k) for k in totals}
        for k in totals:
            totals[k] += counts[k]
        lines.append(f"\n{name}: {counts['CRASH']} crash, {counts['CLEAN']} clean error, {counts['ACCEPTED']} accepted")
        for label, status, output in rows:
            mark = {"CRASH": "XX", "CLEAN": "ok", "ACCEPTED": "? "}[status]
            lines.append(f"  {mark} {status:<8} {_safe(label, 26):<28} {_safe(output)}")
    lines.append(f"\nTOTAL: {totals['CRASH']} crash, {totals['CLEAN']} clean error, {totals['ACCEPTED']} accepted")
    return "\n".join(lines), totals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target")
    ap.add_argument("--experiment", help=f"settings to build the tools with (default {DEFAULT_EXPERIMENT})")
    args = ap.parse_args(argv)
    try:
        tools, exp_path = load_tools(args.target, args.experiment)
    except FuzzError as e:
        print(e, file=sys.stderr)
        return 2
    import approval
    import target_config
    listed = {a["name"] for a in target_config.get(args.target)["action_tools"]}
    skipped = [tool_name(t) for t in tools if tool_name(t) in listed or approval.is_safe_action(t)]
    tools = [t for t in tools if tool_name(t) not in skipped]
    print(f"Fuzzing {len(tools)} tools of '{args.target}'" + (f" built from {exp_path}" if exp_path else "")
          + " (no LLM, no API calls)")
    for name in skipped:
        print(f"  SKIPPED {name}: an action tool (junk input must never trigger a real action); "
              f"checked by check_actions.py")
    text, totals = report(fuzz(tools))
    print(text)
    if totals["CRASH"]:
        print("FAIL: some tools raised on bad input. In the agent, LangGraph's ToolNode catches this and hands the raw\n"
              "exception text to the model; wrap the tools (e.g. validated_tool.py) so they return clean errors.")
        return 1
    print("PASS: no tool crashed." + (" Review the ACCEPTED rows: are those inputs really valid?" if totals["ACCEPTED"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
