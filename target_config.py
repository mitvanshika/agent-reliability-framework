"""
Per-target settings from targets.yaml, with defaults that keep the old
behaviour. Any target may set:

    my_target:
      module: my_module
      function: answer
      description: "one line: what this system does"    # given to the judges, scope addon, attack generator
      judge_set: generic_v1                             # judge wording (default experiment_v2)
      judges: [correctness_gt, completeness, groundedness]   # which judges apply; the others are N/A
      pass_threshold: 6.5                               # final_score >= this counts as a pass (default 7)
      scenarios: scenarios_my_target                    # its main scenario folder (default scenarios/)
      non_retrieval_tools: [send_email]                 # tools that don't fetch information (context_relevance)
      action_tools: [send_email]                        # tools that ACT: must be approval.safe_action (check_actions.py)
      actions_sandbox: my_module:sandbox                # fn(tmpdir) -> restore(): point side effects at tmpdir

Left out, each key means exactly what the framework did before: no
description, every judge of the judge set, threshold 7, scenarios/, and
no non-retrieval tools.
"""

import os

DEFAULT_THRESHOLD = 7
DEFAULT_SCENARIOS = "scenarios"   # = run_suite.SCENARIOS_DIR
KEYS = {"module", "function", "accepts_settings", "description", "tools", "judge_set", "judges",
        "pass_threshold", "scenarios", "non_retrieval_tools", "action_tools", "actions_sandbox"}


class TargetConfigError(ValueError):
    pass


def _targets():
    import run_suite
    return run_suite.load_targets() or {}


def get(target, targets=None) -> dict:
    """The target's settings with defaults filled in (an unknown or missing target -> all defaults)."""
    entry = ((targets if targets is not None else _targets()).get(target) or {}) if target else {}
    return {
        "description": entry.get("description"),
        "judge_set": entry.get("judge_set"),                                            # None = default
        "judges": list(entry["judges"]) if entry.get("judges") is not None else None,   # None = all
        "pass_threshold": float(entry.get("pass_threshold", DEFAULT_THRESHOLD)),
        "scenarios": entry.get("scenarios") or _default_scenarios(),
        "non_retrieval_tools": list(entry.get("non_retrieval_tools") or []),
        "action_tools": action_tools(entry),
    }


def action_tools(entry) -> list:
    """[{"name": ..., "test_args": {...} or None}] from `action_tools:` (names or mappings)."""
    out = []
    for a in (entry or {}).get("action_tools") or []:
        if isinstance(a, str):
            out.append({"name": a, "test_args": None})
        elif isinstance(a, dict) and a.get("name"):
            out.append({"name": a["name"], "test_args": a.get("test_args")})
    return out


def _default_scenarios():
    import run_suite
    return run_suite.SCENARIOS_DIR      # read at call time (tests point it at a temp folder)


def threshold(target, targets=None) -> float:
    return get(target, targets)["pass_threshold"]


def for_meta(meta) -> dict:
    """Settings of the target a run tested (from its run_meta.json)."""
    return get((meta.get("resolved_config") or {}).get("target"))


def is_main_suite(target, scenario_dir) -> bool:
    """True if scenario_dir is this target's own main scenario folder (or not given)."""
    if not scenario_dir:
        return True
    return os.path.normpath(scenario_dir) == os.path.normpath(get(target)["scenarios"])


def validate(target, entry, known_judges) -> list:
    """Problems with one targets.yaml entry (empty list = fine)."""
    problems = []
    if not isinstance(entry, dict):
        return [f"{target}: must be a mapping"]
    unknown = sorted(set(entry) - KEYS)
    if unknown:
        problems.append(f"{target}: unknown keys {unknown} (known: {sorted(KEYS)})")
    if entry.get("judge_set") is not None:
        import judge_sets
        if entry["judge_set"] not in judge_sets.MAIN_SETS:
            problems.append(f"{target}.judge_set: one of {list(judge_sets.MAIN_SETS)} (got {entry['judge_set']!r})")
    if entry.get("judges") is not None:
        j = entry["judges"]
        if not isinstance(j, list) or not j:
            problems.append(f"{target}.judges: a non-empty list of judge names")
        else:
            bad = [n for n in j if n not in known_judges]
            if bad:
                problems.append(f"{target}.judges: unknown {bad} (known: {sorted(known_judges)})")
    if "pass_threshold" in entry:
        t = entry["pass_threshold"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 10:
            problems.append(f"{target}.pass_threshold: a number from 0 to 10 (got {t!r})")
    if "scenarios" in entry and not (isinstance(entry["scenarios"], str) and entry["scenarios"]):
        problems.append(f"{target}.scenarios: a folder path")
    if "non_retrieval_tools" in entry and not (isinstance(entry["non_retrieval_tools"], list)
                                               and all(isinstance(x, str) for x in entry["non_retrieval_tools"])):
        problems.append(f"{target}.non_retrieval_tools: a list of tool names")
    if "action_tools" in entry:
        a = entry["action_tools"]
        if not isinstance(a, list) or not all(
                isinstance(x, str) or (isinstance(x, dict) and isinstance(x.get("name"), str)
                                       and isinstance(x.get("test_args", {}), dict)) for x in a):
            problems.append(f"{target}.action_tools: a list of tool names, or of {{name: ..., test_args: {{...}}}}")
        elif a and not entry.get("tools"):
            problems.append(f"{target}.action_tools needs `tools:` too (how to get the target's tools)")
    if "actions_sandbox" in entry and not (isinstance(entry["actions_sandbox"], str) and ":" in entry["actions_sandbox"]):
        problems.append(f"{target}.actions_sandbox: module:function")
    if "description" in entry and not (isinstance(entry["description"], str) and entry["description"].strip()):
        problems.append(f"{target}.description: a non-empty string")
    return problems
