"""
Names of the judge sets, kept in a tiny module with no LLM imports so
compare.py / promote_baseline.py / run_experiment.py can use it cheaply.

Scores are only comparable when produced by the same judge set: a
final_score averaged over 4 judges (classic) and one averaged over 5
judges with an answer key (experiment) measure different things.
"""

import hashlib
import json

# Track 1 (run_judges.py) and every run judged before Phase 3.
CLASSIC = {
    "name": "classic_v1",
    "llm_judges": ["correctness", "tool_efficiency", "completeness", "groundedness"],
    "checks": [],
}

# judge_experiment.py: answer-key correctness + context relevance, plus tool_match (code, display only).
EXPERIMENT = {
    "name": "experiment_v1",
    "llm_judges": ["correctness_gt", "completeness", "groundedness", "tool_efficiency", "context_relevance"],
    "checks": ["tool_match"],
}

# experiment_v1 with completeness v2: it also sees the tool outputs + answer key, so an honest
# "not available" / escalation counts as complete and an invented answer does not.
EXPERIMENT_V2 = {**EXPERIMENT, "name": "experiment_v2", "versions": {"completeness": "v2"}}

# Adversarial suites (scenarios with an `attack:` field): did it resist, did it invent?
# completeness / tool_efficiency are left out on purpose: refusing is often correct here.
ADVERSARIAL = {
    "name": "adversarial_v1",
    "llm_judges": ["safety", "groundedness"],
    "checks": [],
}

BY_NAME = {js["name"]: js for js in (CLASSIC, EXPERIMENT, EXPERIMENT_V2, ADVERSARIAL)}


def changed_judges(old, new) -> list:
    """Judges/checks of `new` that `old` lacks or runs in a different version."""
    old_names = set(old["llm_judges"]) | set(old.get("checks") or [])
    ov, nv = old.get("versions") or {}, new.get("versions") or {}
    return [n for n in list(new["llm_judges"]) + list(new.get("checks") or [])
            if n not in old_names or ov.get(n) != nv.get(n)]


# every judge/check name, in display order
ALL_NAMES = list(dict.fromkeys(EXPERIMENT["llm_judges"] + ADVERSARIAL["llm_judges"] + CLASSIC["llm_judges"]
                               + EXPERIMENT["checks"]))


def for_scenarios(scenarios) -> dict:
    """The judge set a run should use: adversarial_v1 if its scenarios are attacks."""
    return ADVERSARIAL if any(isinstance(s, dict) and s.get("attack") for s in scenarios) else EXPERIMENT


def key(judge_set) -> str:
    """Canonical identity: name + judges + checks + judge model."""
    js = judge_set or CLASSIC
    return json.dumps({k: js.get(k) for k in ("name", "llm_judges", "checks", "judge_model", "versions")},
                      sort_keys=True)


def same(a, b) -> bool:
    a, b = a or CLASSIC, b or CLASSIC
    if a.get("judge_model") is None or b.get("judge_model") is None:   # classic runs never recorded a model
        return {k: a.get(k) for k in ("name", "llm_judges", "checks")} == {k: b.get(k) for k in ("name", "llm_judges", "checks")}
    return key(a) == key(b)


def describe(judge_set) -> str:
    js = judge_set or CLASSIC
    if js.get("name") == "mixed":
        return "mixed (some scenarios judged with different judge sets)"
    model = f", model {js['judge_model']}" if js.get("judge_model") else ""
    checks = f" + {', '.join(js['checks'])}" if js.get("checks") else ""
    return f"{js['name']} ({', '.join(js['llm_judges'])}{checks}{model})"


def answer_key_hash(scenario: dict) -> str | None:
    """Fingerprint of a scenario's answer key (expected_answer + expected_tools)."""
    if scenario.get("expected_answer") is None and scenario.get("expected_tools") is None:
        return None
    blob = json.dumps({"expected_answer": scenario.get("expected_answer"),
                       "expected_tools": scenario.get("expected_tools")}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def of_judged_traces(traces) -> dict:
    """The judge set used for a run's judged traces; CLASSIC if none recorded one."""
    found = {}
    for t in traces:
        js = t.get("judge_set") or CLASSIC
        found[key(js)] = js
    if not found:
        return CLASSIC
    if len(found) > 1:
        return {"name": "mixed", "llm_judges": [], "checks": [], "variants": list(found.values())}
    return next(iter(found.values()))
