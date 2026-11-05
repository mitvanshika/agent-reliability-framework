import os

import yaml

from trace_utils import load_latest_traces

SCENARIOS_DIR = "scenarios"

VARIANTS = {
    # Same folders as run_judges.VARIANTS. Not imported from there because
    # that would load the judges + Groq client just to list files.
    "strong": {"traces_dir": "traces", "judged_dir": "judged_traces"},
    "weak": {"traces_dir": "traces_weak", "judged_dir": "judged_traces_weak"},
}


def get_all_scenario_ids() -> set[str]:
    ids = set()
    for filename in os.listdir(SCENARIOS_DIR):
        if filename.endswith(".yaml"):
            with open(os.path.join(SCENARIOS_DIR, filename), encoding="utf-8") as f:
                scenario = yaml.safe_load(f)
                ids.add(scenario["id"])
    return ids


def get_judged_status(folder: str) -> dict[str, str]:
    """Returns {scenario_id: status} for every scenario that has a judged file.

    status is one of:
      "ok"          - agent answered and judging succeeded
      "agent_error" - the agent itself crashed, so there was no answer to judge.
                      Re-judging can't fix this; the agent has to be re-run.
      "judge_error" - the agent answered but judging failed (e.g. rate limit).
                      Re-judging can fix this.
    """
    status = {}
    if not os.path.isdir(folder):
        return status
    # Latest run per scenario only (see trace_utils.py).
    for scenario_id, trace in load_latest_traces(folder).items():
        if trace.get("error") is not None:
            s = "agent_error"
        elif trace.get("judge_error") is None and trace.get("final_score") is not None:
            s = "ok"
        else:
            s = "judge_error"
        status[scenario_id] = s
    return status


def get_traced(folder: str) -> dict[str, dict]:
    """{scenario_id: latest trace} for scenarios the agent has run."""
    if not os.path.isdir(folder):
        return {}
    return load_latest_traces(folder)


def main():
    all_ids = get_all_scenario_ids()
    for variant, dirs in VARIANTS.items():
        status = get_judged_status(dirs["judged_dir"])
        traced = get_traced(dirs["traces_dir"])
        unjudged = [sid for sid in all_ids if sid not in status]
        # No trace at all: the agent never ran this scenario (e.g. new YAML).
        never_ran = sorted(sid for sid in unjudged if sid not in traced)
        # Trace exists but was never judged (e.g. run_judges.py was interrupted).
        # A crashed trace can't be judged, so it goes with agent_crashed instead.
        not_judged = sorted(sid for sid in unjudged
                            if sid in traced and traced[sid].get("error") is None)
        judge_failed = sorted(sid for sid in all_ids if status.get(sid) == "judge_error")
        agent_crashed = sorted(sid for sid in all_ids if status.get(sid) == "agent_error"
                               or (sid in unjudged and sid in traced
                                   and traced[sid].get("error") is not None))
        needs_judging = sorted(not_judged + judge_failed)

        problems = len(never_ran) + len(needs_judging) + len(agent_crashed)
        print(f"\n{variant} ({problems} with problems out of {len(all_ids)} total):")
        if never_ran:
            print(f"  agent never ran ({len(never_ran)}): {' '.join(never_ran)}")
        if not_judged:
            print(f"  traced but never judged ({len(not_judged)}): {' '.join(not_judged)}")
        if judge_failed:
            print(f"  judge failed ({len(judge_failed)}): {' '.join(judge_failed)}")
        if agent_crashed:
            print(f"  agent crashed ({len(agent_crashed)}): {' '.join(agent_crashed)}")

        if never_ran:
            print("\n  No trace exists for the 'agent never ran' ones, so there is nothing to judge yet.")
            print("  Run the agent (this re-runs the whole suite for this variant), then judge:")
            print(f"  python run_suite.py {variant}")
            print(f"  python run_judges.py {variant}")
        if needs_judging:
            print(f"\n  Re-judge with:\n  python rejudge_specific.py {variant} {' '.join(needs_judging)}")
        if agent_crashed:
            print("\n  Re-judging can't fix crashed ones (there is no answer to judge).")
            print("  Re-run the agent for them with run_suite.py, then judge again.")
        if not problems:
            print("  none — all judged successfully")


if __name__ == "__main__":
    main()