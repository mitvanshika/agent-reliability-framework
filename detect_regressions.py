import json
import os
import sys
from scipy.stats import fisher_exact

from alerts import send_slack_alert, format_regression_alert


def load_judged_traces(folder):
    results = {}
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(folder, filename)) as f:
            trace = json.load(f)
        scenario_id = trace["scenario_id"]
        results[scenario_id] = trace
    return results


def passed(trace, threshold=7):
    if trace.get("error") is not None:
        return False
    score = trace.get("final_score")
    if score is None:
        return False
    return score >= threshold


def detect_regressions():
    strong = load_judged_traces("judged_traces")
    weak = load_judged_traces("judged_traces_weak")

    shared_ids = sorted(set(strong.keys()) & set(weak.keys()))

    print("Per-scenario comparison:")
    print("-" * 60)
    for scenario_id in shared_ids:
        s_trace = strong[scenario_id]
        w_trace = weak[scenario_id]
        s_score = s_trace.get("final_score")
        w_score = w_trace.get("final_score")
        s_str = f"{s_score:.1f}" if s_score is not None else "ERROR"
        w_str = f"{w_score:.1f}" if w_score is not None else "ERROR"
        flag = ""
        if s_score is not None and w_score is not None and (s_score - w_score) >= 2:
            flag = "  <-- possible regression"
        print(f"{scenario_id:15} strong: {s_str:6} weak: {w_str:6}{flag}")

    strong_pass = sum(1 for t in strong.values() if passed(t))
    strong_fail = len(strong) - strong_pass
    weak_pass = sum(1 for t in weak.values() if passed(t))
    weak_fail = len(weak) - weak_pass

    print("-" * 60)
    print(f"Strong agent pass rate: {strong_pass}/{len(strong)}")
    print(f"Weak agent pass rate:   {weak_pass}/{len(weak)}")

    table = [[strong_pass, strong_fail], [weak_pass, weak_fail]]
    odds_ratio, p_value = fisher_exact(table)

    print(f"\nFisher's exact test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: statistically significant regression detected.")
        alert_text = format_regression_alert(strong_pass, len(strong), weak_pass, len(weak), p_value)
        send_slack_alert(alert_text)
    else:
        print("Result: no statistically significant regression (could be normal variation, or sample size too small to tell).")

    return p_value


if __name__ == "__main__":
    p_value = detect_regressions()
    # Non-zero exit on a real regression — lets this be dropped into
    # a CI pipeline and fail a build automatically.
    sys.exit(1 if p_value < 0.05 else 0)