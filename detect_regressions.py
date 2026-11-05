import json
import os
import sys
from scipy.stats import fisher_exact, mannwhitneyu

from alerts import send_slack_alert, format_regression_alert
from trace_utils import load_latest_traces

JUDGE_NAMES = ["correctness", "tool_efficiency", "completeness", "groundedness"]
NOISE_DIR = "noise"


def load_judged_traces(folder):
    # Latest run per scenario only (see trace_utils.py), so an old
    # run's file can't silently override the current one.
    return load_latest_traces(folder)


def passed(trace, threshold=7):
    if trace.get("error") is not None:
        return False
    score = trace.get("final_score")
    if score is None:
        return False
    return score >= threshold


def load_noise_bands(variants=("strong", "weak"), noise_dir=NOISE_DIR):
    """Score wobble per scenario from calibrate_noise.py output, if any.

    Returns ({scenario_id: band}, [source descriptions]). The band is the
    score range seen across calibration runs; if both variants were
    calibrated, the larger of the two (the conservative choice).
    """
    bands, sources = {}, []
    for variant in variants:
        path = os.path.join(noise_dir, f"{variant}_noise.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        note = ""
        if report.get("mode") == "judge_only":
            note = ", judge noise only: agent noise not included, bands may be too narrow"
        sources.append(f"{path} ({report.get('mode')} mode, {report.get('runs')} runs{note})")
        for scenario_id, st in report.get("scenarios", {}).items():
            if st.get("range") is not None:
                bands[scenario_id] = max(bands.get(scenario_id, 0.0), st["range"])
    return bands, sources


def _judge_field(trace, judge, field):
    return trace.get(f"{judge}_result", {}).get(field, "?")


def reliability_warnings(traces):
    """Judged traces whose evaluation itself is shaky.
    Returns (disagreed, low_confidence), each a list of (scenario_id, detail)."""
    disagreed, low_confidence = [], []
    for scenario_id in sorted(traces):
        trace = traces[scenario_id]
        if trace.get("final_score") is None:
            continue  # never judged (agent crash / judge error): nothing to assess
        if trace.get("disagreement"):
            detail = ", ".join(f"{j} {_judge_field(trace, j, 'score')}" for j in JUDGE_NAMES)
            disagreed.append((scenario_id, detail))
        low = trace.get("low_confidence_judges") or []
        if low:
            detail = ", ".join(f"{j} {_judge_field(trace, j, 'confidence')}" for j in low)
            low_confidence.append((scenario_id, detail))
    return disagreed, low_confidence


def print_evaluation_reliability(variants):
    # Display only: does not affect pass/fail, the statistical tests or the exit code.
    print("\nEvaluation reliability (where the MEASUREMENT is shaky, not the agent):")
    flagged, judged = set(), 0
    for name, traces in variants.items():
        disagreed, low_confidence = reliability_warnings(traces)
        judged += sum(1 for t in traces.values() if t.get("final_score") is not None)
        flagged |= {(name, sid) for sid, _ in disagreed + low_confidence}
        print(f"  {name}:")
        print("    judges disagreed (score spread >= 4):" + ("" if disagreed else " none"))
        for sid, detail in disagreed:
            print(f"      {sid:15} {detail}")
        print("    low-confidence judges (confidence < 0.5):" + ("" if low_confidence else " none"))
        for sid, detail in low_confidence:
            print(f"      {sid:15} {detail}")
    print(f"  {len(flagged)} of {judged} judged results have a reliability warning; "
          "treat those scores with caution.")


def detect_regressions():
    strong = load_judged_traces("judged_traces")
    weak = load_judged_traces("judged_traces_weak")

    shared_ids = sorted(set(strong.keys()) & set(weak.keys()))

    # Display only: noise bands never change pass/fail, the tests or the exit code.
    noise_bands, noise_sources = load_noise_bands()
    if noise_sources:
        print("Noise bands from: " + "; ".join(noise_sources))
        print("A drop no bigger than the scenario's noise band is marked (within noise).\n")

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
        noise = ""
        if scenario_id in noise_bands and s_score is not None and w_score is not None:
            delta = s_score - w_score
            noise = f"  delta {delta:+.1f}, noise {noise_bands[scenario_id]:.1f}"
            if 0 < delta <= noise_bands[scenario_id]:
                noise += " (within noise)"
        print(f"{scenario_id:15} strong: {s_str:6} weak: {w_str:6}{flag}{noise}")

    strong_pass = sum(1 for t in strong.values() if passed(t))
    strong_fail = len(strong) - strong_pass
    weak_pass = sum(1 for t in weak.values() if passed(t))
    weak_fail = len(weak) - weak_pass

    print("-" * 60)
    print(f"Strong agent pass rate: {strong_pass}/{len(strong)}")
    print(f"Weak agent pass rate:   {weak_pass}/{len(weak)}")

    print_evaluation_reliability({"strong": strong, "weak": weak})

    table = [[strong_pass, strong_fail], [weak_pass, weak_fail]]
    odds_ratio, p_value = fisher_exact(table)

    print(f"\nFisher's exact test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: statistically significant regression detected.")
    else:
        print("Result: no statistically significant regression (could be normal variation, or sample size too small to tell).")

    # Secondary check: Mann-Whitney U compares the raw 0-10 score
    # distributions directly, instead of thresholding everything into
    # pass/fail first. Fisher's exact answers "is the pass rate
    # different"; this answers "are the underlying scores different"
    # — related but not identical questions, since thresholding to
    # pass/fail discards some information (a 6.9 and a 7.1 are treated
    # as categorically different even though they're nearly the same).
    strong_scores = [t.get("final_score") for t in strong.values() if t.get("final_score") is not None]
    weak_scores = [t.get("final_score") for t in weak.values() if t.get("final_score") is not None]

    u_pvalue = None
    if strong_scores and weak_scores:
        u_stat, u_pvalue = mannwhitneyu(strong_scores, weak_scores, alternative="two-sided")
        print(f"\nMann-Whitney U test (raw score distributions) p-value: {u_pvalue:.4f}")
        if u_pvalue < 0.05:
            print("Result: score distributions are significantly different.")
        else:
            print("Result: no significant difference in score distributions.")
    else:
        print("\nMann-Whitney U test skipped (no scored data on one or both sides).")

    if p_value < 0.05:
        alert_text = format_regression_alert(strong_pass, len(strong), weak_pass, len(weak), p_value, u_pvalue)
        send_slack_alert(alert_text)

    return p_value


if __name__ == "__main__":
    p_value = detect_regressions()
    # Non-zero exit on a real regression — lets this be dropped into
    # a CI pipeline and fail a build automatically.
    sys.exit(1 if p_value < 0.05 else 0)