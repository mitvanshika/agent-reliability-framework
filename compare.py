"""
Compares one judged experiment run against the promoted baseline for its
target (baselines/<target>.json, see promote_baseline.py).

    python compare.py runs/<run> [--baseline baselines/<target>.json] [--alpha 0.05] [--alert]

Per scenario (paired by scenario_id): score delta, and
    ▲ up / ▼ down  = the change is bigger than that scenario's noise band
    ═ same         = within the noise band
Noise bands come from noise/<target>_noise.json (calibrate_noise.py):
the score range seen when nothing changed, floored at 0.5 because a
range from a few runs underestimates the real spread. Without that file
every scenario uses a default band of 1.0, and the report says so.

An agent crash counts as a FAIL with score 0. A judge error (the agent
answered but judging failed) is not the agent's fault, so that pair is
excluded and listed instead.

Two paired tests:
  McNemar exact  pass/fail flips: b = pass->fail, c = fail->pass,
                 p = binomtest(min(b, c), b + c, 0.5); no flips -> p = 1
  Wilcoxon       signed-rank on paired scores; all differences 0 -> p = 1

Verdict (alpha = 0.05 by default):
  NOT ENOUGH DATA            fewer than 5 pairs
  DECLINED                   a test shows a significant decline
  IMPROVED                   significant improvement and no ▼ scenario
  IMPROVED WITH REGRESSIONS  significant improvement but some ▼
  NO REAL DIFFERENCE         nothing significant (any ▼ still listed)
If the tests disagree the more cautious verdict wins, in the order
DECLINED < NO REAL DIFFERENCE < IMPROVED WITH REGRESSIONS < IMPROVED.

Scores are only compared when both sides were judged with the same
judge set (judge_sets.py) and the same answer key; otherwise compare
refuses: re-judge the run or re-promote the baseline. Per-judge mean
deltas cover whatever judges the set has (incl. the tool_match check).

Saves runs/<run>/comparison.json. Exit code: 0 ok, 1 DECLINED,
2 refused (e.g. targets differ). --alert posts to Slack (alerts.py)
when the verdict is DECLINED or any scenario went ▼.
"""

import argparse
import json
import os
import statistics
import sys
import warnings

from scipy.stats import binomtest, wilcoxon

import judge_run
import judge_sets
from alerts import send_slack_alert
from detect_regressions import passed
from experiment_config import config_hash, diff_configs
from run_experiment import estimate_cost, load_pricing, p95
from trace_utils import load_latest_traces

BASELINES_DIR = "baselines"
NOISE_DIR = "noise"
DEFAULT_BAND = 1.0
MIN_BAND = 0.5
MIN_PAIRS = 5
MIN_WILCOXON_NONZERO = 6   # below this, a two-sided exact Wilcoxon can't reach p < 0.05
VERDICT_ORDER = ["DECLINED", "NO REAL DIFFERENCE", "IMPROVED WITH REGRESSIONS", "IMPROVED"]
SYMBOLS = {"up": "▲", "down": "▼", "same": "═"}
ASCII_SYMBOLS = {"up": "^", "down": "v", "same": "="}


class CompareError(Exception):
    """A reason to refuse comparing (exit code 2)."""


# ---------------------------------------------------------------- loading

def run_target(meta):
    return (meta.get("resolved_config") or {}).get("target")


def suite_key(target, suite="main"):
    """File key per (target, suite): the main suite keeps the plain target name."""
    return target if (suite or "main") == "main" else f"{target}__{suite}"


def baseline_path(target, baselines_dir=None, suite="main"):
    return os.path.join(baselines_dir or BASELINES_DIR, f"{suite_key(target, suite)}.json")


def load_run_records(run_dir):
    """(meta, {scenario_id: record}) for a run_experiment.py run folder.

    record status: ok | agent_error (crash, counts as fail) |
                   judge_error (judging failed) | not_judged (no judged file)
    """
    meta = judge_run.load_meta(run_dir)
    traces = load_latest_traces(os.path.join(run_dir, "traces"))
    judged_dir = os.path.join(run_dir, "judged")
    judged = load_latest_traces(judged_dir) if os.path.isdir(judged_dir) else {}
    import target_config
    threshold = target_config.for_meta(meta)["pass_threshold"]
    records = {}
    for sid in meta.get("scenario_ids") or sorted(traces):
        j, t = judged.get(sid), traces.get(sid, {})
        src = j or t
        if j is None:
            status = "not_judged"
        elif src.get("error") is not None:
            status = "agent_error"
        elif j.get("judge_error") or j.get("final_score") is None:
            status = "judge_error"
        else:
            status = "ok"
        judges = {}
        if j:
            js = j.get("judge_set") or judge_sets.CLASSIC
            for name in js["llm_judges"] + list(js.get("checks") or []):
                r = j.get(f"{name}_result")
                if isinstance(r, dict) and "score" in r and not r.get("na"):   # N/A judges are simply absent
                    judges[name] = r["score"]
        records[sid] = {
            "status": status,
            "final_score": j.get("final_score") if j else None,
            "passed": bool(j) and passed(j, threshold),
            "judges": judges,
            "error": src.get("error"),
            "error_type": src.get("error_type"),
            "judge_error": j.get("judge_error") if j else None,
            "latency_s": src.get("latency_s"),
            "usage": src.get("usage"),
            "answer_key_hash": j.get("answer_key_hash") if j else None,
            "disagreement": j.get("disagreement") if j else None,
            "low_confidence_judges": (j.get("low_confidence_judges") or []) if j else [],
        }
    return meta, records


def run_judge_set(run_dir):
    judged_dir = os.path.join(run_dir, "judged")
    if not os.path.isdir(judged_dir):
        return None
    return judge_sets.of_judged_traces(load_latest_traces(judged_dir).values())


def _judge_order(names):
    known = [n for n in judge_sets.ALL_NAMES if n in names]
    return known + sorted(n for n in names if n not in judge_sets.ALL_NAMES)


def noise_staleness(report, baseline):
    """Why a noise file doesn't describe this baseline (list of reasons; empty = fresh)."""
    reasons = []
    if not report.get("config_hash"):
        reasons.append("it records no config_hash (calibrate with --experiment)")
    elif report["config_hash"] != config_hash(baseline.get("resolved_config") or {}):
        reasons.append(f"it was calibrated on '{report.get('experiment')}' with a different config than "
                       f"baseline '{baseline.get('experiment')}'")
    if not judge_sets.same(report.get("judge_set"), baseline.get("judge_set")):
        reasons.append(f"judge set {judge_sets.describe(report.get('judge_set'))} vs baseline "
                       f"{judge_sets.describe(baseline.get('judge_set'))}")
    return reasons


def load_bands(target, noise_dir=None, baseline=None, suite="main"):
    """({scenario_id: band}, description). Empty dict -> default band everywhere.
    With baseline given, a noise file whose config_hash / judge_set don't match it is stale."""
    path = os.path.join(noise_dir or NOISE_DIR, f"{suite_key(target, suite)}_noise.json")
    if not os.path.isfile(path):
        return {}, f"no {path}: using default band {DEFAULT_BAND} for every scenario"
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    if baseline is not None:
        stale = noise_staleness(report, baseline)
        if stale:
            return {}, (f"WARNING: noise file is stale ({path}: {'; '.join(stale)}). "
                        f"Using default band {DEFAULT_BAND} for every scenario; re-run "
                        f"calibrate_noise.py --experiment <baseline file>.")
    bands = {sid: max(st["range"], MIN_BAND) for sid, st in report.get("scenarios", {}).items()
             if st.get("range") is not None}
    note = f"{path} ({report.get('mode')} mode, {report.get('runs')} runs; floor {MIN_BAND}, default {DEFAULT_BAND} if a scenario is missing)"
    if report.get("mode") == "judge_only":
        note += "; judge noise only, bands may be too narrow"
    return bands, note


# ---------------------------------------------------------------- statistics

def mcnemar_exact(b, c):
    """b = pass->fail, c = fail->pass. Two-sided exact binomial test on the flips."""
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def wilcoxon_paired(diffs):
    """(p, n_nonzero, median_diff, direction, note) for run - baseline differences."""
    nonzero = [d for d in diffs if d != 0]
    median = statistics.median(diffs) if diffs else None
    if not nonzero:
        return 1.0, 0, median, 0, "all differences are 0"
    nz_median = statistics.median(nonzero)
    direction = (nz_median > 0) - (nz_median < 0) or (sum(nonzero) > 0) - (sum(nonzero) < 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = float(wilcoxon(nonzero).pvalue)
    note = "; ".join(str(w.message).split("\n")[0] for w in caught)
    if len(nonzero) < MIN_WILCOXON_NONZERO:
        note = (f"only {len(nonzero)} non-zero pairs (< {MIN_WILCOXON_NONZERO}): Wilcoxon cannot reach "
                f"significance at 0.05 with this few" + (f"; {note}" if note else ""))
    return p, len(nonzero), median, direction, note


def _test_verdict(significant, direction, any_down):
    if not significant or direction == 0:
        return "NO REAL DIFFERENCE"
    if direction < 0:
        return "DECLINED"
    return "IMPROVED WITH REGRESSIONS" if any_down else "IMPROVED"


def compare_records(base, run, bands=None, alpha=0.05):
    """Pure comparison of two {scenario_id: record} dicts (see load_run_records)."""
    bands = bands or {}
    missing_in_run = sorted(set(base) - set(run))
    missing_in_baseline = sorted(set(run) - set(base))
    excluded, rows, judge_pairs = [], [], {}
    for sid in sorted(set(base) & set(run)):
        b, r = base[sid], run[sid]
        bad = [side for side, rec in (("baseline", b), ("run", r)) if rec["status"] in ("judge_error", "not_judged")]
        if bad:
            excluded.append({"scenario_id": sid, "reason": " + ".join(f"{s} {base[sid]['status'] if s == 'baseline' else r['status']}" for s in bad)})
            continue
        b_score = b["final_score"] if b["status"] == "ok" else 0.0   # crash = fail, score 0
        r_score = r["final_score"] if r["status"] == "ok" else 0.0
        band = bands.get(sid, DEFAULT_BAND)
        delta = round(r_score - b_score, 4)
        direction = "down" if delta < -band else "up" if delta > band else "same"
        b_pass, r_pass = b["status"] == "ok" and b["passed"], r["status"] == "ok" and r["passed"]
        flip = "pass->fail" if b_pass and not r_pass else "fail->pass" if r_pass and not b_pass else None
        moved = []
        for name in _judge_order(set(b["judges"]) & set(r["judges"])):
            judge_pairs.setdefault(name, []).append((b["judges"][name], r["judges"][name]))
            if abs(r["judges"][name] - b["judges"][name]) > band:
                moved.append(f"{name} {b['judges'][name]}->{r['judges'][name]}")
        notes = [f"{side} crashed ({rec['error_type'] or 'unknown'})"
                 for side, rec in (("baseline", b), ("run", r)) if rec["status"] == "agent_error"]
        common = set(b["judges"]) & set(r["judges"])
        rows.append({"scenario_id": sid, "baseline_score": b_score, "run_score": r_score, "delta": delta,
                     "band": band, "direction": direction, "baseline_passed": b_pass, "run_passed": r_pass,
                     "flip": flip, "judges_moved": moved, "notes": notes,
                     # for diagnose.py
                     "judge_deltas": {n: round(r["judges"][n] - b["judges"][n], 3) for n in _judge_order(common)},
                     "baseline_judges": b["judges"], "run_judges": r["judges"],
                     "baseline_error_type": b["error_type"] if b["status"] == "agent_error" else None,
                     "run_error_type": r["error_type"] if r["status"] == "agent_error" else None,
                     "run_error": (r["error"] or "")[:300] if r["status"] == "agent_error" else None,
                     "run_disagreement": bool(r.get("disagreement")),
                     "run_low_confidence": r.get("low_confidence_judges") or [],
                     "baseline_disagreement": bool(b.get("disagreement")),
                     "baseline_low_confidence": b.get("low_confidence_judges") or []})

    n = len(rows)
    b_flips = sum(1 for x in rows if x["flip"] == "pass->fail")
    c_flips = sum(1 for x in rows if x["flip"] == "fail->pass")
    p_mcnemar = mcnemar_exact(b_flips, c_flips)
    diffs = [x["delta"] for x in rows]
    p_wilcoxon, n_nonzero, median_diff, w_dir, w_note = wilcoxon_paired(diffs)
    down = [x["scenario_id"] for x in rows if x["direction"] == "down"]
    up = [x["scenario_id"] for x in rows if x["direction"] == "up"]

    m_dir = (c_flips > b_flips) - (c_flips < b_flips)
    v_mcnemar = _test_verdict(p_mcnemar < alpha, m_dir, bool(down))
    v_wilcoxon = _test_verdict(p_wilcoxon < alpha, w_dir, bool(down))
    if n < MIN_PAIRS:
        verdict, why = "NOT ENOUGH DATA", f"only {n} paired scenarios (need at least {MIN_PAIRS})"
    else:
        verdict = min(v_mcnemar, v_wilcoxon, key=VERDICT_ORDER.index)
        why = (f"McNemar -> {v_mcnemar} (p={p_mcnemar:.4f}); Wilcoxon -> {v_wilcoxon} (p={p_wilcoxon:.4f})"
               + ("; tests disagree, the more cautious verdict is used" if v_mcnemar != v_wilcoxon else ""))

    per_judge = {}
    for name in _judge_order(set(judge_pairs)):
        pairs = judge_pairs[name]
        b_mean = statistics.fmean(p[0] for p in pairs)
        r_mean = statistics.fmean(p[1] for p in pairs)
        per_judge[name] = {"baseline_mean": round(b_mean, 3), "run_mean": round(r_mean, 3),
                           "delta": round(r_mean - b_mean, 3), "n": len(pairs),
                           "in_final_score": name not in judge_sets.EXPERIMENT["checks"]}

    return {
        "alpha": alpha,
        "pairs": n,
        "per_judge": per_judge,
        "rows": rows,
        "missing_in_run": missing_in_run,
        "missing_in_baseline": missing_in_baseline,
        "excluded": excluded,
        "counts": {"up": len(up), "down": len(down), "same": n - len(up) - len(down)},
        "down": down,
        "up": up,
        "flips": {"pass_to_fail": b_flips, "fail_to_pass": c_flips},
        "mcnemar": {"b_pass_to_fail": b_flips, "c_fail_to_pass": c_flips, "p": p_mcnemar, "verdict": v_mcnemar},
        "wilcoxon": {"n_nonzero": n_nonzero, "median_diff": median_diff, "p": p_wilcoxon,
                     "verdict": v_wilcoxon, "note": w_note},
        "verdict": verdict,
        "verdict_reason": why,
    }


# ---------------------------------------------------------------- cost / latency / errors

def _pct(new, old):
    if new is None or old is None:
        return None
    return round((new - old) / old * 100, 1) if old else None


def resource_deltas(base_summary, base_settings, run_summary, run_settings, pricing):
    def cost(summary, settings):
        totals = {**summary.get("tokens", {}), "scenarios": summary.get("scenarios", 0),
                  "scenarios_with_usage": summary.get("scenarios_with_usage", 0)}
        return estimate_cost(totals, settings, pricing)[0]

    out = {}
    for key, b, r in [
        ("total_tokens", base_summary["tokens"].get("total_tokens"), run_summary["tokens"].get("total_tokens")),
        ("latency_mean_s", base_summary.get("latency_mean_s"), run_summary.get("latency_mean_s")),
        ("latency_p95_s", base_summary.get("latency_p95_s"), run_summary.get("latency_p95_s")),
        ("agent_errors", base_summary.get("agent_errors"), run_summary.get("agent_errors")),
        ("estimated_cost_usd", cost(base_summary, base_settings), cost(run_summary, run_settings)),
    ]:
        out[key] = {"baseline": b, "run": r,
                    "delta": round(r - b, 6) if r is not None and b is not None else None, "pct": _pct(r, b)}
    kinds = sorted(set(base_summary.get("errors_by_type", {})) | set(run_summary.get("errors_by_type", {})))
    out["errors_by_type"] = {k: {"baseline": base_summary.get("errors_by_type", {}).get(k, 0),
                                 "run": run_summary.get("errors_by_type", {}).get(k, 0)} for k in kinds}
    return out


# ---------------------------------------------------------------- output

def _symbols():
    try:
        "".join(SYMBOLS.values()).encode(sys.stdout.encoding or "ascii")
        return SYMBOLS
    except (UnicodeEncodeError, LookupError):
        return ASCII_SYMBOLS


def _f(x, fmt="{:.1f}"):
    return fmt.format(x) if x is not None else "-"


def print_comparison(result, header_lines=(), show_rows=True):
    sym = _symbols()
    for line in header_lines:
        print(line)
    print(f"Paired: {result['pairs']} scenarios"
          f" | missing in run: {' '.join(result['missing_in_run']) or '-'}"
          f" | missing in baseline: {' '.join(result['missing_in_baseline']) or '-'}")
    for ex in result["excluded"]:
        print(f"Excluded: {ex['scenario_id']} ({ex['reason']}): judging failed, not the agent's fault")
    if show_rows and result["rows"]:
        print(f"\n{'scenario':15} {'base':>5} {'run':>5} {'delta':>6} {'band':>5}     pass/fail   judges moved")
        for x in result["rows"]:
            flip = {"pass->fail": "P -> F", "fail->pass": "F -> P"}.get(x["flip"], "")
            extra = ", ".join(x["judges_moved"] + x["notes"])
            print(f"{x['scenario_id']:15} {x['baseline_score']:5.1f} {x['run_score']:5.1f} {x['delta']:+6.1f} "
                  f"{x['band']:5.1f}  {sym[x['direction']]}  {flip:10}  {extra}".rstrip())
    c = result["counts"]
    print(f"\n{sym['up']} {c['up']}  {sym['down']} {c['down']}  {sym['same']} {c['same']}"
          f"    flips: pass->fail {result['flips']['pass_to_fail']}, fail->pass {result['flips']['fail_to_pass']}")
    if result.get("per_judge"):
        print("Per-judge mean (scenarios scored on both sides; N/A left out):")
        for name, d in result["per_judge"].items():
            label = name if d["in_final_score"] else f"{name} (code check, not in final_score)"
            print(f"  {label:44} {d['baseline_mean']:5.2f} -> {d['run_mean']:5.2f}  ({d['delta']:+.2f}, n={d['n']})")
    m, w = result["mcnemar"], result["wilcoxon"]
    print(f"McNemar exact (pass/fail flips): b={m['b_pass_to_fail']} c={m['c_fail_to_pass']}  p={m['p']:.4f}")
    print(f"Wilcoxon signed-rank (scores):   non-zero pairs={w['n_nonzero']}  "
          f"median diff {_f(w['median_diff'], '{:+.2f}')}  p={w['p']:.4f}")
    if w["note"]:
        print(f"  warning: {w['note']}")


def subset_summary(records, ids):
    """Tokens / latency / errors over just these scenarios, from per-scenario records, so a
    --scenarios subset run is compared with the SAME scenarios of a full baseline."""
    recs = [records[s] for s in ids if s in records]
    usage = [r["usage"] for r in recs if r.get("usage")]
    latencies = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
    errors = {}
    for r in recs:
        if r["status"] == "agent_error":
            kind = r.get("error_type") or "unknown"
            errors[kind] = errors.get(kind, 0) + 1
    return {
        "scenarios": len(recs),
        "scenarios_with_usage": len(usage),
        "tokens": {k: sum(u.get(k, 0) for u in usage) if usage else None
                   for k in ("input_tokens", "output_tokens", "total_tokens")},
        "latency_mean_s": round(statistics.fmean(latencies), 3) if latencies else None,
        "latency_p95_s": p95(latencies),
        "agent_errors": sum(errors.values()),
        "errors_by_type": dict(sorted(errors.items())),
    }


def print_resources(res):
    n = res.get("scenarios")
    print(f"\nCost / latency / errors (run vs baseline{f', same {n} scenarios' if n else ''}):")
    for key, label, fmt in [("total_tokens", "tokens", "{:,.0f}"), ("latency_mean_s", "latency mean", "{:.2f}s"),
                            ("latency_p95_s", "latency p95", "{:.2f}s"), ("agent_errors", "agent errors", "{:.0f}"),
                            ("estimated_cost_usd", "est. cost", "${:.4f}")]:
        d = res[key]
        if d["baseline"] is None and d["run"] is None:
            continue
        delta = "" if d["delta"] is None else (" " + ("+" if d["delta"] >= 0 else "-") + fmt.format(abs(d["delta"])))
        pct = f" ({d['pct']:+.1f}%)" if d["pct"] is not None else ""
        print(f"  {label:13} {_f(d['run'], fmt):>10} vs {_f(d['baseline'], fmt):>10}{delta}{pct}")
    changed = {k: v for k, v in res["errors_by_type"].items() if v["baseline"] != v["run"]}
    if changed:
        print("  errors by type: " + ", ".join(f"{k} {v['baseline']}->{v['run']}" for k, v in changed.items()))


def format_slack(result, run_meta, base, run_dir):
    sym = SYMBOLS
    lines = [f":rotating_light: *Experiment comparison: {result['verdict']}*" if result["verdict"] == "DECLINED"
             else f":warning: *Experiment comparison: {result['verdict']}*",
             f"Run: {run_meta.get('experiment')} ({run_dir}) vs baseline {base.get('experiment')} "
             f"(target {base.get('target')})",
             f"{sym['down']} {len(result['down'])} down: {' '.join(result['down']) or '-'}",
             f"Flips pass->fail {result['flips']['pass_to_fail']}, fail->pass {result['flips']['fail_to_pass']}; "
             f"McNemar p={result['mcnemar']['p']:.4f}, Wilcoxon p={result['wilcoxon']['p']:.4f}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description="Compare a judged run against the promoted baseline.")
    p.add_argument("run_dir")
    p.add_argument("--baseline", help="baseline file (default: baselines/<target>.json)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--alert", action="store_true", help="Slack alert if DECLINED or anything went down")
    args = p.parse_args(argv)
    run_dir = args.run_dir.rstrip("/\\")

    try:
        meta, run_records = load_run_records(run_dir)
        target = run_target(meta)
        suite = meta.get("suite") or "main"
        bpath = args.baseline or baseline_path(target, suite=suite)
        if not os.path.isfile(bpath):
            raise CompareError(f"No baseline at {bpath}. Promote one first: python promote_baseline.py runs/<run>")
        with open(bpath, encoding="utf-8") as f:
            base = json.load(f)
        if (base.get("suite") or "main") != suite:
            raise CompareError(f"Different suites: this run is the '{suite}' suite, baseline {bpath} is the "
                               f"'{base.get('suite') or 'main'}' suite. Promote a baseline for this suite first.")
        if base.get("target") != target:
            raise CompareError(f"Targets differ: run is '{target}', baseline {bpath} is '{base.get('target')}'. "
                               "Comparing different systems scenario by scenario is meaningless.")
        if all(r["status"] == "not_judged" for r in run_records.values()):
            raise CompareError(f"{run_dir} has not been judged yet: python judge_run.py {run_dir}")
        run_js, base_js = run_judge_set(run_dir), base.get("judge_set") or judge_sets.CLASSIC
        if run_js and run_js["name"] == "mixed":
            raise CompareError(f"{run_dir} was judged with a mix of judge sets: re-judge it "
                               f"(python judge_run.py {run_dir}).")
        if not judge_sets.same(run_js, base_js):
            raise CompareError(
                f"Judge sets differ: baseline was judged with {judge_sets.describe(base_js)}, this run with "
                f"{judge_sets.describe(run_js)}. Old and new scores aren't comparable: re-judge or re-promote "
                f"the baseline (python judge_run.py {base.get('run_dir')} --yes, then "
                f"python promote_baseline.py {base.get('run_dir')}).")
        # Only scored answers carry an answer_key_hash. Judge errors, crashes and unjudged
        # scenarios are left out of the comparison anyway, so they can't count as "changed".
        changed_key = sorted(sid for sid in set(base["scenarios"]) & set(run_records)
                             if base["scenarios"][sid].get("status") == "ok" and run_records[sid]["status"] == "ok"
                             and base["scenarios"][sid].get("answer_key_hash") != run_records[sid].get("answer_key_hash"))
        if changed_key:
            raise CompareError(
                f"The answer key changed between the baseline and this run for: {' '.join(changed_key)}. "
                "Scores against different answer keys aren't comparable: re-judge or re-promote the baseline.")
    except CompareError as e:
        print(f"Refused: {e}", file=sys.stderr)
        sys.exit(2)

    bands, band_note = load_bands(target, baseline=base, suite=suite)
    if band_note.startswith("WARNING"):
        print(band_note, file=sys.stderr)
    result = compare_records(base["scenarios"], run_records, bands, args.alpha)

    base_cfg = base.get("resolved_config") or {}
    run_cfg = meta.get("resolved_config") or {}
    config_diff = diff_configs(base_cfg, run_cfg)
    pricing = load_pricing()
    run_summary = judge_run.summarize(run_dir, meta, pricing)
    # Same scenarios on both sides: a --scenarios subset is not compared with a full baseline.
    shared = [x["scenario_id"] for x in result["rows"]] + [e["scenario_id"] for e in result["excluded"]]
    resources = resource_deltas(subset_summary(base["scenarios"], shared), base_cfg.get("settings") or {},
                                subset_summary(run_records, shared), run_cfg.get("settings") or {}, pricing)
    resources["scenarios"] = len(shared)

    print_comparison(result, header_lines=[
        f"Comparing   {meta.get('experiment')}  ({run_dir})",
        f"against     baseline '{base.get('experiment')}' for target '{target}' "
        f"(promoted {base.get('promoted_at', '?')[:10]} from {base.get('run_dir')})"
        + (f', note: "{base["note"]}"' if base.get("note") else ""),
        f"Config diff vs baseline: {'; '.join(config_diff) if config_diff else '(identical config)'}",
        f"Judge set: {judge_sets.describe(base.get('judge_set'))}",
        f"Noise bands: {band_note}",
    ])
    print_resources(resources)
    print(f"\nVERDICT: {result['verdict']}")
    print(f"  {result['verdict_reason']}")
    if result["verdict"] == "NO REAL DIFFERENCE" and result["down"]:
        print(f"  but these scenarios went down beyond their noise band: {' '.join(result['down'])}")
    if result["verdict"] == "IMPROVED WITH REGRESSIONS":
        print(f"  regressions to review: {' '.join(result['down'])}")

    out = {"run_dir": run_dir, "experiment": meta.get("experiment"), "target": target,
           "baseline_file": bpath, "baseline_experiment": base.get("experiment"),
           "baseline_run_dir": base.get("run_dir"), "config_diff": config_diff,
           "noise_bands": band_note, "judge_set": base.get("judge_set") or judge_sets.CLASSIC,
           **result, "resources": resources}

    if result["down"]:
        import diagnose   # imported here: diagnose.py imports this module
        out["diagnosis"] = diagnose.diagnose(out)
        print(f"\nDiagnosis of the {len(out['diagnosis'])} down question(s) "
              f"(details: python diagnose.py {run_dir}; add --draft for a next-experiment file):")
        for d in out["diagnosis"]:
            print(f"  {d['scenario_id']:15} {d['cause']:26} next: {d['suggestion']}")
    path = os.path.join(run_dir, "comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")

    if args.alert:
        if result["verdict"] == "DECLINED" or result["down"]:
            send_slack_alert(format_slack(result, meta, base, run_dir))
        else:
            print("[alerts] nothing declined: no Slack alert sent.")

    if result["verdict"] == "DECLINED":
        sys.exit(1)
    return out


if __name__ == "__main__":
    main()
