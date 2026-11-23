"""
Turns the Financial Due Diligence eval set (its eval/eval_set.json, 29 questions)
into scenario files for this framework: scenarios_fdd/fdd_q01.yaml ... fdd_q29.yaml.
Only READS the Financial DD repo; writes only inside this repo.

    python adapters/convert_fdd_eval.py [path to the Financial DD repo] [--out scenarios_fdd]

The default repo path is ../Financial-Due-Diligence (next to this repo).
Ids keep the eval's own numbering (fdd_q20 = its Q20), so results can be
matched with Financial DD's own eval notes.

Re-run it if the eval set changes; don't edit the files by hand. If an answer key
changes after a baseline is promoted, compare.py notices (answer-key hash) and
refuses to compare until the baseline is re-judged.
"""

import argparse
import hashlib
import json
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert(fdd_repo, out_dir):
    src = os.path.join(fdd_repo, "eval", "eval_set.json")
    with open(src, "rb") as f:
        raw = f.read()
    questions = json.loads(raw.decode("utf-8"))
    fingerprint = hashlib.sha256(raw).hexdigest()[:12]
    os.makedirs(out_dir, exist_ok=True)
    for i, q in enumerate(questions, start=1):
        sid = f"fdd_q{i:02d}"
        scenario = {
            "id": sid,
            "question": q["question"],
            "category": q["type"],              # numeric_lookup, narrative_comprehension, cross_referencing, no_answer, trend
            "expected_answer": q["expected_answer"],
            "difficulty": q.get("difficulty"),
            "reference_source": q.get("source"),   # where the answer is in the filing; for people, not used in scoring
            "generated_from": f"Financial-Due-Diligence eval/eval_set.json #{i} (sha256 {fingerprint})",
        }
        with open(os.path.join(out_dir, f"{sid}.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(scenario, f, sort_keys=False, allow_unicode=True, width=1000)
    return len(questions), fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("fdd_repo", nargs="?", default=os.path.join(REPO, "..", "Financial-Due-Diligence"))
    p.add_argument("--out", default=os.path.join(REPO, "scenarios_fdd"))
    args = p.parse_args()
    n, fingerprint = convert(os.path.abspath(args.fdd_repo), args.out)
    print(f"Wrote {n} scenarios to {args.out} (eval_set.json sha256 {fingerprint})")


if __name__ == "__main__":
    main()