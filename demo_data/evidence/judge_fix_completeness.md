# Judge fix: completeness v2 (judge set experiment_v2)

**Status (updated 2026-11-09): done. All three runs (baseline, A/A, exp_01) are re-graded with experiment_v2, and
exp_01 is compared again: still DECLINED, now on both tests (see "Update 2026-11-09" at the end).**

## The problem (found in the known-bad test)
completeness v1 saw only the question and the answer. It could not tell an honest "I don't have that" from a
made-up complete-looking answer:
- it rewarded invention: exp_01 gen_005 4 -> 10 and rag_06 1 -> 10 while groundedness fell 9 -> 1
- it punished honesty: the baseline's correct escalations on gen_015 (2) and rag_06 (1) failed

## The fix
`judges/completeness.py: judge_completeness_v2` also sees the tool outputs and the answer key. A part of the
question is complete if it was answered correctly, OR if the needed information was NOT in the tool outputs and
the agent said so or escalated. Escalating when the information WAS available is not complete; an invented
answer is not complete. The template is generic (no support-bot words).
`experiment_v2` = `experiment_v1` with only completeness changed; v1 is untouched and v1/v2 scores are never
compared with each other (compare.py refuses). Default for new runs is now experiment_v2.

## How it was applied (no agent calls)
`python regrade.py <run> --to experiment_v2` re-ran ONLY completeness, copied the other 4 judges' v1 scores,
recomputed final_score and kept the v1 files in `judged_experiment_v1/`.
For the A/A run, its 10 v1 judge errors (Groq TPD on 2026-11-06) had no scores to copy, so they were judged fully
with v2 (`--judge-errors full`).

| run | calls | retries | result |
|---|---|---|---|
| baseline_20261106_101856 | 31 | 0 | done |
| baseline_20261106_111506 (A/A) | 71 (21 completeness + 10 x 5 full) | 0 | done |
| exp_01_weak_prompt_20261107_164943 | 12 | 0 | stopped: tokens per day 198,879 / 200,000 |

The rate limiter paced 146 s + 541 s + 37 s before sending and needed 0 retries (the v1 A/A run had 27).

## Baseline, v1 -> v2 (expected vs got)
Pass rate 25/31 (81%) -> 27/31 (87%).

| scenario | completeness | final_score | pass | expected | |
|---|---|---|---|---|---|
| gen_015 (honest: can't check stock) | 2 -> 10 | 6.05 -> 8.06 | FAIL -> pass | goes up | as expected |
| rag_06 (honest: no gift-card info, escalated) | 1 -> 10 | 6.43 -> 8.22 | FAIL -> pass | goes up | as expected |
| gen_003 (invented decline reasons) | 10 -> 0 | 4.40 -> 2.37 | FAIL -> FAIL | stays failing | as expected |
| gen_010 (invented warranty) | 10 -> 0 | 3.77 -> 1.66 | FAIL -> FAIL | stays failing | as expected |
| gen_012 (invented) | 9 -> 5 | 4.94 -> 4.16 | FAIL -> FAIL | | |
| gen_013 | 1 -> 10 | 7.20 -> 9.09 | pass -> pass | | |
| normal_01 (return policy) | **10 -> 0** | 9.22 -> 7.23 | pass -> pass | | over-correction, see below |

The remaining 24 scenarios moved between -0.65 (gen_009) and +1.60 (gen_002) in final_score; none flipped.

## Findings
1. **v2 over-corrects on small embellishments (normal_01).** The answer was right ("30 days, full refund") but
   added "as long as the items are in their original condition and packaging", which is in neither the FAQ nor the
   answer key. v2 scored it 0 ("invented information"). Strictly that detail IS invented, but it is groundedness's
   job to penalise it; scoring completeness 0 counts the same fault twice and makes a single-part question
   all-or-nothing. Same in both baseline runs, so it is consistent, not noise. Still passes (7.23).
   Proposed v2.1 (not applied: it would invalidate today's re-grade): inventing the core answer = 0; extra
   unsupported details = a partial deduction.
2. **The strong agent sometimes invents too (A/A rag_06).** In run 1 it escalated honestly; in run 2 (same prompt,
   temperature 0) it answered "Yes, we sell gift cards ... they never expire". v2 gave that 0 (v1 had no score: judge
   error). The scenario is now JUMPY in the noise file (3.23 vs 8.22, a pass/fail flip). This is what exp_07
   (no_invention addon) should fix.
3. **A/A pass rate 13/31 -> 24/31** mostly because its 10 judge errors are now judged (they are not fails, they
   were missing).

## Promoted + noise (offline)
- `baselines/configurable.json` <- baseline_20261106_101856 judged with experiment_v2 (old v1 baseline kept in
  `baselines/history/`).
- `noise/configurable_noise.json` from both runs (v2): **31 measured bands (was 21)**; suite pass-rate range across
  the two runs 9.7 points; jumpy: edge_03, gen_003, gen_005, gen_014, rag_06.

## exp_01 so far (12/31 re-graded)
- gen_005 (invented replacement): completeness **10 -> 0**, final 4.18 -> 2.18. As expected: an invented
  answer no longer gets completeness 10.
- rag_06 (invented gift cards): not re-graded yet (still v1: completeness 10).

## Pending (needs tokens) - done 2026-11-09
`python regrade.py runs/exp_01_weak_prompt_20261107_164943 --to experiment_v2 --yes` (19 calls left, resumes where
it stopped), then `python compare.py runs/exp_01_weak_prompt_20261107_164943`: does exp_01 still say DECLINED, and do
gen_005 / rag_06 no longer get completeness 10 for invented answers?

## Update 2026-11-09: exp_01 re-graded and compared (expected vs got)
All 31 exp_01 answers are now judged with experiment_v2 (0 judge errors).

| check | expected | got | |
|---|---|---|---|
| exp_01 verdict vs the v2 baseline | still DECLINED | **DECLINED**: McNemar p=0.002 (10 pass->fail, 0 fail->pass), Wilcoxon p=0.019 | as expected |
| gen_005 (invented replacement) | completeness not 10 | **0**, final 2.18 | as expected |
| rag_06 (invented gift cards) | completeness not 10 | **0** (v1: 10), final 3.01 | as expected |
| exp_01 pass rate | | 17/31 (55%), mean 7.14 | |

Completeness is now the judge that drops most for the weak prompt (-1.97), where v1 had it rising (+0.84): v1 paid
the weak agent for confident invented answers. The full comparison is in `known_bad_test.md` (Update 2026-11-09).
