# Financial DD truth test A/A: baseline vs baseline

**Expected:** NO REAL DIFFERENCE (the same Financial DD, settings and questions, run twice).
**Got:** **NO REAL DIFFERENCE** (McNemar p=1.000: 2 pass->fail, 1 fail->pass; Wilcoxon p=0.365).
**Result: PASS.** No false alarm, so a real change can be measured against this baseline.

## Numbers
| | baseline | A/A re-run |
|---|---|---|
| run | `fdd_baseline_20261124_135240` | `fdd_baseline_20261124_154652` |
| pass | 28/29 | 27/29 |
| mean final_score | 9.61 | 9.41 |
| tokens | 133,040 | 126,012 (-5.3%) |
| mean latency | 48.8 s | 41.2 s |

Judge set generic_v2 on both. Bands: `python calibrate_noise.py --from-runs <both runs>` ->
`noise/financial_dd_noise.json`. Suite mean moved 0.20, pass rate 3.5 points. 20 of 29 questions moved less than
0.5; jumpy: fdd_q09, fdd_q10, fdd_q20.

## Why the jumpy questions moved (`stability.py`, from the saved runs)
- **fdd_q20: SYSTEM.** 9.1 -> 4.0 with nothing changed. The first search failed in both runs, so Financial DD asked the
  LLM to split the question into 4 searches, and the LLM wrote them differently: run 1 included **"tax charge"** and
  found 5 relevant chunks out of 10, run 2 did not and found 1. Without the chunk about the one-time $10.2 bn EU tax
  charge, the answer missed the cause. One phrase in an LLM-written search decided the answer.
- **fdd_q25: JUDGE.** The same refusal, word for word, scored 8.6 and then 10.0 (correctness_gt 5 -> 10).
- **fdd_q09, fdd_q10: WORDING + borderline.** Same evidence, reworded answers, scores near the pass mark (7).

## Cost is noisy too
Total tokens moved 5.3% with nothing changed; single questions moved up to +36% (q24) and -29% (q15). So a cost
saving needs to be clearly beyond ~5% in total, and single-question token changes say little.
