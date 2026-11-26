# Judge fix: generic completeness graded its own instructions

**Found on:** Financial DD baseline (`fdd_baseline_20261124_135240`), judge set generic_v1.
**Symptom:** fdd_q10 ("What cybersecurity risks does Apple describe?") got **completeness 0 with confidence 0.97**,
reason: *"The request asked to split the evaluation task into its constituent parts ... the system's output did not
perform any of the requested evaluation steps."*
**Cause:** the judge prompt said "Split the request into its parts". The judge read **its own prompt** as "the
request" and graded Financial DD for not doing the judge's job. (A judge can only judge what it can see, and here it
misread what it saw. Confidence 0.97 did not mean correct.)

## Fix
New judge set **generic_v2** (`judges/generic.py judge_completeness_v2`): the prompt says it is a grader, must not do
the task or grade against its own instructions, and splits **the input the system received** into parts. Everything
else is unchanged; generic_v1 stays for old runs. `regrade.py --to generic_v2` re-ran only completeness (29 calls) and
now keeps the target's own judge list (without that, the re-graded run recorded 5 judges and compare.py would have
refused it next to new 4-judge runs).

## Old -> new (completeness, same answers)
| | generic_v1 | generic_v2 |
|---|---|---|
| fdd_q10 | 0 (final 5.28, FAIL) | 6 (final 6.83, still FAIL: correctness 3, details missing) |
| fdd_q09 | 5 (final 6.79, FAIL) | 10 (final 7.98, pass) |
| fdd_q15 | 10 | 8 (noise) |
| other 26 | unchanged | unchanged |
| pass rate | 27/29 | 28/29 |

fdd_q09 is a side effect, stated openly: v1 penalised the same missing reference details twice (correctness 4 and
completeness 5); v2 judges the question's parts, so missing details count only in correctness_gt.
The baseline was promoted after this re-grade, with a note saying so.
