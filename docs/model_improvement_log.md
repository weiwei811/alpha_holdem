# Six-player model improvement log

Updated: 2026-10-06. Additional Modal allowance: **$5**. Training, model evaluation and benchmark reports ran on Modal; unit tests ran locally.

## What changed

- Added the opt-in structured model: actor-relative seats/history, public betting features, separate policy/value encoders; 689,990 parameters. Existing models remain available.
- Added checkpoint/config checks, training diagnostics, uniform historical sampling and cloud comparison tools.
- Fixed opponent defaults, preemption recovery, container symlinks and checkpoint imports.
- Added an explicit entropy override after full-state restore; preserves model weights and optimizer state.
- Extended the one-hour launcher with selected-checkpoint resume and explicit entropy/sampling/refresh/frozen-pool settings.
- Fixed report serialization and duplicate final-step checkpoint handling. Export verifies the exact evaluated weights. Saved evaluation results are reused safely.
- Added independent evaluation seed options, historical-table tests and combined cloud reporting/model packaging.

## What the tests found

| Check | Result / decision |
|---|---|
| Previous extra hour | No clear gain: −1.43 chips/hand [−4.00, +1.15]. More time alone was insufficient in that run. |
| Ranked vs uniform pilot | Inconclusive; no reason to change sampling defaults. |
| Critic fit, whole-hand holdout | Tiny-set fit EV ≈1.00; held-out EV 0.034 → 0.067, then overfitting. Capacity and gradients work; noisy-return generalization is weak. |
| Equal-policy six-seat sanity | Profit interval includes zero. |
| Matched settings screen | All variants added 240,000 transitions. Lower entropy had the best average but no significant gain; diversity change did not clearly help. |
| Regression tests | Targeted tests: **21 passed**. Full suite before commit: **70 passed, 4 skipped**. Launcher syntax checks passed. |
| Cloud integration | Full optimizer/league resume, checkpoint export and both independent evaluations succeeded. CPU/CUDA verified; actual Mac MPS remains unverified. |

No new reward/bootstrap fault was found. IMPALA's value loss is batch-summed, so large logged values alone are not a scaling bug. Earlier critic samples were about 95% preflop; postflop conclusions remain limited. Entropy alone was not proven causal; the final result measures the complete continuation recipe.

## Final training and performance

Run: `six-player-20261006-lowentropy-final-1h`. Continued the selected checkpoint for one bounded T4 hour: entropy **0.02**, ranked sampling, refresh **0.01**, frozen historical pool. Finished successfully without preemption at **2,828,000 transitions / 707 updates**, with finite weights. Startup used ~15/16 CPU cores and ~17 GB container memory; full GPU utilization was not established.

Two independent test banks, **108,000 total benchmark hands** (7,200 per model/table):

| Opponent table | Starting model | Final model, chips/hand |
|---|---:|---:|
| Random | 18.42 | 25.45 |
| Check/call | 9.88 | 15.62 |
| Structured initialization | 18.64 | 25.63 |
| Legacy trained table | 2.78 | 10.38 |
| Historical mixed table | 7.03 | 13.60 |

Pool average improved **11.35 → 18.14 chips/hand**. Paired gain **+6.79**, 95% interval **[+3.84, +9.73]**. Both independent banks showed a positive supported gain; final profit intervals were positive on all five tables. This verifies stronger benchmark play, not optimal poker or exploitability; only one training seed was tested.

Final weights, matching config and cloud-generated report: `outputs/final_six_player_model/` in the task workspace. Full optimizer/league state remains in Modal volume `alpha-holdem-budget-results`, under the run above. See the README for continuation/evaluation commands.

Packaged the verified inference weights/config/report in `models/six_player_structured_v1/` for Git distribution. README now includes GUI and Python usage. Copied weights were checked byte-for-byte against the verified artifact.

## Budget

Conservative compute reservations (not a billing statement): probe $0.15; settings screen $1.26; report recovery/export $0.01; bounded hour $1.67; two final evaluations $0.26; combined report/package $0.01. **Total ≤$3.36 of $5**. No further training is running or needed for this verified result.

Published dated best checkpoint: `models/six_player_structured_v1/six_player_best_2026-10-06.pkl`. SHA-256 verified identical to the evaluated weights; README usage points to the dated file.
