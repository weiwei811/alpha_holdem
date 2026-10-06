# Six-player model benchmark

Run: `six-player-20261006-lowentropy-final-1h`. 108,000 benchmark hands across two independent deal banks, computed on Modal CPU.

Final pool profit: +18.14 chips/hand; reference: +11.35. Paired improvement: +6.79, 95% interval [+3.84, +9.73].

| Opponent table | Final chips/hand | 95% interval | Reference |
|---|---:|---|---:|
| random | +25.45 | [+20.54, +30.36] | +18.42 |
| check_call | +15.62 | [+9.67, +21.57] | +9.88 |
| structured_initial | +25.63 | [+20.75, +30.51] | +18.64 |
| legacy_table | +10.38 | [+5.95, +14.80] | +2.78 |
| historical_mix | +13.60 | [+8.80, +18.41] | +7.03 |

Settings: structured six-player model, entropy 0.02, ranked historical sampling, opponent refresh 0.01, frozen pool. The final continuation added one bounded T4 hour.

Health: 2,828,000 transitions, 707 updates, 689,990 parameters; finite weights: True.

Repository files: `weights.pkl` for inference and `training_config.json` for the matching architecture. Full optimizer/league state stays in Modal volume `alpha-holdem-budget-results`, under `six-player-20261006-lowentropy-final-1h/league/`; it is not included in this inference package.

Limits: one training seed and fixed benchmark opponents. Positive benchmark profit and improvement do not establish optimal poker play or exploitability. The screen did not isolate a significant entropy effect; the final result measures the complete continuation recipe.
