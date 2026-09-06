# Mississippi State Attendance Model

Generated 2026-09-06.

## Data

- Home games in features: 29
- With attendance (Tier 1 training rows): 23
- With attendance and price (Tier 2 training rows): 18

## Leave-one-out accuracy

| model | rows | RMSE | MAE | R² |
|---|---|---|---|---|
| Season mean (all rows) | 23 | 4698 | 3602 | 0.007 |
| Tier 1 (all rows) | 23 | 3270 | 2724 | 0.519 |
| Season mean (priced rows) | 18 | 5369 | 4571 | -0.190 |
| Price only (priced rows) | 18 | 3652 | 3063 | 0.449 |
| Relative price only (priced rows) | 18 | 2229 | 1602 | 0.795 |
| Tier 1 (priced rows) | 18 | 3698 | 3260 | 0.435 |
| Tier 2 (priced rows) | 18 | 2229 | 1602 | 0.795 |

## Tier 1 feature selection

Top candidate subsets by LOO-RMSE. A smaller subset is preferred when its LOO-RMSE is within 5% of the best.

| features | LOO RMSE |
|---|---|
| opp_ranked + opp_elo + opp_sp | 3270 |
| opp_ranked | 3529 |
| opp_sp | 4193 |
| conf_game | 4213 |
| opp_p4 | 4262 |

## Tier 2 price feature

Production uses relative log-price alone. These alternatives are comparisons, not an automatic selection rule.

| features | LOO RMSE |
|---|---|
| rel_log_price | 2229 |
| log_getin | 3652 |
| opp_ranked + opp_elo + opp_sp + rel_log_price | 2659 |
| opp_ranked + opp_elo + opp_sp + log_getin | 3357 |

## Fitted models

### Tier 1 (game features only)

Features: opp_ranked + opp_elo + opp_sp. Rows: 23. Residual SE: 3120.

| term | coefficient | std err |
|---|---|---|
| intercept | 64794 | ± 7623 |
| opp_ranked | 8048.11 | ± 1990.33 |
| opp_elo | -10.71 | ± 5.37 |
| opp_sp | 185.53 | ± 85.12 |

### Tier 2 (relative price only)

Features: rel_log_price. Rows: 18. Residual SE: 2203.

| term | coefficient | std err |
|---|---|---|
| intercept | 53074 | ± 519 |
| rel_log_price | 6442.80 | ± 749.61 |

## Price-model results by season

Positive bias means overprediction. Sparse-price seasons remain visible.

| season | priced reference games | scored games | RMSE | MAE | bias | inside 80% range |
|---|---|---|---|---|---|---|
| 2023 | 8 | 8 | 1956 | 1365 | -1178 | 7/8 |
| 2024 | 2 | 2 | 4425 | 4341 | 4341 | 0/2 |
| 2025 | 7 | 7 | 1500 | 1037 | -633 | 7/7 |
| 2026 | 7 | 1 | 1978 | 1978 | 1978 | 1/1 |

## Season transfer

Fixed relative-price specification, refitted without the test season. Forward tests train only on earlier seasons. Both retain archived features and full-season price references; neither is a live forecast replay. Folds with fewer than 8 training games are skipped.

| test | season | training games | scored games | RMSE | MAE | bias |
|---|---|---|---|---|---|---|
| Season held out | 2023 | 10 | 8 | 2222 | 1661 | -1648 |
| Season held out | 2024 | 16 | 2 | 4682 | 4611 | 4611 |
| Season held out | 2025 | 11 | 7 | 1382 | 952 | -793 |
| Season held out | 2026 | 17 | 1 | 1978 | 1978 | 1978 |
| Season held out, pooled | all | — | 18 | 2364 | 1730 | — |
| Earlier seasons only | 2024 | 8 | 2 | 5024 | 4956 | 4956 |
| Earlier seasons only | 2025 | 10 | 7 | 1296 | 895 | -622 |
| Earlier seasons only | 2026 | 17 | 1 | 1978 | 1978 | 1978 |
| Earlier seasons only, pooled | all | — | 10 | 2572 | 1816 | — |

## Per-game leave-one-out predictions

| season | date | opponent | price | actual | Tier 1 LOO | Tier 2 LOO |
|---|---|---|---|---|---|---|
| 2023 | 2023-09-02 | SE Louisiana | 8 | 50041 | 48587 | 48862 |
| 2023 | 2023-09-09 | Arizona | 12 | 51648 | 52482 | 51590 |
| 2023 | 2023-09-16 | LSU | 31 | 60084 | 57887 | 57395 |
| 2023 | 2023-09-30 | Alabama | 25 | 60111 | 55604 | 55967 |
| 2023 | 2023-10-07 | Western Michigan | 6 | 47158 | 48594 | 47122 |
| 2023 | 2023-11-04 | Kentucky | 10 | 52329 | 49127 | 50265 |
| 2023 | 2023-11-18 | Southern Miss | 19 | 53855 | 49091 | 54600 |
| 2023 | 2023-11-23 | Ole Miss | 63 | 60412 | 57789 | 60417 |
| 2024 | 2024-08-31 | Eastern Kentucky |  | 48724 | 50143 |  |
| 2024 | 2024-09-14 | Toledo |  | 47412 | 48117 |  |
| 2024 | 2024-09-21 | Florida |  | 49655 | 51935 |  |
| 2024 | 2024-10-19 | Texas A&M | 10 | 50127 | 57832 | 53608 |
| 2024 | 2024-10-26 | Arkansas |  | 49303 | 51192 |  |
| 2024 | 2024-11-02 | Massachusetts |  | 48617 | 49886 |  |
| 2024 | 2024-11-23 | Missouri | 9 | 47824 | 49713 | 53025 |
| 2025 | 2025-09-06 | Arizona State | 32 | 50808 | 56168 | 50883 |
| 2025 | 2025-09-13 | Alcorn State | 26 | 49158 | 49695 | 49577 |
| 2025 | 2025-09-20 | Northern Illinois | 11 | 45803 | 46284 | 43283 |
| 2025 | 2025-09-27 | Tennessee | 95 | 60417 | 55352 | 57539 |
| 2025 | 2025-10-25 | Texas | 48 | 52680 | 55477 | 53538 |
| 2025 | 2025-11-08 | Georgia | 45 | 53017 | 56155 | 53077 |
| 2025 | 2025-11-28 | Ole Miss | 133 | 60417 | 55759 | 59970 |
| 2026 | 2026-09-05 | UL Monroe | 30 | 48771 | 46316 | 50749 |

## Warnings

- 2023 SE Louisiana: no Elo; imputed 862.0
- 2023 SE Louisiana: no SP+; imputed -36.3
- 2024 Eastern Kentucky: no Elo; imputed 660.0
- 2024 Eastern Kentucky: no SP+; imputed -43.0
- 2025 Alcorn State: no Elo; imputed 618.0
- 2025 Alcorn State: no SP+; imputed -46.6
- 2026 Missouri: no 2026 Elo yet; using 2025 final Elo 1750
- 2026 Vanderbilt: no 2026 Elo yet; using 2025 final Elo 1880
- 2026 Tennessee Tech: no Elo; imputed 805.0
- 2026 Tennessee Tech: no SP+; imputed -42.9
- 2023 Southern Miss: missing attendance filled with 53,855 from https://hailstate.com/news/2023/11/18/football-postgame-notes-mississippi-state-vs-southern-miss
- 2026 UL Monroe: missing attendance filled with 48,771 from https://hailstate.com/sports/football/stats/2026/ulm/boxscore/27455
- season 2024: only 2 priced games; sparse historical rows remain in training and diagnostics; new forecasts use Tier 1 until 3 prices are available

## Caveats

- Small sample: a couple of dozen games across three historical seasons. These are retrospective tests, not forecasts recorded before kickoff.
- The target is announced attendance, not ticket scans or ticket sales. Predictions are capped at 60,417; nominal 80% coverage is conditional on model assumptions and has not been established for weeks-ahead forecasts.
- Historical prices were collected from the past-events table; their observation dates are unknown. Current prices may be weeks before kickoff, and price age matters.
- The season reference uses all available prices, including upcoming games. Historical final-price references were not necessarily available on the forecast date. A common percentage change in every price leaves relative-price predictions unchanged.
- The Tier 2 specification is fixed to relative log-price alone. Alternatives are diagnostics and do not automatically change production. It was chosen after examining this small dataset, so its retrospective scores may still be optimistic.
- Tier 1 features are selected using the same leave-one-out scores reported. Its opponent ratings are collinear and cached SP+ values are season-level, not verified pregame snapshots.
- Live Tier 2 forecasts require at least 3 priced games. Sparse historical seasons are retained in training and shown separately in diagnostics rather than dropping difficult outcomes; their season references are less reliable.
- Sellout odds are an uncalibrated model probability of reaching 60,417 announced attendees, not a verified probability that ticket inventory sells out.
