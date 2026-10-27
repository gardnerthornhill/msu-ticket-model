# Mississippi State Attendance Model

Generated 2026-10-27.

## Data

- Home games in features: 29
- With attendance (Tier 1 training rows): 25
- With attendance and price (Tier 2 training rows): 20

## Leave-one-out accuracy

| model | rows | RMSE | MAE | R² |
|---|---|---|---|---|
| Season mean (all rows) | 25 | 5288 | 4106 | -0.095 |
| Tier 1 (all rows) | 25 | 3238 | 2637 | 0.589 |
| Season mean (priced rows) | 20 | 5924 | 5053 | -0.314 |
| Price only (priced rows) | 20 | 3625 | 2952 | 0.508 |
| Relative price only (priced rows) | 20 | 2842 | 1943 | 0.698 |
| Tier 1 (priced rows) | 20 | 3632 | 3081 | 0.506 |
| Tier 2 (priced rows) | 20 | 2842 | 1943 | 0.698 |

## Tier 1 feature selection

Top candidate subsets by LOO-RMSE. A smaller subset is preferred when its LOO-RMSE is within 5% of the best.

| features | LOO RMSE |
|---|---|
| opp_ranked + opp_elo + opp_sp | 3238 |
| opp_ranked | 3510 |
| conf_game | 4310 |
| opp_sp | 4364 |
| opp_p4 | 4391 |

## Tier 2 price feature

Production uses relative log-price alone. These alternatives are comparisons, not an automatic selection rule.

| features | LOO RMSE |
|---|---|
| rel_log_price | 2842 |
| log_getin | 3625 |
| opp_ranked + opp_elo + opp_sp + rel_log_price | 3177 |
| opp_ranked + opp_elo + opp_sp + log_getin | 3296 |

## Fitted models

### Tier 1 (game features only)

Features: opp_ranked + opp_elo + opp_sp. Rows: 25. Residual SE: 3126.

| term | coefficient | std err |
|---|---|---|
| intercept | 66056 | ± 7728 |
| opp_ranked | 8833.10 | ± 1921.08 |
| opp_elo | -11.62 | ± 5.45 |
| opp_sp | 202.27 | ± 87.96 |

### Tier 2 (relative price only)

Features: rel_log_price. Rows: 20. Residual SE: 2856.

| term | coefficient | std err |
|---|---|---|
| intercept | 53591 | ± 639 |
| rel_log_price | 6318.95 | ± 917.32 |

## Price-model results by season

Positive bias means overprediction. Sparse-price seasons remain visible.

| season | priced reference games | scored games | RMSE | MAE | bias | inside 80% range |
|---|---|---|---|---|---|---|
| 2023 | 8 | 8 | 1670 | 1275 | -654 | 8/8 |
| 2024 | 2 | 2 | 4937 | 4860 | 4860 | 0/2 |
| 2025 | 7 | 7 | 1290 | 1083 | -20 | 7/7 |
| 2026 | 7 | 3 | 5126 | 3785 | -2451 | 2/3 |

## Season transfer

Fixed relative-price specification, refitted without the test season. Forward tests train only on earlier seasons. Both retain archived features and full-season price references; neither is a live forecast replay. Folds with fewer than 8 training games are skipped.

| test | season | training games | scored games | RMSE | MAE | bias |
|---|---|---|---|---|---|---|
| Season held out | 2023 | 12 | 8 | 1617 | 1186 | -798 |
| Season held out | 2024 | 18 | 2 | 5195 | 5129 | 5129 |
| Season held out | 2025 | 13 | 7 | 1153 | 981 | 80 |
| Season held out | 2026 | 17 | 3 | 5040 | 3659 | -2709 |
| Season held out, pooled | all | — | 20 | 2832 | 1879 | — |
| Earlier seasons only | 2024 | 8 | 2 | 5024 | 4956 | 4956 |
| Earlier seasons only | 2025 | 10 | 7 | 1296 | 895 | -622 |
| Earlier seasons only | 2026 | 17 | 3 | 5040 | 3659 | -2709 |
| Earlier seasons only, pooled | all | — | 12 | 3396 | 2263 | — |

## Per-game leave-one-out predictions

| season | date | opponent | price | actual | Tier 1 LOO | Tier 2 LOO |
|---|---|---|---|---|---|---|
| 2023 | 2023-09-02 | SE Louisiana | 8 | 50041 | 48424 | 49528 |
| 2023 | 2023-09-09 | Arizona | 12 | 51648 | 52780 | 52169 |
| 2023 | 2023-09-16 | LSU | 31 | 60084 | 58951 | 57922 |
| 2023 | 2023-09-30 | Alabama | 25 | 60111 | 56420 | 56513 |
| 2023 | 2023-10-07 | Western Michigan | 6 | 47158 | 48442 | 47860 |
| 2023 | 2023-11-04 | Kentucky | 10 | 52329 | 49107 | 50886 |
| 2023 | 2023-11-18 | Southern Miss | 19 | 53855 | 49029 | 55111 |
| 2023 | 2023-11-23 | Ole Miss | 63 | 60412 | 58863 | 60417 |
| 2024 | 2024-08-31 | Eastern Kentucky |  | 48724 | 50052 |  |
| 2024 | 2024-09-14 | Toledo |  | 47412 | 47919 |  |
| 2024 | 2024-09-21 | Florida |  | 49655 | 52120 |  |
| 2024 | 2024-10-19 | Texas A&M | 10 | 50127 | 58432 | 54124 |
| 2024 | 2024-10-26 | Arkansas |  | 49303 | 51298 |  |
| 2024 | 2024-11-02 | Massachusetts |  | 48617 | 49840 |  |
| 2024 | 2024-11-23 | Missouri | 9 | 47824 | 49660 | 53548 |
| 2025 | 2025-09-06 | Arizona State | 32 | 50808 | 56717 | 51480 |
| 2025 | 2025-09-13 | Alcorn State | 26 | 49158 | 49575 | 50214 |
| 2025 | 2025-09-20 | Northern Illinois | 11 | 45803 | 45789 | 44291 |
| 2025 | 2025-09-27 | Tennessee | 95 | 60417 | 56146 | 58070 |
| 2025 | 2025-10-25 | Texas | 48 | 52680 | 56066 | 54068 |
| 2025 | 2025-11-08 | Georgia | 45 | 53017 | 56786 | 53621 |
| 2025 | 2025-11-28 | Ole Miss | 133 | 60417 | 56596 | 60417 |
| 2026 | 2026-09-05 | UL Monroe | 30 | 48771 | 47274 | 50772 |
| 2026 | 2026-09-26 | Missouri | 39 | 60417 | 57814 | 51798 |
| 2026 | 2026-10-03 | Alabama | 128 | 60417 | 56289 | 59683 |

## Warnings

- 2023 SE Louisiana: no Elo; imputed 862.0
- 2023 SE Louisiana: no SP+; imputed -36.3
- 2024 Eastern Kentucky: no Elo; imputed 660.0
- 2024 Eastern Kentucky: no SP+; imputed -43.0
- 2025 Alcorn State: no Elo; imputed 618.0
- 2025 Alcorn State: no SP+; imputed -46.6
- 2026 Tennessee Tech: no Elo; imputed 749.0
- 2026 Tennessee Tech: no SP+; imputed -39.1
- 2023 Southern Miss: missing attendance filled with 53,855 from https://hailstate.com/news/2023/11/18/football-postgame-notes-mississippi-state-vs-southern-miss
- 2026 UL Monroe: missing attendance filled with 48,771 from https://hailstate.com/sports/football/stats/2026/ulm/boxscore/27455
- 2026 Missouri: missing attendance filled with 60,417 from https://static.hailstate.com/custompages/stats/fb/2026/ms0926.PDF
- 2026 Alabama: missing attendance filled with 60,417 from https://hailstate.com/sports/football/stats/2026/alabama/boxscore/27571
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
- The cheapest resale listing does not measure primary ticket sales or the mix of home and visiting fans. Home-team momentum, fan campaigns and visiting support can affect turnout without a proportional change in get-in price; these drivers are not explicit inputs in the price model.
