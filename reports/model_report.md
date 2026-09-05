# Mississippi State Attendance Model

Generated 2026-09-05.

## Data

- Home games in features: 29
- With attendance (Tier 1 training rows): 22
- With attendance and price (Tier 2 training rows): 17

## Leave-one-out accuracy

| model | rows | RMSE | MAE | R² |
|---|---|---|---|---|
| Season mean (all rows) | 22 | 4746 | 3607 | 0.008 |
| Tier 1 (all rows) | 22 | 3300 | 2689 | 0.520 |
| Season mean (priced rows) | 17 | 5415 | 4574 | -0.196 |
| Price only (priced rows) | 17 | 3486 | 2859 | 0.504 |
| Relative price only (priced rows) | 17 | 2238 | 1598 | 0.796 |
| Tier 1 (priced rows) | 17 | 3777 | 3287 | 0.418 |
| Tier 2 (priced rows) | 17 | 2238 | 1598 | 0.796 |

## Tier 1 feature selection

Top candidate subsets by LOO-RMSE. A smaller subset is preferred when its LOO-RMSE is within 5% of the best.

| features | LOO RMSE |
|---|---|
| opp_ranked + opp_elo + opp_sp | 3300 |
| opp_ranked | 3612 |
| opp_sp | 4303 |
| conf_game | 4312 |
| opp_p4 | 4365 |

## Tier 2 price feature

Production uses relative log-price alone. These alternatives are comparisons, not an automatic selection rule.

| features | LOO RMSE |
|---|---|
| rel_log_price | 2238 |
| log_getin | 3486 |
| opp_ranked + opp_elo + opp_sp + rel_log_price | 2764 |
| opp_ranked + opp_elo + opp_sp + log_getin | 3570 |

## Fitted models

### Tier 1 (game features only)

Features: opp_ranked + opp_elo + opp_sp. Rows: 22. Residual SE: 3164.

| term | coefficient | std err |
|---|---|---|
| intercept | 66503 | ± 8117 |
| opp_ranked | 8262.97 | ± 2042.20 |
| opp_elo | -11.99 | ± 5.75 |
| opp_sp | 210.12 | ± 93.38 |

### Tier 2 (relative price only)

Features: rel_log_price. Rows: 17. Residual SE: 2221.

| term | coefficient | std err |
|---|---|---|
| intercept | 53183 | ± 539 |
| rel_log_price | 6356.38 | ± 762.56 |

## Price-model results by season

Positive bias means overprediction. Sparse-price seasons remain visible.

| season | priced reference games | scored games | RMSE | MAE | bias | inside 80% range |
|---|---|---|---|---|---|---|
| 2023 | 8 | 8 | 1905 | 1345 | -1065 | 7/8 |
| 2024 | 2 | 2 | 4557 | 4473 | 4473 | 0/2 |
| 2025 | 7 | 7 | 1444 | 1067 | -505 | 7/7 |

## Season transfer

Fixed relative-price specification, refitted without the test season. Forward tests train only on earlier seasons. Both retain archived features and full-season price references; neither is a live forecast replay. Folds with fewer than 8 training games are skipped.

| test | season | training games | scored games | RMSE | MAE | bias |
|---|---|---|---|---|---|---|
| Season held out | 2023 | 9 | 8 | 2139 | 1569 | -1530 |
| Season held out | 2024 | 15 | 2 | 4839 | 4769 | 4769 |
| Season held out | 2025 | 10 | 7 | 1296 | 895 | -622 |
| Season held out, pooled | all | — | 17 | 2366 | 1668 | — |
| Earlier seasons only | 2024 | 8 | 2 | 5024 | 4956 | 4956 |
| Earlier seasons only | 2025 | 10 | 7 | 1296 | 895 | -622 |
| Earlier seasons only, pooled | all | — | 9 | 2630 | 1798 | — |

## Per-game leave-one-out predictions

| season | date | opponent | price | actual | Tier 1 LOO | Tier 2 LOO |
|---|---|---|---|---|---|---|
| 2023 | 2023-09-02 | SE Louisiana | 8 | 50041 | 48199 | 49036 |
| 2023 | 2023-09-09 | Arizona | 12 | 51648 | 52832 | 51728 |
| 2023 | 2023-09-16 | LSU | 31 | 60084 | 58144 | 57447 |
| 2023 | 2023-09-30 | Alabama | 25 | 60111 | 55576 | 56034 |
| 2023 | 2023-10-07 | Western Michigan | 6 | 47158 | 48302 | 47348 |
| 2023 | 2023-11-04 | Kentucky | 10 | 52329 | 49018 | 50409 |
| 2023 | 2023-11-18 | Southern Miss | 19 | 53855 | 48882 | 54698 |
| 2023 | 2023-11-23 | Ole Miss | 63 | 60412 | 58037 | 60417 |
| 2024 | 2024-08-31 | Eastern Kentucky |  | 48724 | 49885 |  |
| 2024 | 2024-09-14 | Toledo |  | 47412 | 47761 |  |
| 2024 | 2024-09-21 | Florida |  | 49655 | 52127 |  |
| 2024 | 2024-10-19 | Texas A&M | 10 | 50127 | 57899 | 53731 |
| 2024 | 2024-10-26 | Arkansas |  | 49303 | 51273 |  |
| 2024 | 2024-11-02 | Massachusetts |  | 48617 | 49729 |  |
| 2024 | 2024-11-23 | Missouri | 9 | 47824 | 49584 | 53166 |
| 2025 | 2025-09-06 | Arizona State | 32 | 50808 | 56010 | 51033 |
| 2025 | 2025-09-13 | Alcorn State | 26 | 49158 | 49367 | 49754 |
| 2025 | 2025-09-20 | Northern Illinois | 11 | 45803 | 45429 | 43559 |
| 2025 | 2025-09-27 | Tennessee | 95 | 60417 | 55270 | 57588 |
| 2025 | 2025-10-25 | Texas | 48 | 52680 | 55290 | 53651 |
| 2025 | 2025-11-08 | Georgia | 45 | 53017 | 56073 | 53194 |
| 2025 | 2025-11-28 | Ole Miss | 133 | 60417 | 55757 | 59989 |

## Warnings

- 2023 SE Louisiana: no Elo; imputed 862.0
- 2023 SE Louisiana: no SP+; imputed -36.3
- 2024 Eastern Kentucky: no Elo; imputed 660.0
- 2024 Eastern Kentucky: no SP+; imputed -43.0
- 2025 Alcorn State: no Elo; imputed 618.0
- 2025 Alcorn State: no SP+; imputed -46.6
- 2026 Missouri: no 2026 Elo yet; using 2025 final Elo 1750
- 2026 Alabama: no 2026 Elo yet; using 2025 final Elo 1857
- 2026 Vanderbilt: no 2026 Elo yet; using 2025 final Elo 1880
- 2026 Auburn: no 2026 Elo yet; using 2025 final Elo 1670
- 2026 Tennessee Tech: no Elo; imputed 805.0
- 2026 Tennessee Tech: no SP+; imputed -42.9
- 2023 Southern Miss: missing attendance filled with 53,855 from https://hailstate.com/news/2023/11/18/football-postgame-notes-mississippi-state-vs-southern-miss
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
