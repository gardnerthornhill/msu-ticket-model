# Mississippi State Attendance Model

Generated 2026-09-03.

## Data

- Home games in features: 22
- With attendance (Tier 1 training rows): 21
- With attendance and price (Tier 2 training rows): 16

## Leave-one-out accuracy

| model | rows | RMSE | MAE | R² |
|---|---|---|---|---|
| Season mean (all rows) | 21 | 4905 | 3790 | -0.017 |
| Tier 1 (all rows) | 21 | 3182 | 2507 | 0.572 |
| Season mean (priced rows) | 16 | 5635 | 4874 | -0.220 |
| Price only (priced rows) | 16 | 3589 | 2965 | 0.505 |
| Tier 1 (priced rows) | 16 | 3681 | 3163 | 0.479 |
| Tier 2 (priced rows) | 16 | 2967 | 2288 | 0.662 |

## Tier 1 feature selection

Top candidate subsets by LOO-RMSE (ties within 0.001 go to fewer features).

| features | LOO RMSE |
|---|---|
| opp_ranked + opp_elo + opp_sp | 3182 |
| opp_ranked + conf_game + opp_elo | 3444 |
| opp_ranked + conf_game | 3479 |
| opp_ranked + conf_game + opp_p4 | 3513 |
| opp_ranked + opp_sp | 3530 |

## Tier 2 price feature

| features | LOO RMSE |
|---|---|
| opp_ranked + opp_elo + opp_sp + rel_log_price | 2967 |
| opp_ranked + opp_elo + opp_sp + log_getin | 3620 |

## Fitted models

### Tier 1 (game features only)

Features: opp_ranked + opp_elo + opp_sp. Rows: 21. Residual SE: 3043.

| term | coefficient | std err |
|---|---|---|
| intercept | 66021 | ± 7813 |
| opp_ranked | 8537.78 | ± 1972.05 |
| opp_elo | -11.90 | ± 5.53 |
| opp_sp | 211.95 | ± 89.83 |

### Tier 2 (game features + price)

Features: opp_ranked + opp_elo + opp_sp + rel_log_price. Rows: 16. Residual SE: 2524.

| term | coefficient | std err |
|---|---|---|
| intercept | 49987 | ± 9309 |
| opp_ranked | 714.63 | ± 2788.57 |
| opp_elo | 1.95 | ± 6.86 |
| opp_sp | -48.36 | ± 120.64 |
| rel_log_price | 6405.24 | ± 1819.33 |

## Per-game leave-one-out predictions

| season | date | opponent | price | actual | Tier 1 LOO | Tier 2 LOO |
|---|---|---|---|---|---|---|
| 2023 | 2023-09-02 | SE Louisiana | 8 | 50041 | 47598 | 49025 |
| 2023 | 2023-09-09 | Arizona | 12 | 51648 | 52381 | 49005 |
| 2023 | 2023-09-16 | LSU | 31 | 60084 | 58128 | 56851 |
| 2023 | 2023-09-30 | Alabama | 25 | 60111 | 55600 | 56114 |
| 2023 | 2023-10-07 | Western Michigan | 6 | 47158 | 47869 | 47302 |
| 2023 | 2023-11-04 | Kentucky | 10 | 52329 | 48633 | 49121 |
| 2023 | 2023-11-23 | Ole Miss | 63 | 60412 | 58018 | 60417 |
| 2024 | 2024-08-31 | Eastern Kentucky |  | 48724 | 49194 |  |
| 2024 | 2024-09-14 | Toledo |  | 47412 | 47348 |  |
| 2024 | 2024-09-21 | Florida |  | 49655 | 51736 |  |
| 2024 | 2024-10-19 | Texas A&M | 10 | 50127 | 57891 | 54962 |
| 2024 | 2024-10-26 | Arkansas |  | 49303 | 50897 |  |
| 2024 | 2024-11-02 | Massachusetts |  | 48617 | 49258 |  |
| 2024 | 2024-11-23 | Missouri | 9 | 47824 | 49224 | 55121 |
| 2025 | 2025-09-06 | Arizona State | 32 | 50808 | 55964 | 52009 |
| 2025 | 2025-09-13 | Alcorn State | 26 | 49158 | 48611 | 50863 |
| 2025 | 2025-09-20 | Northern Illinois | 11 | 45803 | 44819 | 43596 |
| 2025 | 2025-09-27 | Tennessee | 95 | 60417 | 55264 | 57809 |
| 2025 | 2025-10-25 | Texas | 48 | 52680 | 55298 | 54441 |
| 2025 | 2025-11-08 | Georgia | 45 | 53017 | 56106 | 53583 |
| 2025 | 2025-11-28 | Ole Miss | 133 | 60417 | 55783 | 60241 |

## Warnings

- 2023 SE Louisiana: no Elo; imputed 862.0
- 2023 SE Louisiana: no SP+; imputed -36.3
- 2024 Eastern Kentucky: no Elo; imputed 660.0
- 2024 Eastern Kentucky: no SP+; imputed -43.0
- 2025 Alcorn State: no Elo; imputed 618.0
- 2025 Alcorn State: no SP+; imputed -46.6

## Caveats

- Small sample: a couple of dozen games. Coefficients are rough; the leave-one-out numbers are the honest accuracy.
- Attendance is the announced figure and is capped at Davis Wade Stadium capacity (60,417); sellouts flatten the top end.
- The get-in price is the final price recorded by ticketdata near game day, not a price observed weeks out.
- Price levels shift season to season; the season-relative price feature exists for that reason.
