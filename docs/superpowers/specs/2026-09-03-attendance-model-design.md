# Mississippi State Home Attendance Model — Design

Date: 2026-09-03
Status: approved in chat, pending spec review

Note: the seed `data/tickets.csv` and the exploration scripts (`data/build.py`, `data/analyze.py`,
`data/joined.csv`) predate this spec. The first implementation task converts the seed file to the
schema below and moves the exploration scripts to `exploration/`.

## Goal

Predict announced attendance for Mississippi State football home games at Davis Wade Stadium from
(a) game features available from the CollegeFootballData (CFBD) API and (b) the resale get-in
ticket price from ticketdata.com. Two tiers: Tier 1 uses game features only (usable before any
price exists); Tier 2 adds one price feature (usable once a price is listed).

The exploratory analysis on 2023–2025 (16 games with both price and attendance) found get-in
price to be the single best out-of-sample predictor of attendance (leave-one-out R² ≈ 0.5), with
opponent quality (AP-ranked flag, SP+, Elo, conference game) next. Price levels shift by season,
so within-season relative price is expected to generalize better than raw price.

## Non-goals

- Predicting the ticket price itself (the pipeline is built so the target can be swapped later).
- Any team other than Mississippi State, or any venue other than Davis Wade Stadium.
- Automated scraping of ticketdata.com (Cloudflare blocks it; the user hits the site once per update).
- Using the ticketdata "3-day change" column (no signal in the exploration).
- Betting spread as a feature (only available game week, so unusable for upcoming games).
- Weather (CFBD weather endpoint requires a paid tier).

## Data inputs

### Ticket prices — `data/tickets.csv` (hand-maintained)

Columns:

| column     | type            | notes                                                                 |
|------------|-----------------|-----------------------------------------------------------------------|
| opponent   | string          | ticketdata team name without mascot, e.g. `Ole Miss`, `UMass`         |
| date       | YYYY-MM-DD      | local (Central) game date as shown on ticketdata                       |
| getin      | number or blank | get-in price in USD; blank when the site shows none                    |
| observed   | YYYY-MM-DD or blank | date the price was read; blank for the historical seed rows        |

Rules:
- Rows are never deleted by the pipeline. The user appends or edits rows after each game.
- Upcoming games may be listed with the currently displayed price; when the game passes the user
  overwrites `getin` with the final price shown on the past tab.
- Opponent names are mapped to CFBD names through an alias table in `config.py`
  (`UMass` → `Massachusetts`, `Southeastern Louisiana` → `SE Louisiana`, `USM` → `Southern Miss`,
  etc.). An unmapped name that does not match any CFBD home opponent for that season is a hard
  error listing the season's CFBD home opponents so the user can fix the row.
- Duplicate (opponent, date) pairs are a hard error.

### Browser console snippet — `scripts/ticketdata_console.js`

Pasted into the browser dev-tools console on
`https://www.ticketdata.com/performer/mississippi-state-bulldogs-football?tab=past` (and the
upcoming tab). Reads `table tbody tr`, dedupes the three responsive copies of the table by
(event, date), strips the "at Mississippi State ..." suffix and the "Egg Bowl - " prefix, and
prints CSV lines in the `tickets.csv` column order with `observed` set to today. The user copies
the lines into `data/tickets.csv`. No network calls; it only reads the loaded DOM.

### CFBD — `data/cfbd_raw/` (fetched once, refreshed only while a season is open)

Per season `S` present in `tickets.csv`:

| file                 | endpoint                                   | used for                                  |
|----------------------|--------------------------------------------|-------------------------------------------|
| `games_S.json`       | `/games?year=S&team=Mississippi State`     | attendance, pregame Elo, conference flag, kickoff, week, opponent conference/classification |
| `rankings_S.json`    | `/rankings?year=S`                         | AP Top 25 rank of opponent in the game's week |
| `sp_S.json`          | `/ratings/sp?year=S`                       | opponent SP+ rating                        |
| `elo_S.json`         | `/ratings/elo?year=S`                      | current Elo for opponents of upcoming games (pregame Elo is null before kickoff) |

Refresh rule (`fetch` command): a season's files are (re)downloaded when any of these hold:
1. `games_S.json` is missing;
2. any game in the cached file has `completed == false`;
3. any game in the cached file started within the last 14 days (attendance can post late);
4. `--refresh S` was passed.

Otherwise the season is frozen and no requests are made. All four files for a season refresh
together. API key is read from `CFBD_API_KEY` in the environment or a `.env` file at the repo root
(`.env` is git-ignored; `.env.example` documents it). Requests use `Authorization: Bearer <key>`.
A non-200 response is a hard error naming the endpoint and status.

## Feature build — `data/features.csv`

One row per Mississippi State home game (`homeTeam == "Mississippi State"`, regular season and
postseason alike, neutral-site games excluded), for every season in `tickets.csv`, whether or not
a ticket row exists. Join key: (season, CFBD opponent name). CFBD `startDate` is UTC and is
converted to America/Chicago before deriving the local date and kickoff hour; the ticket `date`
must equal that local date, otherwise the row is flagged as a join error.

| feature        | definition                                                                    |
|----------------|-------------------------------------------------------------------------------|
| season, week   | from CFBD                                                                     |
| attendance     | CFBD `attendance`; null for upcoming games or when CFBD has none (target)     |
| conf_game      | CFBD `conferenceGame` as 0/1                                                  |
| opp_p4         | opponent conference in {SEC, Big Ten, Big 12, ACC} → 1 else 0                 |
| opp_fcs        | opponent classification != fbs → 1 else 0                                     |
| opp_ranked     | opponent in AP Top 25 → 1 else 0. Poll used: the season's poll for the game's week if cached; otherwise the latest cached poll for that season with week < game week; otherwise 0 with a warning |
| opp_ap_rank    | rank if ranked, else 30                                                       |
| opp_elo        | pregame Elo from the game record; if null, the opponent's rating from `elo_S.json`; if still null (FCS), season minimum FBS Elo minus 100 |
| opp_sp         | opponent SP+ rating for the season; FCS → season minimum FBS SP+ minus 10     |
| getin          | from tickets.csv; null when absent                                            |
| log_getin      | ln(getin)                                                                     |
| rel_log_price  | log_getin minus the median log_getin over all games in the same season that have a price (including listed prices on upcoming games) |

Capacity constant `CAPACITY = 60417` (the observed sellout figure at Davis Wade; official listed
capacity is 60,311). Used only to clip predictions.

## Models — `ticketmodel/model.py`

Ordinary least squares (statsmodels) with an intercept. Training rows: home games with non-null
attendance (Tier 1) and additionally non-null price (Tier 2).

**Tier 1 feature selection.** Candidate pool `{opp_ranked, conf_game, opp_elo, opp_sp, opp_p4, week}`.
Evaluate every subset of size 1–3 by leave-one-out RMSE on the Tier 1 training rows. Choose the
minimum LOO-RMSE, except that a smaller subset is preferred when its LOO-RMSE is within 5%
(`SELECTION_TOLERANCE`) of the best. Record the top five subsets and their LOO-RMSE in the report.

**Tier 2.** The chosen Tier 1 subset plus exactly one of `{log_getin, rel_log_price}`, chosen by
LOO-RMSE on the Tier 2 training rows. Both alternatives are reported.

**Persistence.** `models/tier1.json` and `models/tier2.json` hold: feature list, coefficients,
intercept, residual standard error, degrees of freedom, training row count, and the training
data hash. `predict` reads these; it never refits.

**Prediction.** Point estimate clipped to `[0, CAPACITY]`. 80% prediction interval from the OLS
prediction variance (observation-level), also clipped.

## Evaluation — `reports/model_report.md`

Written by `train`. Contains:
- Row counts (games, with attendance, with price).
- For each of: season-mean baseline, price-only (`log_getin`), Tier 1, Tier 2 — LOO-RMSE, LOO-MAE,
  LOO-R². The season-mean baseline predicts each held-out game with the mean attendance of the
  other games in its season, falling back to the global mean when the season has no other rows.
- Tier 1 top-five subset table; Tier 2 price-feature comparison.
- Fitted coefficients with standard errors for both tiers.
- Per-game table: season, opponent, price, actual attendance, Tier 1 LOO prediction, Tier 2 LOO
  prediction (blank where no price).
- A caveats section repeating: small n, capacity ceiling, final-price timing, season price shifts.

## Prediction — `reports/predictions.csv`

`predict` scores every home game in `features.csv` with null attendance. Output columns:
season, date, opponent, getin, tier1_pred, tier1_lo, tier1_hi, tier2_pred, tier2_lo, tier2_hi.
Tier 2 columns are blank when no price is listed. Tier 2 is only scored for a game whose season
has at least 3 priced games (`MIN_PRICED_PER_SEASON`), counting past and upcoming games;
otherwise Tier 2 is left blank and a warning names the season. The same table is printed to the
terminal.

## Project layout

```
ticket-model/
  requirements.txt              pandas, numpy, scipy, statsmodels, python-dotenv, pytest
  .env.example                  CFBD_API_KEY=
  README.md                     the after-each-game workflow in five lines
  ticketmodel/
    __init__.py
    config.py                   TEAM, VENUE, CAPACITY, alias table, paths, candidate features
    cfbd.py                     fetch_season(season, force) with the refresh rule; cache I/O
    tickets.py                  load + validate tickets.csv
    features.py                 build_features() -> DataFrame; writes data/features.csv
    model.py                    loo_rmse(), select_tier1(), select_tier2(), fit(), save/load, predict()
    report.py                   write_report()
    cli.py                      `python -m ticketmodel fetch|build|train|predict [--refresh S]`
  scripts/ticketdata_console.js
  data/tickets.csv  data/cfbd_raw/  data/features.csv
  models/tier1.json  models/tier2.json
  reports/model_report.md  reports/predictions.csv
  tests/
    fixtures/                   small synthetic CFBD payloads (2 seasons, ~6 games)
    test_tickets.py  test_features.py  test_model.py  test_cfbd.py  test_cli.py
```

Commands are idempotent. `build` requires cached CFBD files (it does not fetch). `train` runs
`build` first. `predict` requires saved models. `python -m ticketmodel all` runs fetch → build →
train → predict.

## Error handling

Hard errors (non-zero exit, one clear message): missing API key on fetch; non-200 CFBD response;
unmapped opponent; duplicate ticket rows; ticket date not matching the CFBD local date; fewer
than 8 training rows for a tier; `predict` without saved models. Warnings (printed, continue):
home game with no ticket row; opponent with no SP+ or Elo (imputed).

## Testing

pytest, no network. `cfbd.py` is tested with a fake HTTP function injected via parameter.
Coverage targets:
- tickets: valid file loads; blank price → null; duplicate rows error; unknown opponent error.
- features: UTC-to-Central date shift (a 01:00 UTC kickoff maps to the previous local date);
  FCS opponent gets imputed Elo/SP+ and `opp_fcs = 1`; `rel_log_price` centers each season on its
  median including an upcoming priced game; game with no ticket row gets null price.
- model: `loo_rmse` matches a hand-computed 4-row case; selection returns the subset with the
  lowest LOO-RMSE on a fixture where one feature is exact; predictions clipped at CAPACITY;
  saved model round-trips to identical predictions.
- cfbd: refresh rule — frozen season makes zero requests; season with an incomplete game
  refreshes; `--refresh` forces.
- cli: `predict` with a game lacking a price yields blank Tier 2 columns.

## Workflow after each game

1. Open the ticketdata past tab, paste the console snippet, copy the new row into `data/tickets.csv`.
2. `python -m ticketmodel all`
3. Read `reports/model_report.md` and `reports/predictions.csv`.
