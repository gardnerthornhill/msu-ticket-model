# MSU Ticket Model

Predicts announced attendance at Mississippi State home football games from the resale
**get-in ticket price**, using the price as a live read on demand and CollegeFootballData (CFBD)
for the schedule and opponent strength.

The production price model uses **season-relative log get-in price alone**. In the
2023–2025 audit it predicted attendance better than the former combination of price,
AP ranked status, opponent Elo and opponent SP+. The formula stays fixed while its
coefficients are refitted as data arrives; diagnostic comparisons never silently switch
production back to raw prices or extra opponent variables.

## The price model

Tier 2 is ordinary least squares with an intercept and one feature:

`rel_log_price = ln(getin) - median(ln(getin)) within the season`

The reference includes every home game with a recorded price, even upcoming games and
games without attendance. For an even number of prices it is the geometric mean of the
two middle prices. This allows different seasons' dollar price levels to share a model.
A common percentage change in all prices cancels out, so the model cannot measure a
season-wide change in demand from relative price alone.

Predictions are clipped to 0–60,417 announced attendees and have an observation-level
80% prediction interval. The reported sellout percentage is the model's tail probability
above that attendance threshold; it is not calibrated to ticket inventory.

Tier 1 remains a schedule-only fallback. It chooses up to three schedule variables by
leave-one-out error. Its Elo and SP+ coefficients overlap and should not be interpreted
individually. Live Tier 2 forecasts require at least three listed prices in that season.
Sparse historical seasons remain in training and in the per-season diagnostics so hard
outcomes are not discarded; their references are less reliable.

`reports/model_report.md` and the site's explanation page show:

- Common-sample leave-one-out comparisons, including raw and relative price alone.
- The current price formula versus the legacy price-plus-opponent alternatives.
- Price-model error, bias, price coverage and interval coverage by season.
- Whole-season holdouts and forward tests using only earlier seasons for fitting.

These are **retrospective** tests with cached features and full-season price references,
not a record of forecasts made before kickoff. Price observation dates in the original
historical sample are unknown. The fixed price formula was chosen after studying the
same small dataset, so prospective validation is still needed.

## The site

`site/` is a static site generated from the model outputs: a 2026 outlook with the forecast,
80% range and sellout odds for every home game, a track record of every tracked game scored
with its archived pregame forecast where available and leave-one-out otherwise, a page per game
showing how the price moved the number, and a
plain-English explanation of the weights. It is rebuilt by `python3 -m ticketmodel site`
(and by `all`), so the daily Action keeps it current. To look at it locally:

```
python3 -m ticketmodel site
python3 -m http.server -d site 8000      # then open http://localhost:8000
```

Opponent logos are downloaded once from ESPN's CDN by CFBD team id into `logos/` (a missing
logo becomes initials, never an error). The Mississippi State mark is `logos/msu.png`, a
transparent cut of `logos/MSU.png`.

Hosting: `netlify.toml` tells Netlify to publish the prebuilt `site/` folder from `main` with no
build step, so every daily commit redeploys. Set a `SITE_URL` environment variable (the public
URL, no trailing slash) where the site is generated, i.e. as a repository variable for the daily
Action, to emit canonical links, Open Graph URLs and a sitemap; without it the site carries no
absolute links to itself. The pages deliberately carry no link back to the repository.

## How the data flows

```
data/tickets.csv  (latest get-in prices, hand-maintained)
        +
data/cfbd_raw/    (CFBD games, AP polls, SP+, Elo; fetched once per season)
        |
        v
data/features.csv -> models/tier2.json (+ tier1.json) -> reports/model_report.md
                                                       -> reports/train_summary.json
                                                       -> reports/predictions.csv
                                                       -> site/  (with logos/)
```

- **Ticket prices** come from ticketdata.com, which sits behind Cloudflare, so they are pasted in
  by hand. `scripts/ticketdata_console.js` prints ready-to-paste CSV rows from the loaded page.
  Update a row whenever you want a fresh read. Previous values and supplied observation dates are preserved in `data/ticket_history.csv`.
- **CFBD data** is fetched with your own API key and cached. Finished seasons refresh only
  when explicitly forced; open or recently played seasons refresh automatically.
- **Missing attendance** is filled from `data/attendance_overrides.csv`, which records an
  official source for each correction. It does not rewrite the CFBD cache or replace an
  existing CFBD attendance value. Southern Miss 2023 is restored to 53,855 from the official
  Mississippi State postgame notes. Conflicting source values produce a warning.
- **Price history** records distinct price/observation-date versions when prices are updated
  or features are built. `recorded_at` is ingestion time, separate from `observed`. Unknown
  historical dates remain unknown.
- **Forecast history** in `reports/forecast_history.jsonl` preserves the inputs, fitted model,
  season reference and forecast as they existed before kickoff. Changed inputs/models and
  each new forecast day create a snapshot; identical reruns on the same day do not. No
  snapshot is recorded after kickoff, even when the feed still says incomplete. TBD games
  stop being archived at midnight locally on game day. This history supports future
  evaluation at consistent horizons without pretending retrospective fits were live calls.

## Setup

```
pip install -r requirements.txt
cp .env.example .env      # put your CFBD API key in it (free at collegefootballdata.com)
python3 -m ticketmodel all
```

## After each game

1. Open the ticketdata **past** tab, paste `scripts/ticketdata_console.js` into the browser
   dev-tools console, and copy the game's row into `data/tickets.csv`, replacing the earlier
   pre-game row if there was one. Opponent names may include the mascot; the pipeline strips it.
   Leave `getin` blank when the site shows no price.
2. `python3 -m ticketmodel all`
3. Read `reports/predictions.csv` (the `tier2_*` columns) and `reports/model_report.md`, or
   open the rebuilt `site/`.

Once CFBD posts the attendance, the game moves from predictions into the training set on the
next run.

## Starting a new season

The pipeline only fetches and predicts seasons that have at least one row in `data/tickets.csv`.
Before the first game, paste the **upcoming**-tab rows for the new season into the file, then run
`python3 -m ticketmodel all`. Every home game of that season then appears in `data/features.csv`
and `reports/predictions.csv`. Tier 2 stays blank until the season has three priced games, since
the price feature is relative to the season median.

## Daily automation

`.github/workflows/daily-model.yml` runs every morning at 7am Central: it refreshes the
in-progress CFBD season, retrains, re-predicts, and commits any changes. It needs one repository
secret, `CFBD_API_KEY`, set once from this folder:

```
gh secret set CFBD_API_KEY --body "$(cut -d= -f2- .env)"
```

ticketdata.com blocks automated access with Cloudflare (plain HTTP and headless browsers alike
get the "Just a moment" page from a GitHub runner), so prices are pasted in, not scraped. To add
or update prices from any device: open the repo's **Actions** tab, choose **daily-model**, press
**Run workflow**, and paste rows into the box, one per line:

```
Alabama,2026-10-03,91
Auburn,2026-11-14,80,2026-10-01
```

Rows are matched on opponent and date, so pasting a game again updates its price. A missing
observed date becomes today. The same thing locally:

```
python3 -m ticketmodel add-tickets --rows "Alabama,2026-10-03,91"
```

## Commands

```
python3 -m ticketmodel fetch     # CFBD -> data/cfbd_raw/ (per the refresh rule; --refresh 2025 to force)
python3 -m ticketmodel build     # -> data/features.csv
python3 -m ticketmodel train     # -> models/*.json, reports/model_report.md (fixed relative-price production formula)
python3 -m ticketmodel predict   # -> reports/predictions.csv
python3 -m ticketmodel site      # -> site/ (downloads any missing opponent logos)
python3 -m ticketmodel all       # fetch, build, train, predict, site
python3 -m ticketmodel add-tickets --rows "..."   # upsert pasted price rows into data/tickets.csv
python3 -m pytest                # tests, no network
```

## Caveats

- Small sample: only a few seasons and fewer than two dozen priced outcomes.
- The target is announced attendance, not ticket scans or sales.
- Historical prices came from the past-events table and have no verified observation dates.
  Current prices can be weeks before kickoff; retrospective error is not weeks-ahead accuracy.
- Full-season relative price references can contain information unavailable on the historical
  forecast date. SP+ in the schedule fallback is also season-level, not a dated pregame snapshot.
- The sparse 2024 season remains visible in training and diagnostics. Recovering its missing
  prices is preferable to dropping its difficult outcomes.
- Nominal 80% intervals and sellout odds depend on model assumptions; calibration is uncertain.
- MSU record/losses was an exploratory candidate and is not part of the production formula.

## Layout

```
ticketmodel/      config, tickets loader, CFBD fetch/cache, feature build, model, report, site, CLI
ticketmodel/templates, ticketmodel/static   Jinja templates, CSS and JS for the site
scripts/          browser console snippet for ticketdata.com
data/             tickets.csv, ticket_history.csv, attendance_overrides.csv, features.csv, cfbd_raw/
models/           fitted models as JSON
reports/          model_report.md, train_summary.json, predictions.csv, forecast_history.jsonl
logos/            team logos (MSU mark plus ESPN logos by team id)
site/             generated static site (published by Netlify)
exploration/      the original correlation analysis that motivated the model
docs/superpowers/ design spec and implementation plan
tests/            pytest suite (synthetic fixtures, no network)
```
