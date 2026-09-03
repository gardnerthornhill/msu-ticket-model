# MSU Ticket Model

Predicts announced attendance at Mississippi State home football games from the resale
**get-in ticket price**, using the price as a live read on demand and CollegeFootballData (CFBD)
for the schedule and opponent strength.

The headline finding from the 2023–2025 seasons: the get-in price on
[ticketdata.com](https://www.ticketdata.com/performer/mississippi-state-bulldogs-football) tracks
attendance more closely than any schedule feature. Within a season the rank correlation between
price and crowd size was about 0.93–0.95. A price model beats an opponent-quality model out of
sample.

## The price model

The model of interest here is **Tier 2**: ordinary least squares on

| feature | meaning |
|---|---|
| `rel_log_price` | log get-in price minus the season's median log price (so 2023 and 2025 price levels can share one model) |
| `opp_ranked` | opponent in the AP Top 25 that week |
| `opp_elo` | opponent Elo entering the game (last season's final Elo before they have played) |
| `opp_sp` | opponent SP+ rating for the season |

The price coefficient dominates. Predictions are clipped at Davis Wade Stadium capacity (60,417)
and come with an 80% prediction interval.

Leave-one-out accuracy on the 16 games with both a price and an attendance figure:

| model | LOO RMSE | LOO R² |
|---|---|---|
| Season-mean baseline | 5,635 | -0.22 |
| Price only | 3,589 | 0.51 |
| Opponent features only (Tier 1) | 3,681 | 0.48 |
| **Tier 2: opponent features + price** | **2,967** | **0.66** |

A schedule-only model (Tier 1) is fitted alongside as the baseline and as the fallback for games
with no listed price. It is not the focus, and its coefficients should not be read individually:
opponent Elo and SP+ are collinear and partly cancel.

Full accuracy tables, the chosen features, per-game leave-one-out predictions, and caveats are
regenerated into `reports/model_report.md` on every training run.

## The site

`site/` is a static site generated from the model outputs: a 2026 outlook with the forecast,
80% range and sellout odds for every home game, a track record of every tracked game scored
leave-one-out, a page per game showing how much each variable moved the number, and a
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
data/tickets.csv  (get-in prices, hand-maintained)
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
  Prices are frozen at the time you paste them; update a row whenever you want a fresh read.
- **CFBD data** is fetched with your own API key and cached. A finished season is never
  re-downloaded. Only a season with an unfinished game, or a game in the last 14 days, refreshes.

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
python3 -m ticketmodel train     # -> models/*.json, reports/model_report.md
python3 -m ticketmodel predict   # -> reports/predictions.csv
python3 -m ticketmodel site      # -> site/ (downloads any missing opponent logos)
python3 -m ticketmodel all       # fetch, build, train, predict, site
python3 -m ticketmodel add-tickets --rows "..."   # upsert pasted price rows into data/tickets.csv
python3 -m pytest                # tests, no network
```

## Caveats

- Small sample: a couple of dozen games. The leave-one-out numbers are the honest accuracy;
  coefficients are rough.
- The training prices are the final get-in prices recorded near game day. A price pasted weeks
  out is an earlier snapshot and will move; predictions move with it.
- Attendance is the announced figure and is capped at capacity, so sellouts flatten the top end
  and clipped intervals no longer carry their nominal coverage.
- Feature subsets were chosen by the same leave-one-out scores reported, so the headline RMSE
  is a little optimistic.

## Layout

```
ticketmodel/      config, tickets loader, CFBD fetch/cache, feature build, model, report, site, CLI
ticketmodel/templates, ticketmodel/static   Jinja templates, CSS and JS for the site
scripts/          browser console snippet for ticketdata.com
data/             tickets.csv, features.csv, cfbd_raw/
models/           fitted models as JSON
reports/          model_report.md, train_summary.json, predictions.csv
logos/            team logos (MSU mark plus ESPN logos by team id)
site/             generated static site (published by Netlify)
exploration/      the original correlation analysis that motivated the model
docs/superpowers/ design spec and implementation plan
tests/            pytest suite (synthetic fixtures, no network)
```
