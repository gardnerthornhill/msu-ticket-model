# Mississippi State attendance model

Predicts announced attendance at Davis Wade Stadium from CFBD game features (Tier 1) and the
ticketdata.com get-in price (Tier 2). Design: `docs/superpowers/specs/2026-09-03-attendance-model-design.md`.

## Setup

    pip install -r requirements.txt
    cp .env.example .env      # then put your CFBD key in it

## After each game

1. Open the ticketdata past tab in your browser, paste `scripts/ticketdata_console.js` into the
   dev-tools console, and copy the new row(s) into `data/tickets.csv`. Opponent names may include
   the mascot; the pipeline strips it. Leave `getin` blank when the site shows no price.
2. `python3 -m ticketmodel all`
3. Read `reports/model_report.md` (accuracy, chosen features) and `reports/predictions.csv`
   (upcoming home games; Tier 2 columns are blank until a price is listed).

Finished seasons are cached in `data/cfbd_raw/` and never re-downloaded. Only a season with an
unfinished game, or a game in the last 14 days, is refreshed. Force one with
`python3 -m ticketmodel fetch --refresh 2025`.

## Commands

    python3 -m ticketmodel fetch     # CFBD -> data/cfbd_raw/ (per the refresh rule)
    python3 -m ticketmodel build     # -> data/features.csv
    python3 -m ticketmodel train     # -> models/*.json, reports/model_report.md
    python3 -m ticketmodel predict   # -> reports/predictions.csv
    python3 -m pytest                # tests, no network
