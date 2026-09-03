"""Command line: fetch | build | train | predict | all."""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import cfbd
from . import features as feat
from . import model as mdl
from . import report as rpt
from .config import DEFAULT_PATHS, MIN_PRICED_PER_SEASON, MIN_TRAINING_ROWS, Paths
from .tickets import TicketError, load_tickets, upsert_rows

PRED_COLUMNS = ["season", "date", "opponent", "getin", "tier1_pred", "tier1_lo", "tier1_hi", "tier2_pred", "tier2_lo", "tier2_hi"]


def _seasons(paths: Paths) -> list[int]:
    return sorted(set(load_tickets(paths.tickets)["season"]))


def cmd_fetch(paths: Paths, refresh=(), http=None, key=None) -> None:
    force = set(refresh)
    for season in _seasons(paths):
        did = cfbd.fetch_season(season, paths.cache_dir, force=season in force, http=http, key=key)
        print(f"{season}: {'fetched' if did else 'frozen, using cache'}")


def cmd_build(paths: Paths) -> pd.DataFrame:
    tickets = load_tickets(paths.tickets)
    seasons = {s: cfbd.load_season(s, paths.cache_dir) for s in sorted(set(tickets["season"]))}
    df, warnings = feat.build_features(tickets, seasons)
    for w in warnings:
        print(f"warning: {w}")
    paths.features.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.features, index=False)
    print(f"wrote {paths.features} ({len(df)} home games)")
    df.attrs["warnings"] = warnings
    return df


def cmd_train(paths: Paths) -> dict:
    df = cmd_build(paths)
    t1 = df[df["attendance"].notna()].reset_index(drop=True)
    t2 = t1[t1["getin"].notna()].reset_index(drop=True)
    for name, rows in (("Tier 1", t1), ("Tier 2", t2)):
        if len(rows) < MIN_TRAINING_ROWS:
            raise mdl.ModelError(f"{name} needs at least {MIN_TRAINING_ROWS} training rows, have {len(rows)}")
    tier1_cands = mdl.select_tier1(t1)
    tier1_feats = tier1_cands[0]["features"]
    tier2_cands = mdl.select_tier2(t2, tier1_feats)
    tier2_feats = tier2_cands[0]["features"]
    m1, m2 = mdl.fit(t1, tier1_feats), mdl.fit(t2, tier2_feats)
    mdl.save_model(m1, paths.tier1)
    mdl.save_model(m2, paths.tier2)

    tier1_loo, tier2_loo = mdl.loo_metrics(t1, tier1_feats), mdl.loo_metrics(t2, tier2_feats)
    per_game = t1[["season", "date", "opponent", "getin", "attendance"]].copy()
    per_game["tier1_loo"] = np.round(tier1_loo["preds"])
    per_game["tier2_loo"] = np.nan
    per_game.loc[per_game["getin"].notna(), "tier2_loo"] = np.round(tier2_loo["preds"])
    metrics = {
        "Season mean (all rows)": mdl.metrics(t1["attendance"], mdl.season_mean_baseline(t1)),
        "Tier 1 (all rows)": {k: v for k, v in tier1_loo.items() if k != "preds"},
        "Season mean (priced rows)": mdl.metrics(t2["attendance"], mdl.season_mean_baseline(t2)),
        "Price only (priced rows)": {k: v for k, v in mdl.loo_metrics(t2, ["log_getin"]).items() if k != "preds"},
        "Tier 1 (priced rows)": {k: v for k, v in mdl.loo_metrics(t2, tier1_feats).items() if k != "preds"},
        "Tier 2 (priced rows)": {k: v for k, v in tier2_loo.items() if k != "preds"},
    }
    summary = {
        "generated": date.today().isoformat(),
        "counts": {"games": int(len(df)), "with_attendance": int(len(t1)), "with_price": int(len(t2))},
        "metrics": metrics,
        "tier1_candidates": tier1_cands[:5],
        "tier2_candidates": tier2_cands,
        "tier1_model": m1,
        "tier2_model": m2,
        "per_game": per_game,
        "warnings": df.attrs.get("warnings", []),
    }
    rpt.write_report(paths.report, summary)
    print(f"Tier 1 features: {tier1_feats}  LOO-RMSE {tier1_loo['rmse']:.0f}")
    print(f"Tier 2 features: {tier2_feats}  LOO-RMSE {tier2_loo['rmse']:.0f}")
    print(f"wrote {paths.report}")
    return summary


def cmd_predict(paths: Paths) -> pd.DataFrame:
    m1, m2 = mdl.load_model(paths.tier1), mdl.load_model(paths.tier2)
    if not paths.features.exists():
        raise mdl.ModelError(f"no {paths.features}; run build")
    df = pd.read_csv(paths.features)
    priced_per_season = df[df["getin"].notna()].groupby("season").size()
    up = df[(df["completed"] == 0) & df["attendance"].isna()].reset_index(drop=True)
    out = up[["season", "date", "opponent", "getin"]].copy()
    for c in PRED_COLUMNS[4:]:
        out[c] = np.nan
    if len(up):
        p1 = mdl.predict(m1, up)
        out["tier1_pred"], out["tier1_lo"], out["tier1_hi"] = p1["pred"].round(), p1["lo"].round(), p1["hi"].round()
        priced = up["getin"].notna()
        for season, n in priced_per_season.items():
            if n < MIN_PRICED_PER_SEASON and (priced & (up["season"] == season)).any():
                print(f"warning: season {season} has only {n} priced game(s); Tier 2 needs {MIN_PRICED_PER_SEASON}, showing Tier 1 only")
        enough_priced = up["season"].map(priced_per_season).fillna(0) >= MIN_PRICED_PER_SEASON
        tier2 = priced & enough_priced
        if tier2.any():
            p2 = mdl.predict(m2, up[tier2])
            out.loc[tier2, "tier2_pred"] = p2["pred"].round()
            out.loc[tier2, "tier2_lo"] = p2["lo"].round()
            out.loc[tier2, "tier2_hi"] = p2["hi"].round()
    out = out[PRED_COLUMNS]
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.predictions, index=False)
    print(out.to_string(index=False) if len(out) else "no upcoming home games in features.csv")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ticketmodel", description="Mississippi State home attendance model")
    p.add_argument("command", choices=["fetch", "build", "train", "predict", "all", "add-tickets"])
    p.add_argument("--rows", default=None, help="add-tickets: CSV rows opponent,date,getin[,observed], one per line (default: stdin)")
    p.add_argument("--refresh", type=int, nargs="*", default=[], metavar="SEASON", help="force re-download of these seasons")
    p.add_argument("--root", type=Path, default=None, help="repo root (default: this checkout)")
    a = p.parse_args(argv)
    paths = Paths(a.root.resolve()) if a.root else DEFAULT_PATHS
    try:
        if a.command == "add-tickets":
            rows = a.rows if a.rows is not None else sys.stdin.read()
            added, updated = upsert_rows(paths.tickets, rows, date.today().isoformat())
            print(f"tickets: added {added}, updated {updated} -> {paths.tickets}")
        if a.command in ("fetch", "all"):
            cmd_fetch(paths, a.refresh)
        if a.command == "build":
            cmd_build(paths)
        if a.command in ("train", "all"):
            cmd_train(paths)
        if a.command in ("predict", "all"):
            cmd_predict(paths)
    except (TicketError, cfbd.CfbdError, feat.FeatureError, mdl.ModelError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
