"""Command line: fetch | build | train | predict | site | all."""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import cfbd
from . import features as feat
from . import model as mdl
from . import report as rpt
from . import site as web
from .attendance import fill_missing_attendance
from .history import archive_tickets, archive_forecasts
from .config import DEFAULT_PATHS, MIN_TRAINING_ROWS, MIN_PRICED_PER_SEASON, TIER2_FEATURES, Paths
from .tickets import TicketError, load_tickets, upsert_rows

PRED_COLUMNS = ["season", "date", "opponent", "getin", "tier1_pred", "tier1_lo", "tier1_hi", "tier2_pred", "tier2_lo", "tier2_hi",
                "tier1_p_sellout", "tier2_p_sellout"]


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
    df, corrections = fill_missing_attendance(df, paths.attendance_overrides)
    warnings.extend(corrections)
    coverage = df[df["getin"].notna()].groupby("season").size()
    for season, count in coverage.items():
        if count < MIN_PRICED_PER_SEASON:
            warnings.append(f"season {season}: only {count} priced games; sparse historical rows remain in training and diagnostics; new forecasts use Tier 1 until {MIN_PRICED_PER_SEASON} prices are available")
    archive_tickets(tickets, paths.ticket_history)
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
    tier2_feats = list(TIER2_FEATURES)
    m1, m2 = mdl.fit(t1, tier1_feats), mdl.fit(t2, tier2_feats)
    mdl.save_model(m1, paths.tier1)
    mdl.save_model(m2, paths.tier2)

    tier1_loo, tier2_loo = mdl.loo_metrics(t1, tier1_feats), mdl.loo_metrics(t2, tier2_feats)
    per_game = t1[["season", "date", "opponent", "getin", "attendance"]].copy()
    iv1, iv2 = mdl.loo_intervals(t1, tier1_feats), mdl.loo_intervals(t2, tier2_feats)
    per_game["tier1_loo"], per_game["tier1_lo"], per_game["tier1_hi"] = np.round(iv1["pred"]), np.round(iv1["lo"]), np.round(iv1["hi"])
    per_game["tier1_p_sellout"] = iv1["p_sellout"].to_numpy()
    for c in ("tier2_loo", "tier2_lo", "tier2_hi", "tier2_p_sellout"):
        per_game[c] = np.nan
    priced = per_game["getin"].notna()
    per_game.loc[priced, "tier2_loo"] = np.round(iv2["pred"]).to_numpy()
    per_game.loc[priced, "tier2_lo"] = np.round(iv2["lo"]).to_numpy()
    per_game.loc[priced, "tier2_hi"] = np.round(iv2["hi"]).to_numpy()
    per_game.loc[priced, "tier2_p_sellout"] = iv2["p_sellout"].to_numpy()
    metrics = {
        "Season mean (all rows)": mdl.metrics(t1["attendance"], mdl.season_mean_baseline(t1)),
        "Tier 1 (all rows)": {k: v for k, v in tier1_loo.items() if k != "preds"},
        "Season mean (priced rows)": mdl.metrics(t2["attendance"], mdl.season_mean_baseline(t2)),
        "Price only (priced rows)": {k: v for k, v in mdl.loo_metrics(t2, ["log_getin"]).items() if k != "preds"},
        "Relative price only (priced rows)": {k: v for k, v in mdl.loo_metrics(t2, ["rel_log_price"]).items() if k != "preds"},
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
        "validation": mdl.season_validation(t2, tier2_feats),
        "per_season": [
            {"season": int(season), "priced_games": int((df["season"].eq(season) & df["getin"].notna()).sum()),
             **mdl.metrics(g["attendance"], g["tier2_loo"]),
             "bias": float((g["tier2_loo"] - g["attendance"]).mean()),
             "inside": int(((g["attendance"] >= g["tier2_lo"]) & (g["attendance"] <= g["tier2_hi"])).sum())}
            for season, g in per_game[per_game["tier2_loo"].notna()].groupby("season")
        ],
        "warnings": df.attrs.get("warnings", []),
    }
    rpt.write_report(paths.report, summary)
    _write_summary(paths.train_summary, summary)
    print(f"Tier 1 features: {tier1_feats}  LOO-RMSE {tier1_loo['rmse']:.0f}")
    print(f"Tier 2 features: {tier2_feats}  LOO-RMSE {tier2_loo['rmse']:.0f}")
    print(f"wrote {paths.report}")
    return summary


def _write_summary(path: Path, summary: dict) -> None:
    """Persist the training summary as JSON (NaN -> null) so `site` can read it without retraining."""
    per_game = summary["per_game"]
    records = per_game.astype(object).where(per_game.notna(), None).to_dict("records")
    payload = {**summary, "per_game": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=float))


def cmd_predict(paths: Paths) -> pd.DataFrame:
    m1, m2 = mdl.load_model(paths.tier1), mdl.load_model(paths.tier2)
    if not paths.features.exists():
        raise mdl.ModelError(f"no {paths.features}; run build")
    df = pd.read_csv(paths.features)
    out, warnings = mdl.predict_upcoming(df, m1, m2)
    for w in warnings:
        print(f"warning: {w}")
    archived = archive_forecasts(df, out, {"tier1": m1, "tier2": m2}, paths.forecast_history)
    if archived:
        print(f"archived {archived} pregame forecast snapshot(s)")
    out = out[PRED_COLUMNS]
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.predictions, index=False)
    print(out.to_string(index=False) if len(out) else "no upcoming home games in features.csv")
    return out


def cmd_site(paths: Paths, http=None) -> None:
    data = web.build_site(paths, http=http)
    print(f"wrote {paths.site_dir} ({len(data['games'])} games, {len(data['seasons'])} seasons)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ticketmodel", description="Mississippi State home attendance model")
    p.add_argument("command", choices=["fetch", "build", "train", "predict", "site", "all", "add-tickets"])
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
        if a.command in ("site", "all"):
            cmd_site(paths)
    except (TicketError, cfbd.CfbdError, feat.FeatureError, mdl.ModelError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
