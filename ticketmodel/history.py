"""Preserve price observations and pre-kickoff forecasts without inventing historical timestamps."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import TZ


def _json_value(value):
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value


def archive_tickets(tickets: pd.DataFrame, path: Path, now: datetime | None = None) -> int:
    """Keep distinct observed price versions, including unknown-date legacy rows.

    `recorded_at` is ingestion time; `observed` is the supplied price observation
    date. Never fill unknown historical observation dates with today's date.
    """
    now = now or datetime.now(timezone.utc)
    if tickets.empty:
        return 0
    fields = ["opponent", "date", "getin", "observed"]
    incoming = tickets[fields].copy()
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=fields + ["recorded_at"])
    def keys(frame):
        if frame.empty:
            return pd.Series(dtype=str)
        return frame[fields].fillna("").astype(str).agg("|".join, axis=1)
    # Normalize numeric prices so a CSV round trip (32 vs 32.0) does not duplicate history.
    for frame in (incoming, existing):
        frame["getin"] = pd.to_numeric(frame["getin"], errors="raise").map(lambda x: "" if pd.isna(x) else f"{x:g}")
    seen = set(keys(existing))
    fresh = incoming.loc[~keys(incoming).isin(seen)].drop_duplicates(fields).copy()
    if not len(fresh):
        return 0
    fresh["recorded_at"] = now.astimezone(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(fresh)


def archive_forecasts(df: pd.DataFrame, forecasts: pd.DataFrame, models: dict, path: Path,
                      now: datetime | None = None) -> int:
    """Append self-contained pregame snapshots, once per local day/input/model combination.

    Preserve the exact inputs, price reference, fitted model and outputs. Games
    already started are excluded even if CFBD has not marked them completed.
    A TBD kickoff uses midnight locally as a conservative same-day cutoff.
    """
    now = now or datetime.now(timezone.utc)
    local_now = now.astimezone(ZoneInfo(TZ))
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["snapshot_id"])
    entries = []
    for forecast in forecasts.to_dict("records"):
        rows = df[(df["season"] == forecast["season"]) & (df["date"] == forecast["date"])
                  & (df["opponent"] == forecast["opponent"])]
        row = rows.iloc[0]
        start = datetime.fromisoformat(row["date"]).replace(tzinfo=ZoneInfo(TZ))
        if pd.notna(row["kickoff_hr"]):
            start += timedelta(hours=float(row["kickoff_hr"]))
        if row["completed"] or pd.notna(row["attendance"]) or local_now >= start:
            continue
        tier = "tier2" if pd.notna(forecast["tier2_pred"]) else "tier1"
        inputs = _json_value(row.drop(labels=["attendance", "completed"]).to_dict())
        price_reference = float(np.exp(df.loc[df["season"] == row["season"], "log_getin"].median()))
        payload = _json_value({
            "forecast_date": local_now.date().isoformat(), "season": int(row["season"]), "date": row["date"],
            "opponent": row["opponent"], "tier": tier, "inputs": inputs, "season_price_reference": price_reference,
            "model": models[tier], "forecast": {key: forecast[f"{tier}_{key}"] for key in ("pred", "lo", "hi", "p_sellout")},
        })
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if fingerprint not in seen:
            entries.append({"snapshot_id": fingerprint, "recorded_at": now.astimezone(timezone.utc).isoformat(),
                            "days_ahead": (start.date() - local_now.date()).days, **payload})
            seen.add(fingerprint)
    if entries:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, allow_nan=False) + "\n")
    return len(entries)
