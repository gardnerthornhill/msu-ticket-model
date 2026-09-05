import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ticketmodel import model as mdl
from ticketmodel.history import archive_tickets, archive_forecasts


def test_price_history_keeps_unknown_dates_and_versions_without_duplicate_reads(tmp_path):
    path = tmp_path / "ticket_history.csv"
    df = pd.DataFrame({"opponent": ["UL Monroe"], "date": ["2026-09-05"], "getin": [30.0], "observed": [None]})
    assert archive_tickets(df, path) == 1
    before = path.read_bytes()
    assert archive_tickets(df, path) == 0 and path.read_bytes() == before
    df["getin"], df["observed"] = 32, "2026-09-05"
    assert archive_tickets(df, path) == 1
    history = pd.read_csv(path)
    assert history.getin.tolist() == [30, 32]
    assert pd.isna(history.iloc[0].observed)
    assert history.iloc[1].observed == "2026-09-05"
    assert archive_tickets(df.iloc[:0], path) == 0


def forecast_inputs():
    train = pd.DataFrame({"rel_log_price": [-1, -.8, -.6, -.4, 0, .4, .6, 1],
                          "attendance": [47000, 48000, 48500, 50500, 53000, 55000, 57500, 60000]})
    model = mdl.fit(train, ["rel_log_price"])
    row = pd.DataFrame({"season": [2026], "date": ["2026-09-05"], "opponent": ["UL Monroe"],
                        "kickoff_hr": [18.5], "getin": [32.], "log_getin": [np.log(32)],
                        "rel_log_price": [np.log(32 / 46)], "attendance": [np.nan], "completed": [0]})
    prediction = mdl.predict(model, row)
    forecast = row[["season", "date", "opponent", "getin"]].copy()
    for tier in ["tier1", "tier2"]:
        for c in prediction:
            forecast[f"{tier}_{c}"] = prediction[c]
    return row, forecast, {"tier1": model, "tier2": model}


def test_pregame_snapshot_is_reproducible_and_only_changes_when_inputs_or_day_change(tmp_path):
    path = tmp_path / "forecasts.jsonl"
    rows, forecast, models = forecast_inputs()
    now = datetime(2026, 9, 5, 12, tzinfo=timezone.utc)
    assert archive_forecasts(rows, forecast, models, path, now=now) == 1
    assert archive_forecasts(rows, forecast, models, path, now=now + timedelta(hours=1)) == 0
    saved = json.loads(path.read_text())
    reproduced = mdl.predict(saved["model"], pd.DataFrame([saved["inputs"]])).iloc[0]
    assert reproduced["pred"] == saved["forecast"]["pred"]
    assert saved["days_ahead"] == 0 and saved["recorded_at"] == now.isoformat()
    # Same forecast day, new observed price: preserve both versions.
    rows["getin"] = 33
    assert archive_forecasts(rows, forecast, models, path, now=now + timedelta(hours=2)) == 1
    assert len(path.read_text().splitlines()) == 2


def test_never_archive_after_kickoff_even_if_feed_says_incomplete(tmp_path):
    rows, forecast, models = forecast_inputs()
    path = tmp_path / "forecasts.jsonl"
    kickoff = datetime(2026, 9, 5, 23, 30, tzinfo=timezone.utc)
    assert archive_forecasts(rows, forecast, models, path, now=kickoff) == 0
    assert not path.exists()
    rows["kickoff_hr"] = np.nan
    assert archive_forecasts(rows, forecast, models, path, now=datetime(2026, 9, 5, 12, tzinfo=timezone.utc)) == 0
