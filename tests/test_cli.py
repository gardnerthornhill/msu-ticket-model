import json

import pandas as pd
import pytest

from ticketmodel import cli
from ticketmodel.config import Paths


class FakeHttp:
    def __init__(self):
        self.calls = []

    def __call__(self, url, key):
        self.calls.append(url)
        return 200, json.dumps([])


def test_fetch_only_refreshes_open_season(fixture_root):
    paths = Paths(fixture_root)
    http = FakeHttp()
    cli.cmd_fetch(paths, http=http, key="k")
    assert len(http.calls) == 4                      # 2024 has incomplete games; 2023 is frozen
    assert all("2024" in u for u in http.calls)


def test_train_writes_models_and_report(fixture_root):
    paths = Paths(fixture_root)
    summary = cli.cmd_train(paths)
    assert paths.features.exists() and paths.tier1.exists() and paths.tier2.exists() and paths.report.exists()
    assert summary["counts"] == {"games": 12, "with_attendance": 10, "with_price": 8}
    assert set(summary["metrics"]) == {"Season mean (all rows)", "Tier 1 (all rows)", "Season mean (priced rows)",
                                       "Price only (priced rows)", "Tier 1 (priced rows)", "Tier 2 (priced rows)"}
    per_game = summary["per_game"]
    assert len(per_game) == 10
    assert per_game["tier2_loo"].notna().sum() == 8
    assert "Tier 2" in paths.report.read_text()


def test_predict_scores_upcoming_games_and_blanks_tier2_without_price(fixture_root):
    paths = Paths(fixture_root)
    cli.cmd_train(paths)
    out = cli.cmd_predict(paths)
    assert list(out.columns) == ["season", "date", "opponent", "getin", "tier1_pred", "tier1_lo", "tier1_hi",
                                 "tier2_pred", "tier2_lo", "tier2_hi"]
    assert list(out["opponent"]) == ["Rival C", "Rival D"]
    c = out[out["opponent"] == "Rival C"].iloc[0]
    d = out[out["opponent"] == "Rival D"].iloc[0]
    assert 0 < c["tier1_pred"] <= 60417 and c["tier1_lo"] <= c["tier1_pred"] <= c["tier1_hi"]
    assert c["tier2_pred"] > 0 and pd.isna(d["tier2_pred"]) and pd.isna(d["tier2_lo"])
    assert paths.predictions.exists()
    assert pd.read_csv(paths.predictions).shape == (2, 10)


def test_predict_blanks_tier2_when_season_has_too_few_prices(fixture_root, capsys):
    paths = Paths(fixture_root)
    cli.cmd_train(paths)
    df = pd.read_csv(paths.features)
    blank = df["season"].eq(2024) & df["opponent"].isin(["FCS U", "Rival A"])
    df.loc[blank, ["getin", "log_getin", "rel_log_price"]] = float("nan")
    df.to_csv(paths.features, index=False)
    out = cli.cmd_predict(paths)
    c = out[out["opponent"] == "Rival C"].iloc[0]
    assert pd.isna(c["tier2_pred"])
    assert c["tier1_pred"] > 0
    assert "Tier 2 needs 3" in capsys.readouterr().out


def test_predict_before_train_is_a_clean_error(fixture_root, capsys):
    cli.cmd_build(Paths(fixture_root))
    assert cli.main(["predict", "--root", str(fixture_root)]) == 1
    assert "run train" in capsys.readouterr().err


def test_main_all_without_key_fails_cleanly(fixture_root, monkeypatch, capsys):
    def no_key(*args, **kwargs):
        raise cli.cfbd.CfbdError("CFBD_API_KEY is not set")

    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    monkeypatch.setattr(cli.cfbd, "api_key", no_key)
    assert cli.main(["all", "--root", str(fixture_root)]) == 1
    assert "CFBD_API_KEY" in capsys.readouterr().err


def test_add_tickets_command_upserts_rows(fixture_root, capsys):
    paths = Paths(fixture_root)
    rc = cli.main(["add-tickets", "--rows", "Rival D,2024-11-09,55\nRival C,2024-10-26,50", "--root", str(fixture_root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "added 1" in out and "updated 1" in out
    df = pd.read_csv(paths.tickets)
    assert df[(df["opponent"] == "Rival D") & (df["date"] == "2024-11-09")].iloc[0]["getin"] == 55
    assert df[(df["opponent"] == "Rival C") & (df["date"] == "2024-10-26")].iloc[0]["getin"] == 50
    assert len(df) == 12


def test_add_tickets_rejects_bad_rows(fixture_root, capsys):
    assert cli.main(["add-tickets", "--rows", "Rival D,11/09/2024,55", "--root", str(fixture_root)]) == 1
    assert "error:" in capsys.readouterr().err


def test_train_writes_summary_json_with_loo_intervals(fixture_root):
    paths = Paths(fixture_root)
    cli.cmd_train(paths)
    s = json.loads(paths.train_summary.read_text())
    assert {"generated", "counts", "metrics", "tier1_candidates", "tier2_candidates",
            "tier1_model", "tier2_model", "per_game", "warnings"} <= set(s)
    assert len(s["per_game"]) == 10
    g = s["per_game"][0]
    assert {"tier1_loo", "tier1_lo", "tier1_hi", "tier2_loo", "tier2_lo", "tier2_hi"} <= set(g)
    priced = [r for r in s["per_game"] if r["getin"] is not None]
    assert len(priced) == 8
    assert all(r["tier2_lo"] <= r["tier2_loo"] <= r["tier2_hi"] for r in priced)
    assert all(r["tier2_loo"] is None for r in s["per_game"] if r["getin"] is None)


def test_predict_upcoming_reports_sellout_odds(fixture_root):
    paths = Paths(fixture_root)
    cli.cmd_train(paths)
    from ticketmodel import model as mdl
    df = pd.read_csv(paths.features)
    out, warnings = mdl.predict_upcoming(df, mdl.load_model(paths.tier1), mdl.load_model(paths.tier2))
    assert list(out["opponent"]) == ["Rival C", "Rival D"]
    assert out["tier1_p_sellout"].between(0, 1).all()
    c = out[out["opponent"] == "Rival C"].iloc[0]
    assert 0 <= c["tier2_p_sellout"] <= 1 and warnings == []
