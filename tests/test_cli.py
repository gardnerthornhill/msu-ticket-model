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
