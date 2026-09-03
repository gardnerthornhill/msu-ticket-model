import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from ticketmodel import model as mdl
from ticketmodel.config import CAPACITY


def test_loo_matches_hand_computation():
    # Hand-derived: holding out each row and fitting y ~ x on the other three gives [2, 0, 3, 1].
    df = pd.DataFrame({"x": [0, 0, 1, 1], "attendance": [0, 2, 1, 3], "season": [1, 1, 1, 1]})
    preds = mdl.loo_predictions(df, ["x"])
    assert preds.tolist() == pytest.approx([2, 0, 3, 1])
    m = mdl.loo_metrics(df, ["x"])
    assert m["rmse"] == pytest.approx(2.0) and m["mae"] == pytest.approx(2.0)
    assert m["r2"] == pytest.approx(1 - 16 / 5) and m["n"] == 4


def test_loo_predictions_are_clipped_to_capacity():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "attendance": [10, 20, 30, 40, 90000], "season": [1] * 5})
    preds = mdl.loo_predictions(df, ["x"])
    assert preds.max() <= CAPACITY and preds.min() >= 0


def test_season_mean_baseline_falls_back_to_global_mean():
    df = pd.DataFrame({"season": [1, 1, 2], "attendance": [10, 20, 30]})
    assert mdl.season_mean_baseline(df).tolist() == pytest.approx([20, 10, 15])


def test_metrics_r2_nan_when_target_constant():
    m = mdl.metrics([5, 5, 5], [4, 5, 6])
    assert np.isnan(m["r2"]) and m["rmse"] == pytest.approx(np.sqrt(2 / 3))


def exact_tier1_df():
    conf = [0, 1] * 6
    elo = [1200, 1500, 1100, 1800, 1300, 1900, 1000, 1700, 1400, 1600, 1250, 1950]
    return pd.DataFrame({
        "season": [2023] * 6 + [2024] * 6, "week": range(1, 13),
        "conf_game": conf, "opp_elo": elo, "opp_ranked": [0, 0, 0, 1] * 3, "opp_p4": [1, 0, 1, 1] * 3,
        "opp_sp": [3, 9, -4, 20, 1, 22, -8, 15, 6, 12, 2, 25],
        "attendance": [40000 + 5000 * c + 5 * e for c, e in zip(conf, elo)],
    })


def test_select_tier1_finds_exact_subset_and_prefers_fewer_features():
    ranked = mdl.select_tier1(exact_tier1_df())
    assert set(ranked[0]["features"]) == {"conf_game", "opp_elo"}
    assert ranked[0]["rmse"] < 1e-3
    assert len(ranked) == 6 + 15 + 20
    assert ranked == sorted(ranked, key=lambda r: (round(r["rmse"], 3), len(r["features"])))


def test_select_tier2_picks_price_feature_with_lowest_loo():
    df = exact_tier1_df()
    df["getin"] = [8, 30, 6, 60, 10, 70, 20, 90, 25, 80, 15, 100]
    df["log_getin"] = np.log(df["getin"])
    df["rel_log_price"] = df["log_getin"] - df.groupby("season")["log_getin"].transform("median")
    df["attendance"] = 40000 + 5000 * df["conf_game"] + 3000 * df["log_getin"]
    ranked = mdl.select_tier2(df, ["conf_game"])
    assert ranked[0]["price_feature"] == "log_getin" and ranked[0]["rmse"] < 1e-3
    assert ranked[0]["features"] == ["conf_game", "log_getin"]
    assert {r["price_feature"] for r in ranked} == {"log_getin", "rel_log_price"}


def test_fit_recovers_exact_coefficients():
    m = mdl.fit(exact_tier1_df(), ["conf_game", "opp_elo"])
    assert m["intercept"] == pytest.approx(40000, abs=1e-6)
    assert m["coef"]["conf_game"] == pytest.approx(5000, abs=1e-6)
    assert m["coef"]["opp_elo"] == pytest.approx(5, abs=1e-9)
    assert m["resid_se"] == pytest.approx(0, abs=1e-6)
    assert m["n"] == 12 and m["df_resid"] == 9 and len(m["data_hash"]) == 12


def test_fit_rejects_too_few_rows():
    with pytest.raises(mdl.ModelError, match="at least 8"):
        mdl.fit(exact_tier1_df().head(7), ["conf_game"])


def test_predict_interval_matches_statsmodels_and_clips():
    rng = np.random.default_rng(0)
    df = exact_tier1_df()
    df["attendance"] = df["attendance"] + rng.normal(0, 1500, len(df))
    feats = ["conf_game", "opp_elo"]
    m = mdl.fit(df, feats)
    new = pd.DataFrame({"conf_game": [1, 0], "opp_elo": [1650, 1050]})
    out = mdl.predict(m, new)
    X = sm.add_constant(df[feats].astype(float))
    res = sm.OLS(df["attendance"].to_numpy(float), X).fit()
    ref = res.get_prediction(sm.add_constant(new.astype(float), has_constant="add")).summary_frame(alpha=0.2)
    assert out["pred"].to_numpy() == pytest.approx(ref["mean"].to_numpy())
    assert out["lo"].to_numpy() == pytest.approx(ref["obs_ci_lower"].to_numpy())
    assert out["hi"].to_numpy() == pytest.approx(ref["obs_ci_upper"].to_numpy())
    huge = pd.DataFrame({"conf_game": [1], "opp_elo": [1e6]})
    assert mdl.predict(m, huge)[["pred", "lo", "hi"]].to_numpy().max() == CAPACITY


def test_save_load_round_trip(tmp_path):
    df = exact_tier1_df()
    m = mdl.fit(df, ["conf_game", "opp_elo", "week"])
    p = tmp_path / "m.json"
    mdl.save_model(m, p)
    m2 = mdl.load_model(p)
    pd.testing.assert_frame_equal(mdl.predict(m, df), mdl.predict(m2, df))
    assert m2["features"] == ["conf_game", "opp_elo", "week"]


def test_load_missing_model_is_error(tmp_path):
    with pytest.raises(mdl.ModelError, match="run train"):
        mdl.load_model(tmp_path / "nope.json")
