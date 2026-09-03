import pandas as pd

from ticketmodel.report import write_report


def sample_summary():
    return {
        "generated": "2026-09-03",
        "counts": {"games": 12, "with_attendance": 10, "with_price": 8},
        "metrics": {
            "Season mean (all rows)": {"rmse": 5441.0, "mae": 4500.0, "r2": -0.138, "n": 10},
            "Tier 1 (all rows)": {"rmse": 4190.2, "mae": 3300.0, "r2": 0.325, "n": 10},
            "Tier 2 (priced rows)": {"rmse": 3405.0, "mae": 2800.0, "r2": 0.555, "n": 8},
        },
        "tier1_candidates": [{"features": ["opp_ranked"], "rmse": 4190.2}, {"features": ["conf_game", "opp_elo"], "rmse": 4571.0}],
        "tier2_candidates": [{"features": ["opp_ranked", "rel_log_price"], "price_feature": "rel_log_price", "rmse": 3405.0},
                             {"features": ["opp_ranked", "log_getin"], "price_feature": "log_getin", "rmse": 3591.0}],
        "tier1_model": {"features": ["opp_ranked"], "intercept": 48973.0, "coef": {"opp_ranked": 7480.0},
                        "stderr": {"const": 1200.0, "opp_ranked": 1900.0}, "resid_se": 4100.0, "df_resid": 8, "n": 10},
        "tier2_model": {"features": ["opp_ranked", "rel_log_price"], "intercept": 50000.0,
                        "coef": {"opp_ranked": 5000.0, "rel_log_price": 3500.0},
                        "stderr": {"const": 1000.0, "opp_ranked": 1500.0, "rel_log_price": 900.0}, "resid_se": 3200.0, "df_resid": 5, "n": 8},
        "per_game": pd.DataFrame({
            "season": [2023, 2023], "date": ["2023-09-16", "2023-09-30"], "opponent": ["Rival A", "Mid Major"],
            "getin": [31.0, float("nan")], "attendance": [60000.0, 47000.0], "tier1_loo": [56453.0, 48973.0], "tier2_loo": [58000.0, float("nan")],
        }),
        "warnings": ["2024 Rival D: no ticket row"],
    }


def test_report_contains_every_section(tmp_path):
    p = tmp_path / "reports" / "model_report.md"
    write_report(p, sample_summary())
    text = p.read_text()
    for heading in ["# Mississippi State Attendance Model", "## Data", "## Leave-one-out accuracy",
                    "## Tier 1 feature selection", "## Tier 2 price feature", "## Fitted models",
                    "## Per-game leave-one-out predictions", "## Warnings", "## Caveats"]:
        assert heading in text, heading
    assert "| Tier 2 (priced rows) | 8 | 3405 | 2800 | 0.555 |" in text
    assert "opp_ranked + rel_log_price" in text
    assert "| 2023 | 2023-09-30 | Mid Major |  | 47000 | 48973 |  |" in text
    assert "2024 Rival D: no ticket row" in text
    assert "7480" in text and "±" in text
    assert "within 5% of the best" in text
    assert "optimistic" in text
    assert "collinear" in text
    assert "at least 3 priced games" in text
    assert "nominal 80% coverage" in text
