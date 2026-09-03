import numpy as np
import pandas as pd
import pytest

from ticketmodel.features import FEATURE_COLUMNS, FeatureError, ap_rank_lookup, build_features, game_local_date, local_datetime, opponent_rank, resolve_opponent
from ticketmodel.tickets import load_tickets

RANKINGS = [
    {"season": 2025, "seasonType": "regular", "week": 1, "polls": [
        {"poll": "Coaches Poll", "ranks": [{"school": "Texas", "rank": 1}]},
        {"poll": "AP Top 25", "ranks": [{"school": "Texas", "rank": 2}, {"school": "Georgia", "rank": 5}]},
    ]},
    {"season": 2025, "seasonType": "regular", "week": 3, "polls": [
        {"poll": "AP Top 25", "ranks": [{"school": "Georgia", "rank": 4}]},
    ]},
    {"season": 2025, "seasonType": "postseason", "week": 1, "polls": [
        {"poll": "AP Top 25", "ranks": [{"school": "Georgia", "rank": 1}]},
    ]},
]


def test_resolve_exact_name():
    assert resolve_opponent("Ole Miss", {"Ole Miss", "Texas"}) == "Ole Miss"


def test_resolve_strips_mascot_with_longest_match():
    opps = {"Texas", "Texas A&M"}
    assert resolve_opponent("Texas A&M Aggies", opps) == "Texas A&M"
    assert resolve_opponent("Texas Longhorns", opps) == "Texas"


def test_resolve_alias_exact_and_prefix():
    assert resolve_opponent("UMass", {"Massachusetts"}) == "Massachusetts"
    assert resolve_opponent("UMass Minutemen", {"Massachusetts"}) == "Massachusetts"
    assert resolve_opponent("USM Golden Eagles", {"Southern Miss"}) == "Southern Miss"
    assert resolve_opponent("Louisiana Monroe Warhawks", {"UL Monroe"}) == "UL Monroe"


def test_resolve_alias_target_not_a_home_opponent_raises():
    with pytest.raises(FeatureError, match="not a CFBD home opponent"):
        resolve_opponent("UMass", {"Texas"})


def test_resolve_unknown_raises_listing_opponents():
    with pytest.raises(FeatureError, match="does not match.*Texas"):
        resolve_opponent("Nowhere State", {"Texas"})


def test_ap_lookup_keeps_only_regular_season_ap_polls():
    ap = ap_rank_lookup(RANKINGS)
    assert set(ap) == {1, 3}
    assert ap[1] == {"Texas": 2, "Georgia": 5}


def test_rank_exact_week():
    ap = ap_rank_lookup(RANKINGS)
    assert opponent_rank(ap, 3, "Georgia") == (4, None)


def test_rank_falls_back_to_latest_earlier_poll():
    ap = ap_rank_lookup(RANKINGS)
    assert opponent_rank(ap, 5, "Georgia") == (4, None)
    assert opponent_rank(ap, 5, "Texas") == (None, None)  # dropped out by week 3


def test_rank_with_no_poll_warns():
    rank, warn = opponent_rank({}, 2, "Georgia")
    assert rank is None and "no AP poll" in warn


def test_local_datetime_shifts_calendar_day():
    assert local_datetime("2023-10-01T01:00:00.000Z").strftime("%Y-%m-%d %H:%M") == "2023-09-30 20:00"
    assert local_datetime("2025-09-27T20:15:00.000Z").strftime("%Y-%m-%d %H:%M") == "2025-09-27 15:15"


def row(df, season, opp):
    return df[(df["season"] == season) & (df["opponent"] == opp)].iloc[0]


def test_only_home_non_neutral_games_in_order(features):
    df, _ = features
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 12
    assert not {"Rival E", "Bowl Foe"} & set(df["opponent"])
    assert list(df["date"]) == sorted(df["date"])


def test_utc_to_central_date_and_kickoff(features):
    df, _ = features
    r = row(df, 2023, "Rival A")
    assert r["date"] == "2023-09-16" and r["kickoff_hr"] == 20.0
    r = row(df, 2023, "Rival D")
    assert r["date"] == "2023-11-23" and r["kickoff_hr"] == 18.5


def test_fcs_opponent_is_flagged_and_imputed(features):
    df, warnings = features
    r = row(df, 2023, "FCS U")
    assert r["opp_fcs"] == 1 and r["opp_p4"] == 0 and r["conf_game"] == 0
    assert r["opp_elo"] == 1290 - 100
    assert r["opp_sp"] == -5.0 - 10
    assert r["opp_ranked"] == 0 and r["opp_ap_rank"] == 30
    assert any("FCS U" in w and "Elo" in w for w in warnings)


def test_ranked_flag_exact_week_and_fallback(features):
    df, _ = features
    assert row(df, 2023, "Rival A")[["opp_ranked", "opp_ap_rank"]].tolist() == [1, 10]   # week-3 poll
    assert row(df, 2023, "Rival B")["opp_ranked"] == 0                                    # week 7 -> week-3 poll
    assert row(df, 2023, "Rival D")[["opp_ranked", "opp_ap_rank"]].tolist() == [1, 5]    # week-12 poll
    assert row(df, 2024, "Rival A")[["opp_ranked", "opp_ap_rank"]].tolist() == [1, 8]    # week 4 -> week-1 poll
    assert row(df, 2024, "Rival C")["opp_ranked"] == 0                                    # upcoming, week-1 poll


def test_upcoming_game_uses_elo_ratings_and_has_null_attendance(features):
    df, _ = features
    r = row(df, 2024, "Rival C")
    assert r["opp_elo"] == 1500 and pd.isna(r["attendance"]) and r["completed"] == 0
    assert row(df, 2024, "Rival D")["opp_elo"] == 1820


def test_completed_flag(features):
    df, _ = features
    assert df["completed"].sum() == 10


def test_rel_log_price_uses_season_median_including_upcoming(features):
    df, _ = features
    assert row(df, 2024, "Rival A")["rel_log_price"] == pytest.approx(0.0)      # median of log(12, 40, 45)
    assert row(df, 2024, "FCS U")["rel_log_price"] == pytest.approx(np.log(12) - np.log(40))
    assert pd.isna(row(df, 2024, "Rival B")["rel_log_price"])


def test_price_columns_and_observed(features):
    df, _ = features
    assert row(df, 2023, "Rival D")["getin"] == 63
    assert row(df, 2023, "Rival D")["log_getin"] == pytest.approx(np.log(63))
    assert row(df, 2024, "Rival C")["observed"] == "2024-10-20"
    assert row(df, 2023, "Rival D")["observed"] is None or pd.isna(row(df, 2023, "Rival D")["observed"])


def test_missing_ticket_row_gives_null_price_and_warning(features):
    df, warnings = features
    assert pd.isna(row(df, 2024, "Rival D")["getin"])
    assert any("Rival D" in w and "no ticket row" in w for w in warnings)


def test_ticket_date_mismatch_raises(tmp_path, seasons, tickets_text):
    p = tmp_path / "t.csv"
    p.write_text(tickets_text.replace("Rival A,2023-09-16", "Rival A,2023-09-17"))
    with pytest.raises(FeatureError, match="date"):
        build_features(load_tickets(p), seasons)


def test_unknown_opponent_raises(tmp_path, seasons, tickets_text):
    p = tmp_path / "t.csv"
    p.write_text(tickets_text + "Nowhere State,2023-10-21,5,\n")
    with pytest.raises(FeatureError, match="Nowhere State"):
        build_features(load_tickets(p), seasons)


def test_season_without_cfbd_data_raises(tmp_path, seasons, tickets_text):
    p = tmp_path / "t.csv"
    p.write_text(tickets_text + "Rival A,2022-09-10,20,\n")
    with pytest.raises(FeatureError, match="2022"):
        build_features(load_tickets(p), seasons)


def test_tbd_kickoff_uses_calendar_date_and_blank_hour():
    # CFBD stores TBD kickoffs at midnight Eastern (04:00Z / 05:00Z), which is the previous day in Central.
    date, hr = game_local_date({"startDate": "2026-09-26T04:00:00.000Z", "startTimeTBD": True})
    assert date == "2026-09-26" and np.isnan(hr)
    date, hr = game_local_date({"startDate": "2026-09-26T04:00:00.000Z", "startTimeTBD": False})
    assert date == "2026-09-25" and hr == 23.0


def test_build_features_tbd_game_keeps_calendar_date(tickets_df, seasons):
    for g in seasons[2024]["games"]:
        if g["awayTeam"] == "Rival D":
            g["startDate"], g["startTimeTBD"] = "2024-11-09T04:00:00.000Z", True
    df, _ = build_features(tickets_df, seasons)
    r = row(df, 2024, "Rival D")
    assert r["date"] == "2024-11-09" and pd.isna(r["kickoff_hr"])


def test_missing_current_season_elo_falls_back_to_previous_season(tickets_df, seasons):
    # Early in a season CFBD's Elo list only holds teams that have played; use last season's final Elo.
    seasons[2024]["elo"] = [t for t in seasons[2024]["elo"] if t["team"] != "Rival C"]
    df, warnings = build_features(tickets_df, seasons)
    assert row(df, 2024, "Rival C")["opp_elo"] == 1540  # 2023 value
    assert any("Rival C" in w and "2023" in w for w in warnings)
