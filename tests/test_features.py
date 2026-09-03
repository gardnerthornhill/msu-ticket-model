import pytest

from ticketmodel.features import FeatureError, ap_rank_lookup, local_datetime, opponent_rank, resolve_opponent

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
