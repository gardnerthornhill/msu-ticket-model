import json

import pytest

from ticketmodel.tickets import load_tickets

TEAM = "Mississippi State"
CONF = {"FCS U": "SWAC", "Mid Major": "Mid-American", "Rival A": "SEC", "Rival B": "SEC",
        "Rival C": "SEC", "Rival D": "SEC", "Rival E": "SEC", "Bowl Foe": "ACC"}
CLASS = {"FCS U": "fcs"}
TEAM_ID = 344
OPP_ID = {"FCS U": 2016, "Mid Major": 2459, "Rival A": 333, "Rival B": 2, "Rival C": 99,
          "Rival D": 145, "Rival E": 61, "Bowl Foe": 154}


def _game(season, week, opp, start, conf, attendance, elo, completed=True, home=True, neutral=False):
    cls = CLASS.get(opp, "fbs")
    home_team, away_team = (TEAM, opp) if home else (opp, TEAM)
    return {
        "id": abs(hash((season, week, opp))) % 10**8, "season": season, "week": week, "seasonType": "regular",
        "startDate": start, "completed": completed, "neutralSite": neutral, "conferenceGame": conf,
        "attendance": attendance, "homeTeam": home_team, "awayTeam": away_team,
        "homeId": TEAM_ID if home else OPP_ID[opp], "awayId": OPP_ID[opp] if home else TEAM_ID,
        "homeConference": "SEC" if home else CONF[opp], "awayConference": CONF[opp] if home else "SEC",
        "homeClassification": "fbs" if home else cls, "awayClassification": cls if home else "fbs",
        "homePregameElo": 1450 if home else elo, "awayPregameElo": elo if home else 1450,
        "homePoints": 30 if completed else None, "awayPoints": 20 if completed else None,
    }


def _poll(season, week, ranks, season_type="regular"):
    return {"season": season, "seasonType": season_type, "week": week,
            "polls": [{"poll": "AP Top 25", "ranks": [{"school": s, "rank": r} for s, r in ranks]}]}


def _sp(season, ratings):
    rows = [{"year": season, "team": t, "conference": CONF[t], "rating": r} for t, r in ratings]
    rows.append({"year": season, "team": "nationalAverages", "conference": None, "rating": -50.0})
    return rows


def _elo(season, ratings):
    return [{"year": season, "team": t, "conference": CONF[t], "elo": e} for t, e in ratings]


def make_seasons():
    g23 = [
        _game(2023, 1, "FCS U", "2023-09-02T20:00:00.000Z", False, 49000, None),
        _game(2023, 3, "Rival A", "2023-09-17T01:00:00.000Z", True, 60000, 1700),
        _game(2023, 5, "Mid Major", "2023-09-30T16:00:00.000Z", False, 47000, 1300),
        _game(2023, 7, "Rival B", "2023-10-14T23:30:00.000Z", True, 55000, 1600),
        _game(2023, 10, "Rival C", "2023-11-04T16:00:00.000Z", True, 52000, 1550),
        _game(2023, 12, "Rival D", "2023-11-24T00:30:00.000Z", True, 60400, 1800),
        _game(2023, 2, "Rival E", "2023-09-09T23:00:00.000Z", True, 70000, 1650, home=False),
        _game(2023, 14, "Bowl Foe", "2023-12-28T00:00:00.000Z", False, 30000, 1500, neutral=True),
    ]
    g24 = [
        _game(2024, 1, "FCS U", "2024-08-31T22:00:00.000Z", False, 48000, None),
        _game(2024, 2, "Mid Major", "2024-09-07T23:30:00.000Z", False, 47500, 1350),
        _game(2024, 4, "Rival A", "2024-09-21T16:00:00.000Z", True, 58000, 1650),
        _game(2024, 6, "Rival B", "2024-10-05T20:15:00.000Z", True, 53000, 1580),
        _game(2024, 9, "Rival C", "2024-10-26T20:15:00.000Z", True, None, None, completed=False),
        _game(2024, 11, "Rival D", "2024-11-09T20:15:00.000Z", True, None, 1820, completed=False),
    ]
    r23 = [_poll(2023, 1, [("Rival A", 12), ("Rival D", 3)]),
           _poll(2023, 3, [("Rival A", 10), ("Rival D", 4)]),
           _poll(2023, 12, [("Rival D", 5)]),
           _poll(2023, 1, [("Rival D", 1)], season_type="postseason")]
    r24 = [_poll(2024, 1, [("Rival A", 8)])]
    sp23 = _sp(2023, [("Rival A", 20.0), ("Rival B", 12.0), ("Rival C", 10.0), ("Rival D", 22.0),
                      ("Mid Major", -5.0), ("Rival E", 15.0), ("Bowl Foe", 1.0)])
    sp24 = _sp(2024, [("Rival A", 21.0), ("Rival B", 11.0), ("Rival C", 9.0), ("Rival D", 23.0), ("Mid Major", -6.0)])
    elo23 = _elo(2023, [("Rival A", 1710), ("Rival B", 1590), ("Rival C", 1540), ("Rival D", 1810),
                        ("Mid Major", 1290), ("Rival E", 1640), ("Bowl Foe", 1500)])
    elo24 = _elo(2024, [("Rival A", 1660), ("Rival B", 1570), ("Rival C", 1500), ("Rival D", 1820), ("Mid Major", 1340)])
    return {2023: {"games": g23, "rankings": r23, "sp": sp23, "elo": elo23},
            2024: {"games": g24, "rankings": r24, "sp": sp24, "elo": elo24}}


TICKETS_TEXT = "opponent,date,getin,observed\n" + "\n".join([
    "FCS U,2023-09-02,8,", "Rival A,2023-09-16,31,", "Mid Major,2023-09-30,6,",
    "Rival B,2023-10-14,20,", "Rival C,2023-11-04,10,", "Rival D,2023-11-23,63,",
    "FCS U,2024-08-31,12,", "Mid Major,2024-09-07,,", "Rival A,2024-09-21,40,",
    "Rival B,2024-10-05,,", "Rival C,2024-10-26,45,2024-10-20",
]) + "\n"


@pytest.fixture
def seasons():
    return make_seasons()


@pytest.fixture
def tickets_text():
    return TICKETS_TEXT


@pytest.fixture
def tickets_df(tmp_path):
    p = tmp_path / "tickets.csv"
    p.write_text(TICKETS_TEXT)
    return load_tickets(p)


@pytest.fixture
def features(tickets_df, seasons):
    from ticketmodel.features import build_features

    return build_features(tickets_df, seasons)


@pytest.fixture
def fixture_root(tmp_path):
    """A repo-shaped tmp dir with data/tickets.csv and data/cfbd_raw/*.json."""
    data = tmp_path / "data"
    cache = data / "cfbd_raw"
    cache.mkdir(parents=True)
    (data / "tickets.csv").write_text(TICKETS_TEXT)
    for season, payload in make_seasons().items():
        for kind, rows in payload.items():
            (cache / f"{kind}_{season}.json").write_text(json.dumps(rows))
    return tmp_path
