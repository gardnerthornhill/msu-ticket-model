"""Join tickets.csv with cached CFBD data into one feature row per home game."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import ALIASES, P4_CONFERENCES, TEAM, TZ


class FeatureError(ValueError):
    """Unresolvable opponent, date mismatch, or missing season data."""


def resolve_opponent(name: str, cfbd_opponents) -> str:
    """Map a ticketdata opponent name (with or without mascot) to the CFBD name."""
    opponents = set(cfbd_opponents)
    for key, target in ALIASES.items():
        if name == key or name.startswith(key + " "):
            if target in opponents:
                return target
            raise FeatureError(
                f"ticket opponent {name!r} maps to {target!r}, which is not a CFBD home opponent this season {sorted(opponents)}"
            )
    matches = [o for o in opponents if name == o or name.startswith(o + " ")]
    if not matches:
        raise FeatureError(
            f"ticket opponent {name!r} does not match any CFBD home opponent {sorted(opponents)}; "
            "add an alias in config.ALIASES or fix the row"
        )
    return max(matches, key=len)


def ap_rank_lookup(rankings) -> dict[int, dict[str, int]]:
    """{week: {school: rank}} for regular-season AP Top 25 polls."""
    out: dict[int, dict[str, int]] = {}
    for entry in rankings:
        if entry.get("seasonType") != "regular":
            continue
        for poll in entry.get("polls", []):
            if poll.get("poll") == "AP Top 25":
                out[int(entry["week"])] = {r["school"]: int(r["rank"]) for r in poll["ranks"]}
    return out


def opponent_rank(ap: dict, week: int, opponent: str):
    """Rank in the game-week poll, else the latest earlier poll, else None with a warning."""
    if week in ap:
        return ap[week].get(opponent), None
    earlier = [w for w in ap if w < week]
    if earlier:
        return ap[max(earlier)].get(opponent), None
    return None, f"no AP poll cached for week {week}; opp_ranked set to 0"


def local_datetime(start_date_utc: str) -> datetime:
    return datetime.fromisoformat(start_date_utc.replace("Z", "+00:00")).astimezone(ZoneInfo(TZ))


FEATURE_COLUMNS = [
    "season", "week", "date", "kickoff_hr", "opponent", "attendance", "completed",
    "conf_game", "opp_p4", "opp_fcs", "opp_ranked", "opp_ap_rank", "opp_elo", "opp_sp",
    "getin", "observed", "log_getin", "rel_log_price",
]


def build_features(tickets: pd.DataFrame, seasons: dict) -> tuple[pd.DataFrame, list[str]]:
    """One row per home, non-neutral game for every season in `seasons`."""
    warnings: list[str] = []
    missing = sorted(set(tickets["season"]) - set(seasons))
    if missing:
        raise FeatureError(f"no CFBD data cached for seasons {missing}; run fetch")
    rows = []
    for season in sorted(seasons):
        data = seasons[season]
        games = [g for g in data["games"] if g.get("homeTeam") == TEAM and not g.get("neutralSite")]
        home_opps = {g["awayTeam"] for g in games}
        ap = ap_rank_lookup(data["rankings"])
        sp = {t["team"]: float(t["rating"]) for t in data["sp"] if t.get("conference")}
        elo = {t["team"]: float(t["elo"]) for t in data["elo"]}
        sp_floor = (min(sp.values()) - 10) if sp else float("nan")
        elo_floor = (min(elo.values()) - 100) if elo else float("nan")
        season_tix = tickets[tickets["season"] == season]
        tix = {resolve_opponent(t.opponent, home_opps): t for t in season_tix.itertuples(index=False)}
        if len(tix) < len(season_tix):
            raise FeatureError(f"{season}: two ticket rows resolve to the same CFBD opponent")
        for g in sorted(games, key=lambda g: g["startDate"]):
            local = local_datetime(g["startDate"])
            opp = g["awayTeam"]
            r = {
                "season": season, "week": int(g["week"]), "date": local.strftime("%Y-%m-%d"),
                "kickoff_hr": local.hour + local.minute / 60, "opponent": opp,
                "attendance": g.get("attendance"), "completed": int(bool(g.get("completed"))),
                "conf_game": int(bool(g.get("conferenceGame"))),
                "opp_p4": int(g.get("awayConference") in P4_CONFERENCES),
                "opp_fcs": int(g.get("awayClassification") != "fbs"),
            }
            rank, warn = opponent_rank(ap, r["week"], opp)
            if warn:
                warnings.append(f"{season} {opp}: {warn}")
            r["opp_ranked"] = int(rank is not None)
            r["opp_ap_rank"] = rank if rank is not None else 30
            e = g.get("awayPregameElo")
            if e is None:
                e = elo.get(opp)
            if e is None:
                e = elo_floor
                warnings.append(f"{season} {opp}: no Elo; imputed {elo_floor}")
            r["opp_elo"] = float(e)
            s = sp.get(opp)
            if s is None:
                s = sp_floor
                warnings.append(f"{season} {opp}: no SP+; imputed {sp_floor}")
            r["opp_sp"] = float(s)
            t = tix.get(opp)
            if t is None:
                r["getin"], r["observed"] = np.nan, None
                warnings.append(f"{season} {opp}: no ticket row")
            else:
                if t.date != r["date"]:
                    raise FeatureError(
                        f"{season} {t.opponent}: ticket date {t.date} does not match CFBD local date {r['date']}"
                    )
                r["getin"] = t.getin
                r["observed"] = t.observed if isinstance(t.observed, str) else None
            rows.append(r)
    df = pd.DataFrame(rows)
    df["attendance"] = pd.to_numeric(df["attendance"])
    df["log_getin"] = np.log(df["getin"].astype(float))
    df["rel_log_price"] = df["log_getin"] - df.groupby("season")["log_getin"].transform("median")
    return df[FEATURE_COLUMNS], warnings
