"""Join tickets.csv with cached CFBD data into one feature row per home game."""
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import ALIASES, TZ


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
