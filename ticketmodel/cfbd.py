"""Fetch CollegeFootballData responses into data/cfbd_raw/ and freeze finished seasons."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import REFRESH_WINDOW_DAYS, ROOT, TEAM

BASE = "https://api.collegefootballdata.com"
ENDPOINTS = {
    "games": "/games?year={season}&team={team}",
    "rankings": "/rankings?year={season}",
    "sp": "/ratings/sp?year={season}",
    "elo": "/ratings/elo?year={season}",
}
KINDS = list(ENDPOINTS)


class CfbdError(RuntimeError):
    """Missing API key, missing cache file, or non-200 response."""


def api_key(env_path: Path = ROOT / ".env") -> str:
    key = os.environ.get("CFBD_API_KEY")
    if not key and Path(env_path).exists():
        from dotenv import dotenv_values

        key = dotenv_values(env_path).get("CFBD_API_KEY")
    if not key:
        raise CfbdError("CFBD_API_KEY is not set; put it in .env (see .env.example) or the environment")
    return key


def default_http(url: str, key: str) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError) as e:
        raise CfbdError(f"CFBD request failed: {e}")


def cache_path(kind: str, season: int, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{kind}_{season}.json"


def load_cached(kind: str, season: int, cache_dir: Path):
    p = cache_path(kind, season, cache_dir)
    if not p.exists():
        raise CfbdError(f"missing cache file {p}; run `python3 -m ticketmodel fetch`")
    return json.loads(p.read_text())


def load_season(season: int, cache_dir: Path) -> dict:
    return {kind: load_cached(kind, season, cache_dir) for kind in KINDS}


def needs_refresh(season: int, cache_dir: Path, now: datetime | None = None) -> bool:
    if any(not cache_path(kind, season, cache_dir).exists() for kind in KINDS):
        return True
    games = json.loads(cache_path("games", season, cache_dir).read_text())
    if not games:
        return True
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=REFRESH_WINDOW_DAYS)
    for g in games:
        if not g.get("completed", False):
            return True
        if datetime.fromisoformat(g["startDate"].replace("Z", "+00:00")) >= cutoff:
            return True
    return False


def fetch_season(season: int, cache_dir: Path, force: bool = False, http=None, key: str | None = None, now=None) -> bool:
    """Download all four files for a season when the refresh rule says so. Returns True if downloaded."""
    if not force and not needs_refresh(season, cache_dir, now):
        return False
    http = http or default_http
    key = key or api_key()
    payloads = {}
    for kind, template in ENDPOINTS.items():
        url = BASE + template.format(season=season, team=urllib.parse.quote(TEAM))
        status, body = http(url, key)
        if status != 200:
            raise CfbdError(f"CFBD {kind} for {season} returned HTTP {status}: {body[:200]}")
        try:
            payloads[kind] = json.loads(body)
        except json.JSONDecodeError:
            raise CfbdError(f"CFBD {kind} for {season} returned non-JSON: {body[:200]}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for kind, data in payloads.items():
        cache_path(kind, season, cache_dir).write_text(json.dumps(data))
    return True
