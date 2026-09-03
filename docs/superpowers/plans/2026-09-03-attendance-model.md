# Mississippi State Attendance Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A command-line Python pipeline that predicts Mississippi State home-game attendance from CFBD game features (Tier 1) and additionally from the ticketdata.com get-in price (Tier 2), with leave-one-out evaluation, a markdown report, and predictions for upcoming games.

**Architecture:** A small package `ticketmodel/` with one module per responsibility: config, CFBD fetch/cache, ticket CSV loading, feature building, OLS modelling, report writing, and a CLI. Data flows `tickets.csv` + cached CFBD JSON → `features.csv` → two OLS models saved as JSON → `model_report.md` and `predictions.csv`. No network in tests; CFBD HTTP is injected as a function.

**Tech Stack:** Python 3.11, pandas, numpy, scipy, statsmodels, python-dotenv, pytest. Standard-library `urllib` for HTTP, `zoneinfo` for time zones.

**Spec:** `docs/superpowers/specs/2026-09-03-attendance-model-design.md`

## Global Constraints

- Python 3.11; run everything with `python3` from the repo root `/Users/jamesthornhill/Desktop/ticket-model`.
- Tests never touch the network. `cfbd.fetch_season` takes an `http` callable; tests pass a fake.
- `TEAM = "Mississippi State"`, `CAPACITY = 60417`, time zone `America/Chicago`.
- CFBD API key comes only from `CFBD_API_KEY` in the environment or the repo-root `.env` file (git-ignored). Never hard-code it in `ticketmodel/`.
- Refresh rule for a season's CFBD cache: refetch when any of the four files is missing, the games list is empty, any game has `completed == false`, any game started within the last 14 days, or `--refresh SEASON` was passed. Otherwise make zero requests.
- Hard errors exit non-zero with one clear message; they are the exception classes `TicketError`, `CfbdError`, `FeatureError`, `ModelError`.
- Predictions (point and interval) are clipped to `[0, CAPACITY]`. Prediction interval is 80%.
- Tier 1 candidate features: `opp_ranked, conf_game, opp_elo, opp_sp, opp_p4, week`; subsets of size 1–3; chosen by LOO-RMSE, ties within 0.001 broken toward fewer features. Tier 2 = Tier 1 subset + exactly one of `log_getin, rel_log_price`.
- Minimum training rows per tier: 8.
- Commit after every task with the trailer lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN`.

## File Structure

| Path | Responsibility |
|---|---|
| `requirements.txt` | dependencies |
| `.env.example` | documents `CFBD_API_KEY` |
| `README.md` | after-each-game workflow |
| `ticketmodel/__init__.py` | empty |
| `ticketmodel/__main__.py` | `python -m ticketmodel` entry |
| `ticketmodel/config.py` | constants, alias table, `Paths` |
| `ticketmodel/tickets.py` | load and validate `tickets.csv` |
| `ticketmodel/cfbd.py` | fetch, cache, refresh rule |
| `ticketmodel/features.py` | opponent resolution, poll lookup, feature join |
| `ticketmodel/model.py` | LOO, selection, fit, save/load, predict |
| `ticketmodel/report.py` | markdown report |
| `ticketmodel/cli.py` | commands fetch / build / train / predict / all |
| `scripts/ticketdata_console.js` | browser console snippet producing CSV rows |
| `tests/conftest.py` | synthetic two-season CFBD fixture + tickets |
| `tests/test_*.py` | one file per module |
| `exploration/` | the pre-spec analysis scripts, moved out of `data/` |

---

### Task 1: Scaffold, config, and ticket loader (plus seed-data migration)

**Files:**
- Create: `requirements.txt`, `.env.example`, `ticketmodel/__init__.py`, `ticketmodel/config.py`, `ticketmodel/tickets.py`, `tests/__init__.py`, `tests/test_tickets.py`
- Modify: `data/tickets.csv` (new schema)
- Move: `data/build.py`, `data/analyze.py`, `data/joined.csv` → `exploration/`
- Delete: `data/cfbd_raw/{media,talent,lines}_*.json`, `data/cfbd_raw/games_{2021,2022,2026}.json`, `data/cfbd_raw/opp/`

**Interfaces:**
- Produces: `config.TEAM, CAPACITY, TZ, P4_CONFERENCES, ALIASES, CANDIDATE_FEATURES, PRICE_FEATURES, MAX_SUBSET_SIZE, MIN_TRAINING_ROWS, REFRESH_WINDOW_DAYS, INTERVAL, ROOT, Paths, DEFAULT_PATHS`
- Produces: `tickets.TicketError`, `tickets.load_tickets(path) -> pd.DataFrame` with columns `opponent, date (YYYY-MM-DD str), getin (float, NaN when blank), observed (str or NaN), season (int)`

- [ ] **Step 1: Scaffold files**

`requirements.txt`:
```
pandas>=2.0
numpy>=1.24
scipy>=1.10
statsmodels>=0.14
python-dotenv>=1.0
pytest>=7.0
```

`.env.example`:
```
CFBD_API_KEY=
```

`ticketmodel/__init__.py` and `tests/__init__.py`: empty files.

`ticketmodel/config.py`:
```python
"""Project-wide constants and paths."""
from dataclasses import dataclass
from pathlib import Path

TEAM = "Mississippi State"
VENUE = "Davis Wade Stadium"
CAPACITY = 60417  # observed sellout figure; official listed capacity is 60,311
TZ = "America/Chicago"
P4_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC"}

# ticketdata opponent name (exact, or prefix followed by a space) -> CFBD team name
ALIASES = {
    "UMass": "Massachusetts",
    "Southeastern Louisiana": "SE Louisiana",
    "USM": "Southern Miss",
    "Southern Mississippi": "Southern Miss",
}

CANDIDATE_FEATURES = ["opp_ranked", "conf_game", "opp_elo", "opp_sp", "opp_p4", "week"]
PRICE_FEATURES = ["log_getin", "rel_log_price"]
MAX_SUBSET_SIZE = 3
MIN_TRAINING_ROWS = 8
REFRESH_WINDOW_DAYS = 14
INTERVAL = 0.80  # prediction interval coverage


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cfbd_raw"

    @property
    def tickets(self) -> Path:
        return self.data_dir / "tickets.csv"

    @property
    def features(self) -> Path:
        return self.data_dir / "features.csv"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def tier1(self) -> Path:
        return self.models_dir / "tier1.json"

    @property
    def tier2(self) -> Path:
        return self.models_dir / "tier2.json"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def report(self) -> Path:
        return self.reports_dir / "model_report.md"

    @property
    def predictions(self) -> Path:
        return self.reports_dir / "predictions.csv"


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS = Paths(ROOT)
```

- [ ] **Step 2: Migrate seed data**

```bash
mkdir -p exploration
git mv data/build.py data/analyze.py data/joined.csv exploration/
git rm -q -r data/cfbd_raw/opp data/cfbd_raw/media_*.json data/cfbd_raw/talent_*.json data/cfbd_raw/lines_*.json data/cfbd_raw/games_2021.json data/cfbd_raw/games_2022.json data/cfbd_raw/games_2026.json
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("data/tickets.csv")
out = df[["opponent", "date", "getin"]].copy()
out["observed"] = ""
out.to_csv("data/tickets.csv", index=False)
print(out.head(3)); print(len(out), "rows")
EOF
head -3 data/tickets.csv
```
Expected: header `opponent,date,getin,observed`, 22 rows, blank `getin` on five 2024 rows (pandas writes NaN as empty).

- [ ] **Step 3: Write the failing tests**

`tests/test_tickets.py`:
```python
import pandas as pd
import pytest

from ticketmodel.config import DEFAULT_PATHS
from ticketmodel.tickets import TicketError, load_tickets


def write(tmp_path, text):
    p = tmp_path / "tickets.csv"
    p.write_text(text)
    return p


def test_loads_valid_file_and_derives_season(tmp_path):
    p = write(tmp_path, "opponent,date,getin,observed\nOle Miss,2025-11-28,133,\nUMass,2024-11-02,,2026-09-03\n")
    df = load_tickets(p)
    assert list(df.columns) == ["opponent", "date", "getin", "observed", "season"]
    assert df.loc[0, "season"] == 2025 and df.loc[0, "getin"] == 133
    assert pd.isna(df.loc[1, "getin"])
    assert pd.isna(df.loc[0, "observed"]) and df.loc[1, "observed"] == "2026-09-03"


def test_duplicate_rows_error(tmp_path):
    p = write(tmp_path, "opponent,date,getin,observed\nOle Miss,2025-11-28,133,\nOle Miss,2025-11-28,120,\n")
    with pytest.raises(TicketError, match="duplicate"):
        load_tickets(p)


def test_missing_column_error(tmp_path):
    p = write(tmp_path, "opponent,date,getin\nOle Miss,2025-11-28,133\n")
    with pytest.raises(TicketError, match="observed"):
        load_tickets(p)


def test_non_numeric_price_error(tmp_path):
    p = write(tmp_path, "opponent,date,getin,observed\nOle Miss,2025-11-28,$133,\n")
    with pytest.raises(TicketError, match="getin"):
        load_tickets(p)


def test_bad_date_error(tmp_path):
    p = write(tmp_path, "opponent,date,getin,observed\nOle Miss,11/28/2025,133,\n")
    with pytest.raises(TicketError, match="date"):
        load_tickets(p)


def test_seed_file_loads():
    df = load_tickets(DEFAULT_PATHS.tickets)
    assert len(df) >= 22
    assert {2023, 2024, 2025} <= set(df["season"])
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_tickets.py -v`
Expected: ImportError / ModuleNotFoundError for `ticketmodel.tickets`.

- [ ] **Step 5: Implement `ticketmodel/tickets.py`**

```python
"""Load and validate the hand-maintained data/tickets.csv."""
import pandas as pd

REQUIRED = ["opponent", "date", "getin", "observed"]


class TicketError(ValueError):
    """Raised when tickets.csv is malformed."""


def load_tickets(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"opponent": str, "date": str, "observed": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise TicketError(f"{path}: missing columns {missing}; expected {REQUIRED}")
    df = df[REQUIRED].copy()
    df["opponent"] = df["opponent"].str.strip()
    try:
        parsed = pd.to_datetime(df["date"], format="%Y-%m-%d")
    except (ValueError, TypeError) as e:
        raise TicketError(f"{path}: bad date, expected YYYY-MM-DD: {e}") from e
    df["date"] = parsed.dt.strftime("%Y-%m-%d")
    try:
        df["getin"] = pd.to_numeric(df["getin"])
    except (ValueError, TypeError) as e:
        raise TicketError(f"{path}: getin must be a number or blank: {e}") from e
    df["season"] = parsed.dt.year.astype(int)
    dups = df.duplicated(subset=["opponent", "date"], keep=False)
    if dups.any():
        rows = df.loc[dups, ["opponent", "date"]].drop_duplicates().to_dict("records")
        raise TicketError(f"{path}: duplicate ticket rows {rows}")
    return df.reset_index(drop=True)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_tickets.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Scaffold ticketmodel package, config, and ticket loader; migrate seed data

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN"
```

---

### Task 2: CFBD fetch with cache and refresh rule

**Files:**
- Create: `ticketmodel/cfbd.py`, `tests/test_cfbd.py`

**Interfaces:**
- Consumes: `config.TEAM, REFRESH_WINDOW_DAYS, ROOT`
- Produces: `cfbd.CfbdError`, `cfbd.KINDS = ["games", "rankings", "sp", "elo"]`, `cfbd.api_key(env_path=ROOT/".env") -> str`, `cfbd.default_http(url, key) -> (status:int, body:str)`, `cfbd.cache_path(kind, season, cache_dir) -> Path`, `cfbd.load_cached(kind, season, cache_dir)`, `cfbd.load_season(season, cache_dir) -> {kind: data}`, `cfbd.needs_refresh(season, cache_dir, now=None) -> bool`, `cfbd.fetch_season(season, cache_dir, force=False, http=None, key=None, now=None) -> bool`

- [ ] **Step 1: Write the failing tests**

`tests/test_cfbd.py`:
```python
import json
from datetime import datetime, timezone

import pytest

from ticketmodel import cfbd
from ticketmodel.cfbd import KINDS, CfbdError

NOW = datetime(2026, 9, 3, tzinfo=timezone.utc)


def game(start, completed=True):
    return {"startDate": start, "completed": completed, "homeTeam": "Mississippi State"}


def write_season(cache_dir, season, games):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        data = games if kind == "games" else []
        (cache_dir / f"{kind}_{season}.json").write_text(json.dumps(data))


class FakeHttp:
    def __init__(self, status=200):
        self.calls = []
        self.status = status

    def __call__(self, url, key):
        self.calls.append(url)
        return self.status, json.dumps([{"startDate": "2025-09-06T23:30:00.000Z", "completed": True}])


def test_frozen_season_makes_no_requests(tmp_path):
    write_season(tmp_path, 2024, [game("2024-11-23T21:15:00.000Z")])
    http = FakeHttp()
    assert cfbd.fetch_season(2024, tmp_path, http=http, key="k", now=NOW) is False
    assert http.calls == []


def test_incomplete_game_triggers_refresh_of_all_four_files(tmp_path):
    write_season(tmp_path, 2026, [game("2026-09-05T23:30:00.000Z", completed=False)])
    http = FakeHttp()
    assert cfbd.fetch_season(2026, tmp_path, http=http, key="k", now=NOW) is True
    assert len(http.calls) == 4
    assert any("/games?year=2026&team=Mississippi%20State" in u for u in http.calls)
    assert any("/ratings/elo?year=2026" in u for u in http.calls)


def test_recent_game_triggers_refresh(tmp_path):
    write_season(tmp_path, 2026, [game("2026-08-29T20:00:00.000Z")])  # 5 days before NOW
    assert cfbd.needs_refresh(2026, tmp_path, now=NOW) is True


def test_missing_file_triggers_refresh(tmp_path):
    write_season(tmp_path, 2023, [game("2023-11-24T00:30:00.000Z")])
    (tmp_path / "elo_2023.json").unlink()
    assert cfbd.needs_refresh(2023, tmp_path, now=NOW) is True


def test_empty_games_list_triggers_refresh(tmp_path):
    write_season(tmp_path, 2027, [])
    assert cfbd.needs_refresh(2027, tmp_path, now=NOW) is True


def test_force_refreshes_frozen_season(tmp_path):
    write_season(tmp_path, 2024, [game("2024-11-23T21:15:00.000Z")])
    http = FakeHttp()
    assert cfbd.fetch_season(2024, tmp_path, force=True, http=http, key="k", now=NOW) is True
    assert json.loads((tmp_path / "games_2024.json").read_text())[0]["startDate"].startswith("2025")


def test_non_200_is_error_and_writes_nothing(tmp_path):
    http = FakeHttp(status=401)
    with pytest.raises(CfbdError, match="HTTP 401"):
        cfbd.fetch_season(2025, tmp_path, http=http, key="k", now=NOW)
    assert not (tmp_path / "games_2025.json").exists()


def test_load_season_missing_file_is_error(tmp_path):
    with pytest.raises(CfbdError, match="missing cache file"):
        cfbd.load_season(2025, tmp_path)


def test_missing_key_is_error(monkeypatch, tmp_path):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(CfbdError, match="CFBD_API_KEY"):
        cfbd.api_key(env_path=tmp_path / ".env")


def test_key_from_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    (tmp_path / ".env").write_text("CFBD_API_KEY=abc123\n")
    assert cfbd.api_key(env_path=tmp_path / ".env") == "abc123"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cfbd.py -v`
Expected: ModuleNotFoundError for `ticketmodel.cfbd`.

- [ ] **Step 3: Implement `ticketmodel/cfbd.py`**

```python
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
        payloads[kind] = json.loads(body)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for kind, data in payloads.items():
        cache_path(kind, season, cache_dir).write_text(json.dumps(data))
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cfbd.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add ticketmodel/cfbd.py tests/test_cfbd.py
git commit -m "Add CFBD fetch with per-season cache and refresh rule

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN"
```

---

### Task 3: Feature helpers — opponent resolution, AP poll lookup, local time

**Files:**
- Create: `ticketmodel/features.py` (helpers only), `tests/test_features.py` (helper tests)

**Interfaces:**
- Consumes: `config.ALIASES, TZ`
- Produces: `features.FeatureError`, `features.resolve_opponent(name, cfbd_opponents) -> str`, `features.ap_rank_lookup(rankings) -> dict[int, dict[str, int]]`, `features.opponent_rank(ap, week, opponent) -> (rank | None, warning | None)`, `features.local_datetime(start_date_utc: str) -> datetime`

- [ ] **Step 1: Write the failing tests**

`tests/test_features.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_features.py -v`
Expected: ModuleNotFoundError for `ticketmodel.features`.

- [ ] **Step 3: Implement the helpers in `ticketmodel/features.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_features.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add ticketmodel/features.py tests/test_features.py
git commit -m "Add opponent resolution, AP poll lookup, and local time helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN"
```

---

### Task 4: Synthetic fixture and `build_features`

**Files:**
- Create: `tests/conftest.py`
- Modify: `ticketmodel/features.py` (add `FEATURE_COLUMNS`, `build_features`), `tests/test_features.py` (append)

**Interfaces:**
- Consumes: `tickets.load_tickets`, helpers from Task 3, `config.TEAM, P4_CONFERENCES`
- Produces: `features.FEATURE_COLUMNS`, `features.build_features(tickets: pd.DataFrame, seasons: dict[int, dict]) -> (pd.DataFrame, list[str])`. `seasons[season]` is the dict returned by `cfbd.load_season`. Output columns, in order: `season, week, date, kickoff_hr, opponent, attendance, completed, conf_game, opp_p4, opp_fcs, opp_ranked, opp_ap_rank, opp_elo, opp_sp, getin, observed, log_getin, rel_log_price`.
- Fixtures (pytest): `seasons` (dict), `tickets_text` (str), `tickets_df`, `features` (tuple df, warnings), `fixture_root` (tmp root with `data/tickets.csv` and `data/cfbd_raw/*.json` written)

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import json

import pytest

from ticketmodel.tickets import load_tickets

TEAM = "Mississippi State"
CONF = {"FCS U": "SWAC", "Mid Major": "Mid-American", "Rival A": "SEC", "Rival B": "SEC",
        "Rival C": "SEC", "Rival D": "SEC", "Rival E": "SEC", "Bowl Foe": "ACC"}
CLASS = {"FCS U": "fcs"}


def _game(season, week, opp, start, conf, attendance, elo, completed=True, home=True, neutral=False):
    cls = CLASS.get(opp, "fbs")
    home_team, away_team = (TEAM, opp) if home else (opp, TEAM)
    return {
        "id": abs(hash((season, week, opp))) % 10**8, "season": season, "week": week, "seasonType": "regular",
        "startDate": start, "completed": completed, "neutralSite": neutral, "conferenceGame": conf,
        "attendance": attendance, "homeTeam": home_team, "awayTeam": away_team,
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
    rows.append({"year": season, "team": "nationalAverages", "conference": None, "rating": 0.0})
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
```

- [ ] **Step 2: Append the failing tests to `tests/test_features.py`**

```python
import numpy as np
import pandas as pd

from ticketmodel.features import FEATURE_COLUMNS, build_features


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
```

Add `from ticketmodel.tickets import load_tickets` to the imports at the top of the test file.

- [ ] **Step 3: Run tests to verify the new ones fail**

Run: `python3 -m pytest tests/test_features.py -v`
Expected: the 10 helper tests pass; the new tests fail with ImportError on `FEATURE_COLUMNS` / `build_features`.

- [ ] **Step 4: Add `build_features` to `ticketmodel/features.py`**

Add imports at the top:
```python
import numpy as np
import pandas as pd

from .config import ALIASES, P4_CONFERENCES, TEAM, TZ
```
(replace the existing `from .config import ALIASES, TZ` line), then append:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_features.py -v`
Expected: 22 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py ticketmodel/features.py tests/test_features.py
git commit -m "Add synthetic CFBD fixture and build_features join

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN"
```

---

### Task 5: Model core — leave-one-out, baseline, metrics, feature selection

**Files:**
- Create: `ticketmodel/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `config.CAPACITY, CANDIDATE_FEATURES, PRICE_FEATURES, MAX_SUBSET_SIZE`
- Produces: `model.TARGET = "attendance"`, `model.ModelError`, `model.loo_predictions(df, features) -> np.ndarray`, `model.season_mean_baseline(df) -> np.ndarray`, `model.metrics(y, preds) -> {"rmse","mae","r2","n"}`, `model.loo_metrics(df, features) -> metrics + {"preds": np.ndarray}`, `model.select_tier1(df) -> list[{"features","rmse"}]` (best first), `model.select_tier2(df, tier1_features) -> list[{"features","price_feature","rmse"}]` (best first)

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:
```python
import numpy as np
import pandas as pd
import pytest

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
        "attendance": [40000 + 5000 * c + 10 * e for c, e in zip(conf, elo)],
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_model.py -v`
Expected: ModuleNotFoundError for `ticketmodel.model`.

- [ ] **Step 3: Implement the core of `ticketmodel/model.py`**

```python
"""OLS attendance models: leave-one-out evaluation, feature selection, fit, persist, predict."""
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import CANDIDATE_FEATURES, CAPACITY, INTERVAL, MAX_SUBSET_SIZE, MIN_TRAINING_ROWS, PRICE_FEATURES

TARGET = "attendance"


class ModelError(ValueError):
    """Too few rows, or a missing saved model."""


def _design(df: pd.DataFrame, features) -> pd.DataFrame:
    X = df[list(features)].astype(float)
    return sm.add_constant(X, has_constant="add")


def _clip(a):
    return np.clip(np.asarray(a, float), 0, CAPACITY)


def loo_predictions(df: pd.DataFrame, features) -> np.ndarray:
    y = df[TARGET].to_numpy(float)
    n = len(df)
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        res = sm.OLS(y[mask], _design(df.iloc[mask], features)).fit()
        preds[i] = float(res.predict(_design(df.iloc[[i]], features))[0])
    return _clip(preds)


def season_mean_baseline(df: pd.DataFrame) -> np.ndarray:
    y = df[TARGET].to_numpy(float)
    seasons = df["season"].to_numpy()
    idx = np.arange(len(y))
    preds = np.empty(len(y))
    for i in idx:
        others = (seasons == seasons[i]) & (idx != i)
        preds[i] = y[others].mean() if others.any() else y[idx != i].mean()
    return preds


def metrics(y, preds) -> dict:
    y = np.asarray(y, float)
    preds = np.asarray(preds, float)
    resid = y - preds
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y)),
    }


def loo_metrics(df: pd.DataFrame, features) -> dict:
    preds = loo_predictions(df, features)
    out = metrics(df[TARGET], preds)
    out["preds"] = preds
    return out


def _rank(results):
    return sorted(results, key=lambda r: (round(r["rmse"], 3), len(r["features"])))


def select_tier1(df: pd.DataFrame) -> list[dict]:
    results = []
    for k in range(1, MAX_SUBSET_SIZE + 1):
        for combo in itertools.combinations(CANDIDATE_FEATURES, k):
            results.append({"features": list(combo), "rmse": loo_metrics(df, combo)["rmse"]})
    return _rank(results)


def select_tier2(df: pd.DataFrame, tier1_features) -> list[dict]:
    results = []
    for pf in PRICE_FEATURES:
        feats = list(tier1_features) + [pf]
        results.append({"features": feats, "price_feature": pf, "rmse": loo_metrics(df, feats)["rmse"]})
    return _rank(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_model.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ticketmodel/model.py tests/test_model.py
git commit -m "Add LOO evaluation, baseline, metrics, and feature selection

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN"
```

---

### Task 6: Model fit, persistence, and prediction with intervals

**Files:**
- Modify: `ticketmodel/model.py` (append), `tests/test_model.py` (append)

**Interfaces:**
- Produces: `model.fit(df, features) -> dict` with keys `features, intercept, coef {f: float}, stderr {name: float}, resid_se, df_resid, n, xtx_inv (nested list), data_hash`; `model.save_model(model, path)`, `model.load_model(path) -> dict` (raises `ModelError` when missing); `model.predict(model, df) -> pd.DataFrame[pred, lo, hi]` indexed like `df`, clipped.

- [ ] **Step 1: Append the failing tests to `tests/test_model.py`**

```python
import statsmodels.api as sm


def test_fit_recovers_exact_coefficients():
    m = mdl.fit(exact_tier1_df(), ["conf_game", "opp_elo"])
    assert m["intercept"] == pytest.approx(40000, abs=1e-6)
    assert m["coef"]["conf_game"] == pytest.approx(5000, abs=1e-6)
    assert m["coef"]["opp_elo"] == pytest.approx(10, abs=1e-9)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_model.py -v`
Expected: the 6 earlier tests pass; new tests fail with AttributeError (`fit`, `predict`, ...).

- [ ] **Step 3: Append to `ticketmodel/model.py`**

```python
def fit(df: pd.DataFrame, features) -> dict:
    if len(df) < MIN_TRAINING_ROWS:
        raise ModelError(f"need at least {MIN_TRAINING_ROWS} training rows, have {len(df)}")
    X = _design(df, features)
    y = df[TARGET].to_numpy(float)
    res = sm.OLS(y, X).fit()
    Xn = X.to_numpy(float)
    return {
        "features": list(features),
        "intercept": float(res.params["const"]),
        "coef": {f: float(res.params[f]) for f in features},
        "stderr": {k: float(v) for k, v in res.bse.items()},
        "resid_se": float(np.sqrt(res.scale)),
        "df_resid": int(res.df_resid),
        "n": int(len(df)),
        "xtx_inv": np.linalg.pinv(Xn.T @ Xn).tolist(),
        "data_hash": hashlib.sha256(df[list(features) + [TARGET]].to_csv(index=False).encode()).hexdigest()[:12],
    }


def save_model(model: dict, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(model, indent=1))


def load_model(path) -> dict:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise ModelError(f"no saved model at {p}; run train")
    return json.loads(p.read_text())


def predict(model: dict, df: pd.DataFrame) -> pd.DataFrame:
    feats = model["features"]
    X = _design(df, feats).to_numpy(float)
    beta = np.array([model["intercept"]] + [model["coef"][f] for f in feats])
    point = X @ beta
    V = np.asarray(model["xtx_inv"], float)
    se = model["resid_se"] * np.sqrt(1.0 + np.einsum("ij,jk,ik->i", X, V, X))
    t = stats.t.ppf(0.5 + INTERVAL / 2, model["df_resid"])
    return pd.DataFrame(
        {"pred": _clip(point), "lo": _clip(point - t * se), "hi": _clip(point + t * se)}, index=df.index
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_model.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add ticketmodel/model.py tests/test_model.py
git commit -m "Add model fit, JSON persistence, and clipped interval predictions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN"
```

---

### Task 7: Markdown report writer

**Files:**
- Create: `ticketmodel/report.py`, `tests/test_report.py`

**Interfaces:**
- Produces: `report.write_report(path, summary: dict) -> None`. `summary` keys:
  - `generated` (str), `counts` `{games, with_attendance, with_price}`
  - `metrics`: ordered dict of label → `{rmse, mae, r2, n}`
  - `tier1_candidates`: list of `{features, rmse}` (top 5), `tier2_candidates`: list of `{features, price_feature, rmse}`
  - `tier1_model`, `tier2_model`: dicts from `model.fit`
  - `per_game`: DataFrame with columns `season, date, opponent, getin, attendance, tier1_loo, tier2_loo`
  - `warnings`: list[str]

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: ModuleNotFoundError for `ticketmodel.report`.

- [ ] **Step 3: Implement `ticketmodel/report.py`**

```python
"""Write reports/model_report.md from the training summary."""
from pathlib import Path

import pandas as pd

from .config import CAPACITY, TEAM, VENUE

CAVEATS = [
    "Small sample: a couple of dozen games. Coefficients are rough; the leave-one-out numbers are the honest accuracy.",
    f"Attendance is the announced figure and is capped at {VENUE} capacity ({CAPACITY:,}); sellouts flatten the top end.",
    "The get-in price is the final price recorded by ticketdata near game day, not a price observed weeks out.",
    "Price levels shift season to season; the season-relative price feature exists for that reason.",
]


def _num(x, nd=0):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return f"{x:.{nd}f}"


def _feats(fs):
    return " + ".join(fs)


def _model_block(title: str, m: dict) -> list[str]:
    lines = [f"### {title}", "", f"Features: {_feats(m['features'])}. Rows: {m['n']}. Residual SE: {_num(m['resid_se'])}.", "",
             "| term | coefficient | std err |", "|---|---|---|",
             f"| intercept | {_num(m['intercept'])} | ± {_num(m['stderr'].get('const'))} |"]
    for f in m["features"]:
        lines.append(f"| {f} | {_num(m['coef'][f], 2)} | ± {_num(m['stderr'].get(f), 2)} |")
    return lines + [""]


def write_report(path, s: dict) -> None:
    c = s["counts"]
    lines = [f"# {TEAM} Attendance Model", "", f"Generated {s['generated']}.", "",
             "## Data", "",
             f"- Home games in features: {c['games']}", f"- With attendance (Tier 1 training rows): {c['with_attendance']}",
             f"- With attendance and price (Tier 2 training rows): {c['with_price']}", "",
             "## Leave-one-out accuracy", "", "| model | rows | RMSE | MAE | R² |", "|---|---|---|---|---|"]
    for label, m in s["metrics"].items():
        lines.append(f"| {label} | {m['n']} | {_num(m['rmse'])} | {_num(m['mae'])} | {_num(m['r2'], 3)} |")
    lines += ["", "## Tier 1 feature selection", "", "Top candidate subsets by LOO-RMSE (ties within 0.001 go to fewer features).", "",
              "| features | LOO RMSE |", "|---|---|"]
    for r in s["tier1_candidates"]:
        lines.append(f"| {_feats(r['features'])} | {_num(r['rmse'])} |")
    lines += ["", "## Tier 2 price feature", "", "| features | LOO RMSE |", "|---|---|"]
    for r in s["tier2_candidates"]:
        lines.append(f"| {_feats(r['features'])} | {_num(r['rmse'])} |")
    lines += ["", "## Fitted models", ""]
    lines += _model_block("Tier 1 (game features only)", s["tier1_model"])
    lines += _model_block("Tier 2 (game features + price)", s["tier2_model"])
    lines += ["## Per-game leave-one-out predictions", "",
              "| season | date | opponent | price | actual | Tier 1 LOO | Tier 2 LOO |", "|---|---|---|---|---|---|---|"]
    for _, r in s["per_game"].iterrows():
        lines.append(f"| {r['season']} | {r['date']} | {r['opponent']} | {_num(r['getin'])} | {_num(r['attendance'])} "
                     f"| {_num(r['tier1_loo'])} | {_num(r['tier2_loo'])} |")
    lines += ["", "## Warnings", ""]
    lines += [f"- {w}" for w in s["warnings"]] or ["- none"]
    lines += ["", "## Caveats", ""] + [f"- {x}" for x in CAVEATS] + [""]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add ticketmodel/report.py tests/test_report.py
git commit -m "Add markdown model report writer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN"
```

---

### Task 8: CLI, entry point, README, console snippet

**Files:**
- Create: `ticketmodel/cli.py`, `ticketmodel/__main__.py`, `README.md`, `scripts/ticketdata_console.js`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `cli.cmd_fetch(paths, refresh=(), http=None, key=None) -> None`, `cli.cmd_build(paths) -> pd.DataFrame`, `cli.cmd_train(paths) -> dict` (the report summary), `cli.cmd_predict(paths) -> pd.DataFrame`, `cli.main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: ModuleNotFoundError for `ticketmodel.cli`.

- [ ] **Step 3: Implement `ticketmodel/cli.py` and `ticketmodel/__main__.py`**

`ticketmodel/cli.py`:
```python
"""Command line: fetch | build | train | predict | all."""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import cfbd
from . import features as feat
from . import model as mdl
from . import report as rpt
from .config import DEFAULT_PATHS, MIN_TRAINING_ROWS, Paths
from .tickets import TicketError, load_tickets

PRED_COLUMNS = ["season", "date", "opponent", "getin", "tier1_pred", "tier1_lo", "tier1_hi", "tier2_pred", "tier2_lo", "tier2_hi"]


def _seasons(paths: Paths) -> list[int]:
    return sorted(set(load_tickets(paths.tickets)["season"]))


def cmd_fetch(paths: Paths, refresh=(), http=None, key=None) -> None:
    force = set(refresh)
    for season in _seasons(paths):
        did = cfbd.fetch_season(season, paths.cache_dir, force=season in force, http=http, key=key)
        print(f"{season}: {'fetched' if did else 'frozen, using cache'}")


def cmd_build(paths: Paths) -> pd.DataFrame:
    tickets = load_tickets(paths.tickets)
    seasons = {s: cfbd.load_season(s, paths.cache_dir) for s in sorted(set(tickets["season"]))}
    df, warnings = feat.build_features(tickets, seasons)
    for w in warnings:
        print(f"warning: {w}")
    paths.features.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.features, index=False)
    print(f"wrote {paths.features} ({len(df)} home games)")
    df.attrs["warnings"] = warnings
    return df


def cmd_train(paths: Paths) -> dict:
    df = cmd_build(paths)
    t1 = df[df["attendance"].notna()].reset_index(drop=True)
    t2 = t1[t1["getin"].notna()].reset_index(drop=True)
    for name, rows in (("Tier 1", t1), ("Tier 2", t2)):
        if len(rows) < MIN_TRAINING_ROWS:
            raise mdl.ModelError(f"{name} needs at least {MIN_TRAINING_ROWS} training rows, have {len(rows)}")
    tier1_cands = mdl.select_tier1(t1)
    tier1_feats = tier1_cands[0]["features"]
    tier2_cands = mdl.select_tier2(t2, tier1_feats)
    tier2_feats = tier2_cands[0]["features"]
    m1, m2 = mdl.fit(t1, tier1_feats), mdl.fit(t2, tier2_feats)
    mdl.save_model(m1, paths.tier1)
    mdl.save_model(m2, paths.tier2)

    tier1_loo, tier2_loo = mdl.loo_metrics(t1, tier1_feats), mdl.loo_metrics(t2, tier2_feats)
    per_game = t1[["season", "date", "opponent", "getin", "attendance"]].copy()
    per_game["tier1_loo"] = np.round(tier1_loo["preds"])
    per_game["tier2_loo"] = np.nan
    per_game.loc[per_game["getin"].notna(), "tier2_loo"] = np.round(tier2_loo["preds"])
    metrics = {
        "Season mean (all rows)": mdl.metrics(t1["attendance"], mdl.season_mean_baseline(t1)),
        "Tier 1 (all rows)": {k: v for k, v in tier1_loo.items() if k != "preds"},
        "Season mean (priced rows)": mdl.metrics(t2["attendance"], mdl.season_mean_baseline(t2)),
        "Price only (priced rows)": {k: v for k, v in mdl.loo_metrics(t2, ["log_getin"]).items() if k != "preds"},
        "Tier 1 (priced rows)": {k: v for k, v in mdl.loo_metrics(t2, tier1_feats).items() if k != "preds"},
        "Tier 2 (priced rows)": {k: v for k, v in tier2_loo.items() if k != "preds"},
    }
    summary = {
        "generated": date.today().isoformat(),
        "counts": {"games": int(len(df)), "with_attendance": int(len(t1)), "with_price": int(len(t2))},
        "metrics": metrics,
        "tier1_candidates": tier1_cands[:5],
        "tier2_candidates": tier2_cands,
        "tier1_model": m1,
        "tier2_model": m2,
        "per_game": per_game,
        "warnings": df.attrs.get("warnings", []),
    }
    rpt.write_report(paths.report, summary)
    print(f"Tier 1 features: {tier1_feats}  LOO-RMSE {tier1_loo['rmse']:.0f}")
    print(f"Tier 2 features: {tier2_feats}  LOO-RMSE {tier2_loo['rmse']:.0f}")
    print(f"wrote {paths.report}")
    return summary


def cmd_predict(paths: Paths) -> pd.DataFrame:
    m1, m2 = mdl.load_model(paths.tier1), mdl.load_model(paths.tier2)
    if not paths.features.exists():
        raise mdl.ModelError(f"no {paths.features}; run build")
    df = pd.read_csv(paths.features)
    up = df[(df["completed"] == 0) & df["attendance"].isna()].reset_index(drop=True)
    out = up[["season", "date", "opponent", "getin"]].copy()
    for c in PRED_COLUMNS[4:]:
        out[c] = np.nan
    if len(up):
        p1 = mdl.predict(m1, up)
        out["tier1_pred"], out["tier1_lo"], out["tier1_hi"] = p1["pred"].round(), p1["lo"].round(), p1["hi"].round()
        priced = up["getin"].notna()
        if priced.any():
            p2 = mdl.predict(m2, up[priced])
            out.loc[priced, "tier2_pred"] = p2["pred"].round()
            out.loc[priced, "tier2_lo"] = p2["lo"].round()
            out.loc[priced, "tier2_hi"] = p2["hi"].round()
    out = out[PRED_COLUMNS]
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.predictions, index=False)
    print(out.to_string(index=False) if len(out) else "no upcoming home games in features.csv")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ticketmodel", description="Mississippi State home attendance model")
    p.add_argument("command", choices=["fetch", "build", "train", "predict", "all"])
    p.add_argument("--refresh", type=int, nargs="*", default=[], metavar="SEASON", help="force re-download of these seasons")
    p.add_argument("--root", type=Path, default=None, help="repo root (default: this checkout)")
    a = p.parse_args(argv)
    paths = Paths(a.root.resolve()) if a.root else DEFAULT_PATHS
    try:
        if a.command in ("fetch", "all"):
            cmd_fetch(paths, a.refresh)
        if a.command == "build":
            cmd_build(paths)
        if a.command in ("train", "all"):
            cmd_train(paths)
        if a.command in ("predict", "all"):
            cmd_predict(paths)
    except (TicketError, cfbd.CfbdError, feat.FeatureError, mdl.ModelError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
```

`ticketmodel/__main__.py`:
```python
import sys

from .cli import main

sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: 5 passed. Then run the full suite: `python3 -m pytest -q` — expected all green (about 54 tests).

- [ ] **Step 5: Write `scripts/ticketdata_console.js`**

```javascript
// Paste into the browser dev-tools console on
// https://www.ticketdata.com/performer/mississippi-state-bulldogs-football?tab=past  (or ?tab=upcoming)
// Prints CSV rows in data/tickets.csv order: opponent,date,getin,observed
// Reads only the already-loaded DOM; makes no network requests.
(() => {
  const today = new Date().toISOString().slice(0, 10);
  const seen = new Set();
  const lines = ["opponent,date,getin,observed"];
  for (const r of document.querySelectorAll("table tbody tr")) {
    const c = [...r.querySelectorAll("td")].map((td) => td.innerText.trim());
    if (c.length < 8) continue;
    const ev = c[1]
      .replace(/^Egg Bowl - /, "")
      .replace(/ \(Rescheduled.*\)$/, "")
      .replace(/ at Mississippi State Bulldogs( Football)?$/, "");
    const m = c[2].match(/(\d{2})\/(\d{2})\/(\d{4})/);
    if (!m) continue;
    const date = `${m[3]}-${m[1]}-${m[2]}`;
    const key = ev + "|" + date;
    if (seen.has(key)) continue;
    seen.add(key);
    const price = (c[7].match(/\$(\d+(?:\.\d+)?)/) || [])[1] || "";
    const opp = ev.includes(",") ? `"${ev}"` : ev;
    lines.push([opp, date, price, today].join(","));
  }
  console.log(lines.join("\n"));
  return `${lines.length - 1} rows`;
})();
```

- [ ] **Step 6: Write `README.md`**

```markdown
# Mississippi State attendance model

Predicts announced attendance at Davis Wade Stadium from CFBD game features (Tier 1) and the
ticketdata.com get-in price (Tier 2). Design: `docs/superpowers/specs/2026-09-03-attendance-model-design.md`.

## Setup

    pip install -r requirements.txt
    cp .env.example .env      # then put your CFBD key in it

## After each game

1. Open the ticketdata past tab in your browser, paste `scripts/ticketdata_console.js` into the
   dev-tools console, and copy the new row(s) into `data/tickets.csv`. Opponent names may include
   the mascot; the pipeline strips it. Leave `getin` blank when the site shows no price.
2. `python3 -m ticketmodel all`
3. Read `reports/model_report.md` (accuracy, chosen features) and `reports/predictions.csv`
   (upcoming home games; Tier 2 columns are blank until a price is listed).

Finished seasons are cached in `data/cfbd_raw/` and never re-downloaded. Only a season with an
unfinished game, or a game in the last 14 days, is refreshed. Force one with
`python3 -m ticketmodel fetch --refresh 2025`.

## Commands

    python3 -m ticketmodel fetch     # CFBD -> data/cfbd_raw/ (per the refresh rule)
    python3 -m ticketmodel build     # -> data/features.csv
    python3 -m ticketmodel train     # -> models/*.json, reports/model_report.md
    python3 -m ticketmodel predict   # -> reports/predictions.csv
    python3 -m pytest                # tests, no network
```

- [ ] **Step 7: Commit**

```bash
git add ticketmodel/cli.py ticketmodel/__main__.py tests/test_cli.py scripts/ticketdata_console.js README.md
git commit -m "Add CLI, console snippet, and README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN"
```

---

### Task 9: First real run and commit of outputs

**Files:**
- Create: `.env` (git-ignored, holds the key), `data/features.csv`, `models/tier1.json`, `models/tier2.json`, `reports/model_report.md`, `reports/predictions.csv`
- Modify: `data/cfbd_raw/*.json` (adds `elo_*.json`, refreshes 2023–2025 once)

- [ ] **Step 1: Create `.env`**

```bash
printf 'CFBD_API_KEY=<your CFBD key>\n' > .env
git check-ignore .env   # must print .env
```

- [ ] **Step 2: Run the pipeline**

```bash
python3 -m ticketmodel all
```
Expected: `2023: fetched`, `2024: fetched`, `2025: fetched` (each season is missing `elo_*.json`, so all three download once), a few warnings (Southern Miss 2023 has no attendance; FCS opponents imputed), Tier 1 and Tier 2 feature lines, and `no upcoming home games in features.csv` (no 2026 rows in tickets yet).

- [ ] **Step 3: Verify freezing**

```bash
python3 -m ticketmodel fetch
```
Expected: all three seasons print `frozen, using cache`.

- [ ] **Step 4: Sanity-check the report**

```bash
sed -n '1,60p' reports/model_report.md
```
Expected: Tier 2 LOO-RMSE on priced rows below the Tier 1 and season-mean rows on the same 16 rows; chosen Tier 1 features drawn from the candidate pool; 21 rows in the per-game table.

- [ ] **Step 5: Run the full test suite once more and commit outputs**

```bash
python3 -m pytest -q
git add -A
git commit -m "First training run: cached CFBD seasons, features, models, report

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H9gSeUsTE4oQJzfS4HrpdN"
```

---

## Self-review notes

- Spec coverage: inputs (T1, T8 snippet), CFBD caching and refresh rule (T2), features incl. poll rule, imputation, date check, rel_log_price (T3–T4), models and selection (T5–T6), report contents (T7), predictions CSV (T8), layout/commands/README (T8), error classes (T1–T8), first run (T9). One refinement beyond the spec: features carry a `completed` flag and `predict` scores only `completed == 0` games, so a finished game with missing attendance (2023 Southern Miss) is not treated as upcoming.
- Names used consistently: `Paths`, `DEFAULT_PATHS`, `load_tickets`, `fetch_season`, `load_season`, `build_features`, `FEATURE_COLUMNS`, `loo_metrics`, `select_tier1`, `select_tier2`, `fit`, `save_model`, `load_model`, `predict`, `write_report`, `cmd_*`, `main`.
