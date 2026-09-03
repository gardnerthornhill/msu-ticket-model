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
