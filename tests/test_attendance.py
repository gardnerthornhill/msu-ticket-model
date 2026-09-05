import json

import pandas as pd
import pytest

from ticketmodel.attendance import fill_missing_attendance
from ticketmodel.config import Paths
from ticketmodel.features import FeatureError
from ticketmodel import cli


def test_sourced_missing_attendance_survives_rebuild_without_editing_cache(fixture_root):
    paths = Paths(fixture_root)
    cache_path = paths.cache_dir / "games_2023.json"
    games = json.loads(cache_path.read_text())
    games[0]["attendance"] = None
    cache_path.write_text(json.dumps(games))
    original_cache = cache_path.read_bytes()
    paths.attendance_overrides.write_text(
        "season,date,opponent,attendance,source\n2023,2023-09-02,FCS U,49000,https://example.org/official-box-score\n")
    first = cli.cmd_build(paths)
    second = cli.cmd_build(paths)
    assert first.iloc[0].attendance == second.iloc[0].attendance == 49000
    assert cache_path.read_bytes() == original_cache
    assert "official-box-score" in " ".join(first.attrs["warnings"])


def test_attendance_correction_does_not_overwrite_existing_source(fixture_root):
    paths = Paths(fixture_root)
    df = cli.cmd_build(paths)
    paths.attendance_overrides.write_text(
        "season,date,opponent,attendance,source\n2023,2023-09-02,FCS U,48000,https://example.org/official-box-score\n")
    out, warnings = fill_missing_attendance(df, paths.attendance_overrides)
    assert out.iloc[0].attendance == 49000
    assert "keeping CFBD" in warnings[0]


def test_attendance_correction_cannot_label_an_upcoming_game(fixture_root):
    paths = Paths(fixture_root)
    df = cli.cmd_build(paths)
    paths.attendance_overrides.write_text(
        "season,date,opponent,attendance,source\n2024,2024-10-26,Rival C,49000,https://example.org/official-box-score\n")
    with pytest.raises(FeatureError, match="completed game"):
        fill_missing_attendance(df, paths.attendance_overrides)
