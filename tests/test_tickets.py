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
