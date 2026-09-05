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


@pytest.mark.parametrize("price", ["0", "-1", "inf", "-inf"])
def test_rejects_prices_that_cannot_be_logged(tmp_path, price):
    p = write(tmp_path, f"opponent,date,getin,observed\nUL Monroe,2026-09-05,{price},\n")
    with pytest.raises(TicketError, match="positive and finite"):
        load_tickets(p)


def test_rejects_invalid_observation_date(tmp_path):
    p = write(tmp_path, "opponent,date,getin,observed\nUL Monroe,2026-09-05,32,yesterday\n")
    with pytest.raises(TicketError, match="observed date"):
        load_tickets(p)


def test_bad_date_error(tmp_path):
    p = write(tmp_path, "opponent,date,getin,observed\nOle Miss,11/28/2025,133,\n")
    with pytest.raises(TicketError, match="date"):
        load_tickets(p)


def test_seed_file_loads():
    df = load_tickets(DEFAULT_PATHS.tickets)
    assert len(df) >= 22
    assert {2023, 2024, 2025} <= set(df["season"])


from ticketmodel.tickets import upsert_rows


def test_upsert_appends_updates_and_defaults_observed(tmp_path):
    p = write(tmp_path, "opponent,date,getin,observed\nAlabama,2026-10-03,93,2026-09-03\nAuburn,2026-11-14,83,2026-09-03\n")
    added, updated = upsert_rows(p, "opponent,date,getin,observed\nAlabama,2026-10-03,88,\nVanderbilt Commodores,2026-11-07,,2026-09-10\n", today="2026-09-12")
    assert (added, updated) == (1, 1)
    df = load_tickets(p)
    assert list(df.columns) == ["opponent", "date", "getin", "observed", "season"]
    assert len(df) == 3
    alabama = df[df["opponent"] == "Alabama"].iloc[0]
    assert alabama["getin"] == 88 and alabama["observed"] == "2026-09-12"      # updated, blank observed -> today
    assert df[df["opponent"] == "Auburn"].iloc[0]["getin"] == 83               # untouched
    history = pd.read_csv(p.with_name("ticket_history.csv"))
    assert set(history[history.opponent == "Alabama"].getin) == {93, 88}
    vandy = df[df["opponent"] == "Vanderbilt Commodores"].iloc[0]
    assert pd.isna(vandy["getin"]) and vandy["observed"] == "2026-09-10"       # appended, blank price kept


def test_upsert_accepts_three_column_rows_and_blank_lines(tmp_path):
    p = write(tmp_path, "opponent,date,getin,observed\n")
    added, updated = upsert_rows(p, "\nMissouri,2026-09-26,46\n\n", today="2026-09-12")
    assert (added, updated) == (1, 0)
    df = load_tickets(p)
    assert df.iloc[0]["getin"] == 46 and df.iloc[0]["observed"] == "2026-09-12"


def test_upsert_rejects_bad_row_and_leaves_file_unchanged(tmp_path):
    text = "opponent,date,getin,observed\nAlabama,2026-10-03,93,2026-09-03\n"
    p = write(tmp_path, text)
    with pytest.raises(TicketError, match="date"):
        upsert_rows(p, "Alabama,10/03/2026,90\n", today="2026-09-12")
    with pytest.raises(TicketError, match="getin"):
        upsert_rows(p, "Alabama,2026-10-03,$90\n", today="2026-09-12")
    with pytest.raises(TicketError, match="columns"):
        upsert_rows(p, "Alabama,2026-10-03\n", today="2026-09-12")
    assert p.read_text() == text


def test_upsert_with_no_rows_is_a_noop(tmp_path):
    text = "opponent,date,getin,observed\nAlabama,2026-10-03,93,2026-09-03\n"
    p = write(tmp_path, text)
    assert upsert_rows(p, "   \n", today="2026-09-12") == (0, 0)
    assert p.read_text() == text
