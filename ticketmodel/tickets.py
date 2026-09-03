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
