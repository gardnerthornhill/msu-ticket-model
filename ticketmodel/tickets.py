"""Load and validate the hand-maintained data/tickets.csv."""
import csv
import io

import pandas as pd

REQUIRED = ["opponent", "date", "getin", "observed"]


class TicketError(ValueError):
    """Raised when tickets.csv is malformed."""


def load_tickets(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"opponent": str, "date": str, "observed": str})
    return _validate(df, str(path))


def _validate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Coerce and check the four ticket columns; add `season`. Raises TicketError on any bad row."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise TicketError(f"{label}: missing columns {missing}; expected {REQUIRED}")
    df = df[REQUIRED].copy()
    df["opponent"] = df["opponent"].astype(str).str.strip()
    try:
        parsed = pd.to_datetime(df["date"], format="%Y-%m-%d")
    except (ValueError, TypeError) as e:
        raise TicketError(f"{label}: bad date, expected YYYY-MM-DD: {e}") from e
    df["date"] = parsed.dt.strftime("%Y-%m-%d")
    try:
        df["getin"] = pd.to_numeric(df["getin"].replace("", None))
    except (ValueError, TypeError) as e:
        raise TicketError(f"{label}: getin must be a number or blank: {e}") from e
    df["observed"] = df["observed"].where(df["observed"].notna() & (df["observed"].astype(str) != ""), None)
    df["season"] = parsed.dt.year.astype(int)
    dups = df.duplicated(subset=["opponent", "date"], keep=False)
    if dups.any():
        rows = df.loc[dups, ["opponent", "date"]].drop_duplicates().to_dict("records")
        raise TicketError(f"{label}: duplicate ticket rows {rows}")
    return df.reset_index(drop=True)


def upsert_rows(path, rows_text: str, today: str) -> tuple[int, int]:
    """Merge pasted CSV rows (opponent,date,getin[,observed]) into tickets.csv, keyed on
    opponent + date. A blank observed date becomes `today`. Returns (added, updated).
    Validates everything before writing, so a bad row leaves the file untouched."""
    lines = [ln.strip() for ln in rows_text.splitlines() if ln.strip()]
    if lines and lines[0].replace(" ", "").lower() == ",".join(REQUIRED):
        lines = lines[1:]
    if not lines:
        return 0, 0
    rows = []
    for raw in csv.reader(io.StringIO("\n".join(lines))):
        cells = [c.strip() for c in raw]
        if len(cells) == 3:
            cells.append("")
        if len(cells) != 4:
            raise TicketError(f"row {raw} must have 3 or 4 columns: opponent,date,getin[,observed]")
        rows.append(cells)
    new = _validate(pd.DataFrame(rows, columns=REQUIRED), "pasted rows")
    new["observed"] = new["observed"].where(new["observed"].notna(), today)
    existing = load_tickets(path).drop(columns="season")
    keyed = {(r.opponent, r.date): i for i, r in enumerate(existing.itertuples(index=False))}
    added = updated = 0
    for r in new.itertuples(index=False):
        key = (r.opponent, r.date)
        if key in keyed:
            existing.loc[keyed[key], ["getin", "observed"]] = [r.getin, r.observed]
            updated += 1
        else:
            existing.loc[len(existing)] = [r.opponent, r.date, r.getin, r.observed]
            added += 1
    out = existing.copy()
    out["getin"] = out["getin"].map(lambda v: "" if pd.isna(v) else f"{float(v):g}")
    out["observed"] = out["observed"].fillna("")
    out.to_csv(path, index=False)
    return added, updated
