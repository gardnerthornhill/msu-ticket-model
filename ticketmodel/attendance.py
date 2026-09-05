"""Fill missing attendance from explicitly sourced, durable corrections without editing CFBD cache."""
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureError


def fill_missing_attendance(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        return df, []
    fixes = pd.read_csv(path)
    required = {"season", "date", "opponent", "attendance", "source"}
    if not required <= set(fixes.columns):
        raise FeatureError(f"{path}: attendance corrections need {sorted(required)}")
    if fixes.duplicated(["season", "date", "opponent"]).any():
        raise FeatureError(f"{path}: duplicate attendance correction")
    out, warnings = df.copy(), []
    for row in fixes.itertuples(index=False):
        if pd.isna(row.source) or not str(row.source).startswith("https://"):
            raise FeatureError(f"{path}: attendance correction needs an https source")
        try:
            attendance = float(row.attendance)
        except (TypeError, ValueError) as exc:
            raise FeatureError(f"{path}: attendance must be a nonnegative whole number") from exc
        if not np.isfinite(attendance) or attendance < 0 or attendance != int(attendance):
            raise FeatureError(f"{path}: attendance must be a nonnegative whole number")
        match = (out["season"] == row.season) & (out["date"] == row.date) & (out["opponent"] == row.opponent)
        if not match.any() and row.season not in set(out["season"]):
            continue
        if match.sum() != 1 or not out.loc[match, "completed"].eq(1).all():
            raise FeatureError(f"{path}: correction must match one completed game: {row.season} {row.opponent}")
        current = out.loc[match, "attendance"].iloc[0]
        if pd.isna(current):
            out.loc[match, "attendance"] = attendance
            warnings.append(f"{row.season} {row.opponent}: missing attendance filled with {int(attendance):,} from {row.source}")
        elif current != attendance:
            warnings.append(f"{row.season} {row.opponent}: CFBD attendance {current:,.0f} differs from sourced correction {attendance:,.0f}; keeping CFBD")
    return out, warnings
