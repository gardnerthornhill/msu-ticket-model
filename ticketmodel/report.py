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
