"""Write reports/model_report.md from the training summary."""
from pathlib import Path

import pandas as pd

from .config import CAPACITY, TEAM, VENUE

CAVEATS = [
    "Small sample: a couple of dozen games across three historical seasons. These are retrospective tests, not forecasts recorded before kickoff.",
    f"The target is announced attendance, not ticket scans or ticket sales. Predictions are capped at {CAPACITY:,}; "
    "nominal 80% coverage is conditional on model assumptions and has not been established for weeks-ahead forecasts.",
    "Historical prices were collected from the past-events table; their observation dates are unknown. Current prices may be weeks before kickoff, and price age matters.",
    "The season reference uses all available prices, including upcoming games. Historical final-price references were not necessarily available on the forecast date. A common percentage change in every price leaves relative-price predictions unchanged.",
    "The Tier 2 specification is fixed to relative log-price alone. Alternatives are diagnostics and do not automatically change production. It was chosen after examining this small dataset, so its retrospective scores may still be optimistic.",
    "Tier 1 features are selected using the same leave-one-out scores reported. Its opponent ratings are collinear and cached SP+ values are season-level, not verified pregame snapshots.",
    "Live Tier 2 forecasts require at least 3 priced games. Sparse historical seasons are retained in training and shown separately in diagnostics rather than dropping difficult outcomes; their season references are less reliable.",
    f"Sellout odds are an uncalibrated model probability of reaching {CAPACITY:,} announced attendees, not a verified probability that ticket inventory sells out.",
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
    lines += ["", "## Tier 1 feature selection", "",
              "Top candidate subsets by LOO-RMSE. A smaller subset is preferred when its LOO-RMSE is within 5% of the best.", "",
              "| features | LOO RMSE |", "|---|---|"]
    for r in s["tier1_candidates"]:
        lines.append(f"| {_feats(r['features'])} | {_num(r['rmse'])} |")
    lines += ["", "## Tier 2 price feature", "", "Production uses relative log-price alone. These alternatives are comparisons, not an automatic selection rule.",
              "", "| features | LOO RMSE |", "|---|---|"]
    for r in s["tier2_candidates"]:
        lines.append(f"| {_feats(r['features'])} | {_num(r['rmse'])} |")
    lines += ["", "## Fitted models", ""]
    lines += _model_block("Tier 1 (game features only)", s["tier1_model"])
    lines += _model_block("Tier 2 (relative price only)", s["tier2_model"])
    if s.get("per_season"):
        lines += ["## Price-model results by season", "", "Positive bias means overprediction. Sparse-price seasons remain visible.", "",
                  "| season | priced reference games | scored games | RMSE | MAE | bias | inside 80% range |",
                  "|---|---|---|---|---|---|---|"]
        for r in s["per_season"]:
            lines.append(f"| {r['season']} | {r['priced_games']} | {r['n']} | {_num(r['rmse'])} | {_num(r['mae'])} | {_num(r['bias'])} | {r['inside']}/{r['n']} |")
        lines.append("")
    if s.get("validation"):
        lines += ["## Season transfer", "", "Fixed relative-price specification, refitted without the test season. Forward tests train only on earlier seasons. "
                  "Both retain archived features and full-season price references; neither is a live forecast replay. Folds with fewer than 8 training games are skipped.", "",
                  "| test | season | training games | scored games | RMSE | MAE | bias |", "|---|---|---|---|---|---|---|"]
        for mode, label in (("season_holdout", "Season held out"), ("forward", "Earlier seasons only")):
            result = s["validation"][mode]
            for r in result["folds"]:
                lines.append(f"| {label} | {r['season']} | {r['training_n']} | {r['n']} | {_num(r['rmse'])} | {_num(r['mae'])} | {_num(r['bias'])} |")
            if result["metrics"]:
                r = result["metrics"]
                lines.append(f"| {label}, pooled | all | — | {r['n']} | {_num(r['rmse'])} | {_num(r['mae'])} | — |")
        lines.append("")
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
