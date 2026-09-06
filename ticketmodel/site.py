"""Generate the static site in site/ from features.csv, the training summary, the fitted models
and the CFBD cache. Pure formatting: every number here was produced by train or predict."""
import json
import math
import os
import re
import shutil
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import cfbd, logos
from . import model as mdl
from .config import CAPACITY, INTERVAL, SITE_NAME, SITE_URL, TEAM, VENUE, Paths
from .report import CAVEATS

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"

FEATURE_LABELS = {
    "rel_log_price": "Get-in price, relative to the season",
    "log_getin": "Get-in price",
    "opp_ranked": "Opponent ranked in the AP Top 25",
    "opp_elo": "Opponent Elo rating",
    "opp_sp": "Opponent SP+ rating",
    "conf_game": "SEC game",
    "opp_p4": "Power-conference opponent",
    "week": "Week of the season",
}
FEATURE_HELP = {
    "rel_log_price": "The cheapest resale ticket on ticketdata.com, compared with that season's median price on a log scale. "
                     "All listed home games contribute to the reference. A doubling relative to that reference has the same effect in any season.",
    "log_getin": "The cheapest resale ticket on ticketdata.com, on a log scale.",
    "opp_ranked": "1 when the opponent was in that week's AP Top 25, otherwise 0.",
    "opp_elo": "CollegeFootballData's Elo rating for the opponent entering the game. Before an opponent has "
               "played, last season's final Elo stands in.",
    "opp_sp": "Bill Connelly's SP+ rating for the opponent that season. Positive is better than average.",
    "conf_game": "1 for an SEC opponent, otherwise 0.",
    "opp_p4": "1 when the opponent plays in the SEC, Big Ten, Big 12 or ACC.",
    "week": "The week of the season the game falls in.",
}
TIER_LABELS = {"tier2": "Price model", "tier1": "Schedule-only model"}


class SiteError(RuntimeError):
    """Missing inputs for the site build."""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _none(v):
    """NaN-safe float for JSON and templates."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if math.isnan(f) else f


def _int(v):
    v = _none(v)
    return None if v is None else int(round(v))


def date_label(iso: str, year: bool = True) -> str:
    d = datetime.strptime(iso, "%Y-%m-%d")
    return d.strftime("%a, %b %-d, %Y") if year else d.strftime("%a, %b %-d")


def kickoff_label(hour) -> str:
    hour = _none(hour)
    if hour is None:
        return "Kickoff to be announced"
    h, m = int(hour), int(round((hour - int(hour)) * 60))
    suffix = "a.m." if h < 12 else "p.m."
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {suffix} CT" if m else f"{h12} {suffix} CT"


def display_value(feature: str, x: float, row) -> str:
    if feature == "rel_log_price":
        ratio = math.exp(x)
        getin = _none(getattr(row, "getin", None))
        price = f"${getin:,.0f}" if getin is not None else ""
        if abs(x) < 0.05:
            return f"{price}, at the season median"
        return f"{price}, {ratio:.2g}× the season median" if ratio >= 1 else f"{price}, {ratio:.0%} of the season median"
    if feature == "log_getin":
        getin = _none(getattr(row, "getin", None))
        return f"${getin:,.0f}" if getin is not None else ""
    if feature == "opp_ranked":
        rank = _none(getattr(row, "opp_ap_rank", None))
        return f"Yes, No. {int(rank)}" if x >= 0.5 and rank is not None and rank <= 25 else "Unranked"
    if feature == "opp_elo":
        return f"{x:,.0f}"
    if feature == "opp_sp":
        return f"{x:+.1f}"
    if feature in ("conf_game", "opp_p4"):
        return "Yes" if x >= 0.5 else "No"
    if feature == "week":
        return f"Week {x:.0f}"
    return f"{x:g}"


def mean_value(feature: str, m: float) -> str:
    if feature == "rel_log_price":
        return f"{math.exp(m):.2f}× the season reference"
    if feature == "opp_ranked":
        return f"{m:.0%} of games ranked"
    if feature == "opp_elo":
        return f"{m:,.0f}"
    if feature == "opp_sp":
        return f"{m:+.1f}"
    if feature in ("conf_game", "opp_p4"):
        return f"{m:.0%} of games"
    if feature == "week":
        return f"week {m:.0f}"
    return f"{m:.2f}"


def describe_model(m: dict, train: pd.DataFrame) -> dict:
    feats = m["features"]
    means = {f: float(train[f].mean()) for f in feats}
    lo = {f: float(train[f].min()) for f in feats}
    hi = {f: float(train[f].max()) for f in feats}
    return {
        "features": feats,
        "intercept": m["intercept"],
        "coef": m["coef"],
        "stderr": m["stderr"],
        "resid_se": m["resid_se"],
        "df_resid": m["df_resid"],
        "n": m["n"],
        "means": means,
        "range": {f: [lo[f], hi[f]] for f in feats},
        "swing": {f: m["coef"][f] * (hi[f] - lo[f]) for f in feats},
        "baseline": m["intercept"] + sum(m["coef"][f] * means[f] for f in feats),
        "labels": {f: FEATURE_LABELS.get(f, f) for f in feats},
        "help": {f: FEATURE_HELP.get(f, "") for f in feats},
        "mean_labels": {f: mean_value(f, means[f]) for f in feats},
    }


def breakdown(model: dict, row) -> dict:
    terms = []
    for f in model["features"]:
        x = float(getattr(row, f))
        contribution = model["coef"][f] * (x - model["means"][f])
        terms.append({
            "feature": f, "label": model["labels"][f], "value": x,
            "display": display_value(f, x, row), "mean_display": model["mean_labels"][f],
            "contribution": contribution, "weight": model["coef"][f],
        })
    total = model["baseline"] + sum(t["contribution"] for t in terms)
    return {"baseline": model["baseline"], "terms": terms, "total": total, "capped": total > CAPACITY}


def opponent_ids(seasons: list[int], cache_dir: Path) -> dict[tuple[int, str], int]:
    ids = {}
    for season in seasons:
        for g in cfbd.load_cached("games", season, cache_dir):
            if g.get("homeTeam") == TEAM and g.get("awayId") is not None:
                ids[(season, g["awayTeam"])] = int(g["awayId"])
    return ids


def game_notes(warnings: list[str], season: int, opponent: str) -> list[str]:
    prefix = f"{season} {opponent}: "
    notes = []
    for w in warnings:
        if not w.startswith(prefix):
            continue
        text = w[len(prefix):]
        if text == "no ticket row":
            text = "No get-in price was recorded for this game, so the schedule-only model is used."
        else:
            text = text[0].upper() + text[1:] + "."
        notes.append(text)
    return notes


def site_url() -> str:
    """Public URL of the site, from the SITE_URL environment variable, without a trailing slash.
    Empty means: emit no canonical links, no og:url and no sitemap."""
    return os.environ.get("SITE_URL", SITE_URL).strip().rstrip("/")


def recorded_forecasts(path: Path) -> dict[tuple[int, str, str], dict]:
    """Return the latest archived pregame snapshot for each scheduled game."""
    latest = {}
    if not path.exists():
        return latest
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            snapshot = json.loads(line)
            key = (int(snapshot["season"]), snapshot["date"], snapshot["opponent"])
            recorded_at = snapshot["recorded_at"]
            snapshot["tier"]
            snapshot["model"]
            snapshot["forecast"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise SiteError(f"invalid forecast snapshot on line {line_number} of {path}") from exc
        if key not in latest or recorded_at > latest[key]["recorded_at"]:
            latest[key] = snapshot
    return latest


def site_data(paths: Paths, today: str | None = None) -> dict:
    today = today or date.today().isoformat()
    for p in (paths.features, paths.train_summary, paths.tier1, paths.tier2):
        if not Path(p).exists():
            raise SiteError(f"missing {p}; run train and predict first")
    df = pd.read_csv(paths.features)
    summary = json.loads(paths.train_summary.read_text())
    m1, m2 = mdl.load_model(paths.tier1), mdl.load_model(paths.tier2)
    t1 = df[df["attendance"].notna()].reset_index(drop=True)
    t2 = t1[t1["getin"].notna()].reset_index(drop=True)
    models = {"tier1": describe_model(m1, t1), "tier2": describe_model(m2, t2)}
    seasons = sorted(int(s) for s in df["season"].unique())
    ids = opponent_ids(seasons, paths.cache_dir)
    loo = {(int(r["season"]), r["opponent"]): r for r in summary["per_game"]}
    recorded = recorded_forecasts(paths.forecast_history)
    upcoming_df, upcoming_warnings = mdl.predict_upcoming(df, m1, m2)
    upcoming = {(int(r.season), r.opponent): r for r in upcoming_df.itertuples(index=False)}
    priced_per_season = df[df["getin"].notna()].groupby("season").size().to_dict()
    warnings = list(summary.get("warnings", [])) + upcoming_warnings

    games = []
    for row in df.itertuples(index=False):
        season, opp = int(row.season), row.opponent
        key = (season, opp)
        snapshot = recorded.get((season, row.date, opp))
        played = _none(row.attendance) is not None
        priced = _none(row.getin) is not None
        explain = None
        if played and key in loo:
            r = loo[key]
            if snapshot:
                tier = snapshot["tier"]
                forecast = {name: _int(snapshot["forecast"][name]) for name in ("pred", "lo", "hi")}
                forecast["p_sellout"] = _none(snapshot["forecast"]["p_sellout"])
                forecast_kind = "recorded"
                # Use the archived model and inputs so the explanation cannot drift as later data arrives.
                train = t2 if tier == "tier2" else t1
                prior = train[train["date"] < row.date].reset_index(drop=True)
                explain = describe_model(snapshot["model"], prior if len(prior) else train)
                forecast_row = pd.Series(snapshot.get("inputs", row._asdict()))
            else:
                tier = "tier2" if priced and r.get("tier2_loo") is not None else "tier1"
                forecast = {"pred": _int(r[f"{tier}_loo"]), "lo": _int(r[f"{tier}_lo"]), "hi": _int(r[f"{tier}_hi"]),
                            "p_sellout": _none(r[f"{tier}_p_sellout"])}
                forecast_kind = "retrospective"
                forecast_row = row
                # Explain the game with the same leave-one-out fit that produced its forecast.
                train = t2 if tier == "tier2" else t1
                others = train[~((train["season"] == season) & (train["opponent"] == opp))].reset_index(drop=True)
                explain = describe_model(mdl.fit(others, models[tier]["features"], check_rows=False), others)
            status = "played"
        elif key in upcoming:
            u = upcoming[key]
            tier = "tier2" if _none(u.tier2_pred) is not None else "tier1"
            forecast = {"pred": _int(getattr(u, f"{tier}_pred")), "lo": _int(getattr(u, f"{tier}_lo")),
                        "hi": _int(getattr(u, f"{tier}_hi")), "p_sellout": _none(getattr(u, f"{tier}_p_sellout"))}
            status = "upcoming"
            forecast_kind = "current"
            forecast_row = row
        else:
            # Played, but CFBD never posted an attendance figure: forecast with the full model.
            tier = "tier2" if priced and priced_per_season.get(season, 0) >= 3 else "tier1"
            p = mdl.predict(m1 if tier == "tier1" else m2, df[(df["season"] == season) & (df["opponent"] == opp)]).iloc[0]
            forecast = {"pred": _int(p["pred"]), "lo": _int(p["lo"]), "hi": _int(p["hi"]), "p_sellout": _none(p["p_sellout"])}
            status = "unreported"
            forecast_kind = "current"
            forecast_row = row
        actual = _int(row.attendance) if played else None
        error = forecast["pred"] - actual if actual is not None else None
        inside = (forecast["lo"] <= actual <= forecast["hi"]) if actual is not None else None
        opp_id = ids.get(key)
        games.append({
            "season": season, "week": int(row.week), "date": row.date, "date_label": date_label(row.date),
            "date_short": date_label(row.date, year=False), "kickoff": kickoff_label(row.kickoff_hr),
            "opponent": opp, "slug": f"{season}-{slugify(opp)}", "opp_id": opp_id,
            "logo": logos.logo_file(opp_id) if opp_id is not None else None,
            "attendance": actual, "actual": actual, "completed": int(row.completed), "status": status,
            "getin": _none(row.getin), "observed": row.observed if isinstance(row.observed, str) else None,
            "observed_label": date_label(row.observed, year=False) if isinstance(row.observed, str) else None,
            "ranked": int(row.opp_ranked), "ap_rank": _int(row.opp_ap_rank) if int(row.opp_ranked) else None,
            "elo": _none(row.opp_elo), "sp": _none(row.opp_sp), "conf_game": int(row.conf_game),
            "rel_log_price": _none(row.rel_log_price),
            "tier": tier, "tier_label": TIER_LABELS[tier], "forecast": forecast,
            "forecast_kind": forecast_kind,
            "forecast_recorded_at": snapshot["recorded_at"] if forecast_kind == "recorded" else None,
            "forecast_date": snapshot.get("forecast_date") if forecast_kind == "recorded" else None,
            "error": error, "inside": inside, "sellout": actual is not None and actual >= CAPACITY,
            "breakdown": breakdown(explain or models[tier], forecast_row),
            "notes": game_notes(warnings, season, opp),
        })
        count = priced_per_season.get(season, 0)
        if priced and count < 3:
            games[-1]["notes"].append(f"Only {count} prices are recorded for this season. Its price reference is less reliable; historical results remain visible in the evaluation.")
    games.sort(key=lambda g: g["date"])

    played = [g for g in games if g["status"] == "played"]
    errors = np.array([g["error"] for g in played], float)
    track = {
        "n": len(played),
        "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(played) else None,
        "mae": float(np.mean(np.abs(errors))) if len(played) else None,
        "inside": int(sum(1 for g in played if g["inside"])),
        "inside_rate": (sum(1 for g in played if g["inside"]) / len(played)) if played else None,
        "priced": sum(1 for g in played if g["tier"] == "tier2"),
        "recorded": sum(1 for g in played if g["forecast_kind"] == "recorded"),
        "sellouts": sum(1 for g in played if g["sellout"]),
    }
    current = seasons[-1]
    lows = [g["forecast"]["lo"] for g in games] + [g["actual"] for g in games if g["actual"]]
    axis_min = min(40000, math.floor((min(lows) - 500) / 5000) * 5000) if lows else 40000

    def bar_scale(subset):
        top = max([abs(t["contribution"]) for g in subset for t in g["breakdown"]["terms"]] or [1000])
        return math.ceil(top / 1000) * 1000

    bar_max = bar_scale(games)
    bar_max_current = bar_scale([g for g in games if g["season"] == current])
    upcoming_games = [g for g in games if g["status"] == "upcoming" and g["date"] >= today]
    next_game = upcoming_games[0]["slug"] if upcoming_games else None
    return {
        "site_name": SITE_NAME, "site_url": site_url(), "team": TEAM, "venue": VENUE, "capacity": CAPACITY,
        "interval": INTERVAL, "generated": today, "trained": summary.get("generated"),
        "seasons": seasons, "current_season": current, "next_game": next_game,
        "axis_min": axis_min, "bar_max": bar_max, "bar_max_current": bar_max_current,
        "games": games, "track": track, "models": models, "metrics": summary["metrics"],
        "tier1_candidates": summary.get("tier1_candidates", []), "tier2_candidates": summary.get("tier2_candidates", []),
        "counts": summary.get("counts", {}), "caveats": CAVEATS, "warnings": warnings,
        "validation": summary.get("validation", {}), "per_season": summary.get("per_season", []),
    }


# ---------------------------------------------------------------- rendering

def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]), trim_blocks=True,
                      lstrip_blocks=True)
    env.filters["num"] = lambda v: "" if _none(v) is None else f"{float(v):,.0f}"
    env.filters["signed"] = lambda v: "" if _none(v) is None else f"{float(v):+,.0f}"
    env.filters["pct"] = lambda v: "" if _none(v) is None else f"{float(v):.0%}"
    env.filters["money"] = lambda v: "" if _none(v) is None else f"${float(v):,.0f}"
    env.filters["dec"] = lambda v, nd=2: "" if _none(v) is None else f"{float(v):,.{nd}f}"
    env.filters["json"] = lambda v: json.dumps(v)
    env.filters["feat"] = lambda f: FEATURE_LABELS.get(f, f).lower().replace("get-in price, relative to the season", "season-relative price")

    def scale(v, lo, hi, a, b):
        """Map v in [lo, hi] onto [a, b], clamping to the ends."""
        v = min(max(float(v), lo), hi)
        return round(a + (v - lo) / (hi - lo) * (b - a), 2)

    env.globals["scale"] = scale
    env.globals["pos"] = lambda v, data: scale(v, data["axis_min"], data["capacity"], 0, 100)
    return env


def _pages(data: dict) -> list[dict]:
    pages = [
        {"template": "index.html", "out": "index.html", "root": "", "path": "",
         "title": f"{data['current_season']} Mississippi State home attendance forecast",
         "description": f"Predicted crowd, 80% range and sellout odds for every {data['current_season']} Mississippi State "
                        f"home game at {VENUE}, driven by the resale get-in price."},
        {"template": "track-record.html", "out": "track-record.html", "root": "", "path": "track-record.html",
         "title": "Track record: forecast vs. actual attendance",
         "description": "Recorded pregame forecasts where available and leave-one-out estimates for earlier Mississippi "
                        "State home games, with actual attendance and 80% ranges."},
        {"template": "model.html", "out": "model.html", "root": "", "path": "model.html",
         "title": "How the attendance model works",
         "description": "The variables, weights, accuracy and caveats behind the Mississippi State home attendance "
                        "forecast, in plain English."},
    ]
    for g in data["games"]:
        verb = "forecast" if g["status"] != "played" else "forecast vs. actual"
        pages.append({"template": "game.html", "out": f"games/{g['slug']}.html", "root": "../",
                      "path": f"games/{g['slug']}.html", "game": g,
                      "title": f"{g['opponent']} at Mississippi State, {g['date_label']}: attendance {verb}",
                      "description": f"Forecast attendance for {g['opponent']} at {VENUE} on {g['date_label']}, with the "
                                     f"80% range, sellout odds and how each variable moved the number."})
    return pages


def build_site(paths: Paths, http=None, today: str | None = None) -> dict:
    data = site_data(paths, today=today)
    out = paths.site_dir
    (out / "games").mkdir(parents=True, exist_ok=True)
    (out / "logos").mkdir(parents=True, exist_ok=True)

    ids = {g["opp_id"] for g in data["games"] if g["opp_id"] is not None}
    failed = logos.ensure_logos(ids, paths.logos_dir, http=http)
    if failed:
        print(f"warning: no logo for team ids {failed}; using initials")
    for g in data["games"]:
        if g["logo"] and not (paths.logos_dir / g["logo"]).exists():
            g["logo"] = None
    for name in {logos.MSU_FILE} | {g["logo"] for g in data["games"] if g["logo"]}:
        src = paths.logos_dir / name
        if src.exists():
            shutil.copyfile(src, out / "logos" / name)
        elif name == logos.MSU_FILE:
            print(f"warning: {src} not found; the MSU mark will be missing")

    env = _env()
    pages = _pages(data)
    for page in pages:
        html = env.get_template(page["template"]).render(data=data, page=page, root=page["root"])
        target = out / page["out"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html)
    for asset in STATIC.iterdir():
        shutil.copyfile(asset, out / asset.name)
    (out / "data.json").write_text(json.dumps(data, indent=1, default=float))
    url = data["site_url"]
    robots = "User-agent: *\nAllow: /\n\n" + "".join(
        f"User-agent: {bot}\nAllow: /\n\n" for bot in ("OAI-SearchBot", "ChatGPT-User", "PerplexityBot", "ClaudeBot"))
    sitemap = out / "sitemap.xml"
    if url:
        robots += f"Sitemap: {url}/sitemap.xml\n"
        urls = "".join(f"  <url><loc>{url}/{p['path']}</loc><lastmod>{data['generated']}</lastmod></url>\n" for p in pages)
        sitemap.write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + urls + "</urlset>\n")
    elif sitemap.exists():
        sitemap.unlink()
    (out / "robots.txt").write_text(robots)
    nojekyll = out / ".nojekyll"
    if nojekyll.exists():
        nojekyll.unlink()
    return data
