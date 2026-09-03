"""OLS attendance models: leave-one-out evaluation, feature selection, fit, persist, predict."""
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import (
    CANDIDATE_FEATURES,
    CAPACITY,
    INTERVAL,
    MAX_SUBSET_SIZE,
    MIN_TRAINING_ROWS,
    PRICE_FEATURES,
    SELECTION_TOLERANCE,
)

TARGET = "attendance"


class ModelError(ValueError):
    """Too few rows, or a missing saved model."""


def _design(df: pd.DataFrame, features) -> pd.DataFrame:
    X = df[list(features)].astype(float)
    return sm.add_constant(X, has_constant="add")


def _clip(a):
    return np.clip(np.asarray(a, float), 0, CAPACITY)


def loo_predictions(df: pd.DataFrame, features) -> np.ndarray:
    y = df[TARGET].to_numpy(float)
    n = len(df)
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        res = sm.OLS(y[mask], _design(df.iloc[mask], features)).fit()
        preds[i] = float(res.predict(_design(df.iloc[[i]], features)).iloc[0])
    return _clip(preds)


def season_mean_baseline(df: pd.DataFrame) -> np.ndarray:
    y = df[TARGET].to_numpy(float)
    seasons = df["season"].to_numpy()
    idx = np.arange(len(y))
    preds = np.empty(len(y))
    for i in idx:
        others = (seasons == seasons[i]) & (idx != i)
        preds[i] = y[others].mean() if others.any() else y[idx != i].mean()
    return preds


def metrics(y, preds) -> dict:
    y = np.asarray(y, float)
    preds = np.asarray(preds, float)
    resid = y - preds
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y)),
    }


def loo_metrics(df: pd.DataFrame, features) -> dict:
    preds = loo_predictions(df, features)
    out = metrics(df[TARGET], preds)
    out["preds"] = preds
    return out


def _rank(results):
    if not results:
        return results
    # Round for the tolerance comparison (not the reported rmse) so leave-one-out floating-point
    # noise on near-exact fits doesn't make a larger subset look spuriously better than a smaller
    # one that fits just as well; the final rmse tuple element still breaks real ties precisely.
    best = min(round(r["rmse"], 3) for r in results)
    return sorted(
        results,
        key=lambda r: (round(r["rmse"], 3) > best * (1 + SELECTION_TOLERANCE), len(r["features"]), r["rmse"]),
    )


def select_tier1(df: pd.DataFrame) -> list[dict]:
    results = []
    for k in range(1, MAX_SUBSET_SIZE + 1):
        for combo in itertools.combinations(CANDIDATE_FEATURES, k):
            results.append({"features": list(combo), "rmse": loo_metrics(df, combo)["rmse"]})
    return _rank(results)


def select_tier2(df: pd.DataFrame, tier1_features) -> list[dict]:
    results = []
    for pf in PRICE_FEATURES:
        feats = list(tier1_features) + [pf]
        results.append({"features": feats, "price_feature": pf, "rmse": loo_metrics(df, feats)["rmse"]})
    return _rank(results)


def fit(df: pd.DataFrame, features) -> dict:
    if len(df) < MIN_TRAINING_ROWS:
        raise ModelError(f"need at least {MIN_TRAINING_ROWS} training rows, have {len(df)}")
    X = _design(df, features)
    y = df[TARGET].to_numpy(float)
    res = sm.OLS(y, X).fit()
    Xn = X.to_numpy(float)
    return {
        "features": list(features),
        "intercept": float(res.params["const"]),
        "coef": {f: float(res.params[f]) for f in features},
        "stderr": {k: float(v) for k, v in res.bse.items()},
        "resid_se": float(np.sqrt(res.scale)),
        "df_resid": int(res.df_resid),
        "n": int(len(df)),
        "xtx_inv": np.linalg.pinv(Xn.T @ Xn).tolist(),
        "data_hash": hashlib.sha256(df[list(features) + [TARGET]].to_csv(index=False).encode()).hexdigest()[:12],
    }


def save_model(model: dict, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(model, indent=1))


def load_model(path) -> dict:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise ModelError(f"no saved model at {p}; run train")
    return json.loads(p.read_text())


def predict(model: dict, df: pd.DataFrame) -> pd.DataFrame:
    feats = model["features"]
    X = _design(df, feats).to_numpy(float)
    beta = np.array([model["intercept"]] + [model["coef"][f] for f in feats])
    point = X @ beta
    V = np.asarray(model["xtx_inv"], float)
    se = model["resid_se"] * np.sqrt(1.0 + np.einsum("ij,jk,ik->i", X, V, X))
    t = stats.t.ppf(0.5 + INTERVAL / 2, model["df_resid"])
    return pd.DataFrame(
        {"pred": _clip(point), "lo": _clip(point - t * se), "hi": _clip(point + t * se)}, index=df.index
    )
