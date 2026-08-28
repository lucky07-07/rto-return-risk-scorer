"""Evaluation and descriptive-statistics helpers.

Used by ``02`` for driver analysis and by ``05`` for the final report. Rates are
always reported with an interval: a 34% RTO rate on 41 orders and a 34% rate on
4,100 orders are different facts, and a bare bar chart hides that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


def wilson_ci(successes, n, z: float = 1.96):
    """Wilson score interval. Behaves at small n and at rates near 0 or 1."""
    k = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
        denom = 1.0 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        halfwidth = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return centre - halfwidth, centre + halfwidth


def rate_table(
    df: pd.DataFrame, by, target: str = "rto", min_count: int = 1
) -> pd.DataFrame:
    """RTO rate by group, with counts and a Wilson interval."""
    g = df.groupby(by, observed=True)[target]
    out = pd.DataFrame({"n": g.size(), "positives": g.sum()})
    out["rate"] = out["positives"] / out["n"]
    lo, hi = wilson_ci(out["positives"], out["n"])
    out["ci_lo"], out["ci_hi"] = lo, hi
    out["lift_vs_base"] = out["rate"] / df[target].mean()
    return out[out["n"] >= min_count]


def single_feature_auc(
    df: pd.DataFrame, features, target: str = "rto", subset=None
) -> pd.DataFrame:
    """ROC-AUC of each feature used alone.

    A leakage screen as much as a strength ranking. Any single raw feature
    scoring near the whole model's AUC deserves an explanation before it is
    trusted; anything above the Bayes ceiling is a bug.
    """
    frame = df if subset is None else df[subset]
    y = frame[target].to_numpy()
    rows = []
    for f in features:
        x = pd.to_numeric(frame[f], errors="coerce")
        mask = x.notna().to_numpy()
        if mask.sum() < 50 or len(np.unique(y[mask])) < 2 or x[mask].nunique() < 2:
            rows.append({"feature": f, "auc": np.nan, "n_used": int(mask.sum())})
            continue
        auc = roc_auc_score(y[mask], x[mask].to_numpy())
        rows.append(
            {
                "feature": f,
                # Direction-free strength: 0.5 is uninformative either way.
                "auc": max(auc, 1 - auc),
                "signed_auc": auc,
                "n_used": int(mask.sum()),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    )


def variance_inflation(X: pd.DataFrame) -> pd.DataFrame:
    """VIF per column, computed as 1 / (1 - R^2) against the other columns.

    Implemented directly rather than pulled from statsmodels to keep the
    dependency list short.
    """
    cols = [c for c in X.columns if X[c].nunique() > 1]
    Z = X[cols].astype(float)
    Z = Z.fillna(Z.median())
    rows = []
    for c in cols:
        others = [o for o in cols if o != c]
        r2 = LinearRegression().fit(Z[others], Z[c]).score(Z[others], Z[c])
        rows.append({"feature": c, "r2_vs_others": r2,
                     "vif": np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def naive_target_encode(
    train: pd.DataFrame, column: str, target: str = "rto", smoothing: float = 30.0
) -> np.ndarray:
    """The leaky encoding, implemented deliberately so ``02`` can measure it.

    This is what target encoding looks like when it is fitted on the same rows it
    is applied to. It exists **only** to demonstrate the size of the leak. It is
    never used to train anything.
    """
    prior = float(train[target].mean())
    agg = train.groupby(column, observed=True)[target].agg(["sum", "count"])
    mapping = (agg["sum"] + prior * smoothing) / (agg["count"] + smoothing)
    return train[column].map(mapping).fillna(prior).to_numpy()
