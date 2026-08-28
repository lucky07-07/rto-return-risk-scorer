"""Evaluation and descriptive-statistics helpers.

Used by ``02`` for driver analysis and by ``05`` for the final report. Rates are
always reported with an interval: a 34% RTO rate on 41 orders and a 34% rate on
4,100 orders are different facts, and a bare bar chart hides that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


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


# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------


def classification_metrics(y_true, y_prob) -> dict:
    """The metric bundle fixed in PRE_REGISTRATION.md.

    PR-AUC is the selection metric: the positive class is the minority and the
    cost of a miss is asymmetric. Brier and log-loss are here because the
    operating threshold is chosen on a rupee cost curve, which is meaningless on
    scores that are only ordinally correct.

    Accuracy is deliberately absent. On a 17% base rate it rewards predicting
    "no RTO" forever.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    single_class = len(np.unique(y_true)) < 2
    return {
        "pr_auc": np.nan if single_class else average_precision_score(y_true, y_prob),
        "roc_auc": np.nan if single_class else roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "mean_pred": float(y_prob.mean()),
        "base_rate": float(y_true.mean()),
    }


def expanding_window_cv(pipeline, frame, y, n_splits: int = 5, columns=None):
    """Expanding-window cross-validation on chronologically ordered data.

    Every model in the benchmark sees exactly these folds. ``KFold`` would train
    on the future to predict the past; ``TimeSeriesSplit`` on positionally
    time-ordered rows is the honest analogue.

    Returns one metric row per fold. The pipeline is cloned per fold, so the
    target encoder is refitted from that fold's history alone.
    """
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit

    y = np.asarray(y)
    cols = list(columns) if columns is not None else list(frame.columns)
    rows = []
    for fold, (past, future) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(np.arange(len(frame)))
    ):
        est = clone(pipeline)
        est.fit(frame.iloc[past][cols], y[past])
        p = est.predict_proba(frame.iloc[future][cols])[:, 1]
        rows.append(
            {"fold": fold, "n_train": len(past), "n_valid": len(future),
             **classification_metrics(y[future], p)}
        )
    return pd.DataFrame(rows)


def paired_bootstrap_delta(
    y_true, p_a, p_b, metric: str = "pr_auc", n_boot: int = 2000, seed: int = 20260101
) -> dict:
    """Paired bootstrap CI for (metric of A) - (metric of B) on the same rows.

    Two models scored on 5,000 validation orders can differ by 0.003 PR-AUC and
    be indistinguishable. Resampling the *same* rows for both models keeps the
    comparison paired, so the shared difficulty of those rows cancels out.

    A confidence interval spanning zero means the leaderboard gap is not
    evidence of a better model.
    """
    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    fn = average_precision_score if metric == "pr_auc" else roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y_true[idx]
        if len(np.unique(ys)) < 2:
            deltas[i] = np.nan
            continue
        deltas[i] = fn(ys, p_a[idx]) - fn(ys, p_b[idx])

    observed = fn(y_true, p_a) - fn(y_true, p_b)
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return {
        "observed_delta": float(observed),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_two_sided": float(2 * min(np.nanmean(deltas > 0), np.nanmean(deltas < 0))),
        "significant": bool(lo > 0 or hi < 0),
    }
