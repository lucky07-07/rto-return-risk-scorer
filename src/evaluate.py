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


# ---------------------------------------------------------------------------
# Prevalence shift
# ---------------------------------------------------------------------------


def resample_to_prevalence(y_true, target_rate: float, seed: int = 20260101):
    """Row indices of a subsample whose positive rate is ``target_rate``.

    Published Indian city RTO rates span 18% to 35%. A model selected and
    thresholded at one base rate can degrade badly at another, and a cost-optimal
    threshold can invert. To measure that rather than assume it, the evaluation
    set is resampled to each target prevalence.

    Only *subsampling* is used -- one class is thinned, never duplicated or
    synthesised -- so every scored row is a real row with a real prediction. The
    price is a smaller n at the extremes, which is why the study reports n and a
    bootstrap band alongside every point.
    """
    y = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if not 0 < target_rate < 1 or len(pos) == 0 or len(neg) == 0:
        raise ValueError("target_rate must be in (0, 1) and both classes present")

    # Keep whichever class does not need thinning, and thin the other.
    n_pos_if_all_neg = int(round(len(neg) * target_rate / (1 - target_rate)))
    if n_pos_if_all_neg <= len(pos):
        keep_pos = rng.choice(pos, size=max(n_pos_if_all_neg, 1), replace=False)
        keep_neg = neg
    else:
        n_neg = int(round(len(pos) * (1 - target_rate) / target_rate))
        keep_pos = pos
        keep_neg = rng.choice(neg, size=max(min(n_neg, len(neg)), 1), replace=False)

    idx = np.sort(np.concatenate([keep_pos, keep_neg]))
    return idx


def prevalence_curve(y_true, y_prob, rates, seed: int = 20260101,
                     n_boot: int = 200) -> pd.DataFrame:
    """Metrics across a range of base rates, with bootstrap bands.

    PR-AUC moves with prevalence *by definition* -- the no-skill PR-AUC equals the
    base rate -- so the honest thing to report next to it is ``pr_auc_lift``, the
    ratio of PR-AUC to that no-skill floor. A model whose lift falls towards 1.0
    has stopped being useful even if its raw PR-AUC is rising.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows = []
    for r in rates:
        idx = resample_to_prevalence(y_true, r, seed=seed)
        ys, ps = y_true[idx], y_prob[idx]
        m = classification_metrics(ys, ps)

        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            b = rng.integers(0, len(ys), len(ys))
            if len(np.unique(ys[b])) < 2:
                continue
            boots.append(average_precision_score(ys[b], ps[b]))
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))

        rows.append({
            "target_prevalence": r,
            "actual_prevalence": float(ys.mean()),
            "n": int(len(ys)),
            **m,
            "pr_auc_lo": float(lo), "pr_auc_hi": float(hi),
            # PR-AUC's no-skill floor IS the base rate, so raw PR-AUC is not
            # comparable across prevalences. Lift is.
            "pr_auc_lift": float(m["pr_auc"] / ys.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def reliability_curve(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile"):
    """Observed frequency against predicted probability, per bin.

    Quantile bins by default rather than equal-width: with a 15% base rate most
    predictions crowd into the low end, and equal-width bins would report nine
    nearly empty buckets and one that hides everything interesting.

    Brier score answers "are the probabilities good?" with one number. This
    answers "and where are they wrong?", which is what a cost-based threshold
    actually needs -- systematic overconfidence near the operating point costs
    money in a way that a decent aggregate Brier can conceal.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        k, n = int(y_true[m].sum()), int(m.sum())
        lo, hi = wilson_ci(k, n)
        rows.append({
            "bin": b, "n": n,
            "mean_predicted": float(y_prob[m].mean()),
            "observed_rate": k / n,
            "ci_lo": float(lo), "ci_hi": float(hi),
            "gap": float(y_prob[m].mean() - k / n),
        })
    curve = pd.DataFrame(rows)
    # Expected Calibration Error: mean |predicted - observed|, weighted by bin size.
    ece = float((curve.n * curve.gap.abs()).sum() / curve.n.sum()) if len(curve) else np.nan
    return curve, ece


# ---------------------------------------------------------------------------
# Human-readable risk reasons
# ---------------------------------------------------------------------------

# Plain-English templates. A merchant-facing reason has to say what about the
# order is risky, not name a column.
REASON_TEMPLATES = {
    "pincode_te": ("this pincode returns COD orders more often than average",
                   "this pincode has a below-average return rate"),
    "city_te": ("this city returns COD orders more often than average",
                "this city has a below-average return rate"),
    "pincode_prefix3_te": ("this delivery region returns orders more often than average",
                           "this delivery region is lower risk than average"),
    "is_cod": ("paying cash on delivery", "paying online up front"),
    "past_rto_rate": ("this customer has returned orders before",
                      "this customer's past orders were delivered"),
    "past_rto_count": ("this customer has prior returns on record",
                       "this customer has no prior returns"),
    "has_history": ("no prior order history to judge from",
                    "an established order history"),
    "is_first_order": ("this is the customer's first order",
                       "a returning customer"),
    "addr_quality_score": ("the delivery address looks incomplete",
                           "the delivery address is detailed"),
    "addr_has_house_number": ("no house or flat number in the address",
                              "a house number is present"),
    "addr_has_landmark": ("no landmark given to help the courier",
                          "a landmark is given"),
    "addr_gibberish_score": ("the address contains unreadable text",
                             "the address text is clean"),
    "addr_token_count": ("the address is unusually short",
                         "the address is detailed"),
    "delivery_days_est": ("a long courier ETA", "a short courier ETA"),
    "tier_ordinal": ("delivery outside the metros", "delivery into a metro"),
    "order_value": ("the basket value", "the basket value"),
    "log_order_value": ("the basket value", "the basket value"),
    "discount_pct": ("a heavy discount on this order", "little or no discount"),
    "discount_amount": ("a large discount on this order", "little or no discount"),
    "order_velocity_24h": ("several orders from this customer in 24 hours",
                           "normal ordering pace"),
    "account_age_days": ("a new account", "a long-standing account"),
    "log_account_age": ("a new account", "a long-standing account"),
    "is_festive": ("ordered during a festive sale", "ordered outside a sale period"),
    "is_alternate_address": ("shipping to a non-default address",
                             "shipping to the usual address"),
}


ONE_HOT_PREFIXES = ("category_", "state_", "pincode_tier_", "order_value_band_")

_BAND_WORDS = {
    "under_500": "an order under Rs500",
    "500_1000": "an order in the Rs500-1,000 impulse band",
    "1000_plus": "an order above Rs1,000",
}
_TIER_WORDS = {
    "metro": "delivery into a metro",
    "tier_1": "delivery to a tier-1 city",
    "tier_2": "delivery to a tier-2 town",
    "tier_3": "delivery outside the main cities",
}


def _one_hot_reason(name: str) -> str | None:
    """Phrase a one-hot column the order actually belongs to."""
    if name.startswith("category_"):
        return f"{name.removeprefix('category_').replace('_', ' ')} is a higher-return category"
    if name.startswith("state_"):
        return f"delivery to {name.removeprefix('state_')}"
    if name.startswith("pincode_tier_"):
        return _TIER_WORDS.get(name.removeprefix("pincode_tier_"), "the delivery location")
    if name.startswith("order_value_band_"):
        return _BAND_WORDS.get(name.removeprefix("order_value_band_"), "the basket value")
    return None


def risk_reasons(shap_row, feature_names, feature_values=None,
                 top_k: int = 3, direction: str = "up") -> list[str]:
    """Turn one order's SHAP values into sentences a merchant can act on.

    ``direction="up"`` returns what pushed the score UP: a risk explanation that
    leads with mitigating factors is not an explanation, it is a hedge.

    ``direction="down"`` returns what held it DOWN, and is the right choice when
    the decision being explained is *allow*. Listing three faint upward nudges as
    the "reasons" for a 0.8% score invites the obvious question -- if this pincode
    is high-RTO, why is the order low-risk? -- and answers it badly. What a
    merchant needs there is why the order was cleared.

    ``feature_values`` matters more than it looks. A one-hot column can carry a
    positive SHAP value for an order that is **not** in that category -- the
    model is being pushed up by the *absence*. Rendering that as "home_kitchen is
    a higher-return category" on an electronics order is a false statement shown
    to a merchant, so one-hot columns are only ever verbalised when the order
    actually has that value.
    """
    vals = np.asarray(shap_row, dtype=float)
    x = None if feature_values is None else np.asarray(feature_values, dtype=float)
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    upward = direction == "up"

    out: list[str] = []
    order = np.argsort(-vals) if upward else np.argsort(vals)
    for i in order:
        if len(out) >= top_k:
            break
        if (vals[i] <= 0) if upward else (vals[i] >= 0):
            break
        name = feature_names[i]

        if name.startswith(ONE_HOT_PREFIXES):
            # Only speak about a category the order is actually in.
            if x is None or x[i] < 0.5:
                continue
            reason = _one_hot_reason(name)
        else:
            tpl = REASON_TEMPLATES.get(name)
            reason = tpl[0 if upward else 1] if tpl else name.replace("_", " ")

        if reason and reason not in out:
            out.append(reason)
    return out



def prior_shift_correction(y_prob, source_rate: float, target_rate: float):
    """Rescale probabilities from one base rate to another.

    A model trained where 17% of orders return is *miscalibrated* by construction
    in a city where 35% do -- the ranking survives, the probabilities do not, and
    a rupee-cost threshold reads probabilities. Under the standard label-shift
    assumption (P(x|y) unchanged, P(y) changed) the fix is an odds adjustment:

        odds' = odds x [target / (1 - target)] / [source / (1 - source)]

    This is one line and it is the difference between a model that degrades
    gracefully across the published 18-35% range and one that quietly overcharges
    good customers in Patna or under-flags returns in Vadodara.
    """
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-9, 1 - 1e-9)
    ratio = (target_rate / (1 - target_rate)) / (source_rate / (1 - source_rate))
    odds = p / (1 - p) * ratio
    return odds / (1 + odds)


def segment_report(df: pd.DataFrame, y_true, y_prob, threshold: float,
                   segments: dict, min_n: int = 100) -> pd.DataFrame:
    """Per-segment performance, for the honest exception list.

    An aggregate metric is an average over populations the model treats very
    differently. This is where the model is asked to account for itself on the
    slices it finds hardest.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    flagged = y_prob >= threshold

    rows = []
    for name, mask in segments.items():
        m = np.asarray(mask)
        n = int(m.sum())
        if n < min_n or len(np.unique(y_true[m])) < 2:
            rows.append({"segment": name, "n": n, "note": "too small or single-class"})
            continue
        ys, ps, fs = y_true[m], y_prob[m], flagged[m]
        tp = int((fs & (ys == 1)).sum())
        rows.append({
            "segment": name,
            "n": n,
            "base_rate": float(ys.mean()),
            "pr_auc": float(average_precision_score(ys, ps)),
            "pr_auc_lift": float(average_precision_score(ys, ps) / ys.mean()),
            "roc_auc": float(roc_auc_score(ys, ps)),
            "brier": float(brier_score_loss(ys, ps)),
            "precision": tp / max(int(fs.sum()), 1),
            "recall": tp / max(int((ys == 1).sum()), 1),
            "flag_rate": float(fs.mean()),
            "mean_pred_vs_actual": float(ps.mean() - ys.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic-to-real distribution shift
# ---------------------------------------------------------------------------

# Merchant profiles a real deployment would plausibly meet. Our generator fixed
# one order mix; a real merchant has a different one, and the question is whether
# the model survives that. Each profile is a target distribution over a single
# observable dimension -- deliberately one at a time, so a degradation can be
# attributed to something specific rather than to a vague "different data".
POPULATION_PROFILES = {
    "generated mix (baseline)": {},
    "metro-heavy D2C": {"pincode_tier": {"metro": 0.60, "tier_1": 0.30,
                                         "tier_2": 0.08, "tier_3": 0.02}},
    "small-town heavy": {"pincode_tier": {"metro": 0.08, "tier_1": 0.22,
                                          "tier_2": 0.35, "tier_3": 0.35}},
    "fashion-led catalogue": {"category": {"fashion": 0.60, "footwear": 0.15,
                                           "beauty": 0.10, "accessories": 0.10,
                                           "home_kitchen": 0.03, "electronics": 0.02}},
    "electronics-led catalogue": {"category": {"electronics": 0.55, "home_kitchen": 0.20,
                                               "accessories": 0.10, "beauty": 0.05,
                                               "fashion": 0.07, "footwear": 0.03}},
    "COD-dominant (85%)": {"is_cod": {True: 0.85, False: 0.15}},
    "prepaid-led (35% COD)": {"is_cod": {True: 0.35, False: 0.65}},
}


def profile_weights(df: pd.DataFrame, profile: dict) -> np.ndarray:
    """Importance weights that reshape ``df`` to a target marginal distribution.

    ``w_i = target_share(group_i) / observed_share(group_i)``, normalised to mean 1.
    Reweighting rather than resampling keeps every row in play, so the estimate
    does not lose the tail of a small group entirely.
    """
    w = np.ones(len(df), dtype=float)
    for col, target in profile.items():
        observed = df[col].value_counts(normalize=True)
        ratio = df[col].map(
            {k: (v / observed.get(k, np.nan)) for k, v in target.items()}
        ).astype(float)
        # Groups the profile does not mention keep weight 0: the profile is a
        # complete distribution over that column, not a partial nudge.
        w *= ratio.fillna(0.0).to_numpy()
    total = w.sum()
    return w * (len(w) / total) if total > 0 else w


def weighted_metrics(y_true, y_prob, weights) -> dict:
    """PR-AUC / ROC-AUC / Brier under importance weights.

    ``average_precision_score`` and ``roc_auc_score`` both take sample_weight, so
    the reweighted population is scored directly rather than via a resample that
    would add sampling noise on top of the shift being measured.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = w > 0
    y_true, y_prob, w = y_true[keep], y_prob[keep], w[keep]
    if len(np.unique(y_true)) < 2:
        return {"pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan,
                "base_rate": np.nan, "effective_n": 0.0}
    base = float(np.average(y_true, weights=w))
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob, sample_weight=w)),
        "roc_auc": float(roc_auc_score(y_true, y_prob, sample_weight=w)),
        "brier": float(np.average((y_prob - y_true) ** 2, weights=w)),
        "base_rate": base,
        # Kish effective sample size: reweighting hard costs precision, and a
        # profile that leans on 200 real rows should not be read like 5,000.
        "effective_n": float(w.sum() ** 2 / np.sum(w**2)),
    }


def covariate_shift_auc(frame_a: pd.DataFrame, frame_b: pd.DataFrame,
                        columns, seed: int = 20260101) -> dict:
    """How separable are two feature distributions? A domain-classifier probe.

    Fit a model to predict which split a row came from. AUC near 0.5 means the
    two are indistinguishable; AUC near 1.0 means a model trained on one is
    extrapolating on the other. This is the standard cheap test for covariate
    shift, and it is the closest we can get to quantifying synthetic-to-real
    drift without real data -- here it measures TIME drift between splits, which
    is the only real shift we actually possess.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    a = frame_a[list(columns)].apply(pd.to_numeric, errors="coerce")
    b = frame_b[list(columns)].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]

    clf = HistGradientBoostingClassifier(max_iter=120, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=3, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p))

    return {"domain_auc": auc, "n_a": len(a), "n_b": len(b)}
