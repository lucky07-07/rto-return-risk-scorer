"""Feature engineering for the COD return-risk scorer.

Two things live here that the notebooks must not redefine:

* :func:`address_features` -- the address-quality read. The generator knows the
  *true* quality latent; the model only ever sees the rendered string, so these
  features are a deliberately noisy measurement of it.
* :class:`OutOfFoldTargetEncoder` -- pincode historical RTO rate. This is the
  single strongest feature and the single easiest way to leak the label, so it
  is encoded out-of-fold with expanding (time-ordered) folds and guarded by
  ``tests/test_no_leakage.py``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit

VOWELS = set("aeiou")

# Landmark cues an Indian address actually uses.
_LANDMARK_RE = re.compile(
    r"\b(?:near|nr|opp|opposite|behind|beside|next to|landmark|infront|in front)\b",
    re.IGNORECASE,
)
# "H No 42", "Flat 3B", "#42", "42/A", or a bare leading number.
_HOUSE_RE = re.compile(
    r"(?:\bh\.?\s?no\.?\b|\bhouse\s?no\b|\bflat\b|\bplot\b|\bdoor\s?no\b|#\s?\d|"
    r"^\s*\d+[a-z]?\b|\b\d+\s*/\s*\d+\b)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")
_CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]{4,}")


def _gibberish_score(text: str) -> float:
    """Fraction of alphabetic tokens that do not look like a real place word.

    A token counts as gibberish when it is long enough to judge and either has
    almost no vowels or contains a four-consonant run. Crude on purpose -- it
    has to run at checkout latency on a free-text field.
    """
    tokens = _TOKEN_RE.findall(text or "")
    tokens = [t for t in tokens if len(t) >= 4]
    if not tokens:
        return 0.0
    bad = 0
    for t in tokens:
        low = t.lower()
        vowel_ratio = sum(ch in VOWELS for ch in low) / len(low)
        if vowel_ratio < 0.25 or _CONSONANT_RUN_RE.search(low):
            bad += 1
    return bad / len(tokens)


def address_features(address: pd.Series) -> pd.DataFrame:
    """Extract the address-quality block from raw address strings."""
    s = address.fillna("").astype(str)
    out = pd.DataFrame(index=s.index)
    out["addr_token_count"] = s.str.findall(r"\S+").str.len().astype(int)
    out["addr_char_len"] = s.str.len().astype(int)
    out["addr_digit_count"] = s.str.count(r"\d").astype(int)
    out["addr_comma_count"] = s.str.count(",").astype(int)
    out["addr_has_house_number"] = s.str.contains(_HOUSE_RE).astype(int)
    out["addr_has_landmark"] = s.str.contains(_LANDMARK_RE).astype(int)
    out["addr_gibberish_score"] = s.map(_gibberish_score).astype(float)
    # A single composite, useful for EDA and for human-readable risk reasons.
    out["addr_quality_score"] = (
        0.30 * np.clip(out["addr_token_count"] / 12.0, 0, 1)
        + 0.25 * out["addr_has_house_number"]
        + 0.25 * out["addr_has_landmark"]
        + 0.20 * (1.0 - out["addr_gibberish_score"])
    ).round(4)
    return out


TIER_ORDINAL = {"metro": 0, "tier_1": 1, "tier_2": 2, "tier_3": 3}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything derivable from a single order row, with no cross-row fitting.

    Safe to apply to train, val and test alike -- it fits nothing.
    """
    out = df.copy()
    out = pd.concat([out, address_features(out["address_line"])], axis=1)
    out["tier_ordinal"] = out["pincode_tier"].map(TIER_ORDINAL).astype(int)
    out["log_order_value"] = np.log1p(out["order_value"].astype(float))
    out["discount_amount"] = (
        out["order_value"].astype(float) * out["discount_pct"].astype(float)
    ).round(2)
    out["log_account_age"] = np.log1p(out["account_age_days"].astype(float))
    out["is_first_order"] = (out["past_orders"] == 0).astype(int)
    out["pincode_prefix3"] = out["pincode"].astype(str).str[:3]
    return out


# Columns the model is allowed to see.
NUMERIC_FEATURES = [
    "order_value",
    "log_order_value",
    "discount_pct",
    "discount_amount",
    "delivery_days_est",
    "account_age_days",
    "log_account_age",
    "order_velocity_24h",
    "past_orders",
    "past_rto_count",
    "past_rto_rate",
    "tier_ordinal",
    "addr_token_count",
    "addr_char_len",
    "addr_digit_count",
    "addr_comma_count",
    "addr_has_house_number",
    "addr_has_landmark",
    "addr_gibberish_score",
    "addr_quality_score",
    "has_history",
    "is_first_order",
    "is_cod",
    "is_festive",
    "is_alternate_address",
]

CATEGORICAL_FEATURES = [
    "category",
    "order_value_band",
    "pincode_tier",
    "state",
]

# Fitted out-of-fold; added by OutOfFoldTargetEncoder, not by feature engineering.
TARGET_ENCODED_FEATURES = ["pincode_te", "pincode_prefix3_te", "city_te"]

TARGET = "rto"

# Never features. Identifiers, PII, post-outcome fields and generator latents.
FORBIDDEN_FEATURES = [
    "rto",
    "order_id",
    "customer_id",
    "customer_name",
    "phone",
    "address_line",
    "order_ts",
    "signup_date",
    "payment_mode",
    "pincode",
    "city",
    "district",
    "pincode_prefix3",
    "_pincode_rto_prior",
    "_reliability_z",
    "_address_quality",
    "_p_rto_true",
]


def feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET_ENCODED_FEATURES


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoding fitted out-of-fold on time-ordered data.

    ``fit_transform(X, y)`` returns encodings in which **no row contributed to
    its own value**: rows are cut into expanding time folds and each fold is
    encoded from strictly earlier folds only. Rows before the first validation
    fold fall back to the global prior.

    ``transform(X)`` -- used for validation and test -- applies the mapping
    fitted on the full training set, which is the correct, non-leaking choice
    because those rows are strictly later in time.

    The input **must be sorted chronologically**; the expanding folds are
    positional.

    Smoothing::

        te(k) = (sum_k + prior * m) / (count_k + m)

    which shrinks thin pincodes towards the global rate instead of letting a
    single order define a 0% or 100% pincode.
    """

    def __init__(self, columns=("pincode", "pincode_prefix3", "city"),
                 smoothing: float = 30.0, n_splits: int = 5):
        # scikit-learn's clone contract: store constructor arguments unmodified.
        self.columns = columns
        self.smoothing = smoothing
        self.n_splits = n_splits

    @property
    def _cols(self) -> list[str]:
        return list(self.columns)

    def _fit_maps(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        prior = float(np.mean(y))
        maps = {}
        for col in self._cols:
            grp = pd.DataFrame({"k": X[col].to_numpy(), "y": y}).groupby("k")["y"]
            agg = grp.agg(["sum", "count"])
            maps[col] = (agg["sum"] + prior * self.smoothing) / (
                agg["count"] + self.smoothing
            )
        return {"prior": prior, "maps": maps}

    def fit(self, X: pd.DataFrame, y=None):
        y = np.asarray(y, dtype=float)
        state = self._fit_maps(X, y)
        self.prior_ = state["prior"]
        self.maps_ = state["maps"]
        self.feature_names_out_ = [f"{c}_te" for c in self._cols]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self._cols:
            out[f"{col}_te"] = (
                X[col].map(self.maps_[col]).astype(float).fillna(self.prior_)
            )
        return out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        y = np.asarray(y, dtype=float)
        self.fit(X, y)

        out = pd.DataFrame(
            self.prior_, index=X.index,
            columns=[f"{c}_te" for c in self._cols], dtype=float,
        )
        n = len(X)
        n_splits = max(2, min(self.n_splits, n - 1))
        for past_idx, fold_idx in TimeSeriesSplit(n_splits=n_splits).split(
            np.arange(n)
        ):
            state = self._fit_maps(X.iloc[past_idx], y[past_idx])
            for col in self._cols:
                vals = (
                    X.iloc[fold_idx][col]
                    .map(state["maps"][col])
                    .astype(float)
                    .fillna(state["prior"])
                    .to_numpy()
                )
                out.iloc[fold_idx, out.columns.get_loc(f"{col}_te")] = vals
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_te" for c in self._cols], dtype=object)


# ---------------------------------------------------------------------------
# Domain interaction features (opt-in; baseline does not use them)
# ---------------------------------------------------------------------------

# Each of these encodes a specific claim about how COD returns actually happen,
# not an automatic cross-product of every column pair. A tree can in principle
# learn any of them from the base features; a logistic regression cannot learn
# any of them. Whether they earn their place is settled by ablation on
# validation in 05, not by assertion here.
INTERACTION_FEATURES = [
    "ix_cod_x_fashion",        # cash + the highest-return category: the canonical RTO
    "ix_cod_x_impulse_band",   # cash + the Rs500-1,000 band the published curve peaks in
    "ix_cod_x_discount",       # cash + a heavy discount: the impulse-buy signature
    "ix_cod_x_first_order",    # cash + no history to judge the customer on
    "ix_cod_x_tier",           # cash risk grows as you leave the metros
    "ix_cod_x_past_rto",       # a known returner, paying cash again
    "ix_cod_x_log_value",      # cash at high ticket: refusal at the door costs more
    "ix_addr_quality_x_tier",  # a vague address hurts far more outside a metro
    "ix_eta_x_tier",           # a long ETA into a small town: time to change your mind
    "ix_discount_x_fashion",   # discounted fashion: size-gamble buying
]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the domain interactions. Row-local, fits nothing, no leakage surface.

    ``past_rto_rate`` is NaN on a customer's first order, so its interaction is
    filled with 0 -- meaning "no prior-return signal", which is what
    ``is_first_order`` already says. The pair stays consistent.
    """
    out = df.copy()
    cod = out["is_cod"].astype(float)
    tier = out["tier_ordinal"].astype(float)

    out["ix_cod_x_fashion"] = cod * (out["category"] == "fashion").astype(float)
    out["ix_cod_x_impulse_band"] = cod * (out["order_value_band"] == "500_1000").astype(float)
    out["ix_cod_x_discount"] = cod * out["discount_pct"].astype(float)
    out["ix_cod_x_first_order"] = cod * out["is_first_order"].astype(float)
    out["ix_cod_x_tier"] = cod * tier
    out["ix_cod_x_past_rto"] = cod * out["past_rto_rate"].astype(float).fillna(0.0)
    out["ix_cod_x_log_value"] = cod * out["log_order_value"].astype(float)
    out["ix_addr_quality_x_tier"] = (1.0 - out["addr_quality_score"].astype(float)) * tier
    out["ix_eta_x_tier"] = out["delivery_days_est"].astype(float) * tier
    out["ix_discount_x_fashion"] = (
        out["discount_pct"].astype(float) * (out["category"] == "fashion").astype(float)
    )
    return out
