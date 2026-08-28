"""Scoring service logic, shared by the API and the Streamlit demo.

The whole point of this module is that there is exactly **one** scoring path. The
API and the UI both call :func:`score_order`; neither reimplements feature
engineering, neither re-derives a threshold, and neither can drift from what
``05_final_evaluation.ipynb`` actually shipped.

Everything operational is read from ``models/final_model.joblib`` — the fitted
pipeline (preprocessor *and* CatBoost together), the frozen threshold, the frozen
tier cut points and the cost parameters. Nothing is refitted, retuned or
recomputed at demo time.

Defence-only: this scores an order and returns reasons. It has no code path that
captures a payment, blocks an account or contacts anyone.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.costs import ALLOW, BLOCK, FEE, CostParams, apply_tiers
from src.evaluate import risk_reasons
from src.features import add_engineered_features
from src.generate import FESTIVE_WINDOWS, TIER_ETA_DAYS, load_pincode_skeleton, value_band
from src.models import model_input_columns

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_model.joblib"
PINCODE_PATH = ROOT / "data" / "external" / "india_pincodes.csv"

# The demo-facing names for the three tiers. `action` keeps the project's own
# vocabulary so anything already reading it is unaffected; `action_label` is the
# judge-facing one.
TIER_TO_LABEL = {ALLOW: "ALLOW", FEE: "REVIEW", BLOCK: "BLOCK"}
TIER_TO_BAND = {ALLOW: "low", FEE: "medium", BLOCK: "high"}


@dataclass(frozen=True)
class CustomerHistory:
    """What the merchant knows about this customer before the order ships.

    A checkout payload does not carry these — they come from the merchant's own
    order store. An unknown customer scores as a genuine first-time customer,
    which is what ``past_rto_rate = None`` means to the model (`02` measured what
    that dilution costs: 0.029 AUC).
    """

    past_orders: int = 0
    past_rto_count: int = 0
    account_age_days: float = 0.0
    order_velocity_24h: int = 0

    def as_features(self) -> dict[str, Any]:
        has_history = int(self.past_orders > 0)
        return {
            "past_orders": int(self.past_orders),
            "past_rto_count": int(self.past_rto_count),
            "past_rto_rate": (
                self.past_rto_count / self.past_orders if self.past_orders > 0
                else np.nan
            ),
            "has_history": has_history,
            "is_first_order": int(not has_history),
            "account_age_days": float(self.account_age_days),
            "order_velocity_24h": int(self.order_velocity_24h),
        }


@dataclass
class ScoringService:
    """Holds the loaded artifact. Constructed once, at process start."""

    bundle: dict
    pincodes: pd.DataFrame
    model_sha256: str

    @property
    def pipeline(self):
        return self.bundle["pipeline"]

    @property
    def threshold(self) -> float:
        return float(self.bundle["threshold"])

    @property
    def low_cut(self) -> float:
        return float(self.bundle["low_cut"])

    @property
    def high_cut(self) -> float:
        return float(self.bundle["high_cut"])

    @property
    def costs(self) -> CostParams:
        return CostParams(**self.bundle["cost_params"])

    def describe(self) -> dict:
        """What ``/health`` reports, so a judge can confirm this is the real model."""
        return {
            "model_key": self.bundle["model_key"],
            "estimator": type(self.pipeline.named_steps["clf"]).__name__,
            "model_sha256": self.model_sha256,
            "seed": self.bundle["seed"],
            "frozen_threshold": self.threshold,
            "frozen_tier_cuts": {"low_cut": self.low_cut, "high_cut": self.high_cut},
            "cost_params_inr": self.bundle["cost_params"],
            "train_base_rate": self.bundle["train_base_rate"],
            "challenger_accepted": self.bundle.get("challenger_accepted"),
            "n_feature_columns": len(self.bundle["feature_columns"]),
            "source": "models/final_model.joblib (written by 05_final_evaluation.ipynb)",
        }


@functools.lru_cache(maxsize=1)
def get_service() -> ScoringService:
    """Load the shipped artifact once per process.

    ``lru_cache`` rather than a module-level global so that importing this module
    is cheap and the 40k-row pincode directory is only read if something actually
    scores an order.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run 05_final_evaluation.ipynb to produce it; "
            "the demo deliberately refuses to fabricate a model."
        )
    bundle = joblib.load(MODEL_PATH)
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    return ScoringService(
        bundle=bundle,
        pincodes=load_pincode_skeleton(PINCODE_PATH).set_index("pincode"),
        model_sha256=digest,
    )


class UnknownPincode(ValueError):
    """Raised when a pincode is absent from the India Post directory."""


def _lookup_pincode(service: ScoringService, pincode: str) -> dict[str, str]:
    """City, district, state and tier for a real Indian pincode.

    Derived rather than accepted from the caller: a checkout form knows the
    pincode, and asking a judge to type a matching state invites a mismatch the
    model would silently score.
    """
    try:
        row = service.pincodes.loc[str(pincode).strip()]
    except KeyError as exc:
        raise UnknownPincode(
            f"pincode {pincode!r} is not in the India Post directory"
        ) from exc
    if isinstance(row, pd.DataFrame):        # a pincode spanning several rows
        row = row.iloc[0]
    return {
        "city": row["city"], "district": row["district"],
        "state": row["state"], "pincode_tier": row["tier"],
    }


def _is_festive(ts: datetime) -> bool:
    """Whether the order falls inside one of the generator's sale windows."""
    t = pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts)
    return any(pd.Timestamp(s) <= t <= pd.Timestamp(e) for s, e, _, _ in FESTIVE_WINDOWS)


def build_scoring_frame(payload: dict, service: ScoringService | None = None
                        ) -> pd.DataFrame:
    """Turn one order payload into the exact frame the fitted pipeline expects.

    Uses ``src.features.add_engineered_features`` — the same function `01` used to
    build the training data — so the address block, log transforms and tier
    ordinal are computed by identical code rather than by a reimplementation that
    could drift.
    """
    service = service or get_service()
    geo = _lookup_pincode(service, payload["pincode"])

    ts = payload.get("order_ts") or datetime.now(timezone.utc)
    history = payload.get("customer_history") or {}
    hist = CustomerHistory(**history).as_features()

    order_value = float(payload["order_value"])
    tier = geo["pincode_tier"]

    row: dict[str, Any] = {
        "order_id": payload.get("order_id", "ord_adhoc"),
        "customer_id": payload.get("customer_id", "CUST_UNKNOWN"),
        "address_line": payload["address_line"],
        "pincode": str(payload["pincode"]).strip(),
        **geo,
        "category": payload["category"],
        "order_value": order_value,
        "order_value_band": str(value_band(np.array([order_value]))[0]),
        "discount_pct": float(payload.get("discount_pct", 0.0)),
        "is_cod": bool(payload["is_cod"]),
        "payment_mode": "COD" if payload["is_cod"] else "PREPAID",
        "is_festive": bool(payload.get("is_festive", _is_festive(ts))),
        "is_alternate_address": bool(payload.get("is_alternate_address", False)),
        # A courier ETA is known at checkout. If the caller does not supply one,
        # fall back to the tier baseline the generator used.
        "delivery_days_est": float(
            payload.get("delivery_days_est") or TIER_ETA_DAYS[tier]
        ),
        **hist,
    }
    frame = add_engineered_features(pd.DataFrame([row]))
    return frame


def expected_loss_if_shipped(probability: float, order_value: float,
                             costs: CostParams) -> float:
    """Expected rupee loss from shipping this order on COD.

    ``P(RTO) x fn_cost`` — the probability-weighted shipping burn. Deliberately
    *not* the order value: the merchant does not lose the sale price on an RTO,
    they lose forward shipping, reverse shipping, packaging and handling, which
    `config/evidence.yaml` prices flat at Rs200 (range 150-250, sensitivity-tested
    in `05`).
    """
    return round(float(probability) * costs.fn_cost_inr, 2)


def _shap_reasons(service: ScoringService, frame: pd.DataFrame,
                  top_k: int = 3, direction: str = "up") -> list[str]:
    """Human-readable risk reasons for a single order.

    Dispatches by model family exactly as `05` does — TreeExplainer for CatBoost,
    which is exact and fast enough to run per request — then hands the values to
    ``src.evaluate.risk_reasons`` for the plain-English rendering, including its
    guard against verbalising a one-hot column the order is not actually in.
    """
    import shap

    from src.models import to_design_matrix

    pre = service.pipeline.named_steps["pre"]
    clf = service.pipeline.named_steps["clf"]
    X = to_design_matrix(pre, frame)

    try:
        sv = np.asarray(shap.TreeExplainer(clf).shap_values(X))
    except Exception:
        return []                      # explanations are a bonus, never a 500
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    sv = np.atleast_2d(sv)
    return risk_reasons(sv[0], list(X.columns), feature_values=X.iloc[0].to_numpy(),
                        top_k=top_k, direction=direction)


def score_order(payload: dict, service: ScoringService | None = None,
                explain: bool = True) -> dict:
    """Score one order and return the full decision record.

    The single entry point for both the API and the UI. Returns the project's own
    ``action`` vocabulary *and* the demo-facing ``action_label``, so nothing that
    already reads ``action`` breaks.
    """
    service = service or get_service()
    frame = build_scoring_frame(payload, service)

    cols = [c for c in model_input_columns() if c in frame.columns]
    probability = float(service.pipeline.predict_proba(frame[cols])[:, 1][0])

    tier = str(apply_tiers([probability], service.low_cut, service.high_cut)[0])
    costs = service.costs

    return {
        "order_id": frame["order_id"].iat[0],
        "rto_probability": round(probability, 4),
        "band": TIER_TO_BAND[tier],
        "action": tier,
        "action_label": TIER_TO_LABEL[tier],
        # On an ALLOW the honest explanation is why the order was CLEARED. Listing
        # three faint upward nudges on a 0.8% score reads as a contradiction.
        "reasons": (
            _shap_reasons(service, frame,
                          direction="down" if tier == ALLOW else "up")
            if explain else []
        ),
        "expected_loss_if_shipped_inr": expected_loss_if_shipped(
            probability, float(frame["order_value"].iat[0]), costs
        ),
        "flagged_at_binary_threshold": probability >= service.threshold,
        "resolved": {
            "city": frame["city"].iat[0],
            "state": frame["state"].iat[0],
            "pincode_tier": frame["pincode_tier"].iat[0],
            "order_value_band": frame["order_value_band"].iat[0],
        },
    }


# ---------------------------------------------------------------------------
# Sample orders for the demo
# ---------------------------------------------------------------------------

# PRE_REGISTRATION.md commits to the test set being opened once, in 05. Pulling
# demo rows from it would be a second read for presentation convenience, which is
# exactly the kind of small erosion this project spends its effort avoiding. The
# samples therefore come from VALIDATION, and every response says so.
SAMPLE_SOURCE = "data/processed/val.parquet"
SAMPLE_SOURCE_NOTE = (
    "Sampled from the VALIDATION split, not the sealed test set. "
    "PRE_REGISTRATION.md commits to the test set being read only in "
    "05_final_evaluation.ipynb; using it to populate a demo form would be a "
    "second read for presentation convenience."
)


def sample_orders(n_per_tier: int = 2, seed: int = 20260101,
                  service: ScoringService | None = None) -> dict:
    """Real validation orders spanning ALLOW, REVIEW and BLOCK.

    Scored with the shipped model so the tiers are the model's own, then a few of
    each are returned as ready-made payloads — a judge should not have to invent
    a realistic Indian address to see all three outcomes.
    """
    service = service or get_service()
    val = pd.read_parquet(ROOT / SAMPLE_SOURCE)

    cols = [c for c in model_input_columns() if c in val.columns]
    p = service.pipeline.predict_proba(val[cols])[:, 1]
    tiers = apply_tiers(p, service.low_cut, service.high_cut)

    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    for tier in (ALLOW, FEE, BLOCK):
        idx = np.flatnonzero((tiers == tier) & val["is_cod"].to_numpy())
        if idx.size == 0:                      # fall back to any order in the tier
            idx = np.flatnonzero(tiers == tier)
        take = rng.choice(idx, size=min(n_per_tier, idx.size), replace=False)
        for i in sorted(take):
            r = val.iloc[int(i)]
            picked.append({
                "label": TIER_TO_LABEL[tier],
                "expected_action": tier,
                "model_probability": round(float(p[i]), 4),
                "payload": {
                    "order_id": r["order_id"],
                    "customer_id": r["customer_id"],
                    "pincode": r["pincode"],
                    "address_line": r["address_line"],
                    "category": r["category"],
                    "order_value": float(r["order_value"]),
                    "discount_pct": float(r["discount_pct"]),
                    "is_cod": bool(r["is_cod"]),
                    "is_festive": bool(r["is_festive"]),
                    "is_alternate_address": bool(r["is_alternate_address"]),
                    "delivery_days_est": float(r["delivery_days_est"]),
                    "customer_history": {
                        "past_orders": int(r["past_orders"]),
                        "past_rto_count": int(r["past_rto_count"]),
                        "account_age_days": float(r["account_age_days"]),
                        "order_velocity_24h": int(r["order_velocity_24h"]),
                    },
                },
            })
    return {"source": SAMPLE_SOURCE, "note": SAMPLE_SOURCE_NOTE, "orders": picked}
