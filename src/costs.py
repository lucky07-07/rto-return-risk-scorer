"""The rupee cost model and the three-tier decision policy.

A risk score is not a decision. This module turns scores into money, because the
threshold question -- "how sure do we need to be before we act?" -- has no answer
until both kinds of mistake are priced.

The two mistakes are not symmetric and neither is a constant:

**False negative** — an order that returns to origin and we did not flag it. The
merchant burns forward shipping, reverse shipping, packaging and handling. Modelled
as a flat rupee amount because it barely varies with basket value.

**False positive** — a good order we discouraged. This is *not* the order value, and
it is not even the full margin. A customer told "COD is unavailable, please pay
online" mostly pays online. The loss is the contribution margin on the fraction who
walk away instead:

    FP cost = order_value x margin_rate x (1 - prepaid_conversion)

Modelling the FP cost as the whole order value would overstate it several times over
and push the threshold far too high. Every constant here is *assumed*, lives in
``config/evidence.yaml``, and is sensitivity-tested in ``05``.

Defence-only: nothing in this module blocks, charges or contacts anyone. It computes
what a decision would cost. Acting on it is the merchant's call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ALLOW, FEE, BLOCK = "allow_cod", "charge_cod_fee", "disable_cod"
TIERS = [ALLOW, FEE, BLOCK]


@dataclass(frozen=True)
class CostParams:
    """Every rupee assumption in one place, so a reader can disagree precisely."""

    fn_cost_inr: float = 200.0        # shipping burned on an unflagged RTO
    margin_rate: float = 0.25         # contribution margin as a share of order value
    prepaid_conversion: float = 0.45  # blocked-but-good customers who pay online instead
    cod_fee_inr: float = 50.0         # the middle-tier fee
    fee_abandon_rate: float = 0.18    # share of fee-tier orders that do not proceed

    @classmethod
    def from_evidence(cls, evidence: dict) -> "CostParams":
        cm = evidence.get("cost_model", {})

        def get(name, default):
            node = cm.get(name)
            if isinstance(node, dict) and node.get("value") is not None:
                return float(node["value"])
            return default

        return cls(
            fn_cost_inr=get("fn_cost_inr", 200.0),
            margin_rate=get("margin_rate", 0.25),
            prepaid_conversion=get("prepaid_conversion", 0.45),
            cod_fee_inr=get("cod_fee_inr", 50.0),
            fee_abandon_rate=get("fee_abandon_rate", 0.18),
        )


def false_positive_cost(order_value, params: CostParams) -> np.ndarray:
    """Lost contribution margin on a good order we discouraged.

    Per order, not a constant: discouraging a Rs3,000 order costs more than
    discouraging a Rs300 one, and a flat FP cost would hide that entirely.
    """
    v = np.asarray(order_value, dtype=float)
    return v * params.margin_rate * (1.0 - params.prepaid_conversion)


def false_negative_cost(order_value, params: CostParams) -> np.ndarray:
    """Shipping burned on an RTO we failed to flag. Flat in basket value."""
    return np.full(np.shape(order_value), params.fn_cost_inr, dtype=float)


def binary_cost(y_true, y_prob, threshold: float, order_value,
                params: CostParams) -> dict:
    """Total rupee cost of a simple allow / disable-COD rule at one threshold."""
    y = np.asarray(y_true).astype(int)
    flagged = np.asarray(y_prob, dtype=float) >= threshold
    fp_c = false_positive_cost(order_value, params)
    fn_c = false_negative_cost(order_value, params)

    tp = flagged & (y == 1)
    fp = flagged & (y == 0)
    fn = (~flagged) & (y == 1)
    tn = (~flagged) & (y == 0)

    cost_fp = float(fp_c[fp].sum())
    cost_fn = float(fn_c[fn].sum())
    n = len(y)
    return {
        "threshold": float(threshold),
        "tp": int(tp.sum()), "fp": int(fp.sum()),
        "fn": int(fn.sum()), "tn": int(tn.sum()),
        "flag_rate": float(flagged.mean()),
        "precision": float(tp.sum() / max(flagged.sum(), 1)),
        "recall": float(tp.sum() / max((y == 1).sum(), 1)),
        "cost_false_positives_inr": cost_fp,
        "cost_false_negatives_inr": cost_fn,
        "total_cost_inr": cost_fp + cost_fn,
        "cost_per_order_inr": (cost_fp + cost_fn) / n,
    }


def threshold_sweep(y_true, y_prob, order_value, params: CostParams,
                    grid=None) -> pd.DataFrame:
    """Rupee cost across the whole threshold range. The cost minimum is the answer."""
    if grid is None:
        grid = np.round(np.arange(0.01, 1.00, 0.01), 4)
    return pd.DataFrame(
        [binary_cost(y_true, y_prob, t, order_value, params) for t in grid]
    )


def optimal_threshold(y_true, y_prob, order_value, params: CostParams,
                      grid=None) -> tuple[float, pd.DataFrame]:
    """The cost-minimising threshold, and the sweep it came from."""
    sweep = threshold_sweep(y_true, y_prob, order_value, params, grid)
    return float(sweep.loc[sweep.total_cost_inr.idxmin(), "threshold"]), sweep


def no_model_baselines(y_true, order_value, params: CostParams) -> dict:
    """What the policies that need no model cost, in rupees.

    Any model has to beat both of these to be worth deploying at all.
    """
    y = np.asarray(y_true).astype(int)
    fp_c = false_positive_cost(order_value, params)
    fn_c = false_negative_cost(order_value, params)
    return {
        "allow_everything": float(fn_c[y == 1].sum()),
        "block_everything": float(fp_c[y == 0].sum()),
    }


# ---------------------------------------------------------------------------
# Three-tier policy
# ---------------------------------------------------------------------------


def tier_costs(y_true, order_value, params: CostParams) -> pd.DataFrame:
    """Rupee outcome of each tier, per order, for both possible true labels.

    ``allow``  we bear the full shipping loss when the order returns.
    ``fee``    a fraction ``fee_abandon_rate`` of orders do not proceed. Those we
               lose margin on if they were good, and avoid the loss on if they were
               not. The rest proceed, and the fee is collected on delivery, which
               offsets part of the loss. The fee is a *price on risk*, not a
               punishment: it keeps the sale.
    ``block``  COD is withdrawn and prepaid offered. No RTO loss either way; we lose
               margin only on the good customers who walk instead of paying online.
    """
    y = np.asarray(y_true).astype(int)
    v = np.asarray(order_value, dtype=float)
    fp_c = false_positive_cost(v, params)
    fn_c = false_negative_cost(v, params)
    a, f = params.fee_abandon_rate, params.cod_fee_inr

    cost = pd.DataFrame(index=np.arange(len(y)))
    cost[ALLOW] = np.where(y == 1, fn_c, 0.0)
    cost[FEE] = np.where(
        y == 1,
        (1 - a) * fn_c,                        # abandoned bad orders cost nothing
        a * fp_c - (1 - a) * f,                # fee revenue on delivered good orders
    )
    cost[BLOCK] = np.where(y == 1, 0.0, fp_c)
    return cost


def apply_tiers(y_prob, low_cut: float, high_cut: float) -> np.ndarray:
    """Map scores to allow / charge-a-fee / disable-COD."""
    p = np.asarray(y_prob, dtype=float)
    return np.select([p < low_cut, p < high_cut], [ALLOW, FEE], default=BLOCK)


def three_tier_cost(y_true, y_prob, order_value, low_cut: float, high_cut: float,
                    params: CostParams) -> dict:
    """Total cost of a three-tier policy at a given pair of cut points."""
    tiers = apply_tiers(y_prob, low_cut, high_cut)
    costs = tier_costs(y_true, order_value, params)
    total = sum(float(costs.loc[tiers == t, t].sum()) for t in TIERS)
    y = np.asarray(y_true).astype(int)
    out = {
        "low_cut": float(low_cut), "high_cut": float(high_cut),
        "total_cost_inr": total, "cost_per_order_inr": total / len(y),
    }
    for t in TIERS:
        m = tiers == t
        out[f"n_{t}"] = int(m.sum())
        out[f"rto_rate_{t}"] = float(y[m].mean()) if m.any() else np.nan
    return out


def optimise_tiers(y_true, y_prob, order_value, params: CostParams,
                   grid=None) -> tuple[float, float, pd.DataFrame]:
    """Grid-search the two cut points on the rupee cost surface.

    Chosen on **validation** and then applied unchanged to test -- picking cut
    points on the test set would make the reported cost an in-sample number.
    """
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.025), 4)
    rows = []
    for lo in grid:
        for hi in grid:
            if hi <= lo:
                continue
            rows.append(three_tier_cost(y_true, y_prob, order_value, lo, hi, params))
    surface = pd.DataFrame(rows)
    best = surface.loc[surface.total_cost_inr.idxmin()]
    return float(best.low_cut), float(best.high_cut), surface


def sensitivity(y_true, y_prob, order_value, base: CostParams,
                fn_costs=(150, 175, 200, 225, 250),
                margin_rates=(0.15, 0.20, 0.25, 0.30, 0.35)) -> pd.DataFrame:
    """Re-derive the optimal threshold across the assumed-cost ranges.

    ``fn_cost_inr`` is marked *assumed* in ``config/evidence.yaml`` with a stated
    150-250 range, so the honest report is not one threshold but how much the
    threshold moves when the assumption does.
    """
    rows = []
    for fn in fn_costs:
        for mr in margin_rates:
            p = CostParams(
                fn_cost_inr=float(fn), margin_rate=float(mr),
                prepaid_conversion=base.prepaid_conversion,
                cod_fee_inr=base.cod_fee_inr,
                fee_abandon_rate=base.fee_abandon_rate,
            )
            thr, sweep = optimal_threshold(y_true, y_prob, order_value, p)
            row = sweep.loc[sweep.threshold == thr].iloc[0]
            rows.append({
                "fn_cost_inr": fn, "margin_rate": mr,
                "optimal_threshold": thr,
                "cost_per_order_inr": row.cost_per_order_inr,
                "precision": row.precision, "recall": row.recall,
                "flag_rate": row.flag_rate,
            })
    return pd.DataFrame(rows)
