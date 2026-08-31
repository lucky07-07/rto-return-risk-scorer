"""Descriptive threshold sweep, for reporting only.

Why this is not a threshold search
----------------------------------
The operating point is **already frozen**: 0.37, chosen on validation in `05`,
recorded in `reports/results/05_final_metrics.json` under
`frozen_operating_point`. This script does not choose anything. It re-reads the
predictions `05` already wrote and tabulates precision, recall, F1 and rupee
cost across the threshold range, so a reader can see the trade-off curve
instead of a single row.

Nothing here selects a threshold, and nothing downstream consumes the output.
If it did, it would be selecting on test, which the pre-registration forbids.

Run from the repository root:

    python scripts/threshold_sweep.py

Writes `reports/results/05_threshold_sweep.csv`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.costs import CostParams, threshold_sweep  # noqa: E402

RESULTS = ROOT / "reports" / "results"
FINAL = json.loads((RESULTS / "05_final_metrics.json").read_text())
PARAMS = CostParams(**FINAL["cost"]["params"])
FROZEN = FINAL["frozen_operating_point"]["threshold"]

# The grid a reader would ask about, plus the frozen point itself so the table
# always contains the row the rest of the project reports.
GRID = np.round(np.unique(np.concatenate([np.arange(0.05, 0.96, 0.05), [FROZEN]])), 4)


def load(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground truth, predicted probability and order value for one split."""
    frame = pd.read_parquet(ROOT / "data" / "processed" / f"{split}.parquet")
    if split == "test":
        preds = pd.read_parquet(RESULTS / "05_test_predictions.parquet")
        p = preds.set_index("order_id")["p_rto"]
    else:
        preds = pd.read_parquet(RESULTS / "04_tuned_val_predictions.parquet")
        p = preds.set_index("order_id")["09_catboost__flaml"]
    frame = frame.set_index("order_id").loc[p.index]
    return frame["rto"].to_numpy(), p.to_numpy(), frame["order_value"].to_numpy()


rows = []
for split in ("validation", "test"):
    y, p, value = load("val" if split == "validation" else "test")
    sweep = threshold_sweep(y, p, value, PARAMS, grid=GRID)
    denom = sweep["precision"] + sweep["recall"]
    sweep["f1"] = np.where(denom > 0, 2 * sweep["precision"] * sweep["recall"] / denom, 0.0)
    sweep.insert(0, "split", split)
    sweep["is_frozen_operating_point"] = np.isclose(sweep["threshold"], FROZEN)
    rows.append(sweep)

out = pd.concat(rows, ignore_index=True)
cols = ["split", "threshold", "is_frozen_operating_point", "tp", "fp", "fn", "tn",
        "flag_rate", "precision", "recall", "f1", "cost_per_order_inr", "total_cost_inr"]
out = out[cols]
out.to_csv(RESULTS / "05_threshold_sweep.csv", index=False)

# ---- guard: the frozen row must reproduce what 05 already committed ---------
frozen_test = out[(out.split == "test") & out.is_frozen_operating_point].iloc[0]
committed = FINAL["test"]
for name, got, want in (
    ("precision", frozen_test.precision, committed["precision"]),
    ("recall", frozen_test.recall, committed["recall"]),
    ("f1", frozen_test.f1, committed["f1"]),
    ("flag_rate", frozen_test.flag_rate, committed["flag_rate"]),
):
    assert abs(got - want) < 1e-9, f"{name}: sweep {got} != committed {want}"
cm = committed["confusion_matrix"]
assert (int(frozen_test.tp), int(frozen_test.fp), int(frozen_test.fn), int(frozen_test.tn)) == \
       (cm["tp"], cm["fp"], cm["fn"], cm["tn"]), "confusion matrix disagrees with 05"

print(f"frozen threshold {FROZEN} reproduces 05_final_metrics.json exactly\n")
show = ["threshold", "precision", "recall", "f1", "flag_rate", "cost_per_order_inr"]
for split in ("validation", "test"):
    print(split.upper())
    print(out[out.split == split][show].to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))
    print()
print(f"wrote {RESULTS / '05_threshold_sweep.csv'}")
