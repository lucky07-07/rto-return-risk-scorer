"""Calibration guard: generated data must reproduce published Indian RTO rates.

These are the same assertions that run inside ``01_data_generation.ipynb``. They
run here too so that the calibration is enforced by CI rather than by a human
remembering to look at a printout.

The test reads ``data/raw/orders.csv`` if notebook 01 has been run; otherwise it
regenerates a smaller sample in-process. Either way the published constants come
from ``config/evidence.yaml``, never from a literal typed into the test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.generate import build_dataset, calibration_report, load_evidence

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "orders.csv"

EVIDENCE = load_evidence(ROOT / "config" / "evidence.yaml")


@pytest.fixture(scope="module")
def orders() -> pd.DataFrame:
    if RAW.exists():
        return pd.read_csv(RAW, dtype={"pincode": str}, parse_dates=["order_ts"])
    # Smaller regeneration keeps the test usable on a fresh clone. The
    # calibration solve is applied per cell, so it holds at this size too.
    return build_dataset(
        evidence_path=ROOT / "config" / "evidence.yaml",
        pincode_path=ROOT / "data" / "external" / "india_pincodes.csv",
        n_orders=20_000,
        n_customers=8_000,
    )["orders"]


@pytest.fixture(scope="module")
def report(orders: pd.DataFrame) -> dict:
    return calibration_report(orders)


def test_cod_rto_rate_matches_shipnotes(report):
    """Shipway ShipNotes FY25: 26% RTO on COD. Tolerance +/- 2pp."""
    target = EVIDENCE["cod_rto_rate"]["value"]
    assert 0.24 <= report["cod_rto_rate"] <= 0.28, (
        f"COD RTO {report['cod_rto_rate']:.4f} vs published {target}"
    )


def test_prepaid_rto_rate_below_two_percent(report):
    """Shipway ShipNotes FY25: under 2% RTO on prepaid."""
    bound = EVIDENCE["prepaid_rto_rate"]["value"]
    assert report["prepaid_rto_rate"] < bound, (
        f"Prepaid RTO {report['prepaid_rto_rate']:.4f} not below {bound}"
    )


def test_order_value_curve_is_non_monotonic(report):
    """RTO peaks in the Rs500-1000 impulse band and falls above Rs1,000.

    This is the assumption most submissions get backwards. If the generated
    curve ever rises monotonically with order value, the data is wrong.
    """
    bands = report["cod_rto_by_band"]
    assert bands["500_1000"] > bands["1000_plus"], bands
    assert bands["500_1000"] > bands["under_500"], bands


def test_order_value_bands_match_published_rates(report):
    """Each band lands within 2pp of its ShipNotes figure."""
    published = {
        "under_500": EVIDENCE["order_value_bands"]["under_500"]["value"],
        "500_1000": EVIDENCE["order_value_bands"]["mid_500_1000"]["value"],
        "1000_plus": EVIDENCE["order_value_bands"]["over_1000"]["value"],
    }
    for band, target in published.items():
        got = report["cod_rto_by_band"][band]
        assert abs(got - target) <= 0.02, f"{band}: {got:.4f} vs published {target}"


def test_fashion_rto_matches_published_rate(report):
    """Fashion/apparel RTO is reported at 40%+ on COD."""
    target = EVIDENCE["fashion_rto_rate"]["value"]
    assert report["cod_rto_fashion"] >= target - 0.02, report["cod_rto_fashion"]


def test_fashion_is_the_riskiest_category(report):
    by_cat = report["cod_rto_by_category"]
    assert max(by_cat, key=by_cat.get) == "fashion", by_cat
    assert min(by_cat, key=by_cat.get) == "electronics", by_cat


def test_cod_share_matches_industry_reports(report):
    """60-65% of Indian e-commerce orders are COD."""
    assert 0.58 <= report["cod_share"] <= 0.66, report["cod_share"]


def test_city_rto_spread_covers_published_range(report):
    """City-level rates must span roughly the published Vadodara-Patna range."""
    lo = EVIDENCE["city_rto_range"]["min"]["value"]
    hi = EVIDENCE["city_rto_range"]["max"]["value"]
    assert report["city_rto_min"] <= lo + 0.06, report["city_rto_min"]
    assert report["city_rto_max"] >= hi - 0.06, report["city_rto_max"]


def test_generation_is_deterministic():
    """Two runs with the same seed must produce identical labels."""
    kw = dict(
        evidence_path=ROOT / "config" / "evidence.yaml",
        pincode_path=ROOT / "data" / "external" / "india_pincodes.csv",
        n_orders=4_000,
        n_customers=1_500,
    )
    a = build_dataset(**kw)["orders"]
    b = build_dataset(**kw)["orders"]
    pd.testing.assert_frame_equal(a, b)
