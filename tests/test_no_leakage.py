"""Leakage guard: pincode target encoding must be out-of-fold.

Pincode historical RTO rate is the strongest single feature in this problem and
the easiest way to accidentally hand the model its own label. These tests are
the reason the encoder exists as a class instead of three lines in a notebook.

Also guarded here:
* customer history features look strictly backwards;
* the chronological split never puts a later order before an earlier one;
* generator latents and post-outcome fields cannot reach the feature matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FORBIDDEN_FEATURES,
    OutOfFoldTargetEncoder,
    feature_columns,
)
from src.generate import add_customer_history, build_dataset, chronological_split

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def small():
    return build_dataset(
        evidence_path=ROOT / "config" / "evidence.yaml",
        pincode_path=ROOT / "data" / "external" / "india_pincodes.csv",
        n_orders=6_000,
        n_customers=2_000,
    )


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------


def test_encoder_cannot_reproduce_a_one_row_per_key_target():
    """The decisive case.

    Every key appears exactly once, so a leaky encoder returns the label itself
    and a correct out-of-fold encoder cannot do better than the prior. If this
    test ever passes with a near-perfect correlation, the encoding is leaking.
    """
    n = 400
    X = pd.DataFrame({"pincode": [f"{i:06d}" for i in range(n)]})
    y = np.tile([0, 1], n // 2)

    enc = OutOfFoldTargetEncoder(columns=("pincode",), smoothing=0.0)
    oof = enc.fit_transform(X, y)["pincode_te"].to_numpy()

    # A leaky in-fold encoding returns exactly y here. The out-of-fold encoder
    # never sees a key twice, so every row falls back to the prior.
    assert not np.allclose(oof, y), "encoder returned the labels themselves"
    if np.std(oof) > 0:
        assert abs(np.corrcoef(oof, y)[0, 1]) < 0.05, (
            "Out-of-fold encoding correlates with the row's own label -- leaking"
        )
    assert np.allclose(oof, oof[0]), (
        "with one row per key there is nothing to learn out of fold; "
        "any variation means past labels reached the current row"
    )


def test_fit_transform_differs_from_transform_on_training_rows():
    """``fit_transform`` must be out-of-fold; ``transform`` uses the full map."""
    rng = np.random.default_rng(0)
    n = 2_000
    X = pd.DataFrame({"pincode": rng.choice([f"{i:06d}" for i in range(50)], n)})
    y = rng.binomial(1, 0.3, n)

    enc = OutOfFoldTargetEncoder(columns=("pincode",))
    oof = enc.fit_transform(X, y)["pincode_te"].to_numpy()
    full = enc.transform(X)["pincode_te"].to_numpy()

    assert not np.allclose(oof, full), (
        "fit_transform returned the in-fold mapping -- that is the leak"
    )


def test_preprocessor_keeps_the_encoder_out_of_fold(small):
    """The encoder must stay *inside* the ColumnTransformer.

    Lifting it out -- encoding once, then feeding the result to a pipeline -- is
    the leak `02` measures at ~0.17 AUC on pincode. This asserts the placement,
    not just the class.
    """
    from src.features import add_engineered_features
    from src.models import build_preprocessor, to_design_matrix

    train = add_engineered_features(small["splits"]["train"]).reset_index(drop=True)
    pre = build_preprocessor()

    oof = to_design_matrix(pre, train, train["rto"], fit=True)
    in_fold = to_design_matrix(pre, train)

    te_cols = [c for c in oof.columns if c.endswith("_te")]
    assert te_cols, "target-encoded columns missing from the design matrix"
    for c in te_cols:
        assert not np.allclose(oof[c], in_fold[c]), (
            f"{c} is identical whether fitted or transformed -- the encoder is "
            "not running out of fold inside the pipeline"
        )


def test_preprocessor_drops_raw_high_cardinality_keys(small):
    """pincode / city / prefix must survive only as encodings, never raw."""
    from src.features import add_engineered_features
    from src.models import build_preprocessor, to_design_matrix

    train = add_engineered_features(small["splits"]["train"]).reset_index(drop=True)
    X = to_design_matrix(build_preprocessor(), train, train["rto"], fit=True)

    assert not {"pincode", "city", "pincode_prefix3"} & set(X.columns)
    assert X.notna().all().all(), "design matrix must be dense"


def test_encoding_of_unseen_key_falls_back_to_prior():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"pincode": rng.choice(["110001", "560001"], 500)})
    y = rng.binomial(1, 0.4, 500)
    enc = OutOfFoldTargetEncoder(columns=("pincode",)).fit(X, y)

    unseen = pd.DataFrame({"pincode": ["999999"]})
    assert enc.transform(unseen)["pincode_te"].iat[0] == pytest.approx(enc.prior_)


def test_encoder_is_fitted_on_train_only(small):
    """Validation and test rows must be transformed, never fitted on."""
    splits = small["splits"]
    enc = OutOfFoldTargetEncoder(columns=("pincode",))
    enc.fit_transform(splits["train"][["pincode"]], splits["train"]["rto"])
    prior_train_only = enc.prior_

    assert prior_train_only == pytest.approx(splits["train"]["rto"].mean())
    # The fitted prior must not have moved towards the test base rate.
    assert prior_train_only != pytest.approx(small["orders"]["rto"].mean(), abs=1e-12)


# ---------------------------------------------------------------------------
# Customer history
# ---------------------------------------------------------------------------


def test_customer_history_excludes_the_current_order():
    """``past_rto_count`` must never include the row's own outcome."""
    df = pd.DataFrame(
        {
            "order_id": ["ORD000000", "ORD000001", "ORD000002", "ORD000003"],
            "customer_id": ["C1", "C1", "C2", "C1"],
            "order_ts": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "rto": [1, 1, 0, 0],
        }
    )
    out = add_customer_history(df).set_index("order_id")

    assert out.loc["ORD000000", "past_orders"] == 0
    assert np.isnan(out.loc["ORD000000", "past_rto_rate"])
    assert out.loc["ORD000001", "past_rto_count"] == 1
    assert out.loc["ORD000001", "past_rto_rate"] == pytest.approx(1.0)
    assert out.loc["ORD000003", "past_orders"] == 2
    assert out.loc["ORD000003", "past_rto_count"] == 2
    # C2's single order must not see C1's history.
    assert out.loc["ORD000002", "past_orders"] == 0


def test_past_rto_count_never_exceeds_past_orders(small):
    o = small["orders"]
    assert (o["past_rto_count"] <= o["past_orders"]).all()
    assert (o["past_orders"] >= 0).all()


def test_first_order_has_no_history(small):
    o = small["orders"]
    first = o[o["past_orders"] == 0]
    assert first["past_rto_count"].eq(0).all()
    assert first["past_rto_rate"].isna().all()


def test_velocity_counts_only_earlier_orders(small):
    o = small["orders"]
    assert (o["order_velocity_24h"] <= o["past_orders"]).all()


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------


def test_split_is_strictly_chronological(small):
    s = small["splits"]
    assert s["train"]["order_ts"].max() <= s["val"]["order_ts"].min()
    assert s["val"]["order_ts"].max() <= s["test"]["order_ts"].min()


def test_split_sizes_are_70_10_20(small):
    s = small["splits"]
    n = sum(len(v) for v in s.values())
    assert len(s["train"]) / n == pytest.approx(0.70, abs=0.005)
    assert len(s["val"]) / n == pytest.approx(0.10, abs=0.005)
    assert len(s["test"]) / n == pytest.approx(0.20, abs=0.005)


def test_splits_do_not_overlap(small):
    s = small["splits"]
    ids = [set(v["order_id"]) for v in s.values()]
    assert ids[0].isdisjoint(ids[1])
    assert ids[1].isdisjoint(ids[2])
    assert ids[0].isdisjoint(ids[2])


def test_split_is_not_random(small):
    """A random split would interleave timestamps; a chronological one cannot."""
    s = small["splits"]
    assert s["train"]["order_ts"].is_monotonic_increasing
    boundary = s["train"]["order_ts"].max()
    assert (s["test"]["order_ts"] >= boundary).all()


# ---------------------------------------------------------------------------
# Feature hygiene
# ---------------------------------------------------------------------------


def test_no_forbidden_column_is_a_feature():
    """Latents, identifiers, PII and post-outcome fields stay out."""
    feats = set(feature_columns())
    assert feats.isdisjoint(set(FORBIDDEN_FEATURES)), feats & set(FORBIDDEN_FEATURES)


def test_generator_latents_are_prefixed_and_excluded(small):
    """Every latent column carries an underscore prefix and is forbidden."""
    latents = [c for c in small["orders"].columns if c.startswith("_")]
    assert latents, "expected the generator latents to be present for auditing"
    for c in latents:
        assert c in FORBIDDEN_FEATURES, f"{c} is a latent but not forbidden"
    assert set(latents).isdisjoint(set(feature_columns()))


def test_raw_order_log_has_no_true_probability(small):
    """``_p_rto_true`` is the Bayes-optimal score; it must never be shipped."""
    from src.generate import RAW_COLUMNS

    assert "_p_rto_true" not in RAW_COLUMNS
    assert not any(c.startswith("_") for c in RAW_COLUMNS)


# ---------------------------------------------------------------------------
# Domain interaction features
# ---------------------------------------------------------------------------


def test_interaction_features_are_row_local(small):
    """Interactions must be derivable from one row -- no cross-row fitting.

    If they were not, they would need the same out-of-fold treatment as target
    encoding, and the ablation in 05 would have been measuring a leak.
    """
    from src.features import INTERACTION_FEATURES, add_engineered_features
    from src.features import add_interaction_features

    df = add_engineered_features(small["splits"]["train"]).reset_index(drop=True)
    full = add_interaction_features(df)

    # Shuffling and re-slicing must not change any row's interaction values.
    shuffled = df.sample(frac=1.0, random_state=0)
    from_shuffled = add_interaction_features(shuffled).loc[df.index]
    for c in INTERACTION_FEATURES:
        assert np.allclose(full[c], from_shuffled[c]), f"{c} depends on row order"

    # And a single row alone must give the same answer as that row in context.
    one = add_interaction_features(df.iloc[[7]])
    for c in INTERACTION_FEATURES:
        assert np.isclose(one[c].iat[0], full[c].iat[7]), f"{c} needs neighbours"


def test_interaction_features_have_no_nan(small):
    """past_rto_rate is NaN on first orders; its interaction must not propagate it."""
    from src.features import INTERACTION_FEATURES, add_engineered_features
    from src.features import add_interaction_features

    df = add_interaction_features(
        add_engineered_features(small["splits"]["train"]).reset_index(drop=True)
    )
    assert df[INTERACTION_FEATURES].isna().sum().sum() == 0


def test_interactions_are_not_in_the_baseline_feature_set():
    """The shipped model does not use them; the ablation in 05 rejected them."""
    from src.features import INTERACTION_FEATURES
    from src.models import model_input_columns

    assert not set(INTERACTION_FEATURES) & set(model_input_columns())
    assert set(INTERACTION_FEATURES) <= set(model_input_columns(interactions=True))
