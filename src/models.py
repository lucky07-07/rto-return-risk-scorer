"""Preprocessing and the model zoo for the COD return-risk scorer.

The preprocessing pipeline is defined here, once, and imported by ``02`` (which
justifies it), ``03`` (which benchmarks against it), ``04`` (which tunes on top
of it) and ``05`` (which evaluates through it). A pipeline redefined in each
notebook is a pipeline that silently diverges.

The single most important property: the out-of-fold target encoder lives
*inside* the ``ColumnTransformer``. scikit-learn calls ``fit_transform`` on
sub-transformers during ``fit``, so training rows get out-of-fold encodings and
everything passed to ``transform`` later gets the train-fitted mapping. Doing
this outside the pipeline is how target encoding leaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OutOfFoldTargetEncoder,
)

# High-cardinality keys. They never enter the model raw -- only through the
# out-of-fold target encoder, which is why they are in FORBIDDEN_FEATURES.
TARGET_ENCODING_SOURCES = ["pincode", "pincode_prefix3", "city"]

# States with fewer orders than this are pooled into one "infrequent" level
# rather than getting a column that a single fold might never see.
RARE_CATEGORY_MIN_COUNT = 50


def build_preprocessor(
    scale: bool = False,
    te_smoothing: float = 30.0,
    te_n_splits: int = 5,
) -> ColumnTransformer:
    """Assemble the feature matrix from a processed split.

    Parameters
    ----------
    scale:
        Standardise numeric columns. Required by logistic regression and the
        MLP; irrelevant to trees, so it is off by default and switched on per
        model in the benchmark.
    te_smoothing, te_n_splits:
        Passed to :class:`~src.features.OutOfFoldTargetEncoder`.

    Notes
    -----
    ``past_rto_rate`` is NaN for a customer's first ever order. It is imputed
    with the training median and paired with ``has_history`` / ``is_first_order``,
    which are already columns, so "no history" stays distinguishable from
    "history happens to sit at the median".

    The input **must be sorted chronologically** -- the target encoder's folds
    are positional.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler() if scale else "passthrough"),
        ]
    )

    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=RARE_CATEGORY_MIN_COUNT,
        sparse_output=False,
    )

    target_encoded = Pipeline(
        [
            (
                "encode",
                OutOfFoldTargetEncoder(
                    columns=tuple(TARGET_ENCODING_SOURCES),
                    smoothing=te_smoothing,
                    n_splits=te_n_splits,
                ),
            ),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler() if scale else "passthrough"),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("te", target_encoded, TARGET_ENCODING_SOURCES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_input_columns() -> list[str]:
    """Every column the preprocessor reads, in the order it reads them."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET_ENCODING_SOURCES


def to_design_matrix(preprocessor, frame: pd.DataFrame, y=None, fit: bool = False):
    """Apply the preprocessor and return a named DataFrame.

    ``fit=True`` uses ``fit_transform``, which is what produces the *out-of-fold*
    target encodings for training rows. Anything evaluated later must go through
    ``fit=False``.
    """
    X = frame[model_input_columns()]
    if fit:
        arr = preprocessor.fit_transform(X, y)
    else:
        arr = preprocessor.transform(X)
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(np.asarray(arr), columns=names, index=frame.index)
