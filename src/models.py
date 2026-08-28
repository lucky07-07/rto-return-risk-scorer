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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OutOfFoldTargetEncoder,
)

SEED = 20260101

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


# ---------------------------------------------------------------------------
# The model zoo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One benchmark entry: an estimator plus how it wants its inputs."""

    key: str
    label: str
    estimator: object
    scale: bool = False
    family: str = "other"
    note: str = ""


def model_zoo(seed: int = SEED, n_jobs: int = -1) -> dict[str, Entry]:
    """Eleven entries: a majority-class baseline plus ten models.

    Hyperparameters are light, documented defaults -- this is a *benchmark*, not
    a tuning run. Tuning happens in ``04`` on the finalists only, so every model
    here gets the same courtesy: sane defaults, same seed, same features, same
    folds. Handing one model a tuned configuration and the rest their library
    defaults would make the leaderboard meaningless.

    No estimator is class-weighted. That is a deliberate reversal of the note in
    ``02`` and is justified, with numbers, in section 14 of ``03``: reweighting
    buys almost no ranking quality and measurably damages calibration, and this
    system's operating point is chosen on a rupee cost curve that needs
    probabilities to mean what they say.
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    entries = [
        Entry(
            "00_dummy", "Majority baseline", family="baseline",
            estimator=DummyClassifier(strategy="prior"),
            note="predicts the training base rate for every order",
        ),
        Entry(
            "01_logistic", "Logistic Regression", family="linear", scale=True,
            estimator=LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs", random_state=seed
            ),
            note="L2 regularised; the collinearity remedy identified in 02",
        ),
        Entry(
            "02_decision_tree", "Decision Tree", family="tree",
            estimator=DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=50, random_state=seed
            ),
            note="depth-capped; an unbounded tree memorises and tells us nothing",
        ),
        Entry(
            "03_random_forest", "Random Forest", family="bagging",
            estimator=RandomForestClassifier(
                n_estimators=400, min_samples_leaf=20, n_jobs=n_jobs,
                random_state=seed
            ),
        ),
        Entry(
            "04_extra_trees", "Extra Trees", family="bagging",
            estimator=ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=20, n_jobs=n_jobs,
                random_state=seed
            ),
        ),
        Entry(
            "05_gradient_boosting", "Gradient Boosting", family="boosting",
            estimator=GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3,
                subsample=0.9, random_state=seed
            ),
        ),
        Entry(
            "06_hist_gradient_boosting", "HistGradientBoosting", family="boosting",
            estimator=HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, early_stopping=False,
                random_state=seed
            ),
        ),
        Entry(
            "07_xgboost", "XGBoost", family="boosting",
            estimator=XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=5,
                subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                eval_metric="logloss", n_jobs=n_jobs, random_state=seed,
            ),
        ),
        Entry(
            "08_lightgbm", "LightGBM", family="boosting",
            estimator=LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=30, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.9, n_jobs=n_jobs, random_state=seed,
                verbose=-1,
            ),
        ),
        Entry(
            "09_catboost", "CatBoost", family="boosting",
            estimator=CatBoostClassifier(
                iterations=400, learning_rate=0.05, depth=6, random_seed=seed,
                verbose=0, allow_writing_files=False, thread_count=n_jobs,
            ),
        ),
        Entry(
            "10_mlp", "MLP", family="neural", scale=True,
            estimator=MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=400,
                early_stopping=True, n_iter_no_change=15, random_state=seed,
            ),
        ),
    ]
    return {e.key: e for e in entries}


def build_pipeline(entry: Entry, **preprocessor_kwargs) -> Pipeline:
    """Preprocessor + estimator, with the target encoder inside the pipeline."""
    return Pipeline(
        [
            ("pre", build_preprocessor(scale=entry.scale, **preprocessor_kwargs)),
            ("clf", entry.estimator),
        ]
    )
