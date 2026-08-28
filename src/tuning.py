"""Hyperparameter search: one space, two strategies.

``04`` puts **Optuna** (multivariate TPE + Hyperband pruning) head to head against
**FLAML** (cost-frugal BlendSearch/CFO). For that comparison to mean anything the
two have to be given exactly the same problem, so the search space is declared
once here, library-neutrally, and translated into each library's dialect.

What is held identical:

* the search space, parameter for parameter and bound for bound;
* the objective -- mean PR-AUC over the same expanding-window folds of train;
* the seed;
* the **wall-clock budget**. Optuna is trial-native and FLAML is budget-native, so
  a fixed trial count would flatter Optuna and a fixed budget is the fair meeting
  point. Trials actually completed is then a *result*, not a control.

Validation is never touched during search. It is scored once per tuned
configuration, after the search closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEED = 20260101


# ---------------------------------------------------------------------------
# Library-neutral parameter description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """One tunable hyperparameter, described once for both search libraries."""

    name: str
    kind: str                      # "float" | "int" | "cat"
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    choices: tuple = ()

    def optuna_suggest(self, trial):
        if self.kind == "float":
            return trial.suggest_float(self.name, self.low, self.high, log=self.log)
        if self.kind == "int":
            return trial.suggest_int(self.name, self.low, self.high, log=self.log)
        return trial.suggest_categorical(self.name, list(self.choices))

    def flaml_domain(self):
        from flaml import tune

        if self.kind == "float":
            return (tune.loguniform(self.low, self.high) if self.log
                    else tune.uniform(self.low, self.high))
        if self.kind == "int":
            return (tune.lograndint(self.low, self.high + 1) if self.log
                    else tune.randint(self.low, self.high + 1))
        return tune.choice(list(self.choices))


# ---------------------------------------------------------------------------
# The spaces
# ---------------------------------------------------------------------------

# Bounds are deliberately generous but not absurd: wide enough that the two
# search strategies have somewhere to go, narrow enough that neither burns its
# budget in a region no sane practitioner would ship.
SPACES: dict[str, list[Param]] = {
    "01_logistic": [
        Param("C", "float", 1e-3, 1e2, log=True),
        Param("class_weight", "cat", choices=(None,)),  # pinned: see 03 section 14
    ],
    "02_decision_tree": [
        Param("max_depth", "int", 2, 20),
        Param("min_samples_leaf", "int", 5, 500, log=True),
        Param("min_samples_split", "int", 2, 200, log=True),
        Param("ccp_alpha", "float", 1e-6, 1e-2, log=True),
    ],
    "03_random_forest": [
        Param("n_estimators", "int", 100, 800),
        Param("max_depth", "int", 3, 30),
        Param("min_samples_leaf", "int", 1, 100, log=True),
        Param("max_features", "float", 0.1, 1.0),
    ],
    "04_extra_trees": [
        Param("n_estimators", "int", 100, 800),
        Param("max_depth", "int", 3, 30),
        Param("min_samples_leaf", "int", 1, 100, log=True),
        Param("max_features", "float", 0.1, 1.0),
    ],
    "05_gradient_boosting": [
        Param("n_estimators", "int", 50, 500),
        Param("learning_rate", "float", 0.01, 0.3, log=True),
        Param("max_depth", "int", 2, 6),
        Param("subsample", "float", 0.5, 1.0),
        Param("min_samples_leaf", "int", 5, 200, log=True),
    ],
    "06_hist_gradient_boosting": [
        Param("max_iter", "int", 50, 800),
        Param("learning_rate", "float", 0.01, 0.3, log=True),
        Param("max_leaf_nodes", "int", 8, 128, log=True),
        Param("min_samples_leaf", "int", 5, 200, log=True),
        Param("l2_regularization", "float", 1e-6, 10.0, log=True),
        Param("max_features", "float", 0.4, 1.0),
    ],
    "07_xgboost": [
        Param("n_estimators", "int", 100, 900),
        Param("learning_rate", "float", 0.01, 0.3, log=True),
        Param("max_depth", "int", 2, 10),
        Param("min_child_weight", "float", 0.5, 50.0, log=True),
        Param("subsample", "float", 0.5, 1.0),
        Param("colsample_bytree", "float", 0.4, 1.0),
        Param("reg_lambda", "float", 1e-3, 50.0, log=True),
        Param("reg_alpha", "float", 1e-8, 5.0, log=True),
    ],
    "08_lightgbm": [
        Param("n_estimators", "int", 100, 900),
        Param("learning_rate", "float", 0.01, 0.3, log=True),
        Param("num_leaves", "int", 8, 200, log=True),
        Param("min_child_samples", "int", 5, 300, log=True),
        Param("subsample", "float", 0.5, 1.0),
        Param("colsample_bytree", "float", 0.4, 1.0),
        Param("reg_lambda", "float", 1e-3, 50.0, log=True),
        Param("reg_alpha", "float", 1e-8, 5.0, log=True),
    ],
    "09_catboost": [
        Param("iterations", "int", 100, 900),
        Param("learning_rate", "float", 0.01, 0.3, log=True),
        Param("depth", "int", 3, 10),
        Param("l2_leaf_reg", "float", 0.5, 50.0, log=True),
        Param("random_strength", "float", 1e-3, 10.0, log=True),
        Param("bagging_temperature", "float", 0.0, 2.0),
    ],
    "10_mlp": [
        Param("alpha", "float", 1e-6, 1e-1, log=True),
        Param("learning_rate_init", "float", 1e-4, 1e-2, log=True),
        Param("hidden_layer_sizes", "cat",
              choices=((32,), (64,), (64, 32), (128, 64), (128, 64, 32))),
        Param("batch_size", "cat", choices=(128, 256, 512)),
    ],
}

# Parameters LightGBM ignores unless bagging_freq is set.
_FIXED_EXTRAS = {
    "08_lightgbm": {"subsample_freq": 1},
}


def space_for(model_key: str) -> list[Param]:
    if model_key not in SPACES:
        raise KeyError(f"no search space declared for {model_key}")
    return SPACES[model_key]


def space_summary(model_key: str) -> pd.DataFrame:
    """The space as a table, so `04` can show that both libraries got the same one."""
    rows = []
    for p in space_for(model_key):
        rows.append({
            "parameter": p.name,
            "type": p.kind,
            "low": p.low,
            "high": p.high,
            "scale": "log" if p.log else ("-" if p.kind == "cat" else "linear"),
            "choices": str(p.choices) if p.kind == "cat" else "",
        })
    return pd.DataFrame(rows)


def apply_params(entry, params: dict):
    """Clone the zoo entry's estimator with a candidate configuration applied."""
    from sklearn.base import clone

    est = clone(entry.estimator)
    merged = dict(params)
    merged.update(_FIXED_EXTRAS.get(entry.key, {}))
    est.set_params(**merged)
    return est


# ---------------------------------------------------------------------------
# The shared objective
# ---------------------------------------------------------------------------


def make_objective(entry, frame, y, columns, n_splits: int = 3):
    """Mean PR-AUC over expanding-window folds of the TRAIN split.

    Both searches optimise this exact callable. Three folds rather than five:
    the budget buys more configurations that way, and `03` already showed the
    fold-to-fold spread is small enough that three is a usable signal.

    Returns ``(objective_fn, fold_scores_fn)`` where the second yields per-fold
    scores so Optuna's pruner has intermediate values to act on.
    """
    from sklearn.model_selection import TimeSeriesSplit

    from src.models import build_preprocessor
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import average_precision_score

    y = np.asarray(y)
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(len(frame))))
    X = frame[list(columns)]

    def fold_scores(params: dict):
        """Yield PR-AUC per fold, so a pruner can stop a bad config early."""
        for past, future in folds:
            pipe = Pipeline([
                ("pre", build_preprocessor(scale=entry.scale)),
                ("clf", apply_params(entry, params)),
            ])
            pipe.fit(X.iloc[past], y[past])
            p = pipe.predict_proba(X.iloc[future])[:, 1]
            yield average_precision_score(y[future], p)

    def objective(params: dict) -> float:
        return float(np.mean(list(fold_scores(params))))

    return objective, fold_scores


# ---------------------------------------------------------------------------
# Strategy 1 -- Optuna: multivariate TPE + Hyperband
# ---------------------------------------------------------------------------


def run_optuna(entry, frame, y, columns, time_budget_s: float,
               n_splits: int = 3, seed: int = SEED):
    """TPESampler(multivariate=True) + HyperbandPruner on a wall-clock budget.

    ``multivariate=True`` models the joint distribution over parameters rather
    than each independently, which matters for boosters where learning rate and
    tree count trade off against each other.

    Hyperband prunes on the per-fold intermediate scores, so a configuration
    that is clearly poor on fold 0 does not get to spend the budget on folds 1
    and 2.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = space_for(entry.key)
    _, fold_scores = make_objective(entry, frame, y, columns, n_splits=n_splits)

    def objective(trial):
        params = {p.name: p.optuna_suggest(trial) for p in space}
        scores = []
        for step, s in enumerate(fold_scores(params)):
            scores.append(s)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=seed),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=n_splits, reduction_factor=3
        ),
        study_name=f"optuna_{entry.key}",
    )
    study.optimize(objective, timeout=time_budget_s, n_jobs=1,
                   gc_after_trial=True, catch=(Exception,))
    return study


def optuna_history(study) -> pd.DataFrame:
    """Trial-by-trial history with a running best, for the convergence plot."""
    rows = []
    for t in study.trials:
        rows.append({
            "trial": t.number,
            "state": t.state.name,
            "value": t.value if t.value is not None else np.nan,
            "seconds": ((t.datetime_complete - t.datetime_start).total_seconds()
                        if t.datetime_complete and t.datetime_start else np.nan),
            "elapsed_s": ((t.datetime_complete - study.trials[0].datetime_start
                           ).total_seconds()
                          if t.datetime_complete else np.nan),
        })
    hist = pd.DataFrame(rows)
    hist["running_best"] = hist["value"].cummax().ffill()
    return hist


# ---------------------------------------------------------------------------
# Strategy 2 -- FLAML: cost-frugal BlendSearch / CFO
# ---------------------------------------------------------------------------


def run_flaml(entry, frame, y, columns, time_budget_s: float,
              n_splits: int = 3, seed: int = SEED, low_cost_hint: dict | None = None):
    """BlendSearch (CFO + global search) on the identical space and budget.

    FLAML's premise is that evaluation cost varies enormously across a space --
    900 trees cost nine times what 100 do -- so it starts from a cheap
    configuration and expands outward, spending its budget on many cheap trials
    instead of a few expensive ones. ``low_cost_partial_config`` is where that
    prior is expressed; leaving it empty would discard the entire point of the
    method.
    """
    from flaml import tune
    from flaml.tune.searcher.blendsearch import BlendSearch

    space = space_for(entry.key)
    domain = {p.name: p.flaml_domain() for p in space}
    objective, _ = make_objective(entry, frame, y, columns, n_splits=n_splits)

    history = []

    def trainable(config):
        t0 = pd.Timestamp.utcnow()
        score = objective(dict(config))
        history.append({
            "value": score,
            "seconds": (pd.Timestamp.utcnow() - t0).total_seconds(),
            "config": dict(config),
        })
        tune.report(pr_auc=score)

    searcher = BlendSearch(
        space=domain,
        metric="pr_auc",
        mode="max",
        seed=seed,
        low_cost_partial_config=low_cost_hint or {},
    )
    analysis = tune.run(
        trainable,
        search_alg=searcher,
        time_budget_s=time_budget_s,
        num_samples=-1,
        verbose=0,
    )
    hist = pd.DataFrame(history)
    if len(hist):
        hist["trial"] = np.arange(len(hist))
        hist["elapsed_s"] = hist["seconds"].cumsum()
        hist["running_best"] = hist["value"].cummax()
    return analysis, hist


def cheap_corner(model_key: str) -> dict:
    """FLAML's low-cost starting point: the cheapest sane corner of the space.

    Declared per model rather than inferred, so it is visible and arguable.
    """
    hints = {
        "01_logistic": {"C": 1.0},
        "02_decision_tree": {"max_depth": 3, "min_samples_leaf": 100},
        "03_random_forest": {"n_estimators": 100, "max_depth": 6},
        "04_extra_trees": {"n_estimators": 100, "max_depth": 6},
        "05_gradient_boosting": {"n_estimators": 50, "max_depth": 2},
        "06_hist_gradient_boosting": {"max_iter": 50, "max_leaf_nodes": 8},
        "07_xgboost": {"n_estimators": 100, "max_depth": 2},
        "08_lightgbm": {"n_estimators": 100, "num_leaves": 8},
        "09_catboost": {"iterations": 100, "depth": 3},
        "10_mlp": {"alpha": 1e-3},
    }
    return hints.get(model_key, {})
