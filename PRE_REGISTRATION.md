# Pre-Registration

**Return-Risk Scorer · Razorpay AI Buildathon 2026 · Track 02 — AI Risk Manager**

This document fixes the evaluation protocol **before any model is trained**. It is
committed and tagged `pre-registration`. Everything in `03`, `04` and `05` is judged
against what is written here, not against what turned out to look good.

Committed at: notebook `01_data_generation.ipynb` complete and executed;
`02`–`05` not yet written; no model of any kind trained.

---

## 1. The loss class

**COD Return-to-Origin.** One class of loss, as the track requires.

An order is placed with Cash on Delivery, is shipped, and comes back undelivered.
The merchant has burned forward shipping, reverse shipping, packaging and handling,
and has collected nothing.

Out of scope, deliberately: chargebacks, promo abuse, account takeover, prepaid
returns. Those are different loss classes with different base rates and different
cost structures, and mixing them would make every number harder to defend.

## 2. Prediction target

`rto ∈ {0, 1}` — whether the order returns to origin.

Scored **at checkout**, using only what is knowable at checkout. Positive class =
RTO. Base rate 16.4% overall, 25.6% on COD.

## 3. Data and splits — fixed

| | |
|---|---|
| Orders | 50,000 |
| Split | chronological **70 / 10 / 20** → 35,000 train / 5,000 val / 10,000 test |
| Split rule | strictly by `order_ts`. **Never random.** A random split leaks the future into the past. |
| Test set | opened **once**, in `05_final_evaluation.ipynb`, after the final model is chosen |

`02` reads the train split only. `03` and `04` read train and validation only.
No result in `05` may cause a change to `03` or `04`; if it does, that has to be
declared in `WHAT_BROKE.md` and the test set is burned.

## 4. Primary and secondary metrics — fixed in advance

**Model selection metric (validation): PR-AUC.**
Chosen because the positive class is the minority and the cost of a miss is
asymmetric. Accuracy is not reported as a headline anywhere; on a 16.5% base rate
a model that predicts "no RTO" for every order scores 83.5% and is worthless.

Reported alongside, always:

| Metric | Why it is here |
|---|---|
| PR-AUC | ranking quality on the minority class — the selection metric |
| ROC-AUC | comparability with published work |
| **Brier score** | probability quality. A cost-based threshold is meaningless on uncalibrated scores. |
| Reliability diagram | Brier alone hides *where* the miscalibration is |
| Precision / recall / F1 | at the cost-optimal threshold, not at 0.5 |
| Confusion matrix | absolute counts, so the false-positive burden is visible |

## 5. Threshold selection — fixed in advance

The operating threshold is set at the **rupee cost minimum on the validation set**,
never at 0.5 and never at max-F1. The threshold chosen on validation is then applied
unchanged to test.

Cost model:

```
FN  =  Rs 200   an RTO we failed to flag: forward + reverse shipping,
                packaging, handling. Range Rs 150-250, sensitivity-tested.
FP  =  lost contribution margin on a good order we discouraged,
                computed per order value rather than as a flat constant.
```

`config/evidence.yaml` carries both, `fn_cost_inr` marked `assumed` and therefore
sensitivity-tested across 150–250 in `05`.

We will also publish what the **default 0.5 threshold would have cost**, in rupees,
on the same test set.

## 6. Decision policy — three tiers, fixed in advance

| Tier | Action | Set by |
|---|---|---|
| Low | Allow COD | below the lower cut |
| Medium | Charge a COD fee | between the cuts |
| High | Disable COD, offer prepaid | above the upper cut |

Cut points are chosen on validation by expected cost, not by quantile.

## 7. Prevalence-shift study — the centrepiece, declared in advance

Published Indian city RTO rates span **18% (Vadodara) → 35% (Patna)**. A model
selected at one base rate can degrade badly at another, and a cost-optimal threshold
can invert.

We commit in advance to re-evaluating the final model across that whole range by
resampling the test set to each target prevalence, and to publishing the degradation
curve **whatever it shows** — including the prevalence at which the model stops
beating the trivial policy, if there is one.

This is declared before the result is known precisely so that a bad curve cannot be
quietly dropped.

## 8. Leakage controls — fixed in advance

1. **Chronological split.** Asserted in `tests/test_no_leakage.py`.
2. **Out-of-fold target encoding** for pincode / pincode-prefix / city. Fitted on
   train only, with expanding time-ordered folds, so no row contributes to its own
   encoding. Guarded by four tests.
3. **Causal customer history.** `past_orders`, `past_rto_count`, `past_rto_rate`
   are expanding aggregates over that customer's *strictly earlier* orders.
4. **Generator latents are not features.** `_pincode_rto_prior`, `_reliability_z`,
   `_address_quality`, `_p_rto_true` live in `data/interim/` and are named in
   `FORBIDDEN_FEATURES`. A test asserts that no `_`-prefixed column can reach the
   feature matrix.
5. **Bayes ceiling published.** `01` reports the best score achievable from the true
   probability field: **ROC-AUC 0.869, PR-AUC 0.577, Brier 0.097**
   (`reports/results/01_calibration.json`). **A model that beats the ceiling is
   evidence of leakage, not of skill**, and will be treated as a bug.

## 9. Model benchmark — fixed in advance

Eleven entries, same folds, same seed, same feature matrix:

DummyClassifier (majority baseline) · Logistic Regression · Decision Tree ·
Random Forest · Extra Trees · GradientBoosting · HistGradientBoosting · XGBoost ·
LightGBM · CatBoost · MLP

Ranked on validation PR-AUC. **If gradient boosting fails to beat logistic
regression, that is what gets reported.** 2–3 finalists proceed to tuning.

## 10. Tuning — fixed in advance

Two modern approaches, head to head on an **identical budget, search space and
seed**:

1. **Optuna** — `TPESampler(multivariate=True)` + `HyperbandPruner`
2. **FLAML** — cost-frugal `BlendSearch` / `CFO`

Reported: convergence curves, parameter importance, and which approach found the
better configuration for the same compute. The loser is reported too.

## 11. What we will not claim

There is no public dataset of Indian COD orders with RTO outcomes. **No claim of
validation on real merchant data will be made anywhere in this submission.**

The honest framing, which appears in the README:

> Trained and evaluated on synthetic orders built on the real India Post pincode
> directory and calibrated against published Indian RTO statistics (enforced by
> test). We do not claim validation on real merchant data — no such dataset is
> public. Instead we report performance across the full 18–35% RTO range observed
> across Indian cities, so degradation under distribution shift is measured rather
> than assumed.

No metric appears in the README that is not backed by a file in `reports/results/`.

## 12. Defence-only

The system scores an order and returns a probability, human-readable reasons and a
recommended tier. It never captures, refunds, blocks or contacts anyone. The
recommendation is advisory; acting on it is the merchant's decision. Nothing in the
repository is offense-capable.

---

*Fixed before results. Deviations, if any, are recorded in `WHAT_BROKE.md`.*
