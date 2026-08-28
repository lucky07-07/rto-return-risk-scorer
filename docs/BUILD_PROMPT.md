# Build prompt — paste into a fresh context

---

I am building a submission for the **Razorpay AI Buildathon 2026, Track 02 — AI Risk Manager**.
Working directory: `D:\M.TECH\SEM-3\Razor pay`. The scaffold already exists — read the repo first.

## What Razorpay requires (verbatim from razorpay.com/buildathon)

> **02 — AI Risk Manager.** Stop the merchant losing money to fraud, returns and chargebacks.
> Build a working detector, verifier or auto-responder for **one class of loss**, with
> **measured precision and recall on a held-out test set**.
> *The bar:* Honest metrics including **false-positive cost**. **Strictly defense-only:
> anything offense-capable is disqualified.**
> *Why now:* AI-enabled fraud is hitting Indian BFSI while returns and chargebacks quietly eat
> margin. This track surfaces the **risk and ML minded builders** the others miss.

Chosen direction: **Return-risk scorer** (one of their four named example directions).
One class of loss: **COD Return-to-Origin (RTO)**.

This is the ML track. They are screening for ML judgement: proper splits, class handling,
calibration, cost-sensitive thresholds, honest reporting.

## Repo state

Already created and committed: folder structure, `README.md`, `requirements.txt`,
`config/evidence.yaml`, five notebook skeletons with section headers, empty `src/` modules,
two test stubs, and `data/external/india_pincodes.csv` (**real** India Post directory,
39,736 post offices / 23,916 pincodes / 589 districts / 35 states).

Fill in the notebooks and `src/` modules. Do not restructure the repo.

## Data specification

**50,000 orders. Chronological split 70/10/20 → 35,000 train / 5,000 val / 10,000 test.**

Synthetic orders on a real geographic skeleton. Use `Faker("en_IN")` for names, phones and
address strings; seeded `numpy` RNG for everything that determines the label. Customers must
**persist across orders** so `past_rto_rate` is a real history, not a random column.

All grounding constants are in `config/evidence.yaml` with sources. Key ones:

| Parameter | Value | Source |
|---|---|---|
| COD share of orders | 62% | industry reports (60–65%) |
| RTO rate on COD | **26%** | Shipway ShipNotes FY25 |
| RTO rate on prepaid | **<2%** | Shipway ShipNotes FY25 |
| Order value < ₹500 | 25% RTO | ShipNotes FY25 |
| Order value ₹500–1,000 | **28% RTO** | ShipNotes — the "impulse zone" |
| Order value > ₹1,000 | 24% RTO | ShipNotes |
| City range | Vadodara 18% → Patna 35% | ShipNotes city-wise |
| Fashion category | 40%+ | industry |

**The order-value curve is non-monotonic** — RTO peaks in the middle band and falls above
₹1,000. Do not make RTO rise with order value; that is the naive assumption and it is wrong.

Features to engineer (these mirror the signals Razorpay's own RTO Intelligence documents use):
pincode historical RTO rate (**out-of-fold target encoding — this is the main leakage risk**),
address quality (token count, house number present, landmark present, gibberish score),
customer prior RTO count and rate, order value and band, category, COD flag, discount %,
festive-sale flag, order velocity 24h, account age, delivery-days estimate, pincode tier.

## Notebooks

**`01_data_generation.ipynb`** — load the real pincode skeleton, assign city/pincode RTO
priors anchored to the published range, build the customer population, generate 50,000 orders,
assign labels from the grounded drivers with noise so the classes are not trivially separable.
Then **the calibration check**, which must be a real assertion, not a printout:

```python
assert 0.24 <= cod.rto.mean() <= 0.28          # ShipNotes 26%
assert prepaid.rto.mean() < 0.02               # ShipNotes <2%
assert band_rate("500_1000") > band_rate("1000+")   # the non-monotonic peak
```

Mirror these into `tests/test_calibration.py`. Write a SHA-256 manifest of every output.
Chronological split, write `data/raw/` and `data/processed/`.

**`02_eda.ipynb`** — train split only, never touch test. Distributions, RTO drivers by
pincode tier / value band / category / customer history / address quality, correlation and
multicollinearity, outliers, an explicit leakage audit, and the preprocessing pipeline
definition. Save every figure to `reports/figures/`.

**`03_model_training.ipynb`** — benchmark **10 models** plus a majority-class baseline:
DummyClassifier, Logistic Regression, Decision Tree, Random Forest, Extra Trees,
GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, MLP. Same folds, same
seed, same features. Leaderboard on validation by **PR-AUC, ROC-AUC and Brier score** — not
accuracy. Report honestly if gradient boosting fails to beat logistic regression. Pick 2–3
finalists.

**`04_hyperparameter_tuning.ipynb`** — **two modern approaches, head to head on identical
budget, space and seed**:
1. **Optuna** — `TPESampler(multivariate=True)` + `HyperbandPruner`
2. **FLAML** — cost-frugal `BlendSearch` / `CFO`

Report convergence curves, parameter importance, and which approach found a better
configuration for the same compute. Persist tuned models.

**`05_final_evaluation.ipynb`** — select ONE final model **on validation only**, then open the
test set **once**. Report:
- precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix at the cost-optimal threshold
- **calibration**: Brier score and a reliability diagram (this is what most submissions skip,
  and it is what matters for a cost-based decision)
- **cost model in rupees** — FN ≈ ₹200 shipping burned (range 150–250, sensitivity-test it);
  FP = lost margin on a good order that was blocked. Threshold at the **cost minimum**, and
  state what the default 0.5 threshold would have cost instead
- **three-tier decision policy**: allow / charge COD fee / disable COD
- **prevalence-shift study**: re-evaluate across **18% → 35% RTO**, the real range observed
  across Indian cities, and publish the degradation curve. This is the centrepiece — the
  strongest competing submission in this track collapsed to worse-than-random when its base
  rate shifted, and nobody has answered that
- SHAP explanations producing human-readable risk reasons
- comparison plots across all 10 models plus tuned finalists
- an **honest exception list**: where the model underperforms, named and quantified

## Standards

- Every notebook runs top-to-bottom on a clean kernel. Fixed seeds; two runs produce identical output.
- Chronological split only. Never random — it leaks time.
- Out-of-fold target encoding for pincode. `tests/test_no_leakage.py` must guard it.
- No metric appears in the README that is not backed by a file in `reports/results/`.
- Figures to `reports/figures/`, metrics to `reports/results/`, model to `models/`.
- Reusable logic goes in `src/`; notebooks import it rather than redefining it.
- Defence-only: the system scores and recommends. It never captures, refunds, blocks or
  contacts anyone.

## What must not be claimed

There is no public dataset of Indian COD orders with RTO outcomes. **Do not claim validation
on real merchant data.** The honest framing, which belongs in the README:

> Trained and evaluated on synthetic orders built on the real India Post pincode directory
> and calibrated against published Indian RTO statistics (enforced by test). We do not claim
> validation on real merchant data — no such dataset is public. Instead we report performance
> across the full 18–35% RTO range observed across Indian cities, so degradation under
> distribution shift is measured rather than assumed.

Also produce `PRE_REGISTRATION.md` (metrics and protocol fixed before results — commit and
`git tag pre-registration` before training), `DATA_CARD.md`, `ARCHITECTURE.md`, and
`WHAT_BROKE.md` (the buildathon asks what broke and how you recovered — keep it as you go).

Start with `01_data_generation.ipynb`. Show me the calibration check passing before moving on.
