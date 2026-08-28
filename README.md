# Return-Risk Scorer

**Razorpay AI Buildathon 2026 · Track 02 — AI Risk Manager**

Predicts **Return-to-Origin (RTO) risk** for Cash-on-Delivery orders at checkout, explains
which factors drove the score, and returns a graded action — allow, charge a COD fee, or
disable COD.

> It scores risk. It never moves money. See [Defence-only](#defence-only).

---

## The loss class

One class of loss: **COD return-to-origin**.

Roughly 60–65% of Indian e-commerce orders are COD, and about **26% of them come back**
against under 2% for prepaid. For a D2C merchant that is forward shipping, reverse
shipping, packaging and handling burned on every failed delivery — an estimated
₹8,000 crore a year across Indian e-commerce.

## Why a scorer and not a blocker

Blocking COD outright kills conversion on customers who would have paid. The model
therefore produces **three tiers**, not a binary:

| Tier | Action | Rationale |
|---|---|---|
| Low | Allow COD | No friction on good customers |
| Medium | Charge a COD fee | Price the risk instead of losing the sale |
| High | Disable COD, offer prepaid | Avoid the shipping loss entirely |

Thresholds are set at the **cost minimum**, not at 0.5 and not at max-F1.

---

## Data

Synthetic orders on a **real geographic skeleton**.

| Component | Source |
|---|---|
| Pincodes, districts, states | India Post directory — 39,736 post offices, 23,916 pincodes (real) |
| RTO base rates | Published Indian industry statistics (see `config/evidence.yaml`) |
| Orders, customers, addresses | Generated — seeded, reproducible |
| RTO labels | Generated from grounded drivers |

**50,000 orders**, split chronologically **70 / 10 / 20** — 35,000 train, 5,000 validation,
10,000 test.

### Honest statement on the data

> Trained and evaluated on synthetic orders built on the real India Post pincode
> directory and calibrated against published Indian RTO statistics (enforced by test).
> We do not claim validation on real merchant data — no such dataset is public.
> Instead we report performance across the full 18–35% RTO range observed across Indian
> cities, so degradation under distribution shift is measured rather than assumed.

What *is* real is the calibration. The generator is constrained to reproduce published
Indian statistics, and this is enforced as a test rather than asserted in prose:

```bash
pytest tests/test_calibration.py
```

It fails the build if any published rate drifts outside tolerance. Achieved
(`reports/results/01_calibration.json`):

| Statistic | Published | Generated |
|---|---|---|
| COD share of orders | 60–65% | 61.8% |
| RTO on COD | 26% | 25.6% |
| RTO on prepaid | <2% | 1.4% |
| RTO, order < ₹500 | 25% | 24.8% |
| RTO, order ₹500–1,000 | 28% | **27.7%** ← the peak |
| RTO, order > ₹1,000 | 24% | 23.8% |
| RTO, fashion on COD | 40%+ | 39.9% |

The order-value curve is **non-monotonic**: RTO peaks in the ₹500–1,000 impulse band
and *falls* above ₹1,000. Assuming risk rises with order value is the intuitive move,
and the published data contradicts it — so it is asserted, not hoped for.

Full provenance, schema and limitations: [`DATA_CARD.md`](DATA_CARD.md).

### The difficulty ceiling

The labels are Bernoulli draws from a known probability field, so the best score any
model could achieve is computable — and is published up front:

| | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Bayes ceiling | 0.869 | 0.577 | 0.097 |

**A model scoring above the ceiling would be evidence of leakage, not of skill.**

---

## Results

**CatBoost**, tuned by FLAML, selected on validation. Test set opened **once**, at a
threshold and tier cut points frozen before it was opened.

Every number below is backed by [`reports/results/05_final_metrics.json`](reports/results/).

### Held-out test set — 10,000 orders, base rate 14.7%

| | Test | Validation |
|---|---|---|
| PR-AUC | **0.386** (2.62× the no-skill floor) | 0.397 |
| ROC-AUC | 0.806 | 0.810 |
| Brier | **0.106** | 0.109 |
| Expected Calibration Error | **0.012** | 0.013 |
| Precision / Recall / F1 | 0.458 / 0.269 / 0.339 | — |

Confusion matrix at the frozen threshold (0.370): TP 396, FP 468, FN 1,077, TN 8,059.

Validation→test drift is small and in the expected direction, so the model was not
selected on validation noise. The **Bayes ceiling** for this data is PR-AUC 0.555 — the
model captures ~62% of achievable headroom, and a score above the ceiling would be
treated as a leak, not a win.

### Money

| Policy | Cost on 10,000 test orders |
|---|---|
| Allow COD on everything (no model) | ₹294,600 |
| Disable COD on everything (no model) | ₹1,330,855 |
| Model at the default 0.5 threshold | ₹285,954 |
| **Model at the cost minimum (shipped)** | **₹265,119** |

**Using 0.5 out of habit would have cost an extra ₹20,835** on these 10,000 orders.
Saving against the best no-model policy: ₹29,481 (₹2.95/order).

The threshold is genuinely sensitive to the cost assumptions — it moves across
**0.21–0.48** as FN cost ranges ₹150–250 and margin 15–35%. That grid is published so a
merchant can substitute their own numbers rather than inherit ours.

### Three-tier policy

| Tier | Share of orders | Actual RTO rate |
|---|---|---|
| Allow COD | 38% | 1.3% |
| Charge a COD fee | 56% | 20.3% |
| Disable COD | 6% | 47.7% |

The fee tier's size rests on the softest assumptions in the project (fee abandonment
rate, fee level — neither has a published source). `05` says so explicitly and treats
the two-tier threshold as the more defensible result.

### Prevalence shift — 18% to 35%

The centrepiece. A model can hold its ranking and still lose money, because the
threshold stops being the cost minimum. Both are measured:

| Across the published 18–35% range | |
|---|---|
| Minimum PR-AUC lift | **1.89×** the no-skill floor — ranking does not collapse |
| Beats the best no-model policy at every prevalence | **yes** |
| Worst cost regret vs an oracle threshold | ₹16.42/order at 35% |
| Worst regret **after prior correction** | **₹0.08/order** |
| Calibration error, untreated | 0.013 → **0.169** |
| Calibration error, prior-corrected | **0.029** |

What breaks under shift is *calibration*, not ranking — and the fix is one line of
arithmetic ([`prior_shift_correction`](src/evaluate.py)): a merchant who knows their own
RTO rate rescales the scores before thresholding. No retraining, no new labels.

### What we found that was not flattering

- **No model beat logistic regression.** Across 11 benchmarked entries, a paired
  bootstrap put zero models' confidence interval clear of it. Reported because
  `PRE_REGISTRATION.md` committed to reporting it.
- **Hyperparameter tuning made the held-out score worse** (−0.004 PR-AUC), while
  improving the CV objective it was optimising. The tuning objective was overfitted.
- Three engineered features are near-useless, declared weak in `02` *before* any model
  saw them, and confirmed by SHAP in `05`.

Full list, quantified by segment: section 13 of `05` and
[`reports/results/05_exception_list.csv`](reports/results/).

---

## Pipeline

| Notebook | Purpose |
|---|---|
| `01_data_generation.ipynb` | Real pincode skeleton → 50k orders → calibration check → 70/10/20 split |
| `02_eda.ipynb` | Distributions, RTO drivers, leakage audit, preprocessing design |
| `03_model_training.ipynb` | 10 models benchmarked against a majority baseline |
| `04_hyperparameter_tuning.ipynb` | Two modern approaches — Optuna (multivariate TPE + Hyperband) vs FLAML (BlendSearch/CFO) |
| `05_final_evaluation.ipynb` | Test set opened once, cost model, calibration, prevalence-shift study |

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order. `01` writes `data/processed/`; nothing downstream runs without it.

```bash
uvicorn api.main:app --reload
```

```bash
streamlit run app/streamlit_app.py
```

---

## Evaluation approach

Three commitments, made before any model was trained — fixed in
[`PRE_REGISTRATION.md`](PRE_REGISTRATION.md), committed and tagged `pre-registration`
before a single model existed:

**Chronological split.** Train on earlier orders, test on later. A random split leaks time.

**Out-of-fold target encoding.** Pincode historical RTO rate is the strongest feature and
the easiest way to leak the label. It is encoded out-of-fold; `tests/test_no_leakage.py`
guards it.

**Prevalence-shift study.** Real RTO rates range from 18% (Vadodara) to 35% (Patna) across
Indian cities. Performance is reported across that whole range rather than at one base rate,
so degradation under distribution shift is measured instead of assumed.

---

## Defence-only

This system is a **scorer**. By construction it:

- reads order attributes and returns a probability, reasons and a recommended tier
- never captures, refunds, blocks or moves money
- never contacts a customer
- exposes no capability that could be repurposed to commit fraud

The recommended action is advisory. Acting on it is the merchant's decision.

---

## Documents

| | |
|---|---|
| [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md) | Metrics, thresholds and protocol, fixed before any training |
| [`DATA_CARD.md`](DATA_CARD.md) | What is real, what is generated, schema, limitations |
| [`WHAT_BROKE.md`](WHAT_BROKE.md) | Running log of failures and recoveries |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design and data flow |

## Repository layout

```
config/          evidence.yaml — published statistics with sources
data/external/   real India Post pincode directory
data/raw/        generated 50k orders
data/processed/  train / val / test
notebooks/       01 → 05, run in order
src/             importable modules used by the notebooks
tests/           calibration and leakage guards
models/          persisted final model
reports/         figures and metrics
api/             FastAPI scoring service
app/             Streamlit dashboard
```

---

## Author

Anil Kumar · <anilkumarlucky07@gmail.com>
