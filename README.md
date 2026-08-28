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

No public dataset of Indian COD orders with RTO outcomes exists — those labels sit with
merchants and courier aggregators. **This model is therefore not validated on real merchant
data, and no such claim is made.**

What *is* real is the calibration. The generator is constrained to reproduce published
Indian statistics, and this is enforced as a test rather than asserted in prose:

```bash
pytest tests/test_calibration.py
```

It fails the build if the generated 26% COD RTO rate, the sub-2% prepaid rate, or the
non-monotonic order-value curve drift outside tolerance.

---

## Results

Metrics land here once `05_final_evaluation` has been run. The test set is opened once.

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

Three commitments, made before any model was trained (see `PRE_REGISTRATION.md`):

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
