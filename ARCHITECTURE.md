# Architecture

**Return-Risk Scorer · Razorpay AI Buildathon 2026 · Track 02**

> Status: `01`–`05` complete and executed. Nothing in this document is a claim about
> results; those live in `reports/results/` and are summarised in the README.

---

## 1. Shape of the system

```
                      config/evidence.yaml
                   (published stats + sources)
                              │
   data/external/             ▼
   india_pincodes.csv ──► src/generate.py ──► data/raw/orders.csv
   (real India Post)         │                data/interim/latents.csv
                             │                data/processed/{train,val,test}
                             ▼
                    calibration assertions
                    tests/test_calibration.py
                             │
                             ▼
   src/features.py ──► engineered features + out-of-fold target encoding
                             │
                             ▼
   src/models.py ────► 11-model benchmark        (03)
   src/tuning.py ────► Optuna vs FLAML           (04)
   src/costs.py ─────► rupee cost model          (05)
   src/evaluate.py ──► metrics, calibration,     (05)
                       prevalence + order-mix
                       shift, SHAP
                             │
                             ▼
                    models/ + reports/{figures,results}
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              api/main.py       app/streamlit_app.py
              (score + reasons) (merchant view)
```

Notebooks orchestrate and narrate. Every decision that must be identical across
notebooks lives in `src/` and is imported, never redefined.

## 2. Module responsibilities

| Module | Owns | Status |
|---|---|---|
| `src/generate.py` | Geographic skeleton, tier rules, customer population, order stream, the 12-cell calibration solve, labels, causal history, chronological split, SHA-256 manifest | **done** |
| `src/features.py` | Address-quality extraction, row-local engineered features, opt-in domain interaction block, `OutOfFoldTargetEncoder`, the feature/forbidden column lists | **done** |
| `src/models.py` | The 11 benchmark estimators, the shared preprocessing pipeline, fixed-weight blending, cross-fitted calibration | **done** |
| `src/tuning.py` | Shared search space; Optuna and FLAML drivers on identical budget/seed | **done** |
| `src/costs.py` | FN/FP rupee cost, threshold sweep, three-tier cut points | **done** |
| `src/evaluate.py` | Metric bundle, reliability diagram, prevalence resampling, order-mix reweighting, covariate-shift probe, SHAP reasons | **done** |
| `src/serving.py` | The single scoring path shared by the API and the Streamlit app: artifact loading, feature frame assembly, tier decision, SHAP reasons | **done** |
| `src/interpret.py` | Plain-English rewrite of a scored order via the Gemini API, with a deterministic template fallback when the key is absent or the call fails | **done** |

## 3. The two decisions the design is built around

### 3.1 Time is the only split axis

Every split — train/val/test, and the folds inside target encoding — is
**chronological**. A random split lets the model learn from orders placed *after* the
ones it is scored on, which inflates every metric and is invisible in the output.

`tests/test_no_leakage.py` asserts the split is monotone in `order_ts`, that the three
sets are disjoint, and that the train/val/test boundary is never crossed backwards.

### 3.2 Pincode history is the strongest feature and the easiest leak

Pincode historical RTO rate carries most of the geographic signal. Computing it on the
full dataset would hand each row a value its own label helped produce.

`OutOfFoldTargetEncoder` therefore:

* fits **on train only** — validation and test are transformed, never fitted on;
* returns **out-of-fold** values for training rows, using expanding time-ordered folds,
  so no row contributes to its own encoding;
* smooths thin keys towards the global prior
  (`(sum + prior·m) / (count + m)`, `m = 30`), so a pincode with one order cannot
  become a 0% or 100% pincode;
* falls back to the prior for keys unseen in training.

Four tests guard it, including a one-row-per-key case where a leaky encoder would
return the labels themselves.

## 4. Feature groups

| Group | Features |
|---|---|
| Geography | `pincode_te`, `pincode_prefix3_te`, `city_te` (all out-of-fold), `pincode_tier`, `tier_ordinal`, `state` |
| Address quality | token count, char length, digit count, comma count, house-number flag, landmark flag, gibberish score, composite quality score |
| Customer history | `past_orders`, `past_rto_count`, `past_rto_rate`, `has_history`, `is_first_order`, `account_age_days`, `log_account_age`, `order_velocity_24h` |
| Order | `order_value`, `log_order_value`, `order_value_band`, `category`, `discount_pct`, `discount_amount`, `is_cod`, `is_festive`, `is_alternate_address` |
| Interactions *(opt-in, **not** in the shipped model)* | ten domain products in `src.features.INTERACTION_FEATURES`; ablated in `05` and rejected — see `WHAT_BROKE.md` #16 |
| Fulfilment | `delivery_days_est` |

Explicitly **not** features, and asserted so: identifiers, customer name, phone, raw
address text, timestamps, `payment_mode` (redundant with `is_cod`), and every generator
latent (`_pincode_rto_prior`, `_reliability_z`, `_address_quality`, `_p_rto_true`).

## 5. Serving path

Built and deployed. FastAPI serves both the JSON API and the one-page merchant UI from
a single Uvicorn process, in a Docker container on Render.

```
POST /score  { order attributes }
   →  feature assembly (same src/features.py code path as training)
   →  predicted probability from the frozen artifact
   →  expected rupee loss, and a three-tier recommendation from the
      cost-optimal cut points fixed on validation
   →  { risk_score, expected_loss_inr, tier, reasons[], model_version }
```

The probability is used as the model emits it. Post-hoc recalibration was tested —
sigmoid and isotonic — and both were rejected because each raised cost per order; see
`reports/results/05_final_metrics.json`. Expected calibration error is 0.012 on test.

The API returns a score, a tier and human-readable reasons derived from SHAP. It
performs no action.

## 6. Defence-only, by construction

The system reads order attributes and returns a number, a tier and an explanation.

It has no code path that captures a payment, issues a refund, blocks an account,
contacts a customer, or writes to any external system. The recommendation is advisory;
acting on it is the merchant's decision. There is no capability here that could be
repurposed to commit fraud.

## 7. Reproducibility contract

* One seeded `numpy.random.Generator` per pipeline, drawn in a fixed order.
* Faker seeded from the same value; it produces only cosmetic strings.
* Every notebook runs top-to-bottom on a clean kernel.
* `01` re-generates the whole dataset through `src.generate.build_dataset` and asserts
  frame equality against what it just wrote.
* SHA-256 of every output file lands in `reports/results/*_manifest.json`, keyed by
  repo-relative path so two checkouts produce comparable manifests.
* No metric appears in the README that is not backed by a file in `reports/results/`.
* The test set is evaluated only at settings frozen on validation. `05` asserts that
  the challenger experiments left that operating point unchanged, and
  `05_final_metrics.json` separates *evaluations executed* from *decisions informed*.

**Exceptions, stated rather than buried.**

`01` and `02` are bit-reproducible, verified by rerun.

`03` is numerically deterministic but **not** bit-identical — it records wall-clock fit
times, and multi-threaded floating-point reduction in the tree ensembles is not
associative. Two reruns agreed to within 2.2 × 10⁻¹⁶ on every metric.

`04` is **not** reproducible, and cannot be while the comparison it makes is fair: Optuna
is trial-native and FLAML is budget-native, so the two are given an equal **wall-clock**
budget, and trials completed therefore depend on machine load. `05` loads `04`'s persisted
models, so it inherits that upstream.

See [`WHAT_BROKE.md`](WHAT_BROKE.md) #13 for the full table and the measured deviations.
