# Data Card

**Dataset:** 50,000 synthetic Indian e-commerce orders with COD Return-to-Origin labels
**Produced by:** `notebooks/01_data_generation.ipynb` → `src/generate.py`
**Seed:** `20260101` · **Manifest:** `reports/results/01_manifest.json`

---

## 1. What is real and what is generated

| Component | Status | Source |
|---|---|---|
| Pincodes, post offices, districts, states | **Real** | India Post directory — 39,736 post offices, 23,916 pincodes, 589 districts, 35 states/UTs (`data/external/india_pincodes.csv`) |
| Published RTO rates used as calibration targets | **Real** | `config/evidence.yaml`, every constant carries a source |
| City / pincode tier assignment | Rule-based | Curated metro and tier-1 city lists; district-headquarters towns → tier-2; remainder → tier-3 |
| City / pincode RTO propensity | Generated | Drawn inside the published 18–35% range, positioned by tier; Vadodara and Patna pinned to their published values |
| Customers, orders, addresses, names, phones | **Generated** | Faker `en_IN` for strings; seeded NumPy Generator for everything that determines a label |
| RTO labels | **Generated** | Bernoulli draws from a calibrated probability field |

**There is no public dataset of Indian COD orders with RTO outcomes.** Those labels sit
with merchants and courier aggregators. Nothing here is validated against real merchant
data and no such claim is made anywhere in this submission.

## 2. Why tier is a rule and not a measurement

Post-office density in the India Post directory tracks **rural spread**, not
urbanisation: Thrissur has 370 pincodes, Pune has 125. Using it as an urbanisation
proxy would have been wrong in a way that is easy to miss. Tier is therefore assigned
from curated, checkable city lists, and is documented as an assumption rather than
presented as a measurement.

## 3. The generative model

### 3.1 Structure

```
logit P(RTO) = cell_offset(payment_mode, value_band, is_fashion)     <- solved
             + city_prior_term          logit(pincode prior) - logit(mean prior)
             + 0.85 · (−reliability_z)  persistent customer latent
             + 1.30 · (1 − address_quality)
             + 1.00 · log(category_multiplier)
             + 0.90 · (discount_pct − 0.20)
             + 0.28 · is_festive
             + 0.075 · (delivery_days_est − mean)
             + 0.30 · order_velocity_24h
             + 0.32 · (account-age term, centred)
             − 0.45 · is_repeat_buyer
             + N(0, 0.55²)              irreducible noise
```

Labels are then **Bernoulli draws** from `sigmoid(·)`. There is no deterministic rule
to memorise; the classes overlap by construction.

### 3.2 Marginals are calibrated, conditionals are free

Twelve cells — `{COD, prepaid} × {<₹500, ₹500–1k, >₹1k} × {fashion, other}` — each get
a single log-odds intercept found by deterministic bisection, so the cell's expected
rate lands on its published target. Every other driver contributes freely and is not
tuned to any target.

The fashion target is carried as a **ratio** to the COD rate (40 / 26), and the
non-fashion target in each cell is whatever makes that cell's volume-weighted mean
equal its band target. No published rate is counted twice.

### 3.3 Achieved calibration

| Statistic | Published | Generated | Source |
|---|---|---|---|
| COD share of orders | 60–65% | **61.8%** | Industry reports |
| RTO on COD | 26% | **25.6%** | Shipway ShipNotes FY25 |
| RTO on prepaid | <2% | **1.4%** | Shipway ShipNotes FY25 |
| RTO, order < ₹500 | 25% | **24.8%** | ShipNotes FY25 |
| RTO, order ₹500–1,000 | 28% | **27.7%** | ShipNotes FY25 — the impulse zone |
| RTO, order > ₹1,000 | 24% | **23.8%** | ShipNotes FY25 |
| RTO, fashion on COD | 40%+ | **39.9%** | Fashion/apparel reported 40%+ |
| City RTO spread (51 cities ≥100 COD orders) | 18% → 35% | **15.0% → 36.8%** | ShipNotes FY25 city-wise |

Source of truth: `reports/results/01_calibration.json`. Enforced by `assert` inside
the notebook and mirrored in `tests/test_calibration.py`. If the generator drifts, the
build fails; it does not print a warning.

The realised city spread is slightly **wider** than the published endpoints. The
*priors* are drawn strictly inside [0.18, 0.35]; the realised rates are Bernoulli means
over finite per-city volume and additionally carry that city's category and value mix,
so the tails overshoot. That is stated rather than clipped, because clipping the
realised rates back into the published range would be fitting the report to the claim.

**The order-value curve is non-monotonic.** RTO peaks in the ₹500–1,000 impulse band
and *falls* above ₹1,000. Assuming RTO rises with order value is the intuitive move
and the published data contradicts it, so it is asserted rather than hoped for.

## 4. Schema

### `data/raw/orders.csv` — 50,000 × 24

What a merchant's order log would plausibly hold. Nothing post-outcome, nothing latent.

| Column | Type | Notes |
|---|---|---|
| `order_id` | str | `ORD######`, assigned in chronological order |
| `order_ts` | datetime | 2024-01-01 → 2025-06-30, evening-heavy, festive spikes |
| `customer_id` | str | **persistent** across orders — mean 3.4 orders/customer, max 27 |
| `customer_name`, `phone` | str | Faker `en_IN`, cosmetic only, never features |
| `address_line` | str | rendered from the latent quality; the model sees only this string |
| `pincode`, `city`, `district`, `state` | str | **real** India Post values |
| `pincode_tier` | str | metro / tier_1 / tier_2 / tier_3 |
| `category` | str | fashion, footwear, beauty, accessories, home_kitchen, electronics |
| `order_value` | float | INR, lognormal per category, clipped to [99, 60000] |
| `order_value_band` | str | under_500 / 500_1000 / 1000_plus |
| `discount_pct` | float | 0–0.75, boosted in festive windows |
| `payment_mode`, `is_cod` | str, bool | COD / PREPAID |
| `is_festive` | bool | five sale windows incl. Big Billion Days / Diwali |
| `is_alternate_address` | bool | 12% ship to a non-home address |
| `delivery_days_est` | float | courier ETA at checkout; tier baseline + festive congestion |
| `signup_date`, `account_age_days` | datetime, float | clipped to ≥1 day before first order |
| `order_velocity_24h` | int | same customer, **strictly earlier** orders in 24h |
| `rto` | int | **target** |

### `data/processed/{train,val,test}.parquet` — 42 columns

Raw plus causal history plus engineered features.

| Added | Notes |
|---|---|
| `past_orders`, `past_rto_count`, `past_rto_rate`, `has_history` | expanding over that customer's **strictly earlier** orders. First-ever order → `past_rto_rate = NaN`, `has_history = 0`. Imputation is a modelling decision made in 02/03, not baked in here. |
| `addr_token_count`, `addr_char_len`, `addr_digit_count`, `addr_comma_count`, `addr_has_house_number`, `addr_has_landmark`, `addr_gibberish_score`, `addr_quality_score` | extracted from the string by `src.features.address_features` |
| `tier_ordinal`, `log_order_value`, `discount_amount`, `log_account_age`, `is_first_order`, `pincode_prefix3` | row-local, fit nothing |

Target encoding of pincode is **not** in these files. It is fitted out-of-fold inside
the modelling folds in `03`, because that is the only place it can be done without
leaking.

### `data/interim/` — audit only

`latents.csv` holds `_pincode_rto_prior`, `_reliability_z`, `_address_quality`,
`_p_rto_true`. **Using any of these as a feature is leakage by definition.** They are
named in `FORBIDDEN_FEATURES` and a test asserts no `_`-prefixed column can reach the
feature matrix. They exist so the calibration and the Bayes ceiling can be audited.

## 5. Splits

| Split | Rows | Window | RTO rate | COD RTO rate |
|---|---|---|---|---|
| train | 35,000 | 2024-01-01 → 2025-01-22 | 0.170 | 0.266 |
| val | 5,000 | 2025-01-22 → 2025-03-16 | 0.155 | 0.239 |
| test | 10,000 | 2025-03-16 → 2025-06-30 | 0.147 | 0.231 |

Chronological, never random. The base rate drifts downward across the splits because
the festive windows sit inside the train period. **This is a property of the data, not
a bug** — and it is exactly why `05` measures performance across the full 18–35%
prevalence range instead of reporting a single number.

## 6. Difficulty — the Bayes ceiling

Scored against the true probability field the labels were drawn from:

| | |
|---|---|
| ROC-AUC | **0.869** |
| PR-AUC | **0.577** |
| Brier | **0.097** |
| True P(RTO) range | 0.0001 → 0.973, 9.0% of orders above 0.5 |

This is the ceiling, not a result. It is published so that **a model reporting above
it is read as leakage, not skill.**

## 7. Known limitations

1. **Synthetic labels.** The conditional structure is a modelling choice, calibrated
   at the margins but not validated against real outcomes. A model that works here is
   evidence of sound method, not of production performance.
2. **Coefficients are assumed.** Everything in `COEF` except the calibration targets is
   assumed. They are listed above so a reader can disagree with them specifically.
3. **One address per customer** (plus 12% alternate addresses). Real customers change
   addresses more often, which would weaken the address features.
4. **No courier dimension.** Real RTO varies substantially by courier partner; there is
   no public per-courier data to ground it, so it is omitted rather than invented.
5. **No seasonality beyond five sale windows** and a mild secular growth trend.
6. **State field reflects the directory as published** — e.g. Hyderabad appears under
   Andhra Pradesh rather than Telangana. This is real data as-is, not a correction we
   made silently.
7. **The gibberish detector has false positives** on genuine low-vowel Indian place
   tokens. Its marginal signal is weak, which is reported honestly in `02` rather than
   engineered away.

## 8. Reproducibility

One seeded `numpy.random.Generator` drives every label-determining draw, in a fixed
order; Faker is seeded from the same value and produces only cosmetic strings.

`01` ends by regenerating the entire dataset through `src.generate.build_dataset` and
asserting `assert_frame_equal` against what it just produced — all 50,000 rows and 32
columns. SHA-256 for every output file is written to `reports/results/01_manifest.json`.

## 9. Privacy

No real person's data is present. Names, phone numbers and address text are generated
by Faker. Any resemblance to a real customer record is coincidental.
