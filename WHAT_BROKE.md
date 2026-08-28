# What Broke

Kept as we go, not reconstructed afterwards. The buildathon asks what broke and how we
recovered; this is that log, including the things that were embarrassing.

---

## 1. `catboost==1.2.7` and `shap==0.46.0` have no Python 3.13 wheels

**Symptom.** `pip install -r requirements.txt` died building CatBoost's build
dependency `y-py` from source — `maturin failed … cargo rustc exit code 101`. The real
cause was two pins predating Python 3.13.

**Recovery.** Ran `pip install --only-binary=:all: --dry-run` to make pip *report*
which pins had no wheel instead of silently trying to compile them. Bumped
`catboost` 1.2.7 → 1.2.8 and `shap` 0.46.0 → 0.48.0, the earliest versions with cp313
wheels. Also pinned `pyarrow==25.0.1` explicitly rather than relying on it arriving as
a transitive dependency of Streamlit.

**Kept.** `--only-binary=:all:` is now the install command. If a pin ever lacks a wheel
again it fails in two seconds with a clear message instead of after a four-minute
Rust build.

---

## 2. Tier assignment from post-office density was wrong

**Symptom.** The first plan was to infer urbanisation from the India Post directory —
pincodes per city, or post offices per pincode. Checking it against known cities killed
it: Thrissur has 370 pincodes and Pune has 125; Mumbai averages 44.7 post offices per
pincode while Pune averages 1.35.

**Diagnosis.** Post-office density in the directory tracks **rural administrative
spread**, not urbanisation. A model built on it would have had a tier feature that was
confidently backwards, and nothing downstream would have flagged it.

**Recovery.** Tier is now assigned from curated, checkable metro and tier-1 city lists,
with district-headquarters towns as tier-2 and the remainder tier-3. It is documented
in `DATA_CARD.md` as a rule and an assumption, not as a measurement.

**Cost.** About twenty minutes, all of it before any code depended on the answer.

---

## 3. Fashion RTO would not reach the published 40%

**Symptom.** With fashion as a log-odds multiplier (`log(1.55)`), fashion COD RTO came
out at 31.2% against a published 40%+. Raising the coefficient barely helped —
1.0 → 2.0 moved fashion only 31.2% → 36.1% while crushing electronics from 18.1% to
12.3%. The global band calibration was pulling the marginals back every time.

**Diagnosis.** A single global intercept per band cannot hold two marginals at once.
Fashion is 30% of order volume, so a 40% fashion rate and a 26% overall COD rate imply
a ~20% non-fashion rate — reachable, but only if both are solved for explicitly.

**Recovery.** The calibration solve went from 6 cells to 12:
`{COD, prepaid} × {three value bands} × {fashion, other}`. Fashion carries the
published rate as a ratio to the COD rate; the non-fashion target in each cell is
whatever makes that cell's volume-weighted mean equal its band target. Fashion now
lands at 39.9% with every other published rate still held.

**Kept.** Added `fashion_rto_rate` to `config/evidence.yaml` and
`test_fashion_rto_matches_published_rate` to the suite, so this cannot regress
silently.

---

## 4. The leakage test failed *because the encoder was correct*

**Symptom.** `test_encoder_cannot_reproduce_a_one_row_per_key_target` failed with
`assert nan < 0.05`.

**Diagnosis.** The test asserted that the out-of-fold encoding correlates weakly with
the row's own label. With one row per pincode the encoder correctly falls back to the
prior for **every** row, so the encoding is constant, its standard deviation is zero,
and `np.corrcoef` returns NaN. The strongest possible pass was being read as a
failure.

**Recovery.** Rewrote the assertion to handle the degenerate case explicitly: the
encoding must not equal the labels, must be constant when there is nothing to learn
out of fold, and correlates weakly whenever it does vary. The test now fails for the
right reason and passes for the right reason.

**Kept.** A test whose failure mode is ambiguous is worse than no test. This one now
says which of the three properties broke.

---

## 5. Faker city names inside real pincodes

**Symptom.** Spot-checking generated addresses: `H No 72/405, Raval Marg, Alwar,
Coimbatore` on pincode 642139. "Alwar" is a Rajasthan city, dropped by `faker.city()`
into a Tamil Nadu address.

**Diagnosis.** Cosmetic, but it is exactly the kind of tell that makes a reviewer stop
trusting the rest of the data — and the address-quality features read that string.

**Recovery.** Dropped `faker.city()` from the address renderer. The locality component
is now built from the row's **real** city (`Coimbatore Puram`, `New Delhi East`), so
every token in the address is consistent with its pincode.

---

## 6. Notebook cells emitted without trailing newlines

**Symptom.** The first `nbconvert --execute` run died on cell 1 with
`SyntaxError: invalid syntax` on a line reading `import jsonimport sysfrom pathlib …`.

**Diagnosis.** The `.ipynb` `source` field is a list of lines **each ending in `\n`**,
not a list of bare strings. Splitting on `\n` without putting them back concatenated
the whole cell onto one line.

**Recovery.** Fixed the emitter to re-append `\n` to every line but the last. Also
added `id` fields to every cell to clear the `MissingIDFieldWarning` that nbformat now
raises.

---

## 7. Manifest recorded absolute paths

**Symptom.** `reports/results/01_manifest.json` keyed every entry as
`D:/M.TECH/SEM-3/Razor pay/data/raw/orders.csv`.

**Diagnosis.** A manifest exists so two runs can be compared. Absolute paths make it
uncomparable across machines — the exact failure it is supposed to prevent.

**Recovery.** `write_manifest` now takes a `root` and records repo-relative paths.

---

## 8. `order_velocity_24h` is a dead feature and we kept it anyway

**Symptom.** The univariate AUC screen in `02` put `order_velocity_24h` at **0.5008** —
indistinguishable from a coin flip — despite being a real driver in the generator
(`COEF["velocity"] = 0.30`).

**Diagnosis.** Not a bug in the feature. **99.0% of orders have a velocity of zero.**
The generator's order-arrival process spreads 50,000 orders over 20,000 customers and
547 days, so same-customer bursts inside a 24-hour window are vanishingly rare. A
feature that is constant for 99% of rows cannot move a marginal AUC however
informative its tail is.

**What we did not do.** Retune the arrival process to manufacture more bursts. That
would have been fitting the data to make a feature look good, which is the exact
failure mode this submission is supposed to be immune to.

**Recovery.** Kept the feature — a tree can still split on the thin tail, and dropping
columns for a weak *marginal* AUC is how interaction effects get discarded — and
declared it weak in `02`, **before any model has seen it**. Same for
`account_age_days` (AUC 0.508, largely collinear with `has_history`) and
`addr_gibberish_score` (AUC 0.507).

**Kept.** Declaring weak features in advance means a SHAP plot in `05` showing them
contributing nothing reads as a confirmed prediction rather than a discovery. If they
contribute nothing, they go in the honest exception list.

---

## 9. Two misleading lines of my own narrative in `02`

**Symptom.** The customer-history cell printed `past_rto_rate` AUC twice — "on
customers with history" and "on all train rows" — and both read **0.5646**.

**Diagnosis.** `single_feature_auc` drops NaN rows, so "all train rows" silently
became the same subset. The line claimed a comparison it was not making. Separately,
the velocity commentary asserted "97%+ of orders have velocity 0" from memory; the
actual figure is 99.0%.

**Recovery.** The second number now median-imputes exactly as the pipeline does, which
is what the model actually sees: **0.5646 with history → 0.5356 imputed across all
rows**, a real 0.029 AUC cost of dilution. The velocity figure is now computed in the
cell rather than typed.

**Kept.** Any number in prose that is not computed in the cell that prints it is a
number that will eventually be wrong.

---

## 10. DEVIATION: `class_weight='balanced'` promised in `02`, dropped in `03`

**What `02` said.** "`class_weight='balanced'` where the estimator supports it."

**What `03` does.** No class weighting anywhere. This is a deviation from a stated
plan, so it is recorded here with the evidence that overturned it.

**The evidence.** Fitting three models both ways on the validation split:

| Model | Δ PR-AUC | Δ Brier | mean predicted p (true base rate 0.155) |
|---|---|---|---|
| LightGBM | +0.0038 | **+0.0427** | 0.317 |
| Logistic Regression | −0.0021 | **+0.0695** | 0.378 |
| Random Forest | −0.0030 | **+0.0426** | 0.348 |

**Diagnosis.** Reweighting shifts the intercept. It more than doubles the mean
predicted probability against a true base rate of 0.155 and worsens Brier in every
case, while buying essentially nothing in PR-AUC — LightGBM's +0.004 is inside
bootstrap noise.

That trade is backwards for this system. The operating point is chosen at a **rupee
cost minimum** in `05`, which requires probabilities that mean what they say. We never
operate at 0.5, so a balanced-looking confusion matrix *at* 0.5 is worth nothing;
calibration is worth a great deal.

**Recovery.** Class weighting removed from every entry, and `class_weight` pinned to
`None` in the logistic-regression search space so `04` cannot reintroduce it. The
threshold moves on the cost curve instead of the data moving under the threshold.

**`PRE_REGISTRATION.md` impact: none.** It commits to "no resampling" and to Brier as
a first-class metric; this change tightens compliance with both rather than loosening
it.

---

## 11. No model beat logistic regression, and that is the finding

**Symptom.** After benchmarking 11 entries: Extra Trees leads validation PR-AUC at
0.4013, but a paired bootstrap against L2 logistic regression gives **zero models with
a confidence interval clear of zero**. Seven are statistically indistinguishable from
it; two (Decision Tree, HistGradientBoosting) are significantly *worse*. Logistic
regression has the best CV mean (0.4145) and the best Brier (0.1094), and fits in
0.5 seconds against CatBoost's 6.2.

**Diagnosis.** Not a bug. The generator in `01` builds labels from an **additive
log-odds model** plus Bernoulli noise, so a logistic regression on these features is
close to the correct functional form by construction. The boosters have to discover
that additive structure from data and pay variance for flexibility they do not need.

**What we did not do.** Quietly drop the bootstrap and report "Extra Trees wins by
0.0074". `PRE_REGISTRATION.md` commits to reporting this outcome if it occurred, and
it occurred.

**Scope, stated honestly.** This is a property of our synthetic generator, **not**
evidence that gradient boosting is a poor choice for real RTO data — where courier ×
corridor × seller × season interactions are exactly what a linear model misses.

**Recovery.** Logistic regression carried into the finalist set on its merits, not as
a courtesy: within noise of the best, fastest to fit, easiest to explain at checkout
latency. Finalists span three families (linear, bagging, boosting) so `04`'s
search-strategy comparison has something to compare across.

---

## 12. Hyperparameter tuning made the held-out score worse

**Symptom.** All six searches (3 finalists × 2 strategies) improved their objective —
mean CV gain +0.0149 PR-AUC, all six above the `03` default. Only 4 of 6 improved on
**validation**, and the net effect was **−0.0037**: the best untuned entry (Extra Trees,
0.4013) still beat the best tuned one (CatBoost/Optuna, 0.3977). Extra Trees was worst:
+0.0074 on CV, −0.0085 on validation.

**Diagnosis.** The tuning objective was overfitted. Three expanding folds of the train
split are a small, autocorrelated sample; a search running hundreds of trials against it
finds configurations that suit *those* folds. Validation sits later in time and does not
share the quirk, so the gain evaporates.

**What we did not do.** Re-tune against validation. That moves the overfitting one level
up and leaves nothing honest to select on. We also did not drop the untuned entries from
`05`'s selection pool to protect the time already spent tuning — that would be a
sunk-cost decision.

**Recovery.** `05` selects on validation from a pool containing both tuned configurations
and untuned `03` defaults. Added a CV-gain-vs-validation-gain transfer plot so the
failure is visible rather than buried in a summary line.

**What would actually help** (out of scope at this data size): more folds, repeated CV,
or nested CV — all of which cost compute that would have to come out of the search
budget the two strategies are being compared under.

**Also fixed here:** my own narrative described this as "a gain this small", which
describes a small *positive* number. The result was negative. The text now branches on
the sign and states the finding first.

---

## 13. `04` is not bit-reproducible, and cannot be

**Symptom.** Two runs of `04` with identical seeds produced different trial counts —
Optuna 252/163 pruned on the first run, 255/168 on the second.

**Diagnosis.** Not a bug. The budget both strategies receive is **wall clock**, which is
the deliberate fairness choice: Optuna is trial-native and FLAML is budget-native, so an
equal trial count would flatter Optuna. A wall-clock budget means trials completed depend
on machine load.

**Decision.** Keep the wall-clock budget. Fixing the trial count would make the notebook
reproducible and the comparison dishonest.

**Recovery.** Documented prominently in `04` itself: the space, seed, objective and folds
are fixed and the qualitative conclusions are stable across reruns (both strategies land
in the same region; CV gain does not transfer). Exact trial counts and third decimals are
not stable, and **no `04` number is quoted as a headline in the README.**

**Precisely what is and is not reproducible**, since the distinction matters:

| Notebook | Status | Verified how |
|---|---|---|
| `01` | **bit-reproducible** | full rerun leaves `01_manifest.json` byte-identical; the notebook also regenerates the dataset in-process and asserts frame equality |
| `02` | **bit-reproducible** | full rerun leaves `02_eda_summary.json` byte-identical |
| `03` | **numerically deterministic, not bit-identical** | see below |
| `04` | **not reproducible** | wall-clock budget, by design |
| `05` | deterministic **given `04`'s persisted artefacts** — not from scratch | it loads `models/04_*.joblib`, so it inherits `04`'s irreproducibility upstream |

**On `03`, a correction to an earlier claim in this file.** It was first written down here
as "bit-reproducible" on the assumption that fixed seeds were sufficient. They are not,
and a rerun proved it. Two causes:

1. The persisted outputs record **wall-clock fit times** (`fit_seconds`, `cv_seconds`),
   which obviously vary.
2. The tree ensembles run with `n_jobs=-1`, and **multi-threaded floating-point reduction
   is not associative** — partial sums combine in whatever order threads finish.

Measured drift across two full reruns: every leaderboard metric identical to the printed
precision, maximum deviation anywhere **2.2 × 10⁻¹⁶** in the validation predictions and
**2.8 × 10⁻¹⁷** in the class-weight experiment. Finalist selection, the bootstrap
comparison and the Bayes-ceiling check were identical. So the *conclusions* are stable to
machine epsilon; the *files* are not byte-identical, and the honest word is
"deterministic", not "bit-reproducible".

Forcing bit-identity would mean `n_jobs=1` throughout — several times the runtime for a
difference in the sixteenth decimal place. Not worth it, but worth stating rather than
claiming a reproducibility standard the repository does not actually meet.

So the chain from a clean checkout to `05`'s exact test numbers is **not** bit-reproducible,
and saying "everything is reproducible except `04`" would have been too generous. What
survives a rerun is the model family selected, the operating point to within the grid
resolution, and every qualitative conclusion.

---

## 14. The merchant-facing explanations were saying false things

**Symptom.** Spot-checking the example scored orders in `05`:

```
order ORD049801   Rs2,545 electronics, PREPAID
  RECOMMENDATION   allow COD
  because:  home_kitchen is a high-return category; the delivery location; the delivery location
```

Three separate defects in one four-line block.

**Diagnosis.**

1. **A false statement about the order.** The one-hot column `category_home_kitchen`
   carried a positive SHAP value on an *electronics* order — the model was being pushed
   up by the category's *absence*. `risk_reasons` rendered the column name as a fact
   about the order, so a merchant would have been told an electronics order was
   home_kitchen.
2. **A nonsensical recommendation.** The scorer is a COD gate; "allow COD" on an order
   that is already prepaid is not an action anyone can take.
3. **Duplicate reasons.** Two different one-hot columns both rendering as "the delivery
   location".

**Recovery.** `risk_reasons` now takes the feature *values* and only verbalises a one-hot
column when the order actually has that value; reasons are deduplicated; state, tier and
value-band columns get their own phrasing instead of a generic fallback. The example
block prints "not applicable - order is already prepaid" for prepaid orders, and draws its
low-risk examples from COD orders so the allow tier is demonstrated on an order the gate
would actually see.

**Kept.** The explanation layer is the part of this system a human reads and acts on. It
deserved the same scrutiny as the metrics and had been getting less — the numbers were
checked by assertions, the sentences by nobody.

---

## 15. The fee tier is the softest thing in the submission

**Symptom.** The three-tier optimiser put **56% of orders** in the "charge a COD fee"
tier.

**Diagnosis.** Look at the cost model and it is obvious. The fee is modelled as revenue
collected on delivered orders, so it offsets loss, and the only thing restraining the
tier is an assumed `fee_abandon_rate` of 18%. We effectively told the optimiser the fee
was close to free money, and it responded rationally.

`cod_fee_inr` and `fee_abandon_rate` are both marked `assumed` in `config/evidence.yaml`
with **no published source** — the two weakest numbers in the project.

**Recovery.** Not a silent parameter tweak to make the output look reasonable. `05` now
states the artefact explicitly, names the two assumptions responsible, says what would be
needed to fix it (real abandonment-vs-fee-level data from an A/B test a merchant can run
and we cannot), and directs the reader to the two-tier threshold — which rests only on
FN cost and margin — as the more defensible result.

---

## 16. Four improvement attempts, none accepted

**What was tried**, after the baseline was already established and reported. All five
challengers were selected on **validation**; none saw the test set.

| Challenger | val PR-AUC | Δ vs incumbent | 95% paired bootstrap CI | Verdict |
|---|---|---|---|---|
| *incumbent — CatBoost/FLAML* | **0.3970** | — | — | ships |
| Blend 0.50×LogReg + 0.50×CatBoost | 0.3982 | +0.0013 | [−0.0033, +0.0055] | tie |
| Calibrated (isotonic) | 0.3968 | −0.0002 | [−0.0137, +0.0135] | tie |
| Calibrated (sigmoid) | 0.3931 | −0.0039 | [−0.0174, +0.0098] | tie |
| Logistic Regression + interactions | 0.3929 | −0.0041 | [−0.0126, +0.0038] | tie |
| CatBoost + interactions | 0.3924 | −0.0046 | [−0.0155, +0.0075] | tie |

**Zero challengers had a confidence interval clear of zero.** The pre-registered rule
sends a tie to the simpler, already-validated model, so the incumbent ships unchanged.

**Three of these are informative rather than merely null:**

**Interactions.** Ten domain interactions (`src/features.py::INTERACTION_FEATURES`) —
COD×fashion, COD×impulse-band, address-quality×tier and so on. The prediction recorded
*before* running: trees gain little because they can already represent products; logistic
regression gains most because it cannot represent any. Observed on a like-for-like refit:
logistic **−0.0010**, CatBoost **−0.0007**. Both slightly negative.

The cause is structural and worth naming: **`01` builds labels that are additive in
log-odds**, so there is no true interaction in the data to recover. This experiment was
capped before it started. On real merchant data — courier × corridor × seller × season —
the same block would have far more to find. The negative result is about our generator,
not about the technique.

**Calibration made the rupee cost worse.** Not just neutral — worse:

| | own optimal threshold | val cost/order | ECE |
|---|---|---|---|
| uncalibrated (incumbent) | 0.370 | **₹27.30** | **0.0132** |
| calibrated (isotonic) | 0.340 | ₹27.81 | 0.0133 |
| calibrated (sigmoid) | 0.300 | ₹27.65 | 0.0350 |

The incumbent's ECE was already 0.0132 — there was no distortion to remove — and
cross-fitting a calibrator on three time-ordered folds costs more precision than it buys.
Sigmoid actively hurt, nearly tripling ECE by forcing a two-parameter logistic shape onto
a curve that did not need one.

Worth stating because calibration is usually a free win: **here it was not**, and shipping
it "because calibration is good practice" would have cost real money.

**Blending.** The two inputs correlate at **0.990**. The whole weight grid spans
0.3966–0.3982 in PR-AUC — a total range of 0.0016, well inside the bootstrap interval on
either endpoint. Averaging two models that make the same mistakes on the same orders
cannot fix those mistakes; the argmax weight of 0.50 is being read off noise.

**Cost of the exercise:** ~15 minutes of compute and one figure
(`reports/figures/05_challengers.png`, plus `05_blend_weight.png`). No metric moved. The
value is in knowing the baseline was not leaving anything obvious on the table.

---

## 17. The test set was re-executed, and it did not matter

**The risk.** The challenger experiments were added *after* the baseline had been scored
on test and published. Had a challenger been accepted, `05` would have become a second,
decision-informed read of the test set — exactly what `PRE_REGISTRATION.md` bought its
credibility from avoiding.

**What actually happened.** No challenger was accepted, so the model, threshold and tier
cut points are unchanged, and the test cell re-executes an identical computation.

Verified two ways, one of them repeatable:

* `05` section 14 snapshots the operating point *before* the challengers run and
  **asserts** it is unchanged afterwards. If anything below section 2a ever mutates it,
  the notebook fails rather than printing a reassuring sentence.
* Manually, once: `05_final_metrics.json`'s entire `test` block compared **equal** to the
  previously committed version.

`05_final_metrics.json` therefore records `test_evaluations_executed: 2` and
`test_informed_decisions: 1` as separate fields. An earlier draft of that file carried a
single `test_reads: 2` with a note calling this "the second read" — written before the
outcome was known, and left standing it would have contradicted this entry in the one file
a reader is most likely to grep. Caught on an audit pass, not by the tests.

**So the test set informed no decision.** The protocol holds — not because we were careful
afterwards, but because the validation-only rule produced a null result and there was
nothing to swap in.

**Had it gone the other way**, the honest report would have been a second read, with both
models' test numbers published side by side. The notebook contains that branch and says so.

---

## 18. The model is weakest on the merchants most likely to deploy it

**Symptom.** The population-shift study (`05` section 10) reweights the test set to seven
plausible merchant profiles. The ordering is uncomfortable:

| Profile | PR-AUC lift | Cost/order |
|---|---|---|
| Prepaid-led (35% COD) | **4.13×** | **₹16.24** |
| Electronics-led catalogue | 3.28× | ₹17.47 |
| Metro-heavy D2C | 2.74× | ₹23.42 |
| *Generated mix (baseline)* | *2.62×* | *₹26.51* |
| Small-town heavy | 2.36× | ₹32.24 |
| Fashion-led catalogue | 2.17× | ₹34.00 |
| **COD-dominant (85%)** | **1.97×** | **₹35.58** |

**Diagnosis.** Not a paradox — arithmetic. `is_cod` is the single strongest feature (`02`
measured it at AUC 0.711). A book that is almost entirely COD has had the model's best
discriminator flattened out of it, leaving geography, address quality and basket, which
are individually much weaker.

**Why it matters.** A merchant with 85% COD is precisely who needs a COD gate. They get
the *least* per-order help from it. The model still beats every no-model policy on all
seven profiles, so it remains worth deploying — but a COD-heavy merchant should plan on
the bottom of the reported range, not the headline figure.

This was not visible before the population-shift experiment; the aggregate test number
averages across a 62% COD book and hides it. Added to the exception list in `05`.

---

*Open items and deviations from `PRE_REGISTRATION.md` are appended here as they occur.*
