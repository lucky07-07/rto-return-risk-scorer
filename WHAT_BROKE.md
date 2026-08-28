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

*Open items and deviations from `PRE_REGISTRATION.md` are appended here as they occur.
Nothing has deviated so far.*
