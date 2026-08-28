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

*Open items and deviations from `PRE_REGISTRATION.md` will be appended here as they
occur. Nothing has deviated so far.*
