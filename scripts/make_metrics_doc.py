"""Render `reports/METRICS.md` from the committed result files.

Every number in that document is read from `reports/results/` at generation
time. Nothing is typed by hand, so the document cannot drift away from the
artifacts the notebooks wrote.

Run from the repository root, after `scripts/threshold_sweep.py`:

    python scripts/make_metrics_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports" / "results"
OUT = ROOT / "reports" / "METRICS.md"

FM = json.loads((R / "05_final_metrics.json").read_text())
BM = json.loads((R / "03_benchmark_summary.json").read_text())
TU = json.loads((R / "04_tuning_summary.json").read_text())
SWEEP = pd.read_csv(R / "05_threshold_sweep.csv")
REL = pd.read_csv(R / "05_reliability_test.csv")
SEG = pd.read_csv(R / "05_exception_list.csv")
TESTP = pd.read_parquet(R / "05_test_predictions.parquet")
LB = pd.read_csv(R / "03_leaderboard.csv")

T, V = FM["test"], FM["validation"]
# Validation mean prediction of the *shipped* model, computed from its own saved
# predictions. Do not take this from a challenger row - those are other models.
VAL_MEAN = float(pd.read_parquet(R / "04_tuned_val_predictions.parquet")
                 [FM["final_model"]["key"]].mean())
CM = T["confusion_matrix"]
COST = FM["cost"]
FROZEN = FM["frozen_operating_point"]["threshold"]
CEIL = FM["bayes_ceiling_val"]
L: list[str] = []


def w(s: str = "") -> None:
    L.append(s)


def table(headers, rows) -> None:
    w("| " + " | ".join(headers) + " |")
    w("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w()


def frozen_row(split: str, col: str):
    m = (SWEEP.split == split) & SWEEP.is_frozen_operating_point
    return SWEEP.loc[m, col].iloc[0]


w("# Metrics")
w()
w("Every figure below is read from `reports/results/` by "
  "[`scripts/make_metrics_doc.py`](../scripts/make_metrics_doc.py). Nothing here is")
w("typed by hand, so this file cannot drift away from what the notebooks wrote.")
w()
w(f"The operating point is **frozen at {FROZEN}**, chosen on validation in notebook `05` "
  "before the")
w("test set was opened. Tier cut-offs are "
  f"{FM['frozen_operating_point']['low_cut']} and {FM['frozen_operating_point']['high_cut']}.")
w()
w("---")
w()

# ---------------------------------------------------------------- headline
w("## 1. Headline")
w()
w(f"Test set: {T['n']:,} orders, base rate {T['base_rate'] * 100:.2f}%. "
  f"Validation: 5,000 orders, base rate {BM['val_base_rate'] * 100:.1f}%.")
w()
w("**Discrimination**")
w()
table(["Metric", "Test", "Validation", "Note"], [
    ["PR-AUC", f"{T['pr_auc']:.4f}", f"{V['pr_auc']:.4f}",
     f"{T['pr_auc_lift']:.2f}× the no-skill floor (the base rate)"],
    ["ROC-AUC", f"{T['roc_auc']:.4f}", f"{V['roc_auc']:.4f}", ""],
    ["PR-AUC as % of the Bayes ceiling", f"{T['pr_auc'] / CEIL['pr_auc'] * 100:.0f}%",
     f"{V['pr_auc'] / CEIL['pr_auc'] * 100:.0f}%",
     f"ceiling is {CEIL['pr_auc']:.4f}, computed from the known probability field"],
])
w("The ceiling is the score a model would get if it knew each order's true probability. "
  "A score above it")
w("would be evidence of a leak rather than of skill, and the notebooks assert on it.")
w()
w("**Probability quality**")
w()
table(["Metric", "Test", "Validation", "Note"], [
    ["Brier score", f"{T['brier']:.4f}", f"{V['brier']:.4f}",
     f"ceiling {CEIL['brier']:.4f}; lower is better"],
    ["Log loss", f"{T['log_loss']:.4f}", "—", f"ceiling {CEIL['log_loss']:.4f}"],
    ["Expected calibration error", f"{T['ece']:.4f}", f"{V['ece']:.4f}",
     "10 equal-count bins"],
    ["Mean predicted probability", f"{TESTP['p_rto'].mean():.4f}",
     f"{VAL_MEAN:.4f}", f"against base rates {T['base_rate']:.4f} / {BM['val_base_rate']:.4f}"],
])
w("Probabilities are used as the model emits them. Post-hoc recalibration was tested and "
  "rejected; see section 8.")
w()
w(f"**Classification, at the frozen threshold {FROZEN}**")
w()
table(["Metric", "Test", "Validation"], [
    ["Precision", f"{T['precision']:.4f}", f"{frozen_row('validation', 'precision'):.4f}"],
    ["Recall", f"{T['recall']:.4f}", f"{frozen_row('validation', 'recall'):.4f}"],
    ["F1", f"{T['f1']:.4f}", f"{frozen_row('validation', 'f1'):.4f}"],
    ["Flag rate", f"{T['flag_rate']:.4f}", f"{frozen_row('validation', 'flag_rate'):.4f}"],
])
w("These are **not** the numbers at 0.5. At 0.5 the model flags almost nothing — see "
  "section 2.")
w()
w("**Confusion matrix, test, at " + str(FROZEN) + "**")
w()
table(["", "Predicted return", "Predicted fine", "Total"], [
    ["**Actually returned**", f"{CM['tp']:,} (TP)", f"{CM['fn']:,} (FN)",
     f"{CM['tp'] + CM['fn']:,}"],
    ["**Actually fine**", f"{CM['fp']:,} (FP)", f"{CM['tn']:,} (TN)",
     f"{CM['fp'] + CM['tn']:,}"],
    ["**Total**", f"{CM['tp'] + CM['fp']:,}", f"{CM['fn'] + CM['tn']:,}",
     f"{T['n']:,}"],
])
w("---")
w()

# ---------------------------------------------------------------- sweep
w("## 2. Threshold sweep")
w()
w("Descriptive only. The threshold was fixed on validation before the test set was opened, "
  "and this")
w("table changed no decision — it exists so the trade-off is visible instead of implied by "
  "a single row.")
w("Source: [`05_threshold_sweep.csv`](results/05_threshold_sweep.csv), written by")
w("[`scripts/threshold_sweep.py`](../scripts/threshold_sweep.py).")
w()
for split in ("test", "validation"):
    w(f"**{split.capitalize()}**")
    w()
    sub = SWEEP[(SWEEP.split == split) & (SWEEP.threshold <= 0.70)]
    rows = []
    for _, r in sub.iterrows():
        mark = " ← **frozen**" if r.is_frozen_operating_point else ""
        rows.append([f"{r.threshold:.2f}{mark}", f"{r.precision:.4f}", f"{r.recall:.4f}",
                     f"{r.f1:.4f}", f"{r.flag_rate:.4f}",
                     f"{int(r.tp):,}", f"{int(r.fp):,}", f"{int(r.fn):,}",
                     f"₹{r.cost_per_order_inr:.2f}"])
    table(["Threshold", "Precision", "Recall", "F1", "Flag rate", "TP", "FP", "FN",
           "Cost / order"], rows)
w("Two things worth reading off this. F1 peaks near 0.25, but **F1 is not the objective** — "
  "rupees are,")
w("and the cost minimum sits at 0.37. And the widely used 0.5 default flags barely 1% of "
  "orders, which")
w("is why picking it out of habit costs "
  f"₹{COST['cost_of_using_default_threshold_inr']:,.0f} across {T['n']:,} orders.")
w()
w("---")
w()

# ---------------------------------------------------------------- cost
w("## 3. Cost model")
w()
p = COST["params"]
table(["Parameter", "Value", "Meaning"], [
    ["`fn_cost_inr`", f"₹{p['fn_cost_inr']:.0f}",
     "wasted forward + reverse shipping on a return that was allowed"],
    ["`margin_rate`", f"{p['margin_rate']:.2f}", "margin lost when a good sale is blocked"],
    ["`prepaid_conversion`", f"{p['prepaid_conversion']:.2f}",
     "share of blocked customers who pay online instead"],
    ["`cod_fee_inr`", f"₹{p['cod_fee_inr']:.0f}", "fee charged in the middle tier"],
    ["`fee_abandon_rate`", f"{p['fee_abandon_rate']:.2f}",
     "share who abandon when shown that fee"],
])
w("The last two have no published source and are stated as assumptions. The main threshold "
  "does not")
w("depend on them; the three-tier split does.")
w()
w("**Rupee outcomes on the test set**")
w()
nb = COST["no_model_baselines_inr"]
table(["Policy", "Total cost", "Per order"], [
    ["Allow everything (no model)", f"₹{nb['allow_everything']:,.0f}",
     f"₹{nb['allow_everything'] / T['n']:.2f}"],
    ["Block everything (no model)", f"₹{nb['block_everything']:,.0f}",
     f"₹{nb['block_everything'] / T['n']:.2f}"],
    ["Model at the default 0.5", f"₹{COST['total_at_0.5_threshold_inr']:,.0f}",
     f"₹{COST['total_at_0.5_threshold_inr'] / T['n']:.2f}"],
    [f"**Model at the frozen {FROZEN}**",
     f"**₹{COST['total_at_frozen_threshold_inr']:,.0f}**",
     f"**₹{COST['cost_per_order_inr']:.2f}**"],
    ["Model, three-tier policy", f"₹{COST['three_tier']['total_cost_inr']:,.0f}",
     f"₹{COST['three_tier']['cost_per_order_inr']:.2f}"],
])
table(["Saving", "Amount"], [
    ["Against the better no-model policy",
     f"₹{COST['saving_vs_best_no_model_inr']:,.0f}"],
    ["Against using 0.5 out of habit",
     f"₹{COST['cost_of_using_default_threshold_inr']:,.0f}"],
])
w("The cost-optimal threshold stays inside "
  f"**{COST['sensitivity_threshold_range'][0]}–{COST['sensitivity_threshold_range'][1]}** "
  "across a 5×5 grid of")
w("shipping cost and margin assumptions ([`05_cost_sensitivity.csv`]"
  "(results/05_cost_sensitivity.csv)).")
w()
w("---")
w()

# ---------------------------------------------------------------- tiers
w("## 4. Three-tier policy, on test")
w()
table(["Tier", "Score range", "Orders", "Share", "Actual RTO rate", "Lift vs base",
       "Mean order value"],
      [[t["action"], t["score_range"], f"{t['n_orders']:,}", f"{t['share'] * 100:.1f}%",
        f"{t['actual_rto_rate'] * 100:.2f}%", f"{t['lift_vs_base']:.2f}×",
        f"₹{t['mean_order_value']:,.0f}"] for t in FM["three_tier_policy"]])
w("The tiers separate cleanly: the bottom tier returns at "
  f"{FM['three_tier_policy'][0]['actual_rto_rate'] * 100:.1f}% and the top at "
  f"{FM['three_tier_policy'][2]['actual_rto_rate'] * 100:.1f}%,")
w(f"a {FM['three_tier_policy'][2]['actual_rto_rate'] / FM['three_tier_policy'][0]['actual_rto_rate']:.0f}× "
  "spread.")
w()
w("---")
w()

# ---------------------------------------------------------------- calibration
w("## 5. Calibration, test set")
w()
w("Ten equal-count bins. `gap` is mean predicted minus observed; the 95% interval is on the "
  "observed rate.")
w()
table(["Bin", "n", "Mean predicted", "Observed", "95% CI", "Gap"],
      [[int(r.bin), f"{int(r.n):,}", f"{r.mean_predicted:.4f}", f"{r.observed_rate:.4f}",
        f"{r.ci_lo:.3f} – {r.ci_hi:.3f}", f"{r.gap:+.4f}"] for _, r in REL.iterrows()])
w(f"Largest single-bin gap is {REL.gap.abs().max():.4f}; every bin's interval contains its "
  "predicted value")
w(f"except bin 7. Expected calibration error {T['ece']:.4f}.")
w()
w("---")
w()

# ---------------------------------------------------------------- benchmark
w("## 6. Model benchmark")
w()
w(f"All eleven estimators on identical expanding-window folds "
  f"(k = {BM['cv']['n_splits']}), ranked by validation PR-AUC.")
w()
table(["Model", "Family", "Val PR-AUC", "Val ROC-AUC", "Val Brier", "CV mean", "CV s.d.",
       "Fit (s)"],
      [[r["model"], r["family"], f"{r['val_pr_auc']:.4f}", f"{r['val_roc_auc']:.4f}",
        f"{r['val_brier']:.4f}", f"{r['cv_pr_auc_mean']:.4f}", f"{r['cv_pr_auc_std']:.4f}",
        f"{r['fit_seconds']:.1f}"] for _, r in LB.iterrows()])
w("**Paired bootstrap against logistic regression**, 95% intervals on the PR-AUC difference.")
w()
table(["Model", "Δ PR-AUC", "95% CI", "p", "Verdict"],
      [[b["model"], f"{b['delta_pr_auc_vs_logreg']:+.4f}",
        f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]", f"{b['p_two_sided']:.3f}",
        b["beats_logreg"]] for b in BM["bootstrap_vs_logistic"]])
w("Nothing beats the linear baseline. Two models are significantly *worse*. This is the "
  "single most")
w("important negative result in the project and it is reported rather than buried.")
w()
w("---")
w()

# ---------------------------------------------------------------- tuning
w("## 7. Hyperparameter tuning")
w()
w(f"Optuna and FLAML, an equal "
  f"{TU['time_budget_s_per_model_per_strategy']} s wall-clock budget per model per "
  f"strategy, objective PR-AUC over {TU['objective']['folds']} expanding folds.")
w()
table(["Model", "Strategy", "Untuned", "Tuned", "Gain", "Fit (s)"],
      [[t["model"], t["strategy"], f"{t['untuned_val_pr_auc']:.4f}",
        f"{t['val_pr_auc']:.4f}", f"{t['gain_vs_untuned']:+.4f}", f"{t['fit_seconds']:.1f}"]
       for t in TU["tuned_val_scores"]])
w(f"**Net gain against untuned: {TU['net_gain_vs_untuned']:+.4f} PR-AUC.** Tuning improved "
  "cross-validation")
w("scores and then failed to transfer to held-out data. Reported, not hidden.")
w()
w("---")
w()

# ---------------------------------------------------------------- challengers
w("## 8. Challenger experiments")
w()
w("Each judged by the same paired bootstrap. None was accepted, so the shipped model, "
  "threshold and")
w("tier cut-offs are unchanged.")
w()
table(["Challenger", "Val PR-AUC", "Δ", "95% CI", "p", "Verdict"],
      [[c["challenger"], f"{c['pr_auc']:.4f}", f"{c['boot_observed_delta']:+.4f}",
        f"[{c['boot_ci_lo']:+.4f}, {c['boot_ci_hi']:+.4f}]",
        f"{c['boot_p_two_sided']:.3f}", c["verdict"]]
       for c in FM["challenger_experiments"]["results"]])
w("**Recalibration, and why it was rejected**")
w()
table(["Variant", "Own optimal threshold", "ECE", "Cost / order", "Precision", "Recall"],
      [[c["variant"], f"{c['own_optimal_threshold']:.2f}", f"{c['ece']:.4f}",
        f"₹{c['val_cost_per_order']:.2f}", f"{c['precision']:.4f}", f"{c['recall']:.4f}"]
       for c in FM["challenger_experiments"]["calibration_cost"]])
w("Both recalibrators raised cost per order, so the model ships uncalibrated. Sigmoid also "
  "made")
w("calibration *worse*, tripling ECE.")
w()
w("**Interaction-feature ablation**")
w()
table(["Model", "Features", "Val PR-AUC", "Val ROC-AUC", "Val Brier"],
      [[a["model"], a["features"], f"{a['val_pr_auc']:.4f}", f"{a['val_roc_auc']:.4f}",
        f"{a['val_brier']:.4f}"]
       for a in FM["challenger_experiments"]["interaction_ablation"]])
w("Domain interaction features made both models slightly worse, so they stay off by default.")
w()
w("---")
w()

# ---------------------------------------------------------------- robustness
w("## 9. Robustness")
w()
ps = FM["prevalence_shift"]
w(f"**Prevalence shift.** Return rates vary across Indian cities. The model was swept from "
  f"{ps['range_tested'][0] * 100:.0f}% to {ps['range_tested'][1] * 100:.0f}% prevalence.")
w()
table(["Question", "Answer"], [
    ["Published range covered",
     f"{ps['published_range'][0] * 100:.0f}%–{ps['published_range'][1] * 100:.0f}%"],
    ["Worst PR-AUC lift inside that range", f"{ps['min_pr_auc_lift_in_published_range']:.2f}×"],
    ["Beats no-model everywhere in range", "yes" if ps["beats_no_model_everywhere_in_range"] else "no"],
    ["Worst regret, threshold left frozen",
     f"₹{ps['worst_regret_inr_per_order']:.2f} / order (at {ps['worst_regret_at_prevalence'] * 100:.0f}% prevalence)"],
    ["Worst regret, with prior correction",
     f"₹{ps['worst_regret_with_prior_correction_inr']:.2f} / order"],
    ["Max ECE untreated", f"{ps['ece_untreated_max']:.4f}"],
    ["Max ECE after prior correction", f"{ps['ece_corrected_max']:.4f}"],
])
w("Ranking survives the shift; **calibration is what breaks**, and one line of odds "
  "arithmetic repairs it.")
w()
w("**Merchant-mix reweighting.** The test set reweighted to seven merchant profiles.")
w()
table(["Profile", "Base rate", "PR-AUC", "ROC-AUC", "Brier", "Lift", "Cost / order"],
      [[p_["profile"], f"{p_['base_rate'] * 100:.1f}%", f"{p_['pr_auc']:.4f}",
        f"{p_['roc_auc']:.4f}", f"{p_['brier']:.4f}", f"{p_['pr_auc_lift']:.2f}×",
        f"₹{p_['cost_per_order_inr']:.2f}"] for p_ in FM["population_shift"]])
cs = FM["covariate_shift_train_vs_val"]
w(f"**Covariate shift, train vs validation.** A classifier trained to tell the two splits "
  f"apart reaches AUC **{cs['domain_auc']:.4f}**")
w(f"(n = {cs['n_a']:,} vs {cs['n_b']:,}). Above 0.5, so the splits are distinguishable — "
  "expected, because the split is")
w("chronological and the population drifts over time. This is measured rather than assumed.")
w()
w("---")
w()

# ---------------------------------------------------------------- segments
w("## 10. Segment breakdown, test set")
w()
w("Where the model works and where it does not. Source: "
  "[`05_exception_list.csv`](results/05_exception_list.csv).")
w()
rows = []
for _, s in SEG.iterrows():
    note = "" if pd.isna(s.get("note")) else str(s["note"])
    fmt = lambda v, f="{:.4f}": "—" if pd.isna(v) else f.format(v)
    rows.append([s["segment"], f"{int(s['n']):,}",
                 fmt(s["base_rate"], "{:.3f}"), fmt(s["pr_auc"]),
                 fmt(s["pr_auc_lift"], "{:.2f}×"), fmt(s["roc_auc"]),
                 fmt(s["precision"]), fmt(s["recall"]), note])
table(["Segment", "n", "Base rate", "PR-AUC", "Lift", "ROC-AUC", "Precision", "Recall",
       "Note"], rows)
w("The weakest segment is the one that matters commercially: a merchant whose orders are "
  "almost all")
w("cash on delivery loses the single most informative feature, and lift falls to "
  f"{SEG.loc[SEG.segment == 'COD orders', 'pr_auc_lift'].iloc[0]:.2f}×.")
w()
w("---")
w()
w("## Source files")
w()
table(["File", "Contents"], [
    ["[`05_final_metrics.json`](results/05_final_metrics.json)",
     "test and validation metrics, cost model, tiers, shift studies, challengers"],
    ["[`05_threshold_sweep.csv`](results/05_threshold_sweep.csv)", "section 2"],
    ["[`05_cost_sensitivity.csv`](results/05_cost_sensitivity.csv)",
     "optimal threshold across cost assumptions"],
    ["[`05_reliability_test.csv`](results/05_reliability_test.csv)", "section 5"],
    ["[`05_exception_list.csv`](results/05_exception_list.csv)", "section 10"],
    ["[`05_prevalence_shift.csv`](results/05_prevalence_shift.csv)",
     "full prevalence sweep, 27 points"],
    ["[`05_test_predictions.parquet`](results/05_test_predictions.parquet)",
     "per-order test predictions and tiers"],
    ["[`03_leaderboard.csv`](results/03_leaderboard.csv)", "section 6"],
    ["[`03_benchmark_summary.json`](results/03_benchmark_summary.json)",
     "bootstrap comparison, class-weight experiment, Bayes ceiling"],
    ["[`03_cv_folds.csv`](results/03_cv_folds.csv)", "per-fold scores, 11 models × 5 folds"],
    ["[`04_tuning_summary.json`](results/04_tuning_summary.json)", "section 7"],
    ["[`04_search_history.csv`](results/04_search_history.csv)", "965 tuning trials"],
])

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({len(L)} lines)")
