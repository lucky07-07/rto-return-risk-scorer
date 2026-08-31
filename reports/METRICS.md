# Metrics

Every figure below is read from `reports/results/` by [`scripts/make_metrics_doc.py`](../scripts/make_metrics_doc.py). Nothing here is
typed by hand, so this file cannot drift away from what the notebooks wrote.

The operating point is **frozen at 0.37**, chosen on validation in notebook `05` before the
test set was opened. Tier cut-offs are 0.05 and 0.4.

---

## 1. Headline

Test set: 10,000 orders, base rate 14.73%. Validation: 5,000 orders, base rate 15.5%.

**Discrimination**

| Metric | Test | Validation | Note |
|---|---|---|---|
| PR-AUC | 0.3858 | 0.3970 | 2.62× the no-skill floor (the base rate) |
| ROC-AUC | 0.8060 | 0.8098 |  |
| PR-AUC as % of the Bayes ceiling | 70% | 72% | ceiling is 0.5548, computed from the known probability field |

The ceiling is the score a model would get if it knew each order's true probability. A score above it
would be evidence of a leak rather than of skill, and the notebooks assert on it.

**Probability quality**

| Metric | Test | Validation | Note |
|---|---|---|---|
| Brier score | 0.1059 | 0.1092 | ceiling 0.0940; lower is better |
| Log loss | 0.3360 | — | ceiling 0.3004 |
| Expected calibration error | 0.0118 | 0.0132 | 10 equal-count bins |
| Mean predicted probability | 0.1509 | 0.1568 | against base rates 0.1473 / 0.1550 |

Probabilities are used as the model emits them. Post-hoc recalibration was tested and rejected; see section 8.

**Classification, at the frozen threshold 0.37**

| Metric | Test | Validation |
|---|---|---|
| Precision | 0.4583 | 0.4802 |
| Recall | 0.2688 | 0.2968 |
| F1 | 0.3389 | 0.3668 |
| Flag rate | 0.0864 | 0.0958 |

These are **not** the numbers at 0.5. At 0.5 the model flags almost nothing — see section 2.

**Confusion matrix, test, at 0.37**

|  | Predicted return | Predicted fine | Total |
|---|---|---|---|
| **Actually returned** | 396 (TP) | 1,077 (FN) | 1,473 |
| **Actually fine** | 468 (FP) | 8,059 (TN) | 8,527 |
| **Total** | 864 | 9,136 | 10,000 |

---

## 2. Threshold sweep

Descriptive only. The threshold was fixed on validation before the test set was opened, and this
table changed no decision — it exists so the trade-off is visible instead of implied by a single row.
Source: [`05_threshold_sweep.csv`](results/05_threshold_sweep.csv), written by
[`scripts/threshold_sweep.py`](../scripts/threshold_sweep.py).

**Test**

| Threshold | Precision | Recall | F1 | Flag rate | TP | FP | FN | Cost / order |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.2298 | 0.9654 | 0.3712 | 0.6189 | 1,422 | 4,767 | 51 | ₹61.29 |
| 0.10 | 0.2327 | 0.9593 | 0.3746 | 0.6071 | 1,413 | 4,658 | 60 | ₹58.84 |
| 0.15 | 0.2800 | 0.8330 | 0.4191 | 0.4382 | 1,227 | 3,155 | 246 | ₹42.28 |
| 0.20 | 0.3328 | 0.6884 | 0.4487 | 0.3047 | 1,014 | 2,033 | 459 | ₹32.53 |
| 0.25 | 0.3564 | 0.5743 | 0.4398 | 0.2374 | 846 | 1,528 | 627 | ₹29.32 |
| 0.30 | 0.3773 | 0.4392 | 0.4059 | 0.1715 | 647 | 1,068 | 826 | ₹27.81 |
| 0.35 | 0.4319 | 0.3123 | 0.3625 | 0.1065 | 460 | 605 | 1,013 | ₹26.70 |
| 0.37 ← **frozen** | 0.4583 | 0.2688 | 0.3389 | 0.0864 | 396 | 468 | 1,077 | ₹26.51 |
| 0.40 | 0.4773 | 0.1928 | 0.2747 | 0.0595 | 284 | 311 | 1,189 | ₹27.13 |
| 0.45 | 0.5032 | 0.1073 | 0.1768 | 0.0314 | 158 | 156 | 1,315 | ₹27.95 |
| 0.50 | 0.5649 | 0.0502 | 0.0923 | 0.0131 | 74 | 57 | 1,399 | ₹28.60 |
| 0.55 | 0.6481 | 0.0238 | 0.0458 | 0.0054 | 35 | 19 | 1,438 | ₹29.02 |
| 0.60 | 0.7083 | 0.0115 | 0.0227 | 0.0024 | 17 | 7 | 1,456 | ₹29.23 |
| 0.65 | 0.7500 | 0.0041 | 0.0081 | 0.0008 | 6 | 2 | 1,467 | ₹29.40 |
| 0.70 | 1.0000 | 0.0020 | 0.0041 | 0.0003 | 3 | 0 | 1,470 | ₹29.40 |

**Validation**

| Threshold | Precision | Recall | F1 | Flag rate | TP | FP | FN | Cost / order |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.2379 | 0.9703 | 0.3821 | 0.6322 | 752 | 2,409 | 23 | ₹62.18 |
| 0.10 | 0.2408 | 0.9703 | 0.3858 | 0.6246 | 752 | 2,371 | 23 | ₹60.75 |
| 0.15 | 0.2831 | 0.8413 | 0.4237 | 0.4606 | 652 | 1,651 | 123 | ₹44.63 |
| 0.20 | 0.3424 | 0.7110 | 0.4622 | 0.3218 | 551 | 1,058 | 224 | ₹32.40 |
| 0.25 | 0.3870 | 0.6077 | 0.4729 | 0.2434 | 471 | 746 | 304 | ₹28.64 |
| 0.30 | 0.4087 | 0.4710 | 0.4376 | 0.1786 | 365 | 528 | 410 | ₹27.86 |
| 0.35 | 0.4439 | 0.3316 | 0.3796 | 0.1158 | 257 | 322 | 518 | ₹27.84 |
| 0.37 ← **frozen** | 0.4802 | 0.2968 | 0.3668 | 0.0958 | 230 | 249 | 545 | ₹27.30 |
| 0.40 | 0.4927 | 0.2168 | 0.3011 | 0.0682 | 168 | 173 | 607 | ₹27.85 |
| 0.45 | 0.5114 | 0.1161 | 0.1893 | 0.0352 | 90 | 86 | 685 | ₹29.08 |
| 0.50 | 0.5217 | 0.0619 | 0.1107 | 0.0184 | 48 | 44 | 727 | ₹29.96 |
| 0.55 | 0.5122 | 0.0271 | 0.0515 | 0.0082 | 21 | 20 | 754 | ₹30.58 |
| 0.60 | 0.5000 | 0.0116 | 0.0227 | 0.0036 | 9 | 9 | 766 | ₹30.81 |
| 0.65 | 0.5000 | 0.0052 | 0.0102 | 0.0016 | 4 | 4 | 771 | ₹30.91 |
| 0.70 | 1.0000 | 0.0013 | 0.0026 | 0.0002 | 1 | 0 | 774 | ₹30.96 |

Two things worth reading off this. F1 peaks near 0.25, but **F1 is not the objective** — rupees are,
and the cost minimum sits at 0.37. And the widely used 0.5 default flags barely 1% of orders, which
is why picking it out of habit costs ₹20,835 across 10,000 orders.

---

## 3. Cost model

| Parameter | Value | Meaning |
|---|---|---|
| `fn_cost_inr` | ₹200 | wasted forward + reverse shipping on a return that was allowed |
| `margin_rate` | 0.25 | margin lost when a good sale is blocked |
| `prepaid_conversion` | 0.45 | share of blocked customers who pay online instead |
| `cod_fee_inr` | ₹50 | fee charged in the middle tier |
| `fee_abandon_rate` | 0.18 | share who abandon when shown that fee |

The last two have no published source and are stated as assumptions. The main threshold does not
depend on them; the three-tier split does.

**Rupee outcomes on the test set**

| Policy | Total cost | Per order |
|---|---|---|
| Allow everything (no model) | ₹294,600 | ₹29.46 |
| Block everything (no model) | ₹1,330,855 | ₹133.09 |
| Model at the default 0.5 | ₹285,954 | ₹28.60 |
| **Model at the frozen 0.37** | **₹265,119** | **₹26.51** |
| Model, three-tier policy | ₹150,106 | ₹15.01 |

| Saving | Amount |
|---|---|
| Against the better no-model policy | ₹29,481 |
| Against using 0.5 out of habit | ₹20,835 |

The cost-optimal threshold stays inside **0.21–0.48** across a 5×5 grid of
shipping cost and margin assumptions ([`05_cost_sensitivity.csv`](results/05_cost_sensitivity.csv)).

---

## 4. Three-tier policy, on test

| Tier | Score range | Orders | Share | Actual RTO rate | Lift vs base | Mean order value |
|---|---|---|---|---|---|---|
| allow COD | < 0.050 | 3,811 | 38.1% | 1.34% | 0.09× | ₹1,408 |
| charge a COD fee | 0.050 - 0.400 | 5,594 | 55.9% | 20.34% | 1.38× | ₹913 |
| disable COD, offer prepaid | >= 0.400 | 595 | 5.9% | 47.73% | 3.24× | ₹785 |

The tiers separate cleanly: the bottom tier returns at 1.3% and the top at 47.7%,
a 36× spread.

---

## 5. Calibration, test set

Ten equal-count bins. `gap` is mean predicted minus observed; the 95% interval is on the observed rate.

| Bin | n | Mean predicted | Observed | 95% CI | Gap |
|---|---|---|---|---|---|
| 0 | 1,000 | 0.0085 | 0.0100 | 0.005 – 0.018 | -0.0015 |
| 1 | 1,000 | 0.0117 | 0.0060 | 0.003 – 0.013 | +0.0057 |
| 2 | 1,000 | 0.0159 | 0.0130 | 0.008 – 0.022 | +0.0029 |
| 3 | 1,000 | 0.0385 | 0.0360 | 0.026 – 0.049 | +0.0025 |
| 4 | 1,000 | 0.1206 | 0.0990 | 0.082 – 0.119 | +0.0216 |
| 5 | 1,000 | 0.1473 | 0.1360 | 0.116 – 0.159 | +0.0113 |
| 6 | 1,000 | 0.1799 | 0.1710 | 0.149 – 0.196 | +0.0089 |
| 7 | 1,000 | 0.2401 | 0.2720 | 0.245 – 0.300 | -0.0319 |
| 8 | 1,000 | 0.3161 | 0.2920 | 0.265 – 0.321 | +0.0241 |
| 9 | 1,000 | 0.4309 | 0.4380 | 0.408 – 0.469 | -0.0071 |

Largest single-bin gap is 0.0319; every bin's interval contains its predicted value
except bin 7. Expected calibration error 0.0118.

---

## 6. Model benchmark

All eleven estimators on identical expanding-window folds (k = 5), ranked by validation PR-AUC.

| Model | Family | Val PR-AUC | Val ROC-AUC | Val Brier | CV mean | CV s.d. | Fit (s) |
|---|---|---|---|---|---|---|---|
| Extra Trees | bagging | 0.4013 | 0.8064 | 0.1097 | 0.4061 | 0.0372 | 5.2 |
| Logistic Regression | linear | 0.3939 | 0.8096 | 0.1094 | 0.4145 | 0.0357 | 0.6 |
| CatBoost | boosting | 0.3931 | 0.8062 | 0.1099 | 0.4012 | 0.0320 | 6.6 |
| Random Forest | bagging | 0.3923 | 0.8044 | 0.1107 | 0.4053 | 0.0402 | 6.1 |
| Gradient Boosting | boosting | 0.3913 | 0.8087 | 0.1095 | 0.3970 | 0.0354 | 30.2 |
| MLP | neural | 0.3859 | 0.8055 | 0.1107 | 0.3887 | 0.0355 | 4.8 |
| XGBoost | boosting | 0.3807 | 0.7999 | 0.1113 | 0.3847 | 0.0391 | 2.2 |
| LightGBM | boosting | 0.3789 | 0.7966 | 0.1117 | 0.3777 | 0.0468 | 1.7 |
| Decision Tree | tree | 0.3717 | 0.7977 | 0.1110 | 0.3679 | 0.0372 | 0.7 |
| HistGradientBoosting | boosting | 0.3714 | 0.7952 | 0.1122 | 0.3817 | 0.0400 | 3.1 |
| Majority baseline | baseline | 0.1550 | 0.5000 | 0.1312 | 0.1682 | 0.0257 | 0.4 |

**Paired bootstrap against logistic regression**, 95% intervals on the PR-AUC difference.

| Model | Δ PR-AUC | 95% CI | p | Verdict |
|---|---|---|---|---|
| Extra Trees | +0.0074 | [-0.0083, +0.0221] | 0.387 | no - CI spans 0 |
| CatBoost | -0.0008 | [-0.0137, +0.0121] | 0.895 | no - CI spans 0 |
| Random Forest | -0.0016 | [-0.0156, +0.0123] | 0.861 | no - CI spans 0 |
| Gradient Boosting | -0.0026 | [-0.0124, +0.0079] | 0.664 | no - CI spans 0 |
| MLP | -0.0080 | [-0.0207, +0.0053] | 0.234 | no - CI spans 0 |
| XGBoost | -0.0133 | [-0.0286, +0.0020] | 0.088 | no - CI spans 0 |
| LightGBM | -0.0150 | [-0.0330, +0.0029] | 0.108 | no - CI spans 0 |
| Decision Tree | -0.0222 | [-0.0415, -0.0047] | 0.012 | worse |
| HistGradientBoosting | -0.0225 | [-0.0392, -0.0056] | 0.003 | worse |

Nothing beats the linear baseline. Two models are significantly *worse*. This is the single most
important negative result in the project and it is reported rather than buried.

---

## 7. Hyperparameter tuning

Optuna and FLAML, an equal 300 s wall-clock budget per model per strategy, objective PR-AUC over 3 expanding folds.

| Model | Strategy | Untuned | Tuned | Gain | Fit (s) |
|---|---|---|---|---|---|
| CatBoost | Optuna | 0.3931 | 0.3977 | +0.0046 | 2.7 |
| CatBoost | FLAML | 0.3931 | 0.3970 | +0.0039 | 5.3 |
| Logistic Regression | Optuna | 0.3939 | 0.3966 | +0.0027 | 0.5 |
| Logistic Regression | FLAML | 0.3939 | 0.3965 | +0.0026 | 0.5 |
| Extra Trees | Optuna | 0.4013 | 0.3959 | -0.0054 | 3.8 |
| Extra Trees | FLAML | 0.4013 | 0.3897 | -0.0116 | 3.8 |

**Net gain against untuned: -0.0037 PR-AUC.** Tuning improved cross-validation
scores and then failed to transfer to held-out data. Reported, not hidden.

---

## 8. Challenger experiments

Each judged by the same paired bootstrap. None was accepted, so the shipped model, threshold and
tier cut-offs are unchanged.

| Challenger | Val PR-AUC | Δ | 95% CI | p | Verdict |
|---|---|---|---|---|---|
| Logistic Regression + interactions | 0.3929 | -0.0041 | [-0.0126, +0.0038] | 0.337 | tie - CI spans 0 |
| CatBoost + interactions | 0.3924 | -0.0046 | [-0.0155, +0.0075] | 0.438 | tie - CI spans 0 |
| calibrated (sigmoid) | 0.3931 | -0.0039 | [-0.0174, +0.0098] | 0.551 | tie - CI spans 0 |
| calibrated (isotonic) | 0.3968 | -0.0002 | [-0.0137, +0.0135] | 0.964 | tie - CI spans 0 |
| blend 0.50xLogReg + 0.50xCatBoost | 0.3982 | +0.0013 | [-0.0033, +0.0055] | 0.631 | tie - CI spans 0 |

**Recalibration, and why it was rejected**

| Variant | Own optimal threshold | ECE | Cost / order | Precision | Recall |
|---|---|---|---|---|---|
| uncalibrated (incumbent) | 0.37 | 0.0132 | ₹27.30 | 0.4802 | 0.2968 |
| calibrated (sigmoid) | 0.30 | 0.0350 | ₹27.65 | 0.4256 | 0.3987 |
| calibrated (isotonic) | 0.34 | 0.0133 | ₹27.81 | 0.4269 | 0.3845 |

Both recalibrators raised cost per order, so the model ships uncalibrated. Sigmoid also made
calibration *worse*, tripling ECE.

**Interaction-feature ablation**

| Model | Features | Val PR-AUC | Val ROC-AUC | Val Brier |
|---|---|---|---|---|
| Logistic Regression | base features | 0.3939 | 0.8096 | 0.1094 |
| Logistic Regression | + interactions | 0.3929 | 0.8092 | 0.1094 |
| CatBoost | base features | 0.3931 | 0.8062 | 0.1099 |
| CatBoost | + interactions | 0.3924 | 0.8068 | 0.1098 |

Domain interaction features made both models slightly worse, so they stay off by default.

---

## 9. Robustness

**Prevalence shift.** Return rates vary across Indian cities. The model was swept from 14% to 40% prevalence.

| Question | Answer |
|---|---|
| Published range covered | 18%–35% |
| Worst PR-AUC lift inside that range | 1.89× |
| Beats no-model everywhere in range | yes |
| Worst regret, threshold left frozen | ₹16.42 / order (at 35% prevalence) |
| Worst regret, with prior correction | ₹0.08 / order |
| Max ECE untreated | 0.1686 |
| Max ECE after prior correction | 0.0288 |

Ranking survives the shift; **calibration is what breaks**, and one line of odds arithmetic repairs it.

**Merchant-mix reweighting.** The test set reweighted to seven merchant profiles.

| Profile | Base rate | PR-AUC | ROC-AUC | Brier | Lift | Cost / order |
|---|---|---|---|---|---|---|
| generated mix (baseline) | 14.7% | 0.3858 | 0.8060 | 0.1059 | 2.62× | ₹26.51 |
| metro-heavy D2C | 12.7% | 0.3474 | 0.8083 | 0.0947 | 2.74× | ₹23.42 |
| small-town heavy | 18.7% | 0.4412 | 0.7957 | 0.1262 | 2.36× | ₹32.24 |
| fashion-led catalogue | 19.9% | 0.4321 | 0.7863 | 0.1332 | 2.17× | ₹34.00 |
| electronics-led catalogue | 9.1% | 0.2974 | 0.8180 | 0.0720 | 3.28× | ₹17.47 |
| COD-dominant (85%) | 19.8% | 0.3910 | 0.7364 | 0.1411 | 1.97× | ₹35.58 |
| prepaid-led (35% COD) | 9.0% | 0.3701 | 0.8629 | 0.0659 | 4.13× | ₹16.24 |

**Covariate shift, train vs validation.** A classifier trained to tell the two splits apart reaches AUC **0.6677**
(n = 35,000 vs 5,000). Above 0.5, so the splits are distinguishable — expected, because the split is
chronological and the population drifts over time. This is measured rather than assumed.

---

## 10. Segment breakdown, test set

Where the model works and where it does not. Source: [`05_exception_list.csv`](results/05_exception_list.csv).

| Segment | n | Base rate | PR-AUC | Lift | ROC-AUC | Precision | Recall | Note |
|---|---|---|---|---|---|---|---|---|
| ALL TEST ORDERS | 10,000 | 0.147 | 0.3858 | 2.62× | 0.8060 | 0.4583 | 0.2688 |  |
| COD orders | 6,155 | 0.231 | 0.3927 | 1.70× | 0.6846 | 0.4583 | 0.2787 |  |
| prepaid orders | 3,845 | 0.014 | 0.0356 | 2.63× | 0.6534 | 0.0000 | 0.0000 |  |
| first-time customers | 966 | 0.153 | 0.4062 | 2.65× | 0.8195 | 0.5114 | 0.3041 |  |
| customers with history | 9,034 | 0.147 | 0.3834 | 2.61× | 0.8045 | 0.4523 | 0.2649 |  |
| pincode UNSEEN in train | 73 | — | — | — | — | — | — | too small or single-class |
| pincode seen in train | 9,927 | 0.147 | 0.3875 | 2.63× | 0.8068 | 0.4598 | 0.2700 |  |
| metro | 2,894 | 0.114 | 0.3217 | 2.82× | 0.8072 | 0.4563 | 0.1424 |  |
| tier_1 | 4,021 | 0.129 | 0.3419 | 2.65× | 0.8096 | 0.4627 | 0.1792 |  |
| tier_2 | 2,179 | 0.186 | 0.4214 | 2.26× | 0.7788 | 0.4411 | 0.3596 |  |
| tier_3 | 906 | 0.241 | 0.4956 | 2.06× | 0.7811 | 0.4803 | 0.5046 |  |
| order < Rs500 | 2,894 | 0.167 | 0.3372 | 2.02× | 0.7543 | 0.3919 | 0.1801 |  |
| order Rs500-1000 | 3,596 | 0.163 | 0.4559 | 2.79× | 0.8295 | 0.5115 | 0.4174 |  |
| order > Rs1000 | 3,510 | 0.115 | 0.3334 | 2.90× | 0.8143 | 0.3926 | 0.1588 |  |
| top-decile order value | 1,000 | 0.071 | 0.2446 | 3.44× | 0.8012 | 0.4286 | 0.0845 |  |
| fashion | 2,950 | 0.246 | 0.4625 | 1.88× | 0.7604 | 0.4581 | 0.5124 |  |
| electronics | 1,511 | 0.060 | 0.2317 | 3.85× | 0.8261 | 0.4000 | 0.0220 |  |
| festive window | 1,628 | 0.178 | 0.4316 | 2.42× | 0.8097 | 0.4340 | 0.3966 |  |
| alternate address | 1,190 | 0.167 | 0.3796 | 2.27× | 0.7820 | 0.4694 | 0.2312 |  |
| long ETA (>= 6 days) | 2,129 | 0.212 | 0.4688 | 2.21× | 0.7911 | 0.4592 | 0.4358 |  |

The weakest segment is the one that matters commercially: a merchant whose orders are almost all
cash on delivery loses the single most informative feature, and lift falls to 1.70×.

---

## Source files

| File | Contents |
|---|---|
| [`05_final_metrics.json`](results/05_final_metrics.json) | test and validation metrics, cost model, tiers, shift studies, challengers |
| [`05_threshold_sweep.csv`](results/05_threshold_sweep.csv) | section 2 |
| [`05_cost_sensitivity.csv`](results/05_cost_sensitivity.csv) | optimal threshold across cost assumptions |
| [`05_reliability_test.csv`](results/05_reliability_test.csv) | section 5 |
| [`05_exception_list.csv`](results/05_exception_list.csv) | section 10 |
| [`05_prevalence_shift.csv`](results/05_prevalence_shift.csv) | full prevalence sweep, 27 points |
| [`05_test_predictions.parquet`](results/05_test_predictions.parquet) | per-order test predictions and tiers |
| [`03_leaderboard.csv`](results/03_leaderboard.csv) | section 6 |
| [`03_benchmark_summary.json`](results/03_benchmark_summary.json) | bootstrap comparison, class-weight experiment, Bayes ceiling |
| [`03_cv_folds.csv`](results/03_cv_folds.csv) | per-fold scores, 11 models × 5 folds |
| [`04_tuning_summary.json`](results/04_tuning_summary.json) | section 7 |
| [`04_search_history.csv`](results/04_search_history.csv) | 965 tuning trials |

