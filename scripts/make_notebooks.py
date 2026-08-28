import json, os, io

NB = [
 ("01_data_generation", "Data Generation - 50,000 Synthetic COD Orders", [
   "Setup & seeded RNG", "Load real India Post pincode skeleton",
   "Grounding constants from published Indian RTO statistics",
   "Build customer population (persistent identity & history)",
   "Generate 50,000 orders", "Assign RTO labels from grounded drivers",
   "CALIBRATION CHECK - must reproduce published rates",
   "Chronological split 70/10/20", "Write data/raw + data/processed",
   "SHA-256 manifest"]),
 ("02_eda", "Exploratory Data Analysis & Preprocessing", [
   "Load splits (train only - never look at test)", "Schema & dtypes, missingness",
   "Target balance and base rate", "Univariate distributions",
   "RTO rate by pincode tier / city", "RTO rate by order-value band (check the non-monotonic peak)",
   "RTO rate by category, COD vs prepaid", "Customer history effects",
   "Address-quality effects", "Correlation & multicollinearity",
   "Outliers", "Leakage audit", "Preprocessing pipeline definition"]),
 ("03_model_training", "Model Training - 10 Models Benchmarked", [
   "Load train/val", "Feature pipeline (out-of-fold target encoding)",
   "Model 00 - DummyClassifier (majority baseline)",
   "Model 01 - Logistic Regression", "Model 02 - Decision Tree",
   "Model 03 - Random Forest", "Model 04 - Extra Trees",
   "Model 05 - Gradient Boosting", "Model 06 - HistGradientBoosting",
   "Model 07 - XGBoost", "Model 08 - LightGBM", "Model 09 - CatBoost",
   "Model 10 - MLP", "Validation leaderboard (PR-AUC, ROC-AUC, Brier)",
   "Select finalists for tuning", "Persist results"]),
 ("04_hyperparameter_tuning", "Hyperparameter Tuning - Two Modern Approaches", [
   "Load finalists & search space definition",
   "APPROACH 1 - Optuna: multivariate TPE + HyperbandPruner",
   "Optuna study analysis (importance, parallel coordinate, slice)",
   "APPROACH 2 - FLAML: cost-frugal BlendSearch/CFO",
   "Head-to-head: same budget, same space, same seed",
   "Convergence comparison", "Best configuration per approach",
   "Refit on train, score on val", "Persist tuned models"]),
 ("05_final_evaluation", "Final Evaluation - Test Set Opened Once", [
   "Load tuned finalists", "Select ONE final model on validation only",
   "=== OPEN TEST SET (first and only time) ===",
   "Headline metrics: precision, recall, F1, PR-AUC, ROC-AUC",
   "Confusion matrix at cost-optimal threshold",
   "Calibration: Brier score + reliability diagram",
   "Cost model in rupees (FN = shipping burned, FP = lost margin)",
   "Threshold sweep vs cost - choose the three tiers",
   "Three-tier decision policy: allow / COD fee / disable COD",
   "Prevalence-shift study (18% - 35%, the real Indian city range)",
   "SHAP explanations & risk reasons",
   "Model comparison plots (all 10 + tuned)",
   "Honest exception list - where it underperforms",
   "Persist final model + metrics"]),
]

os.makedirs("notebooks", exist_ok=True)
for name, title, sections in NB:
    cells = [{"cell_type":"markdown","metadata":{},
              "source":[f"# {title}\n","\n",
                        "**Razorpay AI Buildathon 2026 - Track 02: AI Risk Manager**\n","\n",
                        "Return-Risk Scorer for COD orders (Return-to-Origin prediction).\n"]}]
    for i, s in enumerate(sections, 1):
        cells.append({"cell_type":"markdown","metadata":{},
                      "source":[f"## {i}. {s}\n"]})
        cells.append({"cell_type":"code","execution_count":None,"metadata":{},
                      "outputs":[],"source":["# TODO\n"]})
    nb = {"cells":cells,
          "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                      "language_info":{"name":"python","version":"3.13"}},
          "nbformat":4,"nbformat_minor":5}
    with io.open(f"notebooks/{name}.ipynb","w",encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("created notebooks/%s.ipynb  (%d sections)" % (name, len(sections)))
