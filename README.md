# Customer Churn Prediction + Explainable AI

A machine-learning pipeline that predicts telecom customer churn and explains
*why* the model makes each prediction using SHAP (SHapley Additive
exPlanations).

## Problem Statement

Customer churn — a subscriber cancelling their service — directly erodes
recurring revenue, and acquiring a new customer typically costs far more than
retaining an existing one. A telecom provider needs to know **which
customers are likely to churn** and **which factors are driving that risk**,
so retention teams can intervene early with the right offer instead of
guessing.

## Objective

Build a binary classifier that predicts whether a customer will churn
(`Yes`/`No`), compare multiple model families under a shared, reproducible
evaluation protocol, select the best-performing model, and explain its
predictions at both the global (population) and individual (single customer)
level using SHAP — so the results are trustworthy enough to act on, not just
accurate.

## Dataset

- **Source:** [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (public, non-sensitive sample data).
- **File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Rows / columns:** 7,043 customers × 21 columns.
- **Target:** `Churn` (`Yes` / `No`).
- **Features:** customer demographics (gender, senior citizen, partner,
  dependents), account information (tenure, contract type, payment method,
  billing), and subscribed services (phone, internet, streaming, security,
  tech support add-ons).

## ML Workflow

```
data/*.csv
   │
   ▼
[1] Clean data (src/data.py)
   │   - drop customerID
   │   - coerce TotalCharges to numeric, fill missing with 0
   │   - map Churn: No -> 0, Yes -> 1
   ▼
[2] Stratified 80/20 train/test split (random_state=42)
   ▼
[3] Preprocess (src/models.py)
   │   - StandardScaler on numeric features
   │   - OneHotEncoder(drop="first") on categorical features
   ▼
[4] Train 3 tuned models (src/train.py)
   │   - Logistic Regression | Random Forest | XGBoost
   ▼
[5] Evaluate (src/evaluation.py)
   │   - Accuracy, Precision, Recall, F1, ROC-AUC
   │   - classification reports, confusion matrix
   ▼
[6] Select best model by ROC-AUC → XGBoost
   ▼
[7] Explain with SHAP (src/explain.py)
       - global feature importance, beeswarm, individual waterfall
```

The exact same cleaning and split logic is used in the exploratory notebook
(`notebooks/customer_churn_eda.ipynb`) and in the production pipeline
(`src/data.py`), so results are consistent between the two.

## Models Used

| Model | Library |
|---|---|
| Logistic Regression | `scikit-learn` |
| Random Forest | `scikit-learn` |
| XGBoost | `xgboost` |

All three share the same preprocessing pipeline (`ColumnTransformer` with
scaling + one-hot encoding) so the comparison isolates the effect of the
model itself.

## Preprocessing

- **Numeric features** (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`): standardized with `StandardScaler`.
- **Categorical features** (15 columns, e.g. `Contract`, `InternetService`, `PaymentMethod`): `OneHotEncoder(handle_unknown="ignore", drop="first")` to avoid the dummy-variable trap.
- Preprocessing is fit only on the training split and applied to the test split via `Pipeline`, preventing test-set leakage.

## Hyperparameter Tuning

Each model was tuned during the notebook exploration stage; the resulting
best parameters are frozen in [`src/config.py`](src/config.py) and applied
in the pipeline via `set_params(...)` so training is deterministic and
reproducible:

| Model | Tuned Parameters |
|---|---|
| Logistic Regression | `C=100`, `solver=liblinear` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `min_samples_split=2`, `min_samples_leaf=2` |
| XGBoost | `n_estimators=200`, `max_depth=3`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=1.0` |

Full parameter values used for the current run are also saved to
`outputs/results/best_model_parameters.json`.

## Final Model Comparison

Test set: 1,409 customers (20% stratified holdout), `random_state=42`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8020 | 0.6498 | 0.5508 | 0.5962 | 0.8409 |
| Random Forest | 0.8020 | 0.6667 | 0.5080 | 0.5766 | 0.8425 |
| **XGBoost** | 0.7999 | 0.6575 | 0.5134 | 0.5766 | **0.8469** |

(Also saved to `outputs/results/final_model_comparison.csv` on every run.)

## Why XGBoost Was Selected

**ROC-AUC** was used as the primary selection metric because churn is an
imbalanced classification problem (~27% churn rate) — accuracy alone would
favor a model that just predicts the majority class, whereas ROC-AUC
measures how well the model ranks churners above non-churners across all
decision thresholds, which is what a retention team actually needs when
choosing how many customers to target with a limited budget.

XGBoost achieves the highest ROC-AUC (0.8469) of the three models, with
accuracy and F1 essentially tied with the alternatives. It also has strong,
well-understood SHAP support (`TreeExplainer`), making its predictions
explainable at both the population and individual-customer level — a
requirement for this project.

## Explainable AI (SHAP)

The selected XGBoost model is explained using `shap.TreeExplainer`:

- **Global feature importance** (`shap_global_feature_importance.png`) — mean absolute SHAP value per feature, ranking the strongest overall churn drivers.
- **Beeswarm plot** (`shap_beeswarm.png`) — shows both the magnitude and direction of each feature's effect across all test customers.
- **Individual customer waterfall** (`shap_individual_customer.png`) — explains one specific customer's prediction, showing which features pushed their churn probability up or down.
- **`shap_feature_importance.csv`** — the same global ranking as plain numbers, for use outside of plots.

Top churn drivers identified by SHAP: `tenure`, `Contract` type (month-to-month vs. one/two-year), `InternetService` (fiber optic), `PaymentMethod` (electronic check), and `MonthlyCharges`.

## Key Findings

- Contract type and tenure dominate churn risk: short-tenure, month-to-month customers churn far more often than long-tenure customers on annual contracts.
- Fiber-optic internet and electronic-check payment are both associated with higher churn probability, both worth investigating from a service/billing-experience angle.
- All three models land in a similar accuracy/F1 range (~0.80 / ~0.58–0.60) — the dataset's signal ceiling is limited by 21 shallow customer attributes, so the meaningful differentiator between models is ranking quality (ROC-AUC), not raw accuracy.
- Recall on the churn class (~0.51–0.55) is moderate: roughly half of churners are missed at the default 0.5 threshold. See Limitations.

## Project Structure

```
customer-churn-ml-xai/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── customer_churn_eda.ipynb      # exploratory data analysis / tuning notebook
├── outputs/
│   ├── figures/                      # ROC curve, confusion matrix, SHAP plots (PNG)
│   ├── models/                       # trained model artifacts (.joblib)
│   └── results/                      # metrics, reports, parameters (CSV/JSON/TXT)
├── src/
│   ├── config.py                     # paths, random_state, tuned hyperparameters
│   ├── data.py                       # loading, cleaning, train/test split
│   ├── models.py                     # preprocessing + model pipelines
│   ├── evaluation.py                 # centralized metrics / reports / confusion matrix
│   ├── train.py                      # trains all 3 models, saves models + results
│   ├── roc_curve.py                  # ROC curve comparison figure
│   ├── confusion_matrix.py           # XGBoost confusion matrix figure
│   └── explain.py                    # SHAP explainability (global + individual)
├── main.py                           # runs the full pipeline end to end
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <this-repo-url>
cd ML_XAI_benchmark

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The dataset is already included at
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. If you're starting from a fresh
copy, make sure the CSV lives at exactly that path.

## How to Run

```bash
python main.py
```

This runs the full pipeline in one command:

1. Trains and evaluates all three tuned models.
2. Saves trained models to `outputs/models/`.
3. Saves the model comparison table, classification reports, confusion
   matrix values, and best parameters to `outputs/results/`.
4. Generates the ROC curve and XGBoost confusion matrix figures.
5. Runs SHAP explainability and saves global + individual explanation
   figures and a feature-importance CSV.

Each script under `src/` can also be run independently (e.g.
`python src/train.py`) as long as `python main.py` has been run at least
once so that trained models exist for the downstream scripts.

## Results

All outputs are written under `outputs/`:

- `outputs/results/final_model_comparison.csv` — the table above.
- `outputs/results/classification_report_*.txt` — per-model precision/recall/F1 by class.
- `outputs/results/xgboost_confusion_matrix.csv` — raw confusion matrix counts for the selected model.
- `outputs/results/best_model_parameters.json` — tuned hyperparameters used for each model.
- `outputs/results/run_summary.json` — best model, best ROC-AUC, split sizes, random state.
- `outputs/results/shap_feature_importance.csv` — global SHAP ranking.
- `outputs/figures/*.png` — ROC curve, confusion matrix, and SHAP plots.

## Limitations

- The dataset is a single static snapshot (no time-series/behavioral history), so the model cannot react to *recent* changes in customer behavior.
- Class imbalance (~27% churn) means recall on churners (~0.51–0.55) is moderate at the default 0.5 threshold; a retention team would likely want to tune the decision threshold to trade precision for recall depending on campaign cost.
- No external validation set (e.g. a different region or time period) — reported metrics reflect a single stratified holdout from the same dataset.
- SHAP explanations describe correlational feature contribution within this model, not proven causal drivers of churn.

## Future Improvements

- Threshold tuning / cost-sensitive learning to improve churn-class recall for a target retention-campaign budget.
- Cross-validation (e.g. stratified k-fold) instead of a single holdout split for more robust metric estimates.
- Probability calibration (e.g. `CalibratedClassifierCV`) if predicted probabilities are used directly for targeting/ranking.
- A simple serving layer (CLI or API endpoint) to score new customers on demand.
- Additional feature engineering (e.g. tenure buckets, service-count aggregates) to see if it improves ranking quality further.
