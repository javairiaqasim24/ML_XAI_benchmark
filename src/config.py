from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "Churn"
ID_COLUMN = "customerID"

# Best parameters obtained during the notebook tuning stage.
BEST_LOGISTIC_PARAMS = {
    "classifier__C": 100,
    "classifier__solver": "liblinear",
}

BEST_RF_PARAMS = {
    "classifier__max_depth": 10,
    "classifier__min_samples_leaf": 2,
    "classifier__min_samples_split": 2,
    "classifier__n_estimators": 200,
}

BEST_XGB_PARAMS = {
    "classifier__colsample_bytree": 1.0,
    "classifier__learning_rate": 0.03,
    "classifier__max_depth": 3,
    "classifier__n_estimators": 200,
    "classifier__subsample": 0.8,
}
