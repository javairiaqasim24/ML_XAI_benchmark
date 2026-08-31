"""
Centralized model evaluation utilities.

Used by both src/train.py (metrics, reports, saved artifacts) and
src/confusion_matrix.py (plot) so evaluation logic lives in one place
instead of being recomputed independently in each script.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from config import RESULTS_DIR


def evaluate_model(model, X_test, y_test):
    """Return the standard metric set (Accuracy, Precision, Recall, F1, ROC-AUC)."""
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }


def get_confusion_matrix(model, X_test, y_test):
    """Return the raw confusion matrix (2x2 array) for a fitted model."""
    pred = model.predict(X_test)
    return confusion_matrix(y_test, pred)


def build_comparison_table(results):
    """Turn a list of per-model metric dicts into the final comparison DataFrame."""
    return pd.DataFrame(results)[
        ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ].round(4)


def save_comparison_table(comparison, results_dir=RESULTS_DIR):
    path = results_dir / "final_model_comparison.csv"
    comparison.to_csv(path, index=False)
    return path


def save_classification_report(model, X_test, y_test, model_name, results_dir=RESULTS_DIR):
    """Save sklearn's classification_report as a text file for one model."""
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, target_names=["Stayed", "Churned"])

    filename = f"classification_report_{model_name.lower().replace(' ', '_')}.txt"
    path = results_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    return path


def save_confusion_matrix_values(model, X_test, y_test, model_name, results_dir=RESULTS_DIR):
    """Save the raw confusion matrix (counts) as a CSV for one model."""
    cm = get_confusion_matrix(model, X_test, y_test)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Stayed", "Actual: Churned"],
        columns=["Predicted: Stayed", "Predicted: Churned"],
    )

    filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
    path = results_dir / filename
    cm_df.to_csv(path)

    return cm, path


def save_best_params(best_params_by_model, results_dir=RESULTS_DIR):
    """Save the tuned hyperparameters used for each model as JSON."""
    import json

    path = results_dir / "best_model_parameters.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best_params_by_model, f, indent=2)

    return path
