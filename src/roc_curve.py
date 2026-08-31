import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from config import FIGURES_DIR, MODELS_DIR
from data import load_and_clean_data, make_train_test


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = load_and_clean_data()
    X_train, X_test, y_train, y_test = make_train_test(df_clean)

    model_files = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
        "Random Forest": MODELS_DIR / "random_forest.joblib",
        "XGBoost": MODELS_DIR / "xgboost.joblib",
    }

    plt.figure(figsize=(8, 6))

    for name, path in model_files.items():
        model = joblib.load(path)
        probability = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probability)
        model_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {model_auc:.4f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Tuned Model Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output = FIGURES_DIR / "roc_curve_model_comparison.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
