import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from config import FIGURES_DIR, MODELS_DIR
from data import load_and_clean_data, make_train_test
from evaluation import get_confusion_matrix


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = load_and_clean_data()
    X_train, X_test, y_train, y_test = make_train_test(df_clean)

    model = joblib.load(MODELS_DIR / "xgboost.joblib")
    cm = get_confusion_matrix(model, X_test, y_test)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed", "Churned"],
    ).plot()

    plt.title("XGBoost — Confusion Matrix")
    plt.tight_layout()

    output = FIGURES_DIR / "xgboost_confusion_matrix.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
