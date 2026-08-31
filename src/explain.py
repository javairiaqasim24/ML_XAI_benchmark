import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from config import DATA_PATH, FIGURES_DIR, MODELS_DIR, RESULTS_DIR
from data import load_and_clean_data, make_train_test


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "xgboost.joblib"
    model = joblib.load(model_path)

    df_clean = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = make_train_test(df_clean)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_test_transformed, "toarray"):
        X_shap = X_test_transformed.toarray()
    else:
        X_shap = X_test_transformed

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_shap)

    # -----------------------------
    # Global SHAP bar plot
    # -----------------------------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_global_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # -----------------------------
    # Global SHAP importance CSV
    # -----------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap_value": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap_value", ascending=False).reset_index(drop=True)

    importance_df.to_csv(
        RESULTS_DIR / "shap_feature_importance.csv",
        index=False,
    )

    # -----------------------------
    # SHAP beeswarm plot
    # -----------------------------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # -----------------------------
    # Individual customer
    # -----------------------------
    customer_index = 0
    customer_shap = shap_values[customer_index]

    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value.item()

    explanation = shap.Explanation(
        values=customer_shap,
        base_values=expected_value,
        data=X_shap[customer_index],
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_individual_customer.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save the selected customer's original row and prediction.
    customer = X_test.iloc[[customer_index]]
    probability = model.predict_proba(customer)[0, 1]
    prediction = model.predict(customer)[0]

    customer.to_csv(
        RESULTS_DIR / "individual_customer_input.csv",
        index=False,
    )

    print("=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Test customers explained: {len(X_test)}")
    print(f"Transformed features: {len(feature_names)}")
    print(f"Selected customer index: {customer_index}")
    print(f"Predicted churn: {prediction}")
    print(f"Churn probability: {probability:.4f}")
    print("\nSaved figures:")
    print(" - shap_global_feature_importance.png")
    print(" - shap_beeswarm.png")
    print(" - shap_individual_customer.png")
    print("\nSaved results:")
    print(" - shap_feature_importance.csv")
    print(" - individual_customer_input.csv")


if __name__ == "__main__":
    main()
