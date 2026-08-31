import json
import joblib

from config import (
    BEST_LOGISTIC_PARAMS,
    BEST_RF_PARAMS,
    BEST_XGB_PARAMS,
    MODELS_DIR,
    RESULTS_DIR,
)
from data import load_and_clean_data, make_train_test
from models import (
    make_preprocessor,
    make_logistic_pipeline,
    make_rf_pipeline,
    make_xgb_pipeline,
)
from evaluation import (
    evaluate_model,
    build_comparison_table,
    save_comparison_table,
    save_classification_report,
    save_confusion_matrix_values,
    save_best_params,
)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = load_and_clean_data()
    X_train, X_test, y_train, y_test = make_train_test(df_clean)

    preprocessor, numerical_features, categorical_features = make_preprocessor(
        X_train
    )

    print("=" * 75)
    print("CUSTOMER CHURN — REPRODUCIBLE TRAINING PIPELINE")
    print("=" * 75)
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set : {X_test.shape}")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # ---------------------------------------------------------
    # Build models
    # ---------------------------------------------------------
    logistic = make_logistic_pipeline(preprocessor)
    rf = make_rf_pipeline(preprocessor)
    xgb = make_xgb_pipeline(preprocessor)

    # Apply the exact best parameters obtained during notebook tuning.
    logistic.set_params(**BEST_LOGISTIC_PARAMS)
    rf.set_params(**BEST_RF_PARAMS)
    xgb.set_params(**BEST_XGB_PARAMS)

    models = {
        "Logistic Regression": logistic,
        "Random Forest": rf,
        "XGBoost": xgb,
    }

    best_params_by_model = {
        "Logistic Regression": BEST_LOGISTIC_PARAMS,
        "Random Forest": BEST_RF_PARAMS,
        "XGBoost": BEST_XGB_PARAMS,
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODELS_DIR / filename)

        save_classification_report(model, X_test, y_test, name)

        print(
            f"{name}: "
            f"Accuracy={metrics['Accuracy']:.4f}, "
            f"Precision={metrics['Precision']:.4f}, "
            f"Recall={metrics['Recall']:.4f}, "
            f"F1={metrics['F1 Score']:.4f}, "
            f"ROC-AUC={metrics['ROC-AUC']:.4f}"
        )

    comparison = build_comparison_table(results)
    save_comparison_table(comparison)

    # Confusion matrix values are only needed for the final selected model (XGBoost).
    save_confusion_matrix_values(models["XGBoost"], X_test, y_test, "XGBoost")

    save_best_params(best_params_by_model)

    best_row = comparison.loc[comparison["ROC-AUC"].idxmax()]
    summary = {
        "best_model": best_row["Model"],
        "best_roc_auc": float(best_row["ROC-AUC"]),
        "test_size": int(len(X_test)),
        "train_size": int(len(X_train)),
        "random_state": 42,
    }

    with open(RESULTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("FINAL TEST SET RESULTS — TUNED MODELS")
    print("=" * 75)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 75)
    print("BEST MODEL")
    print("=" * 75)
    print(f"Model   : {best_row['Model']}")
    print(f"ROC-AUC : {best_row['ROC-AUC']:.4f}")

    return comparison


if __name__ == "__main__":
    main()
