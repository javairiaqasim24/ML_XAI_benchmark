"""
Customer Churn Prediction + Explainable AI
Run this file from the project root:

    python main.py

It trains the three tuned models, evaluates them, saves the models/results,
and then creates the SHAP and evaluation figures.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import main as train_main
from roc_curve import main as roc_main
from confusion_matrix import main as confusion_main
from explain import main as explain_main


if __name__ == "__main__":
    print("\n[1/4] Training and evaluating models...")
    train_main()

    print("\n[2/4] Creating ROC curve...")
    roc_main()

    print("\n[3/4] Creating XGBoost confusion matrix...")
    confusion_main()

    print("\n[4/4] Running SHAP explainability...")
    explain_main()

    print("\n" + "=" * 75)
    print("PROJECT PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 75)
