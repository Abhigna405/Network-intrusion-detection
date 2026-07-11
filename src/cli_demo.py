"""
cli_demo.py
-----------
Classifies a single network connection (by test-set row index, or a
random one) using both trained models, and prints their predictions
side by side along with the true label for comparison.

Usage:
    python src/cli_demo.py --index 42
    python src/cli_demo.py --random
"""

import argparse
import pickle
import numpy as np
import pandas as pd

OUTPUT_DIR = "outputs"


def main():
    parser = argparse.ArgumentParser(description="Network Intrusion Detection CLI Demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Row index in the test set to classify")
    group.add_argument("--random", action="store_true", help="Classify a random test-set row")
    args = parser.parse_args()

    with open(f"{OUTPUT_DIR}/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open(f"{OUTPUT_DIR}/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    X_test = pd.read_csv(f"{OUTPUT_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{OUTPUT_DIR}/y_test.csv").squeeze()

    if args.random:
        idx = np.random.randint(0, len(X_test))
    else:
        idx = args.index

    row = X_test.iloc[[idx]]
    true_label = y_test.iloc[idx]

    rf_pred = rf_model.predict(row)[0]
    rf_proba = rf_model.predict_proba(row)[0, 1]
    xgb_pred = xgb_model.predict(row)[0]
    xgb_proba = xgb_model.predict_proba(row)[0, 1]

    label_map = {0: "NORMAL", 1: "ATTACK"}

    print(f"\nConnection (test set row {idx}):")
    print(row.T.to_string(header=False))

    print(f"\nTrue label:      {label_map[true_label]}")
    print(f"Random Forest:   {label_map[rf_pred]}  (attack probability: {rf_proba:.4f})")
    print(f"XGBoost:         {label_map[xgb_pred]}  (attack probability: {xgb_proba:.4f})")

    if rf_pred == true_label and xgb_pred == true_label:
        print("\nBoth models predicted correctly.")
    elif rf_pred != true_label and xgb_pred != true_label:
        print("\nBoth models predicted incorrectly - interesting edge case to inspect further.")
    else:
        print("\nModels disagree with each other - only one matched the true label.")


if __name__ == "__main__":
    main()
