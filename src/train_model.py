"""
train_model.py
---------------
Trains and evaluates two classical ML models for network intrusion
detection: Random Forest and XGBoost, on the binary task (normal vs
attack) as the primary deliverable, with the multi-class attack-type
label available for deeper analysis.

Why Random Forest AND XGBoost (not just one)?
- Random Forest is a strong, low-variance baseline: robust to noisy
  features, minimal tuning needed, and highly interpretable via feature
  importances - good for a first pass and a sanity check.
- XGBoost (gradient boosting) typically pushes further on tabular data
  by sequentially correcting errors of prior trees, and handles class
  imbalance well via `scale_pos_weight`. Comparing both gives a concrete,
  defensible "why I chose X" story for an interview, rather than
  presenting a single black-box result.

Why treat this as binary (normal vs attack) as the headline metric?
In a real security operations context, the first and most critical
decision is "is this connection worth investigating at all?" - the
multi-class attack type matters for triage afterward, but the binary
decision is where false negatives (missed attacks) and false positives
(alert fatigue) have the most direct operational cost. We therefore
optimize for and report binary metrics (accuracy, precision, recall,
F1, ROC-AUC) as the primary result, using recall as the metric to watch
most closely - missing a real attack (false negative) is far more
costly than a false alarm.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import xgboost as xgb

RNG_SEED = 42
TEST_SIZE = 0.2
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DATA_PATH = "data/network_traffic.csv"
OUTPUT_DIR = "outputs"


def load_and_preprocess(path=DATA_PATH):
    df = pd.read_csv(path)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = [
        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
        "serror_rate", "same_srv_rate", "num_failed_logins", "logged_in",
    ] + [c + "_enc" for c in CATEGORICAL_COLS]

    X = df[feature_cols].copy()
    y = df["is_attack"].copy()
    y_multiclass = df["label"].copy()

    return df, X, y, y_multiclass, feature_cols, encoders


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

    return metrics, y_pred, y_proba, cm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    df, X, y, y_multiclass, feature_cols, encoders = load_and_preprocess()
    print(f"Dataset shape: {X.shape}, Attack rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RNG_SEED, stratify=y
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ---- Random Forest ----
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",  # compensates for attack being the minority class
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_metrics, rf_pred, rf_proba, rf_cm = evaluate_model(
        "Random Forest", rf_model, X_test, y_test
    )

    # ---- XGBoost ----
    print("\nTraining XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # handles class imbalance
        random_state=RNG_SEED,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)
    xgb_metrics, xgb_pred, xgb_proba, xgb_cm = evaluate_model(
        "XGBoost", xgb_model, X_test, y_test
    )

    # ---- Save comparison ----
    comparison = pd.DataFrame({"Random Forest": rf_metrics, "XGBoost": xgb_metrics}).T
    comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
    print("\n=== Model Comparison ===")
    print(comparison.round(4))

    # ---- Persist artifacts ----
    with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    with open(os.path.join(OUTPUT_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(OUTPUT_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    # Save test split for downstream visualization/SHAP scripts
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)

    print(f"\nSaved model artifacts and metrics to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
