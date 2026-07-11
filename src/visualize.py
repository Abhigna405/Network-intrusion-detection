"""
visualize.py
-------------
Generates the visualizations for the submission, reading the artifacts
saved by train_model.py:
    1. outputs/eda_class_distribution.png - attack type distribution
    2. outputs/eda_feature_distributions.png - key feature distributions by class
    3. outputs/confusion_matrices.png - RF vs XGBoost confusion matrices
    4. outputs/roc_curves.png - ROC curves for both models
    5. outputs/feature_importance.png - RF and XGBoost feature importances
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

OUTPUT_DIR = "outputs"
DATA_PATH = "data/network_traffic.csv"


def plot_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    order = df["label"].value_counts().index
    sns.countplot(data=df, x="label", order=order, ax=axes[0], hue="label",
                  palette="viridis", legend=False)
    axes[0].set_title("Attack Type Distribution")
    axes[0].set_xlabel("")
    for container in axes[0].containers:
        axes[0].bar_label(container)

    binary_counts = df["is_attack"].value_counts().sort_index()
    axes[1].pie(binary_counts, labels=["Normal", "Attack"], autopct="%1.1f%%",
                colors=["#4C72B0", "#C44E52"], startangle=90)
    axes[1].set_title("Normal vs Attack (Binary Task)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_class_distribution.png"), dpi=150)
    plt.close()
    print("Saved outputs/eda_class_distribution.png")


def plot_feature_distributions(df):
    features = ["duration", "src_bytes", "count", "serror_rate"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, feat in zip(axes, features):
        for label in ["normal", "dos", "probe", "r2l", "u2r"]:
            subset = df[df["label"] == label][feat]
            clipped = subset.clip(upper=subset.quantile(0.95))
            sns.kdeplot(clipped, label=label, ax=ax, fill=True, alpha=0.25)
        ax.set_title(f"{feat} distribution by attack type")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_feature_distributions.png"), dpi=150)
    plt.close()
    print("Saved outputs/eda_feature_distributions.png")


def plot_confusion_matrices():
    with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "xgb_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)

    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(OUTPUT_DIR, "y_test.csv")).squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, model) in zip(axes, [("Random Forest", rf_model), ("XGBoost", xgb_model)]):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("Saved outputs/confusion_matrices.png")

    return rf_model, xgb_model, X_test, y_test


def plot_roc_curves(rf_model, xgb_model, X_test, y_test):
    plt.figure(figsize=(7, 6))
    for name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model)]:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Random Forest vs XGBoost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    print("Saved outputs/roc_curves.png")


def plot_feature_importance(rf_model, xgb_model, feature_cols):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
    rf_importance.plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Random Forest - Feature Importance")

    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values()
    xgb_importance.plot(kind="barh", ax=axes[1], color="#55A868")
    axes[1].set_title("XGBoost - Feature Importance")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved outputs/feature_importance.png")


def main():
    df = pd.read_csv(DATA_PATH)
    with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    plot_class_distribution(df)
    plot_feature_distributions(df)
    rf_model, xgb_model, X_test, y_test = plot_confusion_matrices()
    plot_roc_curves(rf_model, xgb_model, X_test, y_test)
    plot_feature_importance(rf_model, xgb_model, feature_cols)


if __name__ == "__main__":
    main()
