"""
shap_analysis.py
-----------------
Generates SHAP (SHapley Additive exPlanations) plots for the XGBoost
model, explaining which features drive each prediction and by how much.

Why SHAP matters for intrusion detection specifically:
A security analyst receiving an "attack" alert needs to know WHY the
model flagged a connection, not just that it did - both to trust the
system (avoid blindly acting on a black box) and to triage efficiently
(a flag driven by extreme serror_rate suggests a different response than
one driven by failed login attempts). SHAP values decompose each
prediction into additive per-feature contributions, giving a
mathematically grounded answer to "why did the model say this is an
attack?" rather than just a feature-importance ranking (which shows what
matters on average, not for this specific case).
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

OUTPUT_DIR = "outputs"


def main():
    with open(os.path.join(OUTPUT_DIR, "xgb_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)

    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, "X_test.csv"))

    # Use a sample for speed if the test set is large
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)

    print(f"Computing SHAP values for {sample_size} test samples...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot: global feature importance + direction of effect
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/shap_summary.png")

    # Bar plot: mean absolute SHAP value per feature (cleaner global ranking)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_importance_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved outputs/shap_importance_bar.png")

    # Single-prediction waterfall: explains one specific attack detection
    attack_indices = X_sample.index[xgb_model.predict(X_sample) == 1]
    if len(attack_indices) > 0:
        example_idx = attack_indices[0]
        example_pos = X_sample.index.get_loc(example_idx)

        plt.figure()
        explanation = shap.Explanation(
            values=shap_values[example_pos],
            base_values=explainer.expected_value,
            data=X_sample.iloc[example_pos],
            feature_names=X_sample.columns.tolist(),
        )
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_single_prediction.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved outputs/shap_single_prediction.png (explains one flagged attack)")


if __name__ == "__main__":
    main()
