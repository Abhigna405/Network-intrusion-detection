from flask import Flask, render_template, request, jsonify
import joblib
import json
import os
import numpy as np

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")

model = joblib.load(os.path.join(MODEL_DIR, "nids_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)

app = Flask(__name__)

CATEGORY_INFO = {
    "normal": ("✅ Normal traffic", "no-risk"),
    "DoS": ("🛑 Denial of Service attack", "high-risk"),
    "Probe": ("🔍 Probing / surveillance attack", "med-risk"),
    "R2L": ("🚪 Remote-to-Local attack", "high-risk"),
    "U2R": ("🔓 User-to-Root attack", "high-risk"),
}


@app.route("/")
def index():
    return render_template(
        "index.html",
        categorical_cols=meta["categorical_cols"],
        categorical_options=meta["categorical_options"],
        numeric_cols=meta["numeric_cols"],
        numeric_defaults=meta["numeric_defaults"],
        metrics=meta["metrics"],
        samples=meta["samples"],
        classes=meta["classes"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    row = []
    for col in meta["feature_order"]:
        if col in meta["categorical_cols"]:
            raw_val = str(data.get(col))
            le = encoders[col]
            if raw_val not in le.classes_:
                raw_val = le.classes_[0]
            row.append(le.transform([raw_val])[0])
        else:
            row.append(float(data.get(col, meta["numeric_defaults"].get(col, 0))))

    X = np.array(row).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = target_encoder.inverse_transform([pred_idx])[0]

    label_text, risk_class = CATEGORY_INFO.get(pred_label, (pred_label, "med-risk"))

    class_probs = {
        target_encoder.inverse_transform([i])[0]: round(float(p) * 100, 1)
        for i, p in enumerate(proba)
    }

    return jsonify({
        "prediction": pred_label,
        "label_text": label_text,
        "risk_class": risk_class,
        "confidence": round(float(proba[pred_idx]) * 100, 1),
        "class_probs": class_probs,
        "top_features": meta["top_features"],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
