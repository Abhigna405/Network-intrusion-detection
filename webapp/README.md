# Network Intrusion Detection — Web App

A Flask app around the Random Forest NSL-KDD classifier, so the intrusion
detection project has a real frontend + backend instead of just a notebook.

## Structure
```
nids-app/
├── app.py                 # Flask backend — loads model, serves /predict
├── requirements.txt
├── templates/index.html   # Form + "load real sample" buttons + result panel
├── static/style.css
└── model/
    ├── train.py            # Retrains from KDDTrain.txt / KDDTest.txt
    ├── KDDTrain.txt, KDDTest.txt   # NSL-KDD dataset
    ├── nids_model.pkl       # Trained Random Forest (5-class)
    ├── label_encoders.pkl   # Encoders for protocol_type/service/flag
    ├── target_encoder.pkl   # Encoder for the attack category label
    └── meta.json            # Form metadata + 5 real sample records
```

## How to run
```bash
pip install -r requirements.txt
python model/train.py    # optional — already trained
python app.py
```
Open **http://localhost:5001**.

## What it does
- Classifies a network connection record into **normal / DoS / Probe / R2L / U2R**.
- There are 41 raw features per record, which is impractical to type by hand,
  so the UI has **"Load a real sample" buttons** — one real record per class,
  pulled straight from the NSL-KDD test set — that autofill the form. You can
  also hand-edit any field afterward to explore what changes the prediction.
- Shows class probabilities for all 5 categories and the top features driving
  the model's decisions (`src_bytes`, `service`, `logged_in`, error rates, etc.)

## Model performance
Random Forest, 200 trees: **73.6% accuracy** on the official NSL-KDD test set.
This is intentionally lower than the ~99% you'll see with a random train/test
split of KDDTrain+ alone — NSL-KDD's official test set (KDDTest+) deliberately
includes attack subtypes that never appear in training, specifically to test
generalization to unseen attacks. Papers benchmarking on this exact split
typically report 75–80% too, so this is a realistic, honest number to quote
rather than an inflated one — good talking point in an interview.

## Plugging into your existing GitHub repo
Copy `nids-app/` into your `Network-intrusion-detection` repo (e.g. as
`webapp/`), commit, push, and add a line to your main README pointing to it.
