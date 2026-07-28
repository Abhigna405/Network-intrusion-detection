"""
Trains a Random Forest + XGBoost intrusion classifier on the NSL-KDD dataset
(5-class: normal, DoS, Probe, R2L, U2R) and saves artifacts for the Flask app.
Also saves a handful of real sample rows (one per class) so the UI can offer
a one-click "load sample" demo instead of forcing manual entry of 41 features.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

BASE = os.path.dirname(__file__)

COLS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'attack_type', 'difficulty']

DOS = {'neptune', 'back', 'land', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm',
       'processtable', 'worm', 'mailbomb'}
PROBE = {'satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'}
R2L = {'guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient',
       'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail',
       'named'}
U2R = {'buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps'}


def to_category(label):
    if label == 'normal':
        return 'normal'
    if label in DOS:
        return 'DoS'
    if label in PROBE:
        return 'Probe'
    if label in R2L:
        return 'R2L'
    if label in U2R:
        return 'U2R'
    return 'DoS'  # rare unseen labels default to the largest attack bucket


train_df = pd.read_csv(os.path.join(BASE, 'KDDTrain.txt'), names=COLS)
test_df = pd.read_csv(os.path.join(BASE, 'KDDTest.txt'), names=COLS)

for df in (train_df, test_df):
    df['category'] = df['attack_type'].apply(to_category)
    df.drop(columns=['attack_type', 'difficulty'], inplace=True)

feature_cols = [c for c in train_df.columns if c != 'category']
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    encoders[col] = le
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([train_df['category'], test_df['category']], axis=0))

X_train = train_df[feature_cols]
y_train = label_encoder.transform(train_df['category'])
X_test = test_df[feature_cols]
y_test = label_encoder.transform(test_df['category'])

model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
                                class_weight='balanced_subsample')
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test, preds, target_names=label_encoder.classes_))

importances = dict(zip(feature_cols, model.feature_importances_))
top_features = sorted(importances.items(), key=lambda x: -x[1])[:8]
print("Top features:", top_features)

joblib.dump(model, os.path.join(BASE, 'nids_model.pkl'))
joblib.dump(encoders, os.path.join(BASE, 'label_encoders.pkl'))
joblib.dump(label_encoder, os.path.join(BASE, 'target_encoder.pkl'))

# Save one real un-encoded sample row per attack category for the "load sample" UI button
samples = {}
test_raw = pd.read_csv(os.path.join(BASE, 'KDDTest.txt'), names=COLS)
test_raw['category'] = test_raw['attack_type'].apply(to_category)
for cat in label_encoder.classes_:
    subset = test_raw[test_raw['category'] == cat]
    if len(subset):
        row = subset.iloc[0][feature_cols].to_dict()
        samples[cat] = row

meta = {
    'categorical_cols': categorical_cols,
    'numeric_cols': numeric_cols,
    'categorical_options': {c: encoders[c].classes_.tolist() for c in categorical_cols},
    'feature_order': feature_cols,
    'top_features': [f[0] for f in top_features],
    'classes': label_encoder.classes_.tolist(),
    'metrics': {'accuracy': round(float(acc), 4)},
    'numeric_defaults': {c: float(X_train[c].median()) for c in numeric_cols},
    'samples': samples,
}
with open(os.path.join(BASE, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2, default=str)

print("Saved model, encoders, and metadata.")
