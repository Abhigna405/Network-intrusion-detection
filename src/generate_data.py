"""
generate_data.py
----------------
Generates a synthetic dataset modeled on the structure of NSL-KDD, the
classic benchmark for network intrusion detection research. We simulate
network connection records with the same broad feature families NSL-KDD
uses (basic connection features, content features, traffic features) and
label each record as either 'normal' or one of four attack categories:
DoS, Probe, R2L (Remote-to-Local), and U2R (User-to-Root).

Why synthetic data?
CodTech allows dummy datasets for data science projects. A synthetic
generator keeps this fully reproducible offline (no dataset download
required - avoiding any network/firewall issues) while preserving the
realistic class imbalance and feature structure that makes intrusion
detection a genuinely hard, interesting ML problem: attacks are rare
relative to normal traffic, and different attack types have very
different signatures.
"""

import numpy as np
import pandas as pd
import os

RNG_SEED = 42
N_SAMPLES = 15000

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "telnet", "dns", "ssh", "pop3", "other"]
FLAGS = ["SF", "S0", "REJ", "RSTR", "RSTO"]

# Attack category base rates (normal dominates, matching real-world traffic)
ATTACK_TYPES = {
    "normal": 0.70,
    "dos": 0.18,     # Denial of Service - high volume, short bursts
    "probe": 0.07,   # Port/network scanning - many short low-byte connections
    "r2l": 0.035,    # Remote-to-Local - login attempts from remote hosts
    "u2r": 0.015,    # User-to-Root - rare, privilege escalation after access
}


def generate_connection(rng, label):
    """Generates one network connection record with feature distributions
    that differ meaningfully by attack type, so the classifier has real
    signal to learn from (not pure noise)."""

    protocol = rng.choice(PROTOCOLS, p=[0.75, 0.20, 0.05])
    service = rng.choice(SERVICES)

    if label == "normal":
        duration = max(0, rng.normal(50, 40))
        src_bytes = max(0, rng.normal(500, 300))
        dst_bytes = max(0, rng.normal(1000, 600))
        flag = rng.choice(FLAGS, p=[0.85, 0.05, 0.04, 0.03, 0.03])
        count = rng.integers(1, 10)
        srv_count = rng.integers(1, 10)
        serror_rate = np.clip(rng.normal(0.02, 0.05), 0, 1)
        same_srv_rate = np.clip(rng.normal(0.9, 0.1), 0, 1)
        num_failed_logins = 0
        logged_in = 1

    elif label == "dos":
        # DoS: very short duration, huge connection counts, high error rates
        duration = max(0, rng.normal(1, 2))
        src_bytes = max(0, rng.normal(50, 30))
        dst_bytes = max(0, rng.normal(0, 5))
        flag = rng.choice(["S0", "REJ", "RSTR"], p=[0.6, 0.25, 0.15])
        count = rng.integers(100, 511)
        srv_count = rng.integers(100, 511)
        serror_rate = np.clip(rng.normal(0.9, 0.1), 0, 1)
        same_srv_rate = np.clip(rng.normal(0.95, 0.05), 0, 1)
        num_failed_logins = 0
        logged_in = 0

    elif label == "probe":
        # Probe: many connections, low bytes, scanning different services
        duration = max(0, rng.normal(2, 3))
        src_bytes = max(0, rng.normal(20, 15))
        dst_bytes = max(0, rng.normal(0, 3))
        flag = rng.choice(["S0", "REJ", "SF"], p=[0.5, 0.3, 0.2])
        count = rng.integers(20, 100)
        srv_count = rng.integers(1, 20)
        serror_rate = np.clip(rng.normal(0.5, 0.2), 0, 1)
        same_srv_rate = np.clip(rng.normal(0.2, 0.15), 0, 1)
        num_failed_logins = 0
        logged_in = 0

    elif label == "r2l":
        # R2L: login attempts from remote, some failed logins, moderate duration
        duration = max(0, rng.normal(20, 15))
        src_bytes = max(0, rng.normal(200, 150))
        dst_bytes = max(0, rng.normal(300, 200))
        flag = rng.choice(["SF", "S0"], p=[0.7, 0.3])
        count = rng.integers(1, 15)
        srv_count = rng.integers(1, 10)
        serror_rate = np.clip(rng.normal(0.1, 0.1), 0, 1)
        same_srv_rate = np.clip(rng.normal(0.6, 0.2), 0, 1)
        num_failed_logins = rng.integers(1, 5)
        logged_in = rng.choice([0, 1], p=[0.6, 0.4])

    else:  # u2r
        # U2R: rare, longer duration once access is gained, privilege escalation
        duration = max(0, rng.normal(100, 80))
        src_bytes = max(0, rng.normal(800, 500))
        dst_bytes = max(0, rng.normal(400, 300))
        flag = "SF"
        count = rng.integers(1, 5)
        srv_count = rng.integers(1, 5)
        serror_rate = np.clip(rng.normal(0.05, 0.05), 0, 1)
        same_srv_rate = np.clip(rng.normal(0.8, 0.15), 0, 1)
        num_failed_logins = rng.integers(0, 2)
        logged_in = 1

    return {
        "duration": round(duration, 2),
        "protocol_type": protocol,
        "service": service,
        "flag": flag,
        "src_bytes": int(src_bytes),
        "dst_bytes": int(dst_bytes),
        "count": int(count),
        "srv_count": int(srv_count),
        "serror_rate": round(serror_rate, 3),
        "same_srv_rate": round(same_srv_rate, 3),
        "num_failed_logins": int(num_failed_logins),
        "logged_in": int(logged_in),
        "label": label,
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    os.makedirs("data", exist_ok=True)

    labels = list(ATTACK_TYPES.keys())
    probs = list(ATTACK_TYPES.values())
    sampled_labels = rng.choice(labels, size=N_SAMPLES, p=probs)

    rows = [generate_connection(rng, label) for label in sampled_labels]
    df = pd.DataFrame(rows)

    # Binary label for the main detection task: normal vs any attack
    df["is_attack"] = (df["label"] != "normal").astype(int)

    df.to_csv("data/network_traffic.csv", index=False)

    print(f"Generated {len(df)} connection records")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nBinary class balance:\n{df['is_attack'].value_counts(normalize=True).round(4)}")


if __name__ == "__main__":
    main()
