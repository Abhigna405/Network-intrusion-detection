# Documentation — Network Intrusion Detection

This document goes deeper than the README into the reasoning behind each
design decision — useful for explaining the project in an interview or
project review.

## 1. Problem Framing

Network intrusion detection is a **classification under severe class
imbalance** problem. Real network traffic is overwhelmingly legitimate;
attacks are the minority class, and within attacks, some categories
(U2R — privilege escalation) are rarer still, despite being the most
dangerous. This shapes almost every downstream decision:

- Accuracy alone is a misleading metric (a model predicting "normal"
  100% of the time on our ~70/30 split would score 70% accuracy while
  being completely useless).
- Standard training (unweighted loss) tends to underfit the minority
  class, since the loss function is dominated by the majority class.
- Evaluation must separate "did we get most predictions right" from
  "did we catch the attacks" — these are different questions.

## 2. Why Random Forest and XGBoost (Classical ML, not Deep Learning)

Network intrusion features (connection duration, byte counts, error
rates, categorical protocol/service/flag) are **tabular, hand-engineered
features** — not raw, high-dimensional signals like images or text where
deep learning's representation-learning advantage matters. On tabular
data with well-designed features:

- Tree-based models directly exploit sharp thresholds (e.g. "if `count`
  > 100 and `serror_rate` > 0.8, likely DoS") — this maps naturally to
  how security researchers already think about these features.
- Tree ensembles are far less data-hungry and less sensitive to feature
  scaling than neural networks.
- They train in seconds on this dataset size, with no GPU or heavy
  framework dependency — this matters practically (see the LightFM
  project for what happens when a heavier ML dependency doesn't install
  cleanly on a given machine).

Deep learning can match or exceed tree ensembles on tabular data at
large scale with careful tuning, but for a dataset this size, tree-based
models are both the pragmatic and the performant choice.

## 3. Handling Class Imbalance: Two Different Mechanisms

**Random Forest — `class_weight='balanced'`:**
Automatically reweights each class inversely proportional to its
frequency in the training data, so errors on the minority (attack)
class are penalized more heavily during tree construction. This nudges
the model away from just predicting the majority class everywhere.

**XGBoost — `scale_pos_weight`:**
A single scalar (here, `count(normal) / count(attack)` in the training
set) that upweights the gradient contribution of positive (attack)
examples during boosting. Functionally similar goal to `class_weight`,
but implemented at the gradient-boosting objective level rather than
per-tree bootstrap sampling.

Both are simpler and more standard than alternatives like SMOTE
(synthetic minority oversampling) — they don't fabricate synthetic
minority samples, avoiding the risk of oversampling introducing
unrealistic feature combinations. For a first-pass production system,
class-weighting is the safer, more interpretable default.

## 4. Why Recall Is the Headline Metric (Not Accuracy)

In a security operations context, the cost of each error type is
asymmetric:

- **False Negative** (real attack classified as normal): the attack
  proceeds uninvestigated. Potentially severe — data breach, service
  outage, compromised credentials.
- **False Positive** (normal connection flagged as attack): an analyst
  spends a few minutes reviewing and dismissing a false alarm.

Given this asymmetry, **recall** (of all real attacks, what fraction did
we catch?) is the metric to optimize and monitor most closely, even at
some cost to precision (more false alarms). This is why both models here
use imbalance-handling mechanisms that explicitly trade some precision
for higher recall on the minority class, rather than optimizing raw
accuracy.

## 5. Why Binary Classification as the Primary Task

The dataset includes both a binary label (`is_attack`) and a multi-class
label (`normal`/`dos`/`probe`/`r2l`/`u2r`). This project treats binary
detection as the primary deliverable because:

- It's the first decision point in any real detection pipeline: "does
  this connection need human review at all?"
- Multi-class attack-type labeling is valuable for triage *after* a
  connection is flagged, but is a secondary refinement, not the
  headline capability.
- With only ~1.5% of the dataset being U2R, multi-class evaluation on
  that category alone would be statistically noisy at this dataset
  size — binary framing is the more robust claim to make.

## 6. SHAP: Why Explainability Matters Specifically Here

A feature-importance ranking (built into both RF and XGBoost) tells you
what mattered *on average*, across all predictions. It does **not** tell
you why any *specific* connection was flagged. In a security context,
this distinction is critical: an analyst investigating a specific alert
needs to know what drove that particular decision, not a general
ranking.

SHAP (SHapley Additive exPlanations) decomposes each individual
prediction into additive per-feature contributions, using a
game-theoretic approach (Shapley values from cooperative game theory)
that guarantees the contributions sum exactly to the difference between
the model's prediction and its baseline (average) output. For tree
models specifically, `TreeExplainer` computes this efficiently and
exactly (not approximately), which is why it's used here instead of a
model-agnostic SHAP method.

**Practical value:** if a connection is flagged with `serror_rate` as
the dominant positive SHAP contributor, an analyst immediately knows to
check for a DoS/flood pattern rather than, say, a credential-stuffing
attempt — different SHAP signatures point toward different next
investigative steps.

## 7. Possible Extensions (good talking points if asked "what would you
improve?")

- Hyperparameter tuning via grid/random search or Bayesian optimization
  (Optuna) against a validation split.
- Multi-class attack-type classification as a secondary model, feeding
  from connections already flagged by the binary detector (a two-stage
  pipeline).
- Real-time streaming inference (e.g. via a Kafka consumer + model
  server) rather than static batch evaluation.
- Validate against the real NSL-KDD or CICIDS2017 datasets to confirm
  findings generalize beyond this synthetic generator.
- Ensemble the two models (e.g. weighted average of probabilities) to
  see if combining RF and XGBoost outperforms either alone.
