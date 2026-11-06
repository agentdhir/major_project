# Major Project — Progress Report
## Network Intrusion Detection with Class-Imbalance-Aware Machine Learning

**Dataset:** NSL-KDD (KDDTrain+ 125,973 records / KDDTest+ 22,544 records, 41 features)
**Evaluation protocol:** trained on KDDTrain+, evaluated on the *official disjoint* KDDTest+ split.

---

## 1. Motivation and Research Gap

The reference literature (Ahmad et al., *Trans. Emerging Telecom. Tech.* 2020; Christy et al.,
*Scientific Reports* 2025) reports NSL-KDD accuracies of 96–99%. We found that such figures are
typically obtained by **randomly splitting KDDTrain+**, where train and test come from the same
distribution.

When the *official* KDDTest+ split is used — which deliberately contains attack signatures absent
from training — performance drops sharply. **Reproducing this gap is the first contribution of
this work.**

| Split protocol | Reported/observed accuracy |
|---|---|
| Random split of KDDTrain+ (common in literature) | 96–99% |
| **Official KDDTrain+ → KDDTest+ (this work)** | **77–79%** |

### The root cause (Fig. 1)

| Class | Train | Test | Ratio |
|---|---:|---:|---:|
| Normal | 67,343 | 9,711 | — |
| DoS | 45,927 | 7,460 | — |
| Probe | 11,656 | 2,421 | — |
| **R2L** | **995** | **2,885** | **2.9× more in test** |
| **U2R** | **52** | **67** | **1.3× more in test** |

R2L and U2R are severely under-represented in training yet abundant in test. This
**distribution shift**, not model capacity, is the binding constraint.

---

## 2. Baseline Results (Fig. 2)

Six classifiers, identical preprocessing (one-hot encoding of 3 categorical features → 122
dimensions, standardised).

### Binary classification (Normal vs. Attack)

| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **0.7938** | 0.9683 | 0.6593 | **0.7845** | 0.0285 | 0.9701 |
| Decision Tree | 0.7854 | 0.9648 | 0.6467 | 0.7743 | 0.0312 | 0.8079 |
| KNN (k=5) | 0.7704 | 0.9234 | 0.6507 | 0.7635 | 0.0714 | 0.8197 |
| Random Forest | 0.7648 | 0.9670 | 0.6075 | 0.7462 | 0.0274 | 0.9594 |
| Logistic Regression | 0.7535 | 0.9173 | 0.6232 | 0.7421 | 0.0742 | 0.7801 |
| Naive Bayes | 0.5546 | 0.9801 | 0.2222 | 0.3622 | 0.0060 | 0.7992 |

**Key observation:** precision is high (~0.97) but recall is low (~0.66). The models are
*conservative* — when they flag an attack they are almost always right, but **they miss ~34% of
real attacks**. For an IDS, missed attacks are the costly error.

### Multi-class — per-class breakdown of the best baseline

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| DoS | 0.93 | 0.85 | 0.89 | 7,460 |
| Normal | 0.67 | 0.93 | 0.78 | 9,711 |
| Probe | 0.85 | 0.72 | 0.78 | 2,421 |
| **R2L** | 0.45 | **0.03** | **0.06** | 2,885 |
| **U2R** | 0.70 | 0.31 | 0.43 | 67 |

**This isolates the problem precisely.** DoS and Probe are essentially solved (F1 0.89 / 0.78).
R2L recall of **0.03** means the system detects only 3 of every 100 remote-to-local attacks.
Aggregate accuracy hides this completely.

---

## 3. Proposed Method

A three-component pipeline targeting the rare classes:

1. **SMOTE** — synthetic oversampling of R2L and U2R to 5,000 samples each (k=3 neighbours,
   applied to training data only, never to test).
2. **Cost-sensitive learning** — inverse-frequency sample weights so rare-class errors are
   penalised proportionally.
3. **Tuned XGBoost** — 300 estimators, depth 10, lr 0.08, subsample 0.9, colsample 0.8.

### Ablation study (Fig. 3) — each component isolated

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline XGBoost | 0.7783 | 0.5516 |
| + Cost-Sensitive Weights | 0.7783 | 0.5906 |
| + SMOTE | 0.7790 | 0.5829 |
| **PROPOSED (SMOTE + Cost + Tuned)** | **0.7896** | **0.6320** |

**Macro-F1 improves 0.5516 → 0.6320 (+14.6% relative).** Macro-F1 weights every class equally,
so it is the correct metric here; accuracy is dominated by Normal/DoS and barely moves.

### Per-class recall (Fig. 4) — where the gain lands

| Method | DoS | Normal | Probe | **R2L** | **U2R** |
|---|---:|---:|---:|---:|---:|
| Baseline XGBoost | 0.8434 | 0.9711 | 0.6729 | 0.0648 | 0.1194 |
| + Cost-Sensitive | 0.8324 | 0.9721 | 0.6981 | 0.0655 | 0.2687 |
| + SMOTE | 0.8421 | 0.9711 | 0.6696 | 0.0738 | 0.2239 |
| **PROPOSED** | 0.8420 | 0.9715 | **0.7030** | **0.1244** | **0.3582** |

- **U2R recall: 0.1194 → 0.3582 (3.0×)**
- **R2L recall: 0.0648 → 0.1244 (1.9×)**
- DoS and Normal recall unchanged — **the gain is not bought by degrading majority classes.**

---

## 4. Honest Limitations

Stated explicitly rather than hidden:

1. **R2L recall is still only 0.12.** SMOTE interpolates between existing minority samples, so it
   cannot synthesise attack signatures that never appear in training. This is a ceiling of
   resampling, not a tuning failure.
2. **Overall accuracy (~0.79) is far below the 96%+ in the literature.** We argue this is the
   *honest* number under the official disjoint protocol, and that the higher published figures
   reflect an easier evaluation setup.
3. **Normal-class precision is 0.70** — false alarms remain a practical concern for deployment.

---

## 5. Next Steps

1. **Feature selection** — mutual information / RFE to cut 122 dimensions, reducing inference
   latency (the "lightweight" requirement in the VANET reference paper).
2. **Anomaly detection for R2L/U2R** — an Autoencoder or One-Class SVM trained only on Normal
   traffic can flag *unseen* attack types that supervised resampling provably cannot.
3. **Hybrid architecture** — supervised classifier for DoS/Probe (already strong) + anomaly
   detector for rare/unseen classes. This directly targets the documented limitation.
4. **Cross-dataset validation** — repeat on CICIDS2017 / UNSW-NB15 to test generalisation.
5. **Statistical validation** — 5-fold CV with mean ± std and significance testing.

---

## 6. Reproducibility

```
data/                    NSL-KDD (auto-downloaded)
src/data_loader.py       Loading, attack taxonomy, preprocessing
src/experiment.py        Baseline comparison (6 models)
src/proposed.py          Ablation study of proposed method
src/figures.py           Figure generation
results/                 CSV metrics + 5 publication figures
```

Run: `python src/experiment.py binary` → `multiclass` → `python src/proposed.py` → `python src/figures.py`

All results use `random_state=42`. Figures at 300 DPI.
