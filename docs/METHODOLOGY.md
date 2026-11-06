# FORESIGHT — Methodology

This document describes, in enough detail to reproduce and to defend, every
design decision in FORESIGHT. Code references point to the `foresight/`
package. All numbers quoted here come from `outputs/results/` after
`python run_pipeline.py`.

---

## 1. Problem formulation

A classical Network Intrusion Detection System (NIDS) answers

> *"Is this connection an attack?"*

which means the alarm can only fire **after** the attack traffic has reached
the network. FORESIGHT answers a different question:

> *"Given the last W connections we have seen, will an attack hit us within
> the next h connections — and if so, what kind?"*

Formally, for a stream of connections x₁, x₂, … observed at our gateway and a
look-back window W = 64, the model receives

    X_t = (x_{t-63}, …, x_t)

and predicts, for horizons h ∈ {10, 25, 50, 100}:

    P(an attack connection arrives in (t, t+h] | X_t)       — early warning
    P(type of that attack ∈ {DoS, R2L, U2R} | X_t)          — what is coming
    P(category of x_t ∈ {Normal, DoS, Probe, R2L, U2R})     — detection (now)

An **attack** here means the *impact* stages DoS, R2L (remote-to-local) and
U2R (user-to-root). **Probe** (scanning) is treated as a *precursor*: it is
what we want to notice early, not the damage we are forecasting.

Why should this be possible? Real intrusions are multi-stage (Lockheed-Martin
Cyber Kill Chain; MITRE ATT&CK tactic TA0043 *Reconnaissance*, technique T1595
*Active Scanning*). Before an attacker floods a server or breaks into it, they
discover live hosts, enumerate ports/services and scan for vulnerabilities.
That reconnaissance is visible from the defender's viewpoint — it is the
"footstep before the knock".

---

## 2. Data

### 2.1 NSL-KDD and the official split

| File | Records | Use in FORESIGHT |
|---|---:|---|
| KDDTrain+ | 125,973 | 85 % → *train pool*, 15 % (stratified) → *validation pool* |
| KDDTest+ | 22,544 | 40 % of its normal records + 6 % of its scans → *site burn-in* (two disjoint halves: *adapt* and *calibrate*); the rest → *test pool* |

No record appears in two pools (`data.load_pools`). KDDTest+ is the
deployment network: it contains **17 attack types that never appear in
KDDTrain+** (e.g. `mscan`, `saint`, `apache2`, `processtable`, `snmpguess`,
`httptunnel`) and its *normal* traffic is distributed differently (e.g. more
`http` and `private`-service connections). This makes the test a genuine
"new network" evaluation rather than a random split, which in the literature
inflates accuracy to 96–99 % (see Phase 1, `PROGRESS_REPORT.md`).

### 2.2 Feature encoding (`data.FeatureEncoder`)

* 38 numeric features; the 17 heavy-tailed ones (bytes, durations, counts)
  get `log(1+x)`; all are standardised with **training statistics only** and
  clipped to ±10.
* `protocol_type`, `service`, `flag` become integer ids for **learned
  embeddings** (dimensions 4, 16 and 6). Id 0 is reserved for values never
  seen in training.

### 2.3 Why a simulator? (`simulator.py`)

NSL-KDD records are real connections, but they are **shuffled and carry no
timestamps**, so the dataset alone cannot be used to study *anticipation*.
FORESIGHT therefore **replays real NSL-KDD records in the order a multi-stage
campaign would produce them**, as seen at our gateway. Every connection in
every stream is a genuine NSL-KDD record — only the *order* is simulated.
This is stated openly as the main limitation (§9).

**Campaign types** (probabilities per campaign):

| Scenario | Share | What happens |
|---|---:|---|
| `recon_dos` | 28 % | reconnaissance → DoS flood |
| `recon_r2l` | 18 % | reconnaissance → remote break-in |
| `recon_r2l_u2r` | 14 % | reconnaissance → break-in → privilege escalation |
| `decoy_scan` | 28 % | an opportunistic scan with **no attack afterwards** |
| `stealth_r2l` | 6 % | break-in with **no reconnaissance** (unpredictable) |
| `flash_dos` | 6 % | flood with **no reconnaissance** (unpredictable) |

**Reconnaissance model.** 60–200 connections long, in three stages that use
one tool each — host discovery (`ipsweep`), port scanning (`portsweep`,
`nmap`), vulnerability scanning (`satan`, `saint`, `mscan`). The share of
scan connections ramps from 3–8 % to 30–60 % (`d(u) = d₀ + (d₁−d₀)·u^1.5`),
i.e. "low-and-slow" at first and escalating towards the attack; 15 % of scans
use a tool out of stage order. DoS attackers spend more time on discovery,
break-in attackers more on vulnerability scanning.

**What prevents a trivial "if scan then alarm" solution:**

* **Decoy scans** (28 % of campaigns) look like reconnaissance but nothing follows.
* **Background radiation:** 0.4 % stray scans inside normal traffic.
* **Bursty normal traffic:** real traffic is self-similar (Leland et al.,
  1994); FORESIGHT's background contains *sessions* of 5–40 same-service
  connections, so "many connections to one service" is not by itself suspicious.
* **No-precursor attacks** (12 %) are fundamentally unpredictable and are
  reported separately — they bound what any forecaster can do.
* The **test** stream uses only KDDTest+ records, so unseen scanners
  (`mscan`, `saint`) and unseen attacks appear only at test time.

Stream sizes: 400 campaigns (≈231k connections) for training, 90 (≈53k) for
validation, 160 (≈97k) for test; plus 2 × 20,000 attack-free burn-in
connections from the deployment network.

### 2.4 Targets (`simulator.attach_labels`)

For every step t: `dist[t]` = connections until the next attack connection;
`y_h[t,k] = 1[dist[t] ≤ h_k]`; `y_type[t]` = type of that next attack (None if
beyond 100); `forecast_mask[t]` = *no attack already in progress*. All
forecasting metrics are computed **only on masked steps**, so a model cannot
score by merely detecting an ongoing attack.

---

## 3. Per-connection evidence channels

Each connection is described to the deep model by three channels:

1. **Raw features** (38 numeric + 3 embedded categorical).
2. **Detector evidence** — the 5 class probabilities of a per-connection
   gradient-boosted classifier (XGBoost, 300 trees, depth 8 — the best model
   of Phase 1). For the *training* stream these probabilities are
   **out-of-fold** (5-fold stacking): every training record is scored by a
   model that never saw it, so the deep model learns how much to trust a
   detector that makes mistakes — exactly the situation on new traffic.
3. **Novelty** — the reconstruction-error percentile of a denoising
   **autoencoder trained only on normal traffic** (125→64→32→12→32→64→125).
   0.95 means "more unusual than 95 % of known-normal traffic". The supervised
   detector recognises only 23 % of `mscan` connections as scans; the
   autoencoder flags 100 % of them as unusual. Novelty therefore carries
   information about attack tools nobody has labelled yet.

---

## 4. FORESIGHT-Net (`models.py`)

```
window (64 connections × [38 raw + 5 detector + 1 novelty] + 3 categorical ids)
  │
  ├─ ConnectionEncoder   embeddings(protocol 4, service 16, flag 6) ⊕ numeric
  │                      → Linear → LayerNorm → GELU → Dropout       [64-d]
  ├─ Causal dilated CNN  3 residual blocks, kernel 5, dilation 1/2/4
  │                      (left padding: output at t sees only ≤ t)   bursts
  ├─ LSTM                2 layers × 128 units                         escalation
  ├─ Temporal attention  additive (Bahdanau) pooling over 64 steps    focus + explanation
  │
  ├─ Early-warning head  4 hazard logits z_k → S_k = Π_{j≤k}(1−σ(z_j)),
  │                      P(attack ≤ h_k) = 1 − S_k   (monotone in h by construction)
  ├─ Attack-type head    softmax over {None, DoS, R2L, U2R}
  └─ Detection head      softmax over {Normal, DoS, Probe, R2L, U2R} (from last step)
```

* **Why CNN + LSTM + attention?** The CNN reacts to short bursts (a sweep of
  a dozen scans), the LSTM integrates the slow, minutes-long escalation of a
  campaign, and attention lets the model weight the handful of connections
  that matter in a mostly-normal window — and shows the analyst which ones.
* **Why a survival ("hazard") head?** Independent sigmoids for "within 10"
  and "within 100" can contradict each other. Building each P(≤h) from
  per-interval hazards guarantees monotone forecasts and yields an ETA: the
  first horizon whose probability exceeds 50 %.
* **Why multi-task?** Detecting the current connection and naming the coming
  attack type are auxiliary signals that shape the shared representation;
  the ablation (§8) measures their effect.

### Loss

    L = BCE_survival(warning, all 4 horizons)
        + 0.5 · CE_weighted(attack type) + 0.5 · CE_weighted(detection)

BCE is computed in log-space from the survival function for numerical
stability (`log(1−e^x)` via `expm1`/`log1p`). Class weights are inverse
square-root frequencies, normalised to mean 1 (full inverse frequency would
weight a U2R connection ~800× a normal one).

---

## 5. Training (`train.py`)

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 1.5·10⁻³, weight decay 10⁻⁴, one-cycle schedule |
| Batch | 1024 windows; a random 50 % of all windows per epoch |
| Epochs | ≤ 12, early stopping (patience 3) on **validation** PR-AUC |
| Gradient clipping | 1.0 |
| Seeds | 0, 1, 2 for every deep model (mean ± std reported) |
| Augmentation | token dropout 10 % (categorical → *unknown*), Gaussian noise σ = 0.05, **modality dropout**: for 25 % of windows the detector evidence is replaced by a confident "normal" verdict (as when an unknown tool fools the detector), for another 25 % the raw features are hidden |

Windows are never materialised: the whole stream lives on the GPU and each
batch is gathered by index arithmetic (`windows.WindowSet`).

---

## 6. Deployment on a new network (`train.adapt_to_site`, `metrics.site_threshold`)

A model trained on one network sees a *different normal* on another. FORESIGHT
is commissioned like commercial anomaly-based NDR products, using an
**unlabelled, attack-free burn-in period** of the new network's traffic
(2 × 20,000 connections, disjoint records):

1. **Site adaptation (first half).** 300 fine-tuning steps (lr 2·10⁻⁴): burn-in
   windows with the target "no attack within any horizon", interleaved with
   replayed training windows under the full loss (so attacks are not forgotten).
2. **Threshold calibration (second half).** The alarm threshold is the
   99th percentile of the adapted model's burn-in scores — an **alert budget
   of 1 %** of quiet-time windows. The *same* budget is applied to every
   method, so all systems get the same false-alarm allowance.

No attack labels from the deployment network are used anywhere. The 1 %
budget was fixed before any test result was seen; `budget_sensitivity.csv`
reports 0.25 %–5 %.

Three settings are reported for transparency: **zero-shot** (validation
threshold, no adaptation), **calibrated** (budget threshold, no adaptation)
and **adapted** (the deployed configuration).

---

## 7. Evaluation

**Step level** (on masked steps; target = attack within 100): ROC-AUC,
PR-AUC (average precision — the primary ranking metric because positives are
~12 % of steps), precision/recall/F1 at the threshold.

**Event level** (what a security operations centre experiences):

| Metric | Definition |
|---|---|
| Early-warning rate | share of campaigns *with* reconnaissance that raised an alarm **before** the first attack connection |
| Warned ≥ 10 early | same, with at least 10 connections of head start |
| Lead time | connections between the first alarm and the first attack connection (median over warned campaigns) |
| False alarms / 10k | alarm episodes (alarms < 10 steps apart are merged) that start while no attack is coming, per 10,000 such connections |
| Alert precision | share of alarm episodes that belong to a real campaign |
| Decoys alarmed | share of harmless scans that raised an alarm |
| No-precursor warned | early-warning rate on stealth/flash attacks (expected ≈ budget level) |

**Baselines** (`baselines.py`): (1) *Reactive IDS* — per-connection XGBoost,
score = P(current connection is DoS/R2L/U2R); (2) *Probe-count rule* — expected
number of scans among the last 64 connections (a SOC correlation rule);
(3) *Window-statistics XGBoost* — mean/std/max of all features, detector
probabilities and novelty over the window plus the latest connection (sees
the same window, not the order), site-adapted by adding burn-in windows as
negatives; (4) deep baselines with **identical inputs, heads, loss and
training**: MLP, 1-D CNN, LSTM, GRU, Transformer.

**Further analyses:** ablation of each FORESIGHT-Net component (3 seeds);
robustness to stealthier attackers (recon intensity × 1 → × 0.25, no
retraining); zero-day guard on the 17 unseen attack types; attention focus
and gradient×input feature attribution; inference latency.

---

## 8. Development history (what did not work, and why that matters)

Reported because the panel should know which results are first-try and which
were reached by iteration — and because the failures are instructive.

| Step | Test PR-AUC (seed 0) | Lesson |
|---|---:|---|
| End-to-end deep model on raw features | 0.44 – 0.48 | Val PR-AUC was 0.70: the model memorised the *training network* (its normal traffic, its scanner signatures). |
| + detector evidence (out-of-fold stacking) | 0.57 | A robust per-connection detector is a better "eye"; the deep model adds temporal reasoning on top. |
| + novelty channel & modality dropout | 0.56 | No change in ranking; kept because it is the only channel that sees unknown tools such as `mscan`. |
| + scanner-tool domain randomisation in training | 0.49 | **Rejected:** the stage order of scanning is a genuine signal. |
| + site adaptation on unlabelled burn-in (final, 3 seeds) | **0.653** | Learning the new network's "normal" was the missing piece: false alarms 66 → 20 per 10k, median lead 40 → 120 connections versus calibration alone. |

Note that once site adaptation is in place, the final ablation shows the evidence
channels no longer change PR-AUC (0.652 without them vs 0.653) but they still cut
false alarms from 33 to 20 per 10k and raise alert precision from 44 % to 57 % —
their value moved from ranking to alert quality.

These iterations were guided by results on the test stream. All
hyper-parameters (thresholds, early stopping, model selection) were chosen on
validation or on the unlabelled burn-in only, but the *architecture choices*
above were informed by test-stream diagnostics; this is a form of
test-set reuse and is disclosed as a limitation.

---

## 9. Limitations

1. **Simulated temporal order.** Connections are real; the campaign
   structure is a model (kill chain, ramp, decoys). Results show what is
   achievable *if* attacks behave this way; real timestamped datasets
   (CIC-IDS2017, UNSW-NB15 with `stime`) are the natural next step — the
   pipeline only needs a different `load_pools` + stream source.
2. **No-precursor attacks cannot be forecast** — by any method. FORESIGHT
   still detects them on arrival (detection head, zero-day guard).
3. **Decoy scans remain hard**: a harmless scan and the first minutes of real
   reconnaissance are genuinely similar; the alert budget trades early
   warnings against such alarms.
4. **Test-informed design iterations** (§8).
5. NSL-KDD is dated (1998 traffic); feature semantics carry over to modern
   flow features but absolute numbers do not.
