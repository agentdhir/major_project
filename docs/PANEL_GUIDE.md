# FORESIGHT — Panel Guide

How to present FORESIGHT to the evaluation panel: the story, the timing, the
live demo, the numbers to know, and model answers to the questions examiners
usually ask. The slide-by-slide script is in the **speaker notes** of
`presentation/FORESIGHT_Major_Project_Presentation.pptx`
(PowerPoint → View → Notes).

---

## 1. The story in 60 seconds (memorise this)

> "A normal intrusion detection system is a smoke alarm: it rings when there
> is already fire. Real attacks, however, come in stages — the attacker first
> scans our network to find live machines, open ports and weak services, and
> only then floods or breaks in. Those scans pass through our own gateway, so
> from the defender's viewpoint we can see the attacker's footsteps.
>
> FORESIGHT is a deep-learning system that reads the last 64 connections at the
> gateway as a sequence and forecasts the probability that an attack will hit
> within the next 10, 25, 50 or 100 connections, which type of attack it will
> be, and why it thinks so.
>
> We trained it on one network and tested it on another with 17 attack types
> it had never seen. To make that work we combined a per-connection detector,
> an autoencoder that only knows normal traffic, and a short unlabelled
> 'learning period' on the new network. It warns for almost every attack
> campaign that has a reconnaissance phase, on average more than 100
> connections before the attack arrives, and explains every warning."

## 2. Suggested timing (15-minute slot)

| Min | Slides | Content | Speaker |
|---|---|---|---|
| 0:00 | 1–2 | Title, outline | Raghav |
| 0:45 | 3–6 | Big idea, kill chain, problem, objectives | Raghav |
| 3:30 | 7–8 | Literature and gap | Raghav |
| 5:00 | 9–11 | Dataset, simulator, formulation | Rudra |
| 7:00 | 12–14 | Architecture, FORESIGHT-Net, generalisation journey | Nikhil |
| 9:30 | 15–16 | Tools, implementation | Aayush |
| 10:00 | 17–25 | Results: main table, four numbers, trade-off, case studies, ablation, robustness, zero-day, type/ETA/detection, explainability | Nikhil / Aayush |
| 13:00 | 26 | Live demo (or the screenshot if the laptop misbehaves) | Aayush |
| 14:00 | 27–32 | Limitations, future work, conclusion, learnings, references, Q&A | Raghav |

If time is short, skip slides 21 (ablation) and 24 (type/ETA/detection) and keep
them as backup for questions — but keep slide 22 (robustness): it is the strongest
argument for the deep model. Adjust the speaker split to your team.

## 3. Live demo checklist

Before the session:

1. Plug the laptop in (the GPU is throttled on battery; the demo also runs on CPU).
2. `streamlit run app/dashboard.py` — wait until it opens in the browser and the
   first chart appears (models load once, ~5–10 s).
3. Keep a second terminal ready with `python monitor.py` as a backup.
4. A demo link that opens directly on a warning:
   `http://localhost:8501/?scenario=recon_dos&t=520`

During the demo (≈3 min):

1. *Reconnaissance → DoS*, stealth 1.0, **Play**. Narrate: grey normal traffic →
   orange scans begin → blue forecast rises → threat level turns **amber (EARLY
   WARNING)** → only later the red attack region starts. The chart prints
   "warned N connections early".
2. **Pause** while amber: show expected attack type, ETA, the attention bars
   (orange bars = scans the model focused on) and the "Why?" feature chart.
3. Switch to *Harmless scan (no attack follows)* and play it — discuss honestly
   whether it warns (decoys are the main source of false alarms).
4. Optional: stealth 0.4 — slower reconnaissance, later warning.

Every prediction at connection *t* uses only connections *t−63 … t*; the
shaded ground-truth phases are drawn for the audience, the model never sees them.

## 4. Numbers to know

All numbers are on the **test network** (KDDTest+ records, 160 campaigns,
17 unseen attack types), deployed configuration (site-adapted, 1 % alert
budget), mean of 3 seeds. Source: `outputs/results/RESULTS.md`.

<!-- NUMBERS:START -->
| What | Number |
|---|---|
| Campaigns with reconnaissance warned before the attack | **99 %** (98.7 ± 0.4) |
| Median head start | **120 connections** (scan-count rule 84, window XGBoost 88) |
| Head start vs window XGBoost at the same false-alarm level (~20 per 10k) | **+36 %** (120 vs 88) |
| False alarms per 10,000 quiet connections | 19.6 (rule 6.8, window XGBoost 20.1, other deep models 21.8–42.8) |
| Alert precision | 56.5 % — best of the deep models (rule 70 %) |
| PR-AUC (step-level ranking) | 0.653 ± 0.008 (best: 1D-CNN 0.687; rule 0.633; reactive IDS 0.167) |
| Very stealthy attackers (×0.25 scan intensity) | warns **88 %** of campaigns, 104 conns early (rule 52 %, window XGBoost 70 %) |
| Zero-shot → calibrated → adapted (false alarms / 10k) | 66 → 12 (but lead only 40) → **19.6 with lead 120** |
| Attack type forecast before impact | 70 % accuracy, macro-F1 0.69 (best of deep models) |
| ETA within one horizon bin | 87 % |
| R2L detection recall with context | 9 % → **23 %** (upper bound; simulated campaign structure helps) |
| Ablation | every component raises alert precision (56.5 % vs 43–53 % without) |
| Zero-day guard on 17 unseen attack types (3,666 conns) | AE AUC 0.956; hybrid detects 84 % at 5 % FPR; AE catches 61 % of R2L vs 50 % |
| Attention focus | scans get **1.8×** their share of attention |
| Model size / speed | 349,653 parameters · ~74 s training · **2.7 ms per forecast on CPU** |
| Decoy scans that still trigger a warning | 86 % (honest weak spot) |
| Data | 125,973 train + 22,544 test records; 17 attack types only in test; 160 test campaigns |
<!-- NUMBERS:END -->

## 5. Questions the panel is likely to ask — with answers

### About the idea

**Q1. How can you predict an attack that has not happened yet?**
Most attacks are multi-stage. Before the damaging step (flood, break-in) the
attacker performs reconnaissance — host discovery, port scanning, vulnerability
scanning (MITRE ATT&CK TA0043, T1595). Those scans cross our gateway. We do not
predict the future from nothing; we recognise the *preparation* of an attack
and estimate when the attack itself will follow.

**Q2. Isn't scanning already an attack? Then you are just detecting it.**
Scanning causes no damage and is extremely common (search engines, researchers,
misconfigured hosts). A classical IDS either ignores it or alarms on all of it.
FORESIGHT's job is to decide *which* scanning activity is the beginning of a real
campaign — 28 % of our campaigns are harmless decoy scans — and to forecast when
and which impact will follow. That is a forecasting problem, not detection.

**Q3. What does "from our viewpoint" mean?**
The model sees only what the defender sees: the stream of connection records at
our own gateway. No threat-intelligence feeds, no attacker-side data.

**Q4. What exactly is predicted?**
Four probabilities — attack within 10, 25, 50, 100 connections — plus the attack
type (DoS / R2L / U2R) and the class of the current connection. The alarm uses the
100-connection horizon; the ETA is the first horizon whose probability exceeds 50 %.

### About the data

**Q5. NSL-KDD has no timestamps. Where do your sequences come from?**
From a simulator that replays real NSL-KDD records in the order a kill-chain
campaign would produce them. Every connection is a genuine record; only the
order is simulated. We say this openly — it is our main limitation — and the
pipeline is built so that a real timestamped dataset (CIC-IDS2017,
UNSW-NB15) only needs a different stream source.

**Q6. Then isn't the model just learning your simulator's rules?**
We guarded against that: (1) 28 % decoy scans and 0.4 % stray scans make "scan ⇒
alarm" wrong; (2) bursty normal traffic; (3) 12 % of attacks have no warning at
all; (4) the test stream uses only test-network records — scanners like `mscan`
and 17 attack types that never occur in training; (5) the per-connection content
is real, so the model must recognise real traffic. A simple scan-count rule is
included as a baseline precisely to show what "learning the simulator" would give.

**Q7. Why NSL-KDD and not a newer dataset?**
It has the official train/test split with unseen attack types, which is ideal for
testing generalisation, and it continues our Phase-1 work. Its age is a
limitation we state; moving to CIC-IDS2017 is future work.

**Q8. Why not split randomly like most papers?**
Random splits put near-identical records in train and test and give 96–99 %
accuracy; on the official split the same models fall to ~79 % (our Phase 1). We
want the honest number.

### About the model

**Q9. Why deep learning? Why not just XGBoost?**
The signal is temporal: escalating intensity, the order of scanning stages, bursts
and pauses. Sequence models read order directly; tree models need hand-made window
statistics. We compare against exactly that (window-statistics XGBoost) and report
the results honestly — the deep models give a longer head start; the classical
model is competitive on ranking. We also *use* XGBoost inside FORESIGHT as a
per-connection detector: the best of both.

**Q10. Explain the architecture in one breath.**
Embed each connection → causal dilated CNN for short bursts → 2-layer LSTM for
slow escalation → attention to focus on the few relevant connections and explain
the alarm → three heads: survival-style early warning, attack type, detection.

**Q11. What is the "survival" or "hazard" head?**
Instead of four independent sigmoids, the network predicts the chance the attack
starts in each interval (0–10], (10–25], (25–50], (50–100]. The probability of no
attack yet is the product of "not in each interval"; one minus that is
P(attack ≤ h). Result: forecasts are always consistent (P(≤25) ≥ P(≤10)) and give an ETA.

**Q12. What is out-of-fold stacking and why did you need it?**
The deep model receives the per-connection detector's probabilities. If the
detector scored its own training records, it would be almost always right and the
deep model would learn to trust it blindly. With 5-fold out-of-fold predictions
every training record is scored by a detector that never saw it — realistic,
imperfect evidence, as on new traffic.

**Q13. What is modality dropout?**
During training, for 25 % of windows we replace the detector's opinion by a
confident "normal" — as if an attacker used a tool the detector does not know — and
for another 25 % we hide the raw features. The model learns to fall back on the
remaining channels, especially the autoencoder's novelty score.

**Q14. How many parameters, how long to train, how fast?**
See §4. Roughly 350k parameters, under two minutes of training on a laptop GPU,
a few milliseconds per forecast on a CPU — real-time capable.

**Q15. Why window 64 and horizon 100?**
64 connections cover the decisive part of a reconnaissance phase (60–200
connections) while staying fast; 100 connections is long enough to be actionable.
Both are configurable in `foresight/config.py`.

### About evaluation and results

**Q16. Why PR-AUC and not accuracy?**
Only ~12 % of moments precede an attack; accuracy would reward always saying
"no attack". PR-AUC measures how well dangerous moments are ranked above quiet
ones. For operators we add event-level metrics: early-warning rate, lead time,
false alarms per 10k connections, alert precision.

**Q17. Is FORESIGHT-Net the best model on every metric?**
No, and we show it. Once all deep backbones share our evidence channels and site
adaptation they are close; some simpler backbones match or exceed it on PR-AUC.
FORESIGHT-Net's strengths are the balance of head start versus false alarms
(fewest false alarms and best alert precision of all deep models, with one of the
longest head starts), the best attack-type forecast and the built-in attention
explanation. The 1D-CNN ranks best on PR-AUC (0.687 vs 0.653) but raises ~50 % more
false alarms. The bigger contribution is the framework — forecasting formulation,
evidence channels, site adaptation and evaluation — which lifts every backbone.

**Q18. The scan-count rule has fewer false alarms. Why not just use it?**
It is a strong, honest baseline and it is very conservative. Its warnings come
later (smaller head start) and it cannot say which attack is coming or when. In
practice a SOC could run both; the operating-curve slide shows the trade-off at
every false-alarm level.

**Q19. How did you set the alarm threshold without test labels?**
On the new network the system first observes unlabelled normal traffic (burn-in).
The threshold is set so that only 1 % of those windows would alarm — the same
alert budget for every method. No test labels are used. The 1 % value was fixed
before testing; other budgets are reported in `budget_sensitivity.csv`.

**Q20. What is site adaptation? Isn't that training on the test set?**
It uses only *unlabelled* normal traffic of the new network, from records that are
disjoint from the test stream — like the "learning period" of commercial
anomaly-detection products. It never sees an attack label or a test-stream record.
Without it the deep models produce many false alarms on the new network
(zero-shot column of `RESULTS.md`).

**Q21. Your design changed after looking at test results — is that cheating?**
We disclose it (METHODOLOGY §8). Thresholds, early stopping and model selection
used only validation or unlabelled data. The architecture iterations (hybrid
evidence, site adaptation) were informed by test diagnostics, which is a form of
test-set reuse; a fully clean protocol would hold out a second, untouched test
network — listed as future work.

**Q22. What happens with attacks that have no reconnaissance?**
They cannot be forecast — by anyone. We include 12 % such attacks and report that
they are warned only at the false-alarm rate. They are still detected on arrival
by the detection head and the zero-day guard.

**Q23. What about slow, stealthy attackers?**
Slide 22: we re-simulated the test network with reconnaissance at 25–75 % of the
normal intensity and re-ran the trained models without retraining. At 25 %
intensity FORESIGHT-Net still warns for 88 % of campaigns, 104 connections early;
the scan-count rule falls to 52 % (58 early) and window XGBoost to 70 %. Counting
rules need many scans in the window; a sequence model integrates weak evidence.

**Q24. What are false alarms per 10k connections in real terms?**
They are relative numbers for comparing methods on our stream (one stream step is
one connection sampled at the gateway). Real gateways see thousands of
connections per second, so in deployment you would aggregate per source host and
apply the alert budget the SOC can handle.

**Q25. How do you know the attention explanation is meaningful?**
We measure it: across 9,475 alarmed reconnaissance windows, scan connections
receive 32 % of the attention while being 17 % of the connections — 1.8× their share
(slide 25). We also give a
second, independent explanation — gradient × input attribution.

### About the zero-day guard

**Q26. How does an autoencoder detect unknown attacks?**
It is trained to compress and rebuild only normal connections. Anything unlike
normal traffic is rebuilt badly; the reconstruction error, converted to a
percentile of normal traffic, is an attack-agnostic anomaly score.

**Q27. Did the autoencoder beat the supervised model?**
Partly. The autoencoder has the best AUC on the novel attack types (0.956 vs 0.951)
and catches more R2L (61 % vs 50 %), but at the fixed 5 % validation threshold its
false-positive rate drifts to 9.4 % on the new network; the hybrid detects 84 % of
novel attacks at 5.7 %. Its key role in FORESIGHT is as an input channel: the scanner `mscan` exists only in the test set — the
supervised detector recognises 23 % of its connections as scans, the autoencoder
flags 100 % as unusual.

### About implementation and practice

**Q28. Can this run on a real network?**
The inference API (`foresight/inference.py`) takes connection records. A packet
capture front-end (CICFlowMeter-style feature extraction) would feed it; that is
future work. Inference speed is not the bottleneck.

**Q29. What would you do when a warning fires?**
Show the analyst the forecast, type, ETA and evidence; with approval, rate-limit or
sandbox the scanning source, raise logging on the targeted services, or tighten
firewall rules for the predicted attack type.

**Q30. How reproducible is it?**
One command (`python run_pipeline.py`) rebuilds everything from the raw files:
streams, 30 deep-model trainings with fixed seeds, baselines, tables, figures. The
slides are generated from the result files.

**Q31. What is new compared with your mid-term (Phase 1)?**
Phase 1 classified each connection (reactive) with classical ML and showed the
honest ~79 % accuracy and the R2L weakness. Phase 2 reformulates the task as
forecasting with deep sequence models, adds the zero-day guard, deployment on a new
network, explanations and a dashboard. Phase 1's XGBoost lives on inside
FORESIGHT as the per-connection detector.

**Q32. What was the hardest part?**
Generalisation. The first deep model looked excellent on validation and was clearly
worse on the new network. Diagnosing why (different normal traffic, unseen
scanners) and fixing it (evidence channels, site adaptation) was the core research
work — and the slide we are proudest of.

**Q33. What did you try that did not work?**
Randomising which scanning tool is used during training — it lowered test PR-AUC,
showing the order of scanning stages is genuine signal. Also, the novelty channel
did not improve ranking on its own; we kept it for unknown tools.

**Q34. Individual question: what exactly did *you* do?**
Each member should be able to explain their own module end-to-end, including one
design decision and one problem they solved (see the contribution slide).

**Q35. If you had six more months?**
Real timestamped traffic (CIC-IDS2017), per-attacker forecasting, live packet
capture, automated response with analyst approval, a second untouched test
network, and statistical significance tests.

## 6. Do's and don'ts

* **Do** state the simulator limitation yourselves, early (slide 10) — it disarms
  the most likely attack on the work.
* **Do** call the scan-count rule a strong baseline; never belittle baselines.
* **Do** refer to the official split and the 17 unseen attack types repeatedly —
  it is the strongest evidence of honest evaluation.
* **Don't** claim 99 % accuracy or that FORESIGHT "stops all attacks".
* **Don't** say the deep model is best at everything — say what the table says.
* **Don't** read the slides; use the speaker notes as a guide, not a script.
