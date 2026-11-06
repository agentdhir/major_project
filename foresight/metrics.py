"""
metrics.py — how we decide whether an early-warning system is any good.

Two complementary views:

1. STEP-LEVEL (standard ML metrics), on steps where no attack is in progress:
     ROC-AUC, PR-AUC (average precision), precision / recall / F1 at the
     chosen threshold for the target "attack within the next 100 connections".

2. EVENT-LEVEL (what a security operations centre actually cares about):
     * Early-Warning Rate (EWR): share of attack campaigns that triggered an
       alarm BEFORE the first attack connection arrived.
     * Lead time: how many connections before the attack the FIRST alarm
       fired (bigger = more time to block the attacker).
     * False alarms per 10k connections: alarm episodes that start while no
       attack is coming (normal traffic or a decoy scan).
     * Decoy false-alarm rate: share of harmless scans that raised an alarm.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from .config import WindowConfig


def best_f1_threshold(score, y):
    """Threshold that maximises F1 (chosen on VALIDATION, applied to test)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def step_metrics(score, y, thr):
    """Standard binary metrics at a fixed threshold."""
    pred = score > thr
    return dict(roc_auc=float(roc_auc_score(y, score)),
                pr_auc=float(average_precision_score(y, score)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)))


def alarm_episodes(alarm: np.ndarray, merge_gap: int = 10):
    """Group consecutive alarm steps into episodes [(start, end), ...].

    Alarms separated by fewer than `merge_gap` quiet steps are merged, so a
    flickering probability counts as one alert, as an analyst would see it.
    """
    on = np.flatnonzero(alarm)
    if len(on) == 0:
        return []
    breaks = np.flatnonzero(np.diff(on) > merge_gap)
    starts = np.r_[on[0], on[breaks + 1]]
    ends = np.r_[on[breaks], on[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def negative_region(stream, wcfg: WindowConfig):
    """Steps where an alarm is definitely FALSE: no attack in progress, no
    attack coming within the alarm horizon, and not inside the recon of a
    real campaign (an alarm during real recon is an early, correct warning)."""
    k = wcfg.alarm_horizon_idx
    return (stream.forecast_mask & (stream.y_h[:, k] == 0)
            & (stream.phase != 1) & (stream.phase != 4))


def event_metrics(stream, score, thr, wcfg: WindowConfig = None):
    """Campaign-level early-warning metrics for a score series over a stream.

    `score` must have one value per stream step (steps without a full
    window should be filled with 0).
    """
    wcfg = wcfg or WindowConfig()
    alarm = score > thr
    camps = stream.campaigns
    leads, warned_pred, warned_unpred = [], [], []
    per_campaign = []
    for r in camps.itertuples():
        if r.onset < 0:
            continue
        z0 = r.recon_start if r.has_recon else max(0, r.onset - wcfg.lead_search)
        zone = alarm[z0:r.onset]
        hit = bool(zone.any())
        lead = int(r.onset - (z0 + int(np.argmax(zone)))) if hit else 0
        (warned_pred if r.predictable else warned_unpred).append(hit)
        if r.predictable and hit:
            leads.append(lead)
        per_campaign.append(dict(id=r.id, scenario=r.scenario, warned=hit,
                                 lead=lead, attack_type=r.attack_type))
    # Decoys: any alarm during the scan or shortly after it.
    decoys = camps[camps.scenario == "decoy_scan"]
    decoy_hits = [bool(alarm[r.recon_start:r.end + 31].any())
                  for r in decoys.itertuples()]
    # Alarm episodes: false ones start in the negative region, true ones
    # start during a real campaign (recon/attack) or when an attack is due.
    neg = negative_region(stream, wcfg)
    pos_zone = stream.y_h[:, wcfg.alarm_horizon_idx] == 1
    for r in camps.itertuples():
        if r.onset >= 0:
            z0 = r.recon_start if r.has_recon else max(0, r.onset - wcfg.lead_search)
            pos_zone[z0:r.end + 1] = True
    eps = alarm_episodes(alarm)
    n_fa = sum(1 for s, _ in eps if neg[s])
    n_true = sum(1 for s, _ in eps if pos_zone[s] and not neg[s])
    # "Actionable" warnings: at least 10 connections of head start.
    pred_rows = [p for p, c in zip(per_campaign,
                                   camps[camps.onset >= 0].itertuples())
                 if c.predictable]
    ewr_10 = (float(np.mean([p["warned"] and p["lead"] >= 10 for p in pred_rows]))
              if pred_rows else 0.0)
    leads = np.array(leads) if leads else np.array([0])
    return dict(
        ewr=float(np.mean(warned_pred)) if warned_pred else 0.0,
        ewr_10=ewr_10,
        lead_median=float(np.median(leads)),
        lead_mean=float(np.mean(leads)),
        ewr_unpredictable=float(np.mean(warned_unpred)) if warned_unpred else 0.0,
        decoy_alarm_rate=float(np.mean(decoy_hits)) if decoy_hits else 0.0,
        false_alarms_per_10k=float(n_fa / max(neg.sum(), 1) * 1e4),
        alert_precision=float(n_true / max(n_true + n_fa, 1)),
        step_fpr=float(alarm[neg].mean()) if neg.any() else 0.0,
        n_predictable=len(warned_pred), n_unpredictable=len(warned_unpred),
        n_decoys=len(decoy_hits),
        per_campaign=per_campaign,
    )


def evaluate_scores(stream, score, thr, wcfg: WindowConfig = None):
    """Step-level + event-level metrics in one dict (per-campaign dropped)."""
    wcfg = wcfg or WindowConfig()
    k = wcfg.alarm_horizon_idx
    m = stream.forecast_mask.copy()
    m[:wcfg.window - 1] = False
    res = step_metrics(score[m], stream.y_h[m, k], thr)
    ev = event_metrics(stream, score, thr, wcfg)
    ev.pop("per_campaign")
    res.update(ev)
    res["threshold"] = float(thr)
    return res


def site_threshold(site_score, budget: float, wcfg: WindowConfig = None):
    """Site self-calibration with a common ALERT BUDGET.

    A model trained on one network sees different "normal" traffic on a new
    network (KDDTest+ normal traffic differs from KDDTrain+), so a raw
    probability threshold learned on validation does not transfer.

    Instead, every method observes a short UNLABELLED burn-in period of the
    new network's traffic and sets its threshold so that it would have been
    alarming on only `budget` (e.g. 1 %) of those windows:

        threshold = quantile(burn-in scores, 1 - budget)

    Every system therefore gets the SAME false-alarm allowance, no attack
    labels from the new network are used, and this is exactly how anomaly-
    based products are commissioned (a "learning period" on site).
    """
    wcfg = wcfg or WindowConfig()
    return float(np.quantile(site_score[wcfg.window - 1:], 1 - budget))


def operating_curve(stream, score, thresholds, wcfg: WindowConfig = None):
    """EWR and false alarms for a sweep of thresholds (trade-off curve)."""
    rows = []
    for t in thresholds:
        ev = event_metrics(stream, score, t, wcfg)
        rows.append(dict(threshold=float(t), ewr=ev["ewr"],
                         fa_per_10k=ev["false_alarms_per_10k"],
                         decoy=ev["decoy_alarm_rate"],
                         lead_median=ev["lead_median"]))
    return rows
