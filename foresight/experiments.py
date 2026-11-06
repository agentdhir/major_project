"""
experiments.py — runs every experiment of the project and writes all results.

    python run_pipeline.py                 # everything
    python run_pipeline.py --quick         # small smoke test (separate folder)
    python run_pipeline.py --stage deep    # one stage (stages cache outputs)

Stages, in order (each one reads what the previous ones cached):
  data       simulate train / val / test / site-burn-in streams
  baselines  reactive IDS, probe-count rule, window-statistics XGBoost
  deep       6 deep backbones x 3 seeds (FORESIGHT-Net is one of them)
  ablation   FORESIGHT-Net without attention / without CNN / single-task
  robust     stealthier attackers (inference only, no retraining)
  zeroday    autoencoder zero-day guard on attack types never seen in training
  explain    attention focus, feature importance, case studies
  latency    inference speed (is it real-time?)
  summary    tables (CSV + Markdown) and headline numbers (JSON)

EVALUATION PROTOCOL (applies to every method)
  * thresholds are tuned on the VALIDATION stream (KDDTrain+ records);
  * "zero-shot": that threshold is used as-is on the test network;
  * "site-calibrated": every method gets the same alert budget (alarm on at
    most 1 % of 20k UNLABELLED burn-in connections; metrics.site_threshold);
  * the TEST stream (KDDTest+, 17 unseen attack types) is used only for the
    final numbers — never for training, early stopping or threshold choice.
"""
import pickle
import shutil
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, f1_score, recall_score,
                             roc_curve)

from . import config as C
from .baselines import (RecordClassifier, WindowXGB, out_of_fold_proba,
                        probe_rule_score, reactive_score)
from .data import FeatureEncoder, load_pools
from .explain import attention_focus, feature_names, global_feature_importance
from .metrics import (best_f1_threshold, evaluate_scores, event_metrics,
                      operating_curve, site_threshold)
from .simulator import CampaignSimulator, Stream, build_stream
from .train import (adapt_to_site, load_checkpoint, predict, save_checkpoint,
                    train_model)
from .utils import ensure_dirs, get_device, load_json, log, save_json
from .windows import WindowSet
from .zero_day import (EmpiricalCDF, anomaly_score, train_autoencoder,
                       zero_day_report)

W = C.WindowConfig()
K = W.alarm_horizon_idx
STREAMS_FILE = C.CACHE_DIR / "streams.pkl"
ENCODER_FILE = C.MODEL_DIR / "encoder.pkl"
RECORD_XGB_FILE = C.MODEL_DIR / "record_xgb.json"
DEPLOY_FILE = C.MODEL_DIR / "foresight_net.pt"


class Settings:
    """Run-size settings (the --quick flag shrinks everything)."""

    def __init__(self, quick=False):
        self.quick = quick
        self.n_campaigns = (dict(train=60, val=25, test=40) if quick
                            else dict(C.N_CAMPAIGNS))
        self.seeds = (0,) if quick else C.SEEDS
        self.tcfg = (C.TrainConfig(max_epochs=2, patience=1) if quick
                     else C.TrainConfig())
        self.site_burnin = 4000 if quick else C.SITE_BURNIN
        self.adapt_steps = 40 if quick else C.ADAPT_STEPS
        self.stealth = (1.0, 0.35) if quick else C.STEALTH_FACTORS


# ============================================================== data
def stage_data(S: Settings):
    """Simulate every stream once and cache it (so all models see the same)."""
    pools = load_pools()
    enc = FeatureEncoder().fit(pools["train"])
    enc.save(ENCODER_FILE)
    streams = {}
    for name in ("train", "val", "test"):
        streams[name] = build_stream(pools[name], enc, S.n_campaigns[name],
                                     C.STREAM_SEEDS[name])
    # Unlabelled burn-in on the deployment site: background traffic only,
    # two disjoint halves (adaptation / threshold calibration).
    for name in ("adapt", "site"):
        streams[name] = build_stream(pools[name], enc, 0, C.STREAM_SEEDS[name],
                                     lead_in=S.site_burnin)
    # Per-connection evidence from the gradient-boosted detector (Phase 1's
    # best model).  Training-stream records get OUT-OF-FOLD probabilities;
    # every other stream is scored by the detector trained on all of train.
    log("  per-connection detector: out-of-fold evidence for training...")
    oof = out_of_fold_proba(enc, pools["train"], n_folds=3 if S.quick else 5)
    streams["train"].evidence = oof[streams["train"].rows]
    rec = RecordClassifier(enc).fit(pools["train"])
    rec.model.save_model(RECORD_XGB_FILE)
    for name in ("val", "test", "site", "adapt"):
        streams[name].evidence = rec.proba(streams[name]).astype(np.float32)
    # Novelty evidence from the zero-day autoencoder (trained on NORMAL
    # training traffic only).  Percentiles are relative to held-out normal
    # validation traffic, so 0.95 = "more unusual than 95 % of normal".
    log("  zero-day autoencoder (normal traffic only)...")
    tr_n = pools["train"][pools["train"].category == "Normal"]
    va_n = pools["val"][pools["val"].category == "Normal"]
    ae = train_autoencoder(enc.onehot(*enc.transform(tr_n)),
                           enc.onehot(*enc.transform(va_n)),
                           epochs=6 if S.quick else 40)
    torch.save(ae.state_dict(), C.MODEL_DIR / "autoencoder.pt")
    ref = anomaly_score(ae, enc.onehot(*enc.transform(va_n)))
    np.save(C.MODEL_DIR / "autoencoder_val_normal_scores.npy", ref)
    cdf = EmpiricalCDF(ref)
    for s in streams.values():
        s.novelty = cdf(anomaly_score(ae, enc.onehot(s.num, s.cat))).astype(np.float32)
    for name, s in streams.items():
        log(f"  {name:5s} stream: {len(s):7,d} connections, "
            f"{len(s.campaigns):3d} campaigns, "
            f"attack-coming rate {s.y_h[s.forecast_mask, K].mean():.3f}")
    with open(STREAMS_FILE, "wb") as f:
        pickle.dump(dict(streams=streams, pools=pools), f, protocol=5)
    # Dataset / stream statistics for the report.
    stats = dict(
        pools={k: dict(records=len(v), **v["category"].value_counts().to_dict())
               for k, v in pools.items()},
        streams={k: dict(connections=len(s), campaigns=len(s.campaigns),
                         scenarios=s.campaigns.scenario.value_counts().to_dict()
                         if len(s.campaigns) else {},
                         phase_share={C_: float(v) for C_, v in zip(
                             ["background", "recon", "attack", "decoy", "lull"],
                             np.bincount(s.phase, minlength=5) / len(s))},
                         category_share=dict(zip(C.CATEGORIES, (np.bincount(
                             s.cat_id, minlength=5) / len(s)).tolist())),
                         unique_records=int(len(np.unique(s.rows))))
                 for k, s in streams.items()},
        novel_test_labels=sorted(set(pools["test"].label) - set(pools["train"].label)
                                 - set(pools["val"].label)),
    )
    save_json(stats, C.RESULT_DIR / "data_stats.json")
    return streams, pools, enc


def load_data():
    with open(STREAMS_FILE, "rb") as f:
        d = pickle.load(f)
    return d["streams"], d["pools"], FeatureEncoder.load(ENCODER_FILE)


# ============================================================== shared eval
def evaluate_method(name, scores: dict, streams: dict, extra=None,
                    adapted_scores: dict = None):
    """Threshold on val, then zero-shot, site-calibrated and (optionally)
    site-adapted test metrics.  Methods that cannot adapt report their
    calibrated numbers in the 'adapted' slot."""
    sv, st, ss = scores["val"], scores["test"], scores["site"]
    vm = streams["val"].forecast_mask.copy()
    vm[:W.window - 1] = False
    thr_val = best_f1_threshold(sv[vm], streams["val"].y_h[vm, K])
    thr_site = site_threshold(ss, C.ALERT_BUDGET)
    res = dict(method=name,
               val=evaluate_scores(streams["val"], sv, thr_val),
               zero_shot=evaluate_scores(streams["test"], st, thr_val),
               calibrated=evaluate_scores(streams["test"], st, thr_site),
               thr_val=thr_val, thr_site=thr_site)
    if adapted_scores is not None:
        thr_ad = site_threshold(adapted_scores["site"], C.ALERT_BUDGET)
        res["adapted"] = evaluate_scores(streams["test"], adapted_scores["test"], thr_ad)
        res["thr_adapted"] = thr_ad
    else:
        res["adapted"], res["thr_adapted"] = res["calibrated"], thr_site
    res.update(extra or {})
    return res


def save_scores(tag, scores: dict):
    np.savez_compressed(C.CACHE_DIR / "scores" / f"{tag}.npz",
                        **{k: v.astype(np.float32) for k, v in scores.items()})


def load_scores(tag):
    z = np.load(C.CACHE_DIR / "scores" / f"{tag}.npz")
    return {k: z[k] for k in z.files}


# ============================================================== baselines
def stage_baselines(S: Settings):
    streams, pools, enc = load_data()
    # Per-connection detector outputs were computed in the data stage
    # (out-of-fold for the training stream).
    probs = {k: s.evidence for k, s in streams.items()}
    results = []
    # 1. reactive IDS
    sc = {k: reactive_score(p) for k, p in probs.items()}
    save_scores("Reactive IDS (XGBoost)", sc)
    results.append(evaluate_method("Reactive IDS (XGBoost)", sc, streams))
    # 2. probe-count rule
    sc = {k: probe_rule_score(p, W.window) for k, p in probs.items()}
    save_scores("Probe-count rule", sc)
    results.append(evaluate_method("Probe-count rule", sc, streams))
    # 3. window-statistics XGBoost
    log("Training window-statistics XGBoost...")
    t0 = time.time()
    wx = WindowXGB(W.window, K).fit(streams["train"], probs["train"])
    wx.model.save_model(C.MODEL_DIR / "window_xgb.json")
    sc = {k: wx.score(s, probs[k]) for k, s in streams.items()}
    save_scores("Window-stats XGBoost", sc)
    wxa = WindowXGB(W.window, K).fit(streams["train"], probs["train"],
                                     site_stream=streams["adapt"],
                                     site_proba=probs["adapt"])
    wxa.model.save_model(C.MODEL_DIR / "window_xgb_adapted.json")
    sca = {k: wxa.score(streams[k], probs[k]) for k in ("test", "site")}
    save_scores("Window-stats XGBoost_adapted", sca)
    results.append(evaluate_method("Window-stats XGBoost", sc, streams,
                                   dict(train_seconds=time.time() - t0), sca))
    for r in results:
        _print_result(r)
    save_json(results, C.RESULT_DIR / "baselines.json")
    # Detection quality of the per-connection model on the test stream.
    st = streams["test"]
    pred = probs["test"].argmax(1)
    save_json(_detection_metrics(st.cat_id, pred, W.window - 1),
              C.RESULT_DIR / "detection_record_xgb.json")


def _print_result(r):
    c = r["adapted"]
    log(f"  {r['method']:26s} PR-AUC {c['pr_auc']:.3f}  EWR {c['ewr']:.3f}  "
        f"lead {c['lead_median']:5.1f}  FA/10k {c['false_alarms_per_10k']:6.2f}  "
        f"decoy {c['decoy_alarm_rate']:.2f}  alert-prec {c['alert_precision']:.2f}"
        f"  | zero-shot FA/10k {r['zero_shot']['false_alarms_per_10k']:.2f}")


def _detection_metrics(y_true, y_pred, start):
    yt, yp = y_true[start:], y_pred[start:]
    rec = recall_score(yt, yp, average=None, labels=range(5), zero_division=0)
    return dict(macro_f1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                accuracy=float((yt == yp).mean()),
                recall=dict(zip(C.CATEGORIES, rec.tolist())))


# ============================================================== deep
def _deep_extra_metrics(stream: Stream, pred: dict):
    """Attack-type forecast, ETA and detection metrics on the test stream."""
    t0 = W.window - 1
    m = stream.forecast_mask.copy()
    m[:t0] = False
    pos = m & (stream.y_h[:, K] == 1)
    # Which attack is coming? (DoS / R2L / U2R among positive steps)
    yt = stream.y_type[pos]
    yp = pred["type_prob"][pos][:, 1:].argmax(1) + 1
    type_m = dict(accuracy=float((yt == yp).mean()),
                  macro_f1=float(f1_score(yt, yp, average="macro", zero_division=0)),
                  confusion=confusion_matrix(yt, yp, labels=[1, 2, 3]).tolist())
    # When? first horizon whose probability passes 0.5
    hz = np.asarray(W.horizons)
    true_bin = np.searchsorted(hz, stream.dist[pos])          # 0..3
    p = pred["prob"][pos]
    over = p >= 0.5
    pred_bin = np.where(over.any(1), over.argmax(1), len(hz) - 1)
    eta = dict(bin_accuracy=float((pred_bin == true_bin).mean()),
               within_one_bin=float((np.abs(pred_bin - true_bin) <= 1).mean()))
    det = _detection_metrics(stream.cat_id, pred["det_prob"].argmax(1), t0)
    return dict(type_forecast=type_m, eta=eta, detection=det)


def _train_and_eval(display, tag, mcfg, S, streams, enc, seed):
    out_json = C.RESULT_DIR / "runs" / f"{tag}.json"
    if out_json.exists():
        log(f"  {tag}: cached, skipping")
        return load_json(out_json)
    model, hist, info = train_model(streams["train"], streams["val"],
                                    len(enc.numeric), enc.cat_sizes, mcfg,
                                    S.tcfg, W, seed=seed, verbose=True)
    dev = get_device()
    preds = {k: predict(model, WindowSet(streams[k], W.window, dev,
                                         mcfg.use_evidence))
             for k in ("val", "test", "site")}
    scores = {k: p["prob"][:, K] for k, p in preds.items()}
    save_scores(tag, scores)
    # Site adaptation on the unlabelled burn-in (see train.adapt_to_site).
    adapted = adapt_to_site(model, streams["adapt"], streams["train"], mcfg,
                            S.tcfg, W, steps=S.adapt_steps, lr=C.ADAPT_LR,
                            seed=seed)
    preds_ad = {k: predict(adapted, WindowSet(streams[k], W.window, dev,
                                              mcfg.use_evidence))
                for k in ("test", "site")}
    scores_ad = {k: p["prob"][:, K] for k, p in preds_ad.items()}
    save_scores(tag + "_adapted", scores_ad)
    res = evaluate_method(display, scores, streams,
                          dict(seed=seed, tag=tag, info=info, history=hist),
                          scores_ad)
    if mcfg.multitask:
        res.update(_deep_extra_metrics(streams["test"], preds_ad["test"]))
    save_checkpoint(C.MODEL_DIR / "runs" / f"{tag}.pt", model, mcfg, W,
                    info["n_num"], enc.cat_sizes,
                    dict(val=res["thr_val"], site=res["thr_site"]),
                    dict(val_pr_auc=info["best_val_pr_auc"]))
    save_checkpoint(C.MODEL_DIR / "runs" / f"{tag}_adapted.pt", adapted, mcfg, W,
                    info["n_num"], enc.cat_sizes,
                    dict(val=res["thr_val"], site=res["thr_adapted"]),
                    dict(val_pr_auc=info["best_val_pr_auc"], adapted=True))
    del adapted
    save_json(res, out_json)
    _print_result(res)
    del model
    torch.cuda.empty_cache()
    return res


def stage_deep(S: Settings):
    streams, _, enc = load_data()
    # Train the proposed model first so its results are available early.
    order = sorted(C.DEEP_MODELS.items(), key=lambda kv: kv[1] != "foresight")
    for display, bb in order:
        for seed in S.seeds:
            log(f"=== {display} (seed {seed}) ===")
            _train_and_eval(display, f"{bb}_s{seed}", C.ModelConfig(backbone=bb),
                            S, streams, enc, seed)
    _select_deployed(S)


def _select_deployed(S: Settings):
    """Deploy the FORESIGHT-Net seed with the best VALIDATION PR-AUC."""
    runs = [load_json(C.RESULT_DIR / "runs" / f"foresight_s{s}.json") for s in S.seeds]
    best = max(runs, key=lambda r: r["info"]["best_val_pr_auc"])
    shutil.copy(C.MODEL_DIR / "runs" / f"{best['tag']}_adapted.pt", DEPLOY_FILE)
    save_json(dict(tag=best["tag"], seed=best["seed"],
                   val_pr_auc=best["info"]["best_val_pr_auc"],
                   thr_val=best["thr_val"], thr_site=best["thr_site"],
                   thr_adapted=best["thr_adapted"], site_adapted=True),
              C.MODEL_DIR / "deployed.json")
    log(f"Deployed model: {best['tag']} (val PR-AUC {best['info']['best_val_pr_auc']:.4f})")


def stage_ablation(S: Settings):
    streams, _, enc = load_data()
    for display, over in C.ABLATIONS.items():
        for seed in S.seeds:
            log(f"=== FORESIGHT-Net {display} (seed {seed}) ===")
            slug = "abl_" + display.replace("w/o ", "no_").replace(" ", "_").replace("-", "")
            _train_and_eval(f"FORESIGHT-Net {display}", f"{slug}_s{seed}",
                            C.ModelConfig(backbone="foresight", **over),
                            S, streams, enc, seed)


# ============================================================== robustness
def stage_robust(S: Settings):
    """Re-simulate the test network with stealthier (slower, sparser) recon
    and re-evaluate the already-trained models — no retraining."""
    streams, pools, enc = load_data()
    from xgboost import XGBClassifier
    rec = RecordClassifier(enc)
    rec.model = XGBClassifier()
    rec.model.load_model(RECORD_XGB_FILE)
    wx = WindowXGB(W.window, K)
    wx.model = XGBClassifier()
    wx.model.load_model(C.MODEL_DIR / "window_xgb_adapted.json")
    base = {r["method"]: r for r in load_json(C.RESULT_DIR / "baselines.json")}
    dev = get_device()
    ae, cdf = _load_autoencoder(enc, dev)
    rows = []
    for f in S.stealth:
        st = build_stream(pools["test"], enc, S.n_campaigns["test"],
                          C.STREAM_SEEDS["test"], C.SimConfig(stealth_factor=f))
        rp = rec.proba(st)
        st.evidence = rp.astype(np.float32)
        st.novelty = cdf(anomaly_score(ae, enc.onehot(st.num, st.cat))).astype(np.float32)
        cands = {"Probe-count rule": probe_rule_score(rp, W.window),
                 "Window-stats XGBoost": wx.score(st, rp)}
        for name, sc in cands.items():
            r = evaluate_scores(st, sc, base[name]["thr_adapted"])
            rows.append(dict(method=name, seed=0, stealth=f, **_slim(r)))
        for display in ("LSTM", "Transformer", "FORESIGHT-Net"):
            bb = C.DEEP_MODELS[display]
            for seed in S.seeds:
                model, ck, _ = load_checkpoint(
                    C.MODEL_DIR / "runs" / f"{bb}_s{seed}_adapted.pt", dev)
                ws = WindowSet(st, W.window, dev, ck["model_cfg"]["use_evidence"])
                p = predict(model, ws)["prob"][:, K]
                r = evaluate_scores(st, p, ck["threshold"]["site"])
                rows.append(dict(method=display, seed=seed, stealth=f, **_slim(r)))
                del model
        log(f"  stealth x{f}: done")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULT_DIR / "robustness_raw.csv", index=False)
    agg = df.groupby(["method", "stealth"]).agg(["mean", "std"]).drop(columns="seed")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(C.RESULT_DIR / "robustness.csv", index=False)


def _load_autoencoder(enc, dev):
    from .zero_day import AutoEncoder
    ae = AutoEncoder(len(enc.onehot_names)).to(dev)
    ae.load_state_dict(torch.load(C.MODEL_DIR / "autoencoder.pt", map_location=dev))
    ae.eval()
    return ae, EmpiricalCDF(np.load(C.MODEL_DIR / "autoencoder_val_normal_scores.npy"))


def _slim(r):
    keys = ("roc_auc", "pr_auc", "f1", "ewr", "ewr_10", "lead_median",
            "false_alarms_per_10k", "decoy_alarm_rate", "alert_precision")
    return {k: r[k] for k in keys}


# ============================================================== zero-day
def stage_zeroday(S: Settings):
    _, pools, enc = load_data()
    from xgboost import XGBClassifier
    rec = RecordClassifier(enc)
    rec.model = XGBClassifier()
    rec.model.load_model(RECORD_XGB_FILE)

    def X(df):
        return enc.onehot(*enc.transform(df))
    tr_n = pools["train"][pools["train"].category == "Normal"]
    va_n = pools["val"][pools["val"].category == "Normal"]
    ae, _ = _load_autoencoder(enc, get_device())     # trained in stage data
    test = pools["test"]
    Xte = X(test)
    known = set(pools["train"].label) | set(pools["val"].label)
    y_att = (test.category != "Normal").values.astype(int)
    novel = ((~test.label.isin(known)) & (y_att == 1)).values
    # Raw scores of each detector.
    sup_te = 1 - rec.proba_records(*enc.transform(test))[:, 0]
    sup_va = 1 - rec.proba_records(*enc.transform(va_n))[:, 0]
    ae_te = anomaly_score(ae, Xte)
    ae_va = anomaly_score(ae, X(va_n))
    F_sup, F_ae = EmpiricalCDF(sup_va), EmpiricalCDF(ae_va)
    hyb_te = np.maximum(F_sup(sup_te), F_ae(ae_te))
    hyb_va = np.maximum(F_sup(sup_va), F_ae(ae_va))
    scores = {"Supervised XGBoost": sup_te, "Autoencoder (zero-day guard)": ae_te,
              "Hybrid (XGBoost + AE)": hyb_te}
    # Thresholds = 95th percentile of VALIDATION normal traffic (5 % FPR).
    thr = {"Supervised XGBoost": np.quantile(sup_va, 0.95),
           "Autoencoder (zero-day guard)": np.quantile(ae_va, 0.95),
           "Hybrid (XGBoost + AE)": np.quantile(hyb_va, 0.95)}
    rows = zero_day_report(y_att, novel, test.category.values, scores, thr)
    pd.DataFrame(rows).to_csv(C.RESULT_DIR / "zero_day.csv", index=False)
    # ROC curves (normal vs NOVEL attacks) for the figure.
    curves = {}
    normal = y_att == 0
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(np.r_[np.zeros(normal.sum()), np.ones(novel.sum())],
                                np.r_[s[normal], s[novel]])
        keep = np.unique(np.linspace(0, len(fpr) - 1, 400).astype(int))
        curves[name] = dict(fpr=fpr[keep].tolist(), tpr=tpr[keep].tolist())
    save_json(dict(curves=curves, n_novel=int(novel.sum()),
                   n_normal=int(normal.sum()),
                   novel_types=sorted(test.label[novel].unique())),
              C.RESULT_DIR / "zero_day_curves.json")
    for r in rows:
        log(f"  {r['method']:30s} AUC(all) {r['auc_all']:.3f}  AUC(novel) "
            f"{r['auc_novel']:.3f}  DR(novel) {r['dr_novel']:.3f}  "
            f"test FPR {r['test_fpr']:.3f}")


# ============================================================== explain
def stage_explain(S: Settings):
    streams, _, enc = load_data()
    dev = get_device()
    model, ck, _ = load_checkpoint(DEPLOY_FILE, dev)
    thr = ck["threshold"]["site"]          # adapted model's site threshold
    st = streams["test"]
    ws = WindowSet(st, W.window, dev, ck["model_cfg"]["use_evidence"])
    FEATURES = feature_names(ck["n_num"])
    pred = predict(model, ws, with_attn=True)
    p = pred["prob"][:, K]
    # 1. Does attention focus on the scan traffic?  (alarmed recon windows)
    valid = np.arange(len(st)) >= W.window - 1
    steps = np.flatnonzero((st.phase == 1) & (p > thr) & valid)
    if len(steps) < 20:     # tiny smoke-test runs: fall back to all recon steps
        steps = np.flatnonzero((st.phase == 1) & valid)
    offs = np.arange(-W.window + 1, 1)
    cats = st.cat_id[steps[:, None] + offs[None, :]]
    mass, share, ratio = attention_focus(pred["attn"][steps].astype(np.float32), cats)
    # 2. Which features drive warnings?
    rng = np.random.default_rng(0)
    sample = rng.choice(steps, min(len(steps), 300 if S.quick else 2000), replace=False)
    imp = global_feature_importance(model, ws, np.sort(sample), K)
    order = np.argsort(-imp)
    # 3. Representative case studies: campaign whose lead is closest to the
    #    median of its scenario (NOT the best one), plus one random decoy.
    ev = event_metrics(st, p, thr)
    pc = pd.DataFrame(ev["per_campaign"])
    cases = []
    for scen in ("recon_dos", "recon_r2l_u2r", "recon_r2l"):
        sub = pc[(pc.scenario == scen) & pc.warned]
        if len(sub):
            med = sub.lead.median()
            cases.append(int(sub.iloc[(sub.lead - med).abs().argsort().iloc[0]].id))
    decoys = st.campaigns[st.campaigns.scenario == "decoy_scan"]
    if len(decoys):
        cases.append(int(decoys.iloc[rng.integers(len(decoys))].id))
    np.savez_compressed(C.CACHE_DIR / "deployed_test_pred.npz", **{
        k: v for k, v in pred.items()})
    save_json(dict(attention_mass_on_scans=mass, scan_share=share,
                   attention_ratio=ratio, n_windows=int(len(steps)),
                   feature_importance={FEATURES[i]: float(imp[i]) for i in order},
                   case_campaigns=cases, threshold=thr,
                   per_campaign=ev["per_campaign"]),
              C.RESULT_DIR / "explain.json")
    log(f"  attention on scans {mass:.3f} vs share {share:.3f} (x{ratio:.1f}); "
        f"top features: {[FEATURES[i] for i in order[:5]]}")
    # Operating curves (threshold sweep) for every method, seed 0.
    curves = {}
    qs = np.r_[np.linspace(0.50, 0.99, 40), np.linspace(0.991, 0.9995, 15)]
    tags = {"Reactive IDS (XGBoost)": "Reactive IDS (XGBoost)",
            "Probe-count rule": "Probe-count rule",
            "Window-stats XGBoost": "Window-stats XGBoost_adapted"}
    tags.update({d: f"{bb}_s0_adapted" for d, bb in C.DEEP_MODELS.items()})
    for name, tag in tags.items():
        sc = load_scores(tag)
        thr_grid = np.quantile(sc["site"][W.window - 1:], qs)
        curves[name] = operating_curve(st, sc["test"], thr_grid)
    save_json(curves, C.RESULT_DIR / "operating_curves.json")
    # Alert-budget sensitivity (the 1 % budget was fixed before testing; the
    # other budgets show how the conclusions change with the operating point).
    rows = []
    for name, tag in tags.items():
        sc = load_scores(tag)
        for b in (0.0025, 0.005, 0.01, 0.02, 0.05):
            ev = event_metrics(st, sc["test"], np.quantile(sc["site"][W.window - 1:], 1 - b))
            rows.append(dict(method=name, budget_pct=b * 100, ewr=ev["ewr"],
                             lead_median=ev["lead_median"],
                             false_alarms_per_10k=ev["false_alarms_per_10k"],
                             alert_precision=ev["alert_precision"],
                             decoy_alarm_rate=ev["decoy_alarm_rate"]))
    pd.DataFrame(rows).to_csv(C.RESULT_DIR / "budget_sensitivity.csv", index=False)


# ============================================================== latency
def stage_latency(S: Settings):
    streams, _, enc = load_data()
    rows = {}
    st = streams["test"]
    for devname in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        dev = torch.device(devname)
        model, ck, _ = load_checkpoint(DEPLOY_FILE, dev)
        ws = WindowSet(st, W.window, dev, ck["model_cfg"]["use_evidence"])
        t = ws.steps[:200]
        with torch.no_grad():
            for i in range(10):                       # warm-up
                model(*ws.batch(t[i:i + 1])[:2])
            if devname == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(len(t)):
                model(*ws.batch(t[i:i + 1])[:2])
            if devname == "cuda":
                torch.cuda.synchronize()
            single = (time.perf_counter() - t0) / len(t) * 1000
            big = ws.steps[:8192]
            t0 = time.perf_counter()
            for i in range(0, len(big), 2048):
                model(*ws.batch(big[i:i + 2048])[:2])
            if devname == "cuda":
                torch.cuda.synchronize()
            thr = len(big) / (time.perf_counter() - t0)
        rows[devname] = dict(ms_per_window_batch1=single,
                             windows_per_second_batched=thr)
        log(f"  {devname}: {single:.2f} ms per window (batch 1), "
            f"{thr:,.0f} windows/s batched")
    rows["params"] = sum(p.numel() for p in model.parameters())
    save_json(rows, C.RESULT_DIR / "latency.json")


# ============================================================== summary
def _ms(values, pct=False):
    v = np.asarray(values, float)
    f = 100 if pct else 1
    if len(v) == 1:
        return f"{v[0] * f:.1f}" if pct else f"{v[0]:.3f}"
    return (f"{v.mean() * f:.1f} ± {v.std() * f:.1f}" if pct
            else f"{v.mean():.3f} ± {v.std():.3f}")


def _table(results_by_method: dict, setting: str):
    cols = [("ROC-AUC", "roc_auc", False), ("PR-AUC", "pr_auc", False),
            ("F1", "f1", False), ("Early-warning rate %", "ewr", True),
            ("Warned ≥10 conns early %", "ewr_10", True),
            ("Median lead (conns)", "lead_median", None),
            ("False alarms /10k", "false_alarms_per_10k", None),
            ("Alert precision %", "alert_precision", True),
            ("Decoys alarmed %", "decoy_alarm_rate", True),
            ("No-precursor attacks warned %", "ewr_unpredictable", True)]
    rows = []
    for name, runs in results_by_method.items():
        row = {"Method": name}
        for label, key, pct in cols:
            vals = [r[setting][key] for r in runs]
            if pct is None:
                v = np.asarray(vals, float)
                row[label] = (f"{v.mean():.1f}" if len(v) == 1
                              else f"{v.mean():.1f} ± {v.std():.1f}")
            else:
                row[label] = _ms(vals, pct)
        rows.append(row)
    return pd.DataFrame(rows)


def _collect(prefix_map, seeds):
    out = {}
    for name, tag in prefix_map.items():
        runs = []
        for s in seeds:
            f = C.RESULT_DIR / "runs" / f"{tag}_s{s}.json"
            if f.exists():
                runs.append(load_json(f))
        if runs:
            out[name] = runs
    return out


def stage_summary(S: Settings):
    base = {r["method"]: [r] for r in load_json(C.RESULT_DIR / "baselines.json")}
    deep = _collect({d: bb for d, bb in C.DEEP_MODELS.items()}, S.seeds)
    allm = {**base, **deep}
    md = ["# FORESIGHT — Results", "",
          "All numbers on the TEST stream (KDDTest+ records, 17 unseen attack "
          "types). Deep models: mean ± std over seeds "
          f"{list(S.seeds)}. Thresholds tuned on validation only.", ""]
    for setting, title in (
            ("adapted", "Site-adapted + calibrated (main result: deployed configuration)"),
            ("calibrated", "Site-calibrated threshold only (no adaptation)"),
            ("zero_shot", "Zero-shot threshold (no calibration, no adaptation)")):
        df = _table(allm, setting)
        df.to_csv(C.RESULT_DIR / f"table_main_{setting}.csv", index=False)
        md += [f"## {title}", "", df.to_markdown(index=False), ""]
    # Ablation
    abl = {"FORESIGHT-Net (full)": deep.get("FORESIGHT-Net", [])}
    for display in C.ABLATIONS:
        slug = "abl_" + display.replace("w/o ", "no_").replace(" ", "_").replace("-", "")
        runs = _collect({display: slug}, S.seeds).get(display, [])
        if runs:
            abl[f"FORESIGHT-Net {display}"] = runs
    if len(abl) > 1:
        df = _table(abl, "adapted")
        df.to_csv(C.RESULT_DIR / "table_ablation.csv", index=False)
        md += ["## Ablation (site-adapted)", "", df.to_markdown(index=False), ""]
    # Type forecast / ETA / detection for deep models
    rows = []
    for name, runs in deep.items():
        rows.append({"Model": name,
                     "Attack-type accuracy": _ms([r["type_forecast"]["accuracy"] for r in runs]),
                     "Attack-type macro-F1": _ms([r["type_forecast"]["macro_f1"] for r in runs]),
                     "ETA exact bin": _ms([r["eta"]["bin_accuracy"] for r in runs]),
                     "ETA within ±1 bin": _ms([r["eta"]["within_one_bin"] for r in runs]),
                     "Detection macro-F1": _ms([r["detection"]["macro_f1"] for r in runs])})
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULT_DIR / "table_forecast_detail.csv", index=False)
    md += ["## What and when: attack-type forecast, ETA and detection", "",
           df.to_markdown(index=False), ""]
    # Detection: per-connection XGBoost vs FORESIGHT detection head
    det_x = load_json(C.RESULT_DIR / "detection_record_xgb.json")
    fr = deep.get("FORESIGHT-Net", [])
    det_rows = [dict(Model="Per-connection XGBoost (reactive)",
                     **{c: f"{det_x['recall'][c]:.3f}" for c in C.CATEGORIES},
                     **{"Macro-F1": f"{det_x['macro_f1']:.3f}"})]
    if fr:
        det_rows.append(dict(Model="FORESIGHT-Net detection head (context-aware)",
                             **{c: _ms([r["detection"]["recall"][c] for r in fr])
                                for c in C.CATEGORIES},
                             **{"Macro-F1": _ms([r["detection"]["macro_f1"] for r in fr])}))
    df = pd.DataFrame(det_rows)
    df.to_csv(C.RESULT_DIR / "table_detection.csv", index=False)
    md += ["## Per-class detection recall on the test stream", "",
           df.to_markdown(index=False), ""]
    # Zero-day, robustness, latency, explainability
    for fname, title in (("zero_day.csv", "Zero-day guard (KDDTest+ records)"),
                         ("robustness.csv", "Robustness to stealthier attackers"),
                         ("budget_sensitivity.csv", "Alert-budget sensitivity (seed 0)")):
        f = C.RESULT_DIR / fname
        if f.exists():
            df = pd.read_csv(f)
            md += [f"## {title}", "", df.round(3).to_markdown(index=False), ""]
    (C.RESULT_DIR / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    # Headline numbers (used by README and slides)
    head = {}
    if fr:
        cal = [r["adapted"] for r in fr]
        head["foresight"] = {k: float(np.mean([c[k] for c in cal])) for k in cal[0]}
        head["foresight_std"] = {k: float(np.std([c[k] for c in cal])) for k in cal[0]}
    for name, runs in base.items():
        head[name] = runs[0]["adapted"]
    for name, runs in deep.items():
        cal = [r["adapted"] for r in runs]
        head[name] = {k: float(np.mean([c[k] for c in cal])) for k in cal[0]}
    save_json(head, C.RESULT_DIR / "headline.json")
    log("Wrote " + str(C.RESULT_DIR / "RESULTS.md"))


STAGES = dict(data=stage_data, baselines=stage_baselines, deep=stage_deep,
              ablation=stage_ablation, robust=stage_robust,
              zeroday=stage_zeroday, explain=stage_explain,
              latency=stage_latency, summary=stage_summary)


def run(stages, quick=False):
    ensure_dirs()
    S = Settings(quick)
    for name in stages:
        log(f"########## STAGE: {name} ##########")
        t0 = time.time()
        STAGES[name](S)
        log(f"########## {name} finished in {time.time() - t0:.0f}s ##########")
