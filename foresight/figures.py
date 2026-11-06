"""
figures.py — every figure used in the README, report and presentation.

    python run_pipeline.py --stage figures

Reads only the cached outputs of the experiment stages; writes PNGs to
outputs/figures/.  Style: one highlight colour (blue) for the proposed
FORESIGHT-Net, muted greys for comparison methods unless a chart needs
several identities, recessive grid, direct labels instead of legends where
possible.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .metrics import event_metrics  # noqa: E402
from .utils import load_json, log  # noqa: E402

# ------------------------------------------------------------------ style
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
GREY, LIGHT, INK, INK2 = "#9aa0a6", "#d9dce0", "#0b0b0b", "#52514e"
CAT_COL = {0: "#b8bec6", 1: RED, 2: ORANGE, 3: VIOLET, 4: INK}   # per category id
METHOD_COL = {"FORESIGHT-Net": BLUE, "Probe-count rule": ORANGE,
              "Window-stats XGBoost": AQUA, "LSTM": YELLOW,
              "Transformer": MAGENTA, "Reactive IDS (XGBoost)": VIOLET,
              "GRU": GREEN, "1D-CNN": GREY, "MLP": "#c3c2b7"}
PHASE_FILL = {1: (YELLOW, 0.16, "reconnaissance"), 2: (RED, 0.16, "attack"),
              3: (VIOLET, 0.12, "decoy scan")}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": "#c9ccd1", "axes.labelcolor": INK2, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#eceef1",
    "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "legend.frameon": False, "lines.linewidth": 2.0,
})
W = C.WindowConfig()
K = W.alarm_horizon_idx


def _save(fig, name):
    fig.savefig(C.FIG_DIR / name, facecolor="white")
    plt.close(fig)
    log(f"  figure: {name}")


def _load():
    import pickle
    with open(C.CACHE_DIR / "streams.pkl", "rb") as f:
        d = pickle.load(f)
    return d["streams"], d["pools"]


def _scores(tag):
    z = np.load(C.CACHE_DIR / "scores" / f"{tag}.npz")
    return {k: z[k] for k in z.files}


def _runs(prefix, seeds):
    out = []
    for s in seeds:
        f = C.RESULT_DIR / "runs" / f"{prefix}_s{s}.json"
        if f.exists():
            out.append(load_json(f))
    return out


def _shade_phases(ax, phase, lo, hi, labels=True):
    """Shade contiguous true phases inside [lo, hi)."""
    seen = set()
    ph = phase[lo:hi]
    for p, (col, a, name) in PHASE_FILL.items():
        idx = np.flatnonzero(ph == p)
        if not len(idx):
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        for s, e in zip(starts, ends):
            ax.axvspan(lo + s, lo + e + 1, color=col, alpha=a, lw=0,
                       label=name if (labels and name not in seen) else None)
            seen.add(name)


def _raster(ax, st, lo, hi):
    """One tick per connection, coloured by its TRUE category."""
    x = np.arange(lo, hi)
    c = st.cat_id[lo:hi]
    for cid in (0, 2, 1, 3, 4):
        m = c == cid
        if m.any():
            ax.vlines(x[m], 0, 1, color=CAT_COL[cid], lw=0.9 if cid else 0.5)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ figures
def fig_dataset(pools):
    order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    tr = pd.concat([pools["train"], pools["val"]])["category"].value_counts().reindex(order)
    te = pd.concat([pools["test"], pools["site"]])["category"].value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(order))
    ax.bar(x - 0.2, tr.values, 0.38, color=GREY, label="KDDTrain+ (training network)")
    ax.bar(x + 0.2, te.values, 0.38, color=BLUE, label="KDDTest+ (deployment network)")
    for i in range(len(order)):
        ax.text(i - 0.2, tr.values[i] * 1.12, f"{tr.values[i]:,}", ha="center", fontsize=8.5, color=INK2)
        ax.text(i + 0.2, te.values[i] * 1.12, f"{te.values[i]:,}", ha="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(x, order)
    ax.set_ylabel("connection records (log scale)")
    known = set(pools["train"].label) | set(pools["val"].label)
    novel = sorted(set(pools["test"].label) - known)
    ax.set_title(f"NSL-KDD: official split — {len(novel)} attack types in the "
                 "test network never appear in training")
    ax.legend(loc="upper right")
    ax.set_ylim(10, 4e5)
    _save(fig, "fig01_dataset.png")


def fig_hero(st, fs, react, thr_f, thr_r, cid):
    """Reactive IDS vs FORESIGHT on one representative campaign."""
    c = st.campaigns.set_index("id").loc[cid]
    lo = max(0, int(c.recon_start) - 150)
    hi = min(len(st), int(c.end) + 120)
    x = np.arange(lo, hi)
    fig, axs = plt.subplots(3, 1, figsize=(10.5, 6.2), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.5, 1.2, 1.6]))
    _raster(axs[0], st, lo, hi)
    axs[0].set_title("What arrives at our gateway (each tick = one real KDDTest+ "
                     "connection: grey normal, orange scan, red attack)", loc="left",
                     fontsize=10.5)
    for ax, s, thr, col, name in ((axs[1], react, thr_r, VIOLET, "Reactive IDS (per-connection XGBoost)"),
                                  (axs[2], fs, thr_f, BLUE, "FORESIGHT-Net: P(attack within next 100 connections)")):
        _shade_phases(ax, st.phase, lo, hi, labels=(ax is axs[2]))
        ax.plot(x, s[lo:hi], color=col, lw=1.8)
        ax.axhline(thr, color=INK2, ls="--", lw=1)
        ax.text(lo + 2, thr + 0.03, "alarm threshold", ha="left", fontsize=8.5, color=INK2)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(name, loc="left", fontsize=10.5)
        al = np.flatnonzero(s[lo:hi] > thr) + lo
        first = al[al >= c.recon_start]
        if len(first):
            f = int(first[0])
            ax.axvline(f, color=col, lw=1.2, alpha=0.8)
            lead = int(c.onset) - f
            txt = (f"first alarm {lead} connections BEFORE the attack" if lead > 0
                   else ("alarm only once the attack has started" if lead > -5
                         else f"alarm {abs(lead)} connections after the attack started"))
            ax.annotate(txt, xy=(f, 0.82), xytext=(f - 8, 0.82), ha="right",
                        fontsize=9.5, color=INK, fontweight="bold")
    axs[2].axvline(c.onset, color=RED, lw=1.5)
    axs[2].legend(loc="center left", fontsize=8.5)
    axs[2].set_xlabel("connection number in the test stream")
    nice = {"recon_dos": "reconnaissance → DoS", "recon_r2l": "reconnaissance → break-in (R2L)",
            "recon_r2l_u2r": "reconnaissance → R2L → root (U2R)"}
    fig.suptitle(f"Reactive detection vs. FORESIGHT forecasting — "
                 f"{nice.get(c.scenario, c.scenario)} campaign",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig02_reactive_vs_foresight.png")


def fig_cases(st, fs, thr, cases):
    titles = {"recon_dos": "Reconnaissance → DoS flood",
              "recon_r2l": "Reconnaissance → remote break-in (R2L)",
              "recon_r2l_u2r": "Full kill chain: recon → R2L → root (U2R)",
              "decoy_scan": "Harmless decoy scan (no attack follows)"}
    camps = st.campaigns.set_index("id")
    fig, axs = plt.subplots(len(cases), 1, figsize=(10.5, 2.2 * len(cases)))
    axs = np.atleast_1d(axs)
    for ax, cid in zip(axs, cases):
        c = camps.loc[cid]
        lo = max(0, int(c.recon_start) - 120)
        hi = min(len(st), int(c.end) + 150)
        x = np.arange(lo, hi)
        _shade_phases(ax, st.phase, lo, hi)
        ax.plot(x, fs[lo:hi], color=BLUE, lw=1.7)
        ax.axhline(thr, color=INK2, ls="--", lw=1)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("P(attack ≤100)", fontsize=9)
        msg = titles.get(c.scenario, c.scenario)
        al = np.flatnonzero(fs[lo:hi] > thr) + lo
        if c.onset >= 0:
            ax.axvline(c.onset, color=RED, lw=1.4)
            pre = al[(al >= c.recon_start) & (al < c.onset)]
            if len(pre):
                ax.axvline(pre[0], color=BLUE, lw=1, alpha=0.7)
                msg += f"   —   warned {int(c.onset - pre[0])} connections early"
        else:
            msg += ("   —   no alarm raised" if not len(al[(al >= c.recon_start) &
                                                           (al <= c.end + 30)])
                    else "   —   alarm raised (false alarm)")
        ax.set_title(msg, loc="left", fontsize=10.5)
        ax.legend(loc="upper left", fontsize=8)
    axs[-1].set_xlabel("connection number in the test stream")
    fig.tight_layout()
    _save(fig, "fig03_case_studies.png")


def fig_comparison(table_rows):
    """Three metrics side by side, FORESIGHT highlighted."""
    df = pd.DataFrame(table_rows)
    metrics = [("pr_auc", "PR-AUC (step level)", False),
               ("alert_precision", "Alert precision (%)", True),
               ("ewr", "Campaigns warned before impact (%)", True)]
    df = df.sort_values("pr_auc")
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    y = np.arange(len(df))
    for ax, (k, title, pct) in zip(axs, metrics):
        v = df[k].values * (100 if pct else 1)
        e = df[k + "_std"].values * (100 if pct else 1)
        cols = [BLUE if m == "FORESIGHT-Net" else GREY for m in df.method]
        ax.barh(y, v, xerr=e, color=cols, height=0.62, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        for i, (vi, ei) in enumerate(zip(v, e)):
            ax.text(vi + ei + (1.5 if pct else 0.015), i, f"{vi:.1f}" if pct else f"{vi:.3f}",
                    va="center", fontsize=8.5, color=INK, fontweight="bold" if df.method.iloc[i] == "FORESIGHT-Net" else None)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 115 if pct else 1.0)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, df.method)
    fig.suptitle("Early-warning performance on the unseen test network "
                 "(site-adapted, 1 % alert budget; mean ± std over 3 seeds)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    _save(fig, "fig04_model_comparison.png")


def fig_operating(curves, points):
    """Two views of the threshold trade-off (Pareto frontiers)."""
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    show = ["FORESIGHT-Net", "Window-stats XGBoost", "Probe-count rule", "LSTM",
            "Transformer", "Reactive IDS (XGBoost)"]
    for name in show:
        if name not in curves:
            continue
        cv = pd.DataFrame(curves[name]).sort_values("fa_per_10k")
        fa = np.maximum(cv.fa_per_10k.values, 0.5)
        kw = dict(where="post", color=METHOD_COL[name], label=name,
                  lw=2.6 if name == "FORESIGHT-Net" else 1.5,
                  zorder=3 if name == "FORESIGHT-Net" else 2)
        # best value achievable while staying at or below x false alarms
        axs[0].step(fa, np.maximum.accumulate(cv.ewr.values) * 100, **kw)
        axs[1].step(fa, np.maximum.accumulate(cv.lead_median.values), **kw)
        if name in points:
            f, e, l = points[name]
            for ax, v in ((axs[0], e * 100), (axs[1], l)):
                ax.plot(max(f, 0.5), v, "o", ms=8, color=METHOD_COL[name],
                        mec="white", mew=2, zorder=4)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlim(0.5, 200)
        ax.set_xlabel("false alarms per 10,000 connections (lower is better)")
    axs[0].set_ylim(0, 103)
    axs[0].set_ylabel("campaigns warned in advance (%)")
    axs[0].set_title("Warning rate vs. false alarms", loc="left", fontsize=11)
    axs[1].set_ylabel("median lead time (connections)")
    axs[1].set_title("Head start vs. false alarms", loc="left", fontsize=11)
    axs[1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Operating curves on the test network (threshold sweep; best value at "
                 "each false-alarm level; dots = deployed 1 % budget)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    _save(fig, "fig05_operating_curve.png")


def fig_lead(leads: dict):
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    names = list(leads)
    data = [leads[n] for n in names]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.6), showfliers=False)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(METHOD_COL.get(n, GREY))
        patch.set_alpha(0.85 if n == "FORESIGHT-Net" else 0.5)
        patch.set_edgecolor("white")
    for i, d in enumerate(data, 1):
        ax.text(max(d) + 4 if len(d) else 4, i, f"median {np.median(d):.0f}  (n={len(d)})",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(1, len(names) + 1), names)
    ax.set_xlabel("lead time: connections between first alarm and the first attack connection")
    ax.set_title("How much warning does the defender get? (warned campaigns, seed 0)", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, "fig06_lead_time.png")


def fig_ablation(rows):
    df = pd.DataFrame(rows)
    metrics = [("pr_auc", "PR-AUC", False), ("alert_precision", "Alert precision (%)", True),
               ("false_alarms_per_10k", "False alarms / 10k", False)]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    y = np.arange(len(df))[::-1]
    for ax, (k, title, pct) in zip(axs, metrics):
        v = df[k].values * (100 if pct else 1)
        e = df[k + "_std"].values * (100 if pct else 1)
        cols = [BLUE if i == 0 else GREY for i in range(len(df))]
        ax.barh(y, v, xerr=e, color=cols, height=0.6, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        for yi, vi, ei in zip(y, v, e):
            ax.text(vi + ei + max(v) * 0.02, yi, f"{vi:.1f}" if (pct or k.startswith("false")) else f"{vi:.3f}",
                    va="center", fontsize=8.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, max(v + e) * 1.25)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, df.variant)
    fig.suptitle("Ablation: what each component of FORESIGHT-Net contributes "
                 "(test network, mean ± std over 3 seeds)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    _save(fig, "fig07_ablation.png")


def fig_robust(df):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for ax, k, title in ((axs[0], "ewr", "Campaigns warned in advance (%)"),
                         (axs[1], "pr_auc", "PR-AUC")):
        for name in ["FORESIGHT-Net", "Window-stats XGBoost", "Probe-count rule",
                     "LSTM", "Transformer"]:
            d = df[df.method == name].sort_values("stealth", ascending=False)
            if not len(d):
                continue
            f = 100 if k == "ewr" else 1
            m, s = d[f"{k}_mean"].values * f, d[f"{k}_std"].fillna(0).values * f
            ax.plot(d.stealth, m, "-o", ms=4, color=METHOD_COL[name], label=name,
                    lw=2.6 if name == "FORESIGHT-Net" else 1.6)
            ax.fill_between(d.stealth, m - s, m + s, color=METHOD_COL[name], alpha=0.12, lw=0)
        ax.invert_xaxis()
        ax.set_xlabel("attacker reconnaissance intensity (1 = normal → 0.25 = very stealthy)")
        ax.set_title(title, loc="left", fontsize=11)
    axs[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Robustness: stealthier ('low-and-slow') attackers, no retraining",
                 x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    _save(fig, "fig08_robustness.png")


def fig_zero_day(curves, table):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    cols = {"Supervised XGBoost": GREY, "Autoencoder (zero-day guard)": ORANGE,
            "Hybrid (XGBoost + AE)": BLUE}
    for name, cv in curves["curves"].items():
        auc = table.set_index("method").loc[name, "auc_novel"]
        axs[0].plot(cv["fpr"], cv["tpr"], color=cols[name], lw=2.2 if "Hybrid" in name else 1.6,
                    label=f"{name}  (AUC {auc:.3f})")
    axs[0].plot([0, 1], [0, 1], color=LIGHT, lw=1, ls=":")
    axs[0].set_xlabel("false-positive rate on test normal traffic")
    axs[0].set_ylabel("detection rate of NOVEL attacks")
    axs[0].set_title(f"ROC: {curves['n_novel']:,} connections of attack\n"
                     "types never seen in training", loc="left", fontsize=10.5)
    axs[0].legend(fontsize=8.5, loc="lower right")
    cats = ["dr_DoS", "dr_Probe", "dr_R2L", "dr_U2R", "dr_novel"]
    labels = ["DoS", "Probe", "R2L", "U2R", "Novel types"]
    x = np.arange(len(cats))
    for i, (name, col) in enumerate(cols.items()):
        r = table.set_index("method").loc[name]
        axs[1].bar(x + (i - 1) * 0.26, [r[c] * 100 for c in cats], 0.25, color=col, label=name)
    axs[1].set_xticks(x, labels)
    axs[1].set_ylabel("detection rate (%)")
    axs[1].set_title("Detection rate by category at 5 % false-positive\n"
                     "rate (threshold set on validation normal traffic)",
                     loc="left", fontsize=10.5)
    axs[1].legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    axs[1].set_ylim(0, 105)
    axs[1].grid(axis="x", visible=False)
    fig.suptitle("Zero-day guard: an autoencoder that only knows 'normal'",
                 x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    _save(fig, "fig09_zero_day.png")


def fig_attention(st, pred, thr, explain):
    p = pred["prob"][:, K]
    steps = np.flatnonzero((st.phase == 1) & (p > thr) & (np.arange(len(st)) >= W.window - 1))
    if not len(steps):
        return
    # a representative alarmed window: median probe share among alarmed recon windows
    offs = np.arange(-W.window + 1, 1)
    share = (st.cat_id[steps[:, None] + offs] == 2).mean(1)
    t = int(steps[np.argsort(share)[len(share) // 2]])
    att = pred["attn"][t].astype(float)
    cats = st.cat_id[t + offs]
    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    ax.bar(offs, att, color=[CAT_COL[c] for c in cats], width=0.85)
    ax.set_xlabel("connection position in the look-back window (0 = now)")
    ax.set_ylabel("attention weight")
    ax.set_title(f"Where FORESIGHT looks when it raises a warning (one representative window)\n"
                 f"Across {explain['n_windows']:,} alarmed windows, scan connections (orange) "
                 f"receive {explain['attention_ratio']:.1f}× their share of attention",
                 loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig10_attention.png")


def fig_features(explain):
    imp = pd.Series(explain["feature_importance"]).sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    cols = [BLUE if ("detector" in n or "novelty" in n) else GREY for n in imp.index]
    ax.barh(imp.index, imp.values * 100, color=cols, height=0.65)
    for i, v in enumerate(imp.values * 100):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8.5)
    ax.set_xlabel("share of the warning explained (gradient × input, %)")
    ax.set_title("What drives a warning? Top 15 inputs\n(blue = evidence channels: detector "
                 "probabilities & autoencoder novelty)", loc="left", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    _save(fig, "fig11_feature_importance.png")


def fig_training(histories):
    fig, ax = plt.subplots(figsize=(8.2, 4))
    for name, h in histories.items():
        d = pd.DataFrame(h)
        ax.plot(d.epoch, d.val_pr_auc, "-o", ms=3, color=METHOD_COL.get(name, GREY),
                lw=2.4 if name == "FORESIGHT-Net" else 1.4, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation PR-AUC")
    ax.set_title("Training curves (seed 0; early stopping on validation PR-AUC)", loc="left")
    ax.legend(fontsize=8.5, ncol=2)
    _save(fig, "fig12_training_curves.png")


def fig_detection(det_x, det_f):
    cats = C.CATEGORIES
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(len(cats))
    ax.bar(x - 0.2, [det_x["recall"][c] * 100 for c in cats], 0.38, color=GREY,
           label=f"Per-connection XGBoost (macro-F1 {det_x['macro_f1']:.3f})")
    ax.bar(x + 0.2, [det_f["recall"][c] * 100 for c in cats], 0.38, color=BLUE,
           label=f"FORESIGHT detection head, uses context (macro-F1 {det_f['macro_f1']:.3f})")
    for i, c in enumerate(cats):
        ax.text(i - 0.2, det_x["recall"][c] * 100 + 1.5, f"{det_x['recall'][c] * 100:.0f}", ha="center", fontsize=8.5)
        ax.text(i + 0.2, det_f["recall"][c] * 100 + 1.5, f"{det_f['recall'][c] * 100:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, cats)
    ax.set_ylabel("recall on the test stream (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Detecting the current connection: does temporal context help?", loc="left")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig13_detection_recall.png")


def fig_type_confusion(conf):
    cm = np.asarray(conf, float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    labels = ["DoS", "R2L", "U2R"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else INK, fontsize=11)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("forecast type")
    ax.set_ylabel("attack that actually arrived")
    ax.set_title("Which attack is coming?\n(row-normalised, before impact)", fontsize=11)
    ax.grid(False)
    _save(fig, "fig14_type_confusion.png")


def fig_overview(st, fs, thr, lo=None, n=7000):
    if lo is None:
        lo = int(st.campaigns.start.iloc[min(10, len(st.campaigns) - 1)])
    hi = min(len(st), lo + n)
    x = np.arange(lo, hi)
    fig, axs = plt.subplots(2, 1, figsize=(12, 3.8), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.35, 1]))
    _raster(axs[0], st, lo, hi)
    _shade_phases(axs[1], st.phase, lo, hi)
    axs[1].plot(x, fs[lo:hi], color=BLUE, lw=1.1)
    axs[1].axhline(thr, color=INK2, ls="--", lw=1)
    axs[1].set_ylabel("P(attack ≤100)")
    axs[1].set_xlabel("connection number in the test stream")
    axs[1].legend(loc="upper right", fontsize=8, ncol=3)
    axs[0].set_title(f"{n:,} consecutive connections of the test network: forecast vs. "
                     "ground truth (shaded, hidden from the model)", loc="left", fontsize=10.5)
    fig.tight_layout()
    _save(fig, "fig15_stream_overview.png")


# ------------------------------------------------------------------ driver
def make_all(quick=False):
    seeds = (0,) if quick else C.SEEDS
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    streams, pools = _load()
    st = streams["test"]
    fig_dataset(pools)
    dep = load_json(C.MODEL_DIR / "deployed.json")
    explain = load_json(C.RESULT_DIR / "explain.json")
    pred = dict(np.load(C.CACHE_DIR / "deployed_test_pred.npz"))
    fs = pred["prob"][:, K]
    thr = explain["threshold"]
    base = {r["method"]: r for r in load_json(C.RESULT_DIR / "baselines.json")}
    react = _scores("Reactive IDS (XGBoost)")["test"]
    cases = explain["case_campaigns"]
    if cases:
        fig_hero(st, fs, react, thr, base["Reactive IDS (XGBoost)"]["thr_site"], cases[0])
        fig_cases(st, fs, thr, cases)
    # ---- comparison table rows (mean/std over seeds)
    rows = []
    for name, r in base.items():
        c = r["adapted"]
        rows.append(dict(method=name, **{k: c[k] for k in ("pr_auc", "alert_precision", "ewr")},
                         **{k + "_std": 0.0 for k in ("pr_auc", "alert_precision", "ewr")}))
    for name, bb in C.DEEP_MODELS.items():
        runs = _runs(bb, seeds)
        if runs:
            row = dict(method=name)
            for k in ("pr_auc", "alert_precision", "ewr"):
                v = [x["adapted"][k] for x in runs]
                row[k], row[k + "_std"] = float(np.mean(v)), float(np.std(v))
            rows.append(row)
    fig_comparison(rows)
    # ---- operating curves + deployed points (seed 0)
    curves = load_json(C.RESULT_DIR / "operating_curves.json")
    points = {n: (r["adapted"]["false_alarms_per_10k"], r["adapted"]["ewr"],
                  r["adapted"]["lead_median"]) for n, r in base.items()}
    for name, bb in C.DEEP_MODELS.items():
        rr = _runs(bb, (0,))
        if rr:
            points[name] = (rr[0]["adapted"]["false_alarms_per_10k"], rr[0]["adapted"]["ewr"],
                            rr[0]["adapted"]["lead_median"])
    fig_operating(curves, points)
    # ---- lead-time distributions (seed 0)
    leads = {}
    for name, tag, thr_m in (
            ("Probe-count rule", "Probe-count rule", base["Probe-count rule"]["thr_adapted"]),
            ("Window-stats XGBoost", "Window-stats XGBoost_adapted",
             base["Window-stats XGBoost"]["thr_adapted"]),
            ("LSTM", "lstm_s0_adapted", None), ("Transformer", "transformer_s0_adapted", None),
            ("FORESIGHT-Net", "foresight_s0_adapted", None)):
        try:
            sc = _scores(tag)["test"]
        except FileNotFoundError:
            continue
        if thr_m is None:
            thr_m = load_json(C.RESULT_DIR / "runs" /
                              f"{tag.replace('_adapted', '')}.json")["thr_adapted"]
        pc = event_metrics(st, sc, thr_m)["per_campaign"]
        leads[name] = [p["lead"] for p in pc if p["warned"] and p["lead"] > 0]
    fig_lead(leads)
    # ---- ablation
    abl_rows = []
    full = _runs("foresight", seeds)
    variants = [("FORESIGHT-Net (full)", full)]
    for display in C.ABLATIONS:
        slug = "abl_" + display.replace("w/o ", "no_").replace(" ", "_").replace("-", "")
        variants.append((display, _runs(slug, seeds)))
    for name, runs in variants:
        if not runs:
            continue
        row = dict(variant=name)
        for k in ("pr_auc", "alert_precision", "false_alarms_per_10k"):
            v = [x["adapted"][k] for x in runs]
            row[k], row[k + "_std"] = float(np.mean(v)), float(np.std(v))
        abl_rows.append(row)
    if len(abl_rows) > 1:
        fig_ablation(abl_rows)
    # ---- robustness
    f = C.RESULT_DIR / "robustness.csv"
    if f.exists():
        fig_robust(pd.read_csv(f))
    # ---- zero-day
    if (C.RESULT_DIR / "zero_day.csv").exists():
        fig_zero_day(load_json(C.RESULT_DIR / "zero_day_curves.json"),
                     pd.read_csv(C.RESULT_DIR / "zero_day.csv"))
    # ---- explainability
    fig_attention(st, pred, thr, explain)
    fig_features(explain)
    # ---- training curves
    hist = {}
    for name, bb in C.DEEP_MODELS.items():
        rr = _runs(bb, (0,))
        if rr:
            hist[name] = rr[0]["history"]
    fig_training(hist)
    # ---- detection & type forecast (deployed seed)
    det_x = load_json(C.RESULT_DIR / "detection_record_xgb.json")
    dep_run = load_json(C.RESULT_DIR / "runs" / f"{dep['tag']}.json")
    fig_detection(det_x, dep_run["detection"])
    fig_type_confusion(dep_run["type_forecast"]["confusion"])
    fig_overview(st, fs, thr)
    log(f"All figures written to {C.FIG_DIR}")
