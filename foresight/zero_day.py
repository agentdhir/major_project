"""
zero_day.py — the Zero-Day Guard: an autoencoder that only knows "normal".

Motivation (from Phase 1 of this project): supervised models — even with
SMOTE — cannot recognise attack types that never occur in training.  On the
official KDDTest+ split, 17 attack types are brand new.

An AUTOENCODER is trained to compress and reconstruct NORMAL connections only.
Anything it reconstructs badly is "unlike anything normal we have seen", so
its reconstruction error is an attack-agnostic anomaly score.

The guard is fused with the supervised per-connection classifier:
    hybrid = max( F_sup(score_sup), F_ae(score_ae) )
where F_x is the empirical CDF of each score on held-out NORMAL validation
traffic — i.e. "how unusual is this compared with known-normal traffic".
An alert fires if EITHER detector finds the connection unusual.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .utils import get_device, log, set_seed


class AutoEncoder(nn.Module):
    """Symmetric dense autoencoder with a 12-dimensional bottleneck."""

    def __init__(self, d_in, bottleneck=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(),
                                     nn.Linear(64, 32), nn.GELU(),
                                     nn.Linear(32, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, 32), nn.GELU(),
                                     nn.Linear(32, 64), nn.GELU(),
                                     nn.Linear(64, d_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_normal_train, X_normal_val, epochs=40, batch=512,
                      lr=1e-3, patience=5, seed=0):
    """Fit on normal traffic only; early-stop on held-out normal traffic."""
    set_seed(seed)
    dev = get_device()
    Xt = torch.tensor(X_normal_train, device=dev)
    Xv = torch.tensor(X_normal_val, device=dev)
    model = AutoEncoder(Xt.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad = np.inf, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), batch):
            xb = Xt[perm[i:i + batch]]
            # light input noise = denoising autoencoder (more robust scores)
            loss = ((model(xb + 0.05 * torch.randn_like(xb)) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(Xv) - Xv) ** 2).mean().item()
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    log(f"  autoencoder: {ep} epochs, val reconstruction MSE {best:.4f}")
    return model


@torch.no_grad()
def anomaly_score(model, X, batch=8192):
    """Per-connection reconstruction error (higher = more anomalous)."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch], device=dev)
        out.append(((model(xb) - xb) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out)


class EmpiricalCDF:
    """Maps a raw score to 'fraction of normal traffic scoring lower'."""

    def __init__(self, normal_scores):
        self.ref = np.sort(np.asarray(normal_scores))

    def __call__(self, s):
        return np.searchsorted(self.ref, s, side="right") / len(self.ref)


def zero_day_report(y_attack, is_novel, category, scores: dict, thresholds: dict):
    """AUCs and detection rates for each scoring method on KDDTest+.

    y_attack : 1 = attack, 0 = normal (per record)
    is_novel : True for attack records whose TYPE never appears in training
    thresholds: per-method threshold giving 5 % FPR on validation normals
    """
    normal = y_attack == 0
    rows = []
    for name, s in scores.items():
        thr = thresholds[name]
        row = dict(method=name,
                   auc_all=roc_auc_score(y_attack, s),
                   auc_novel=roc_auc_score(np.r_[np.zeros(normal.sum()),
                                                 np.ones(is_novel.sum())],
                                           np.r_[s[normal], s[is_novel]]),
                   test_fpr=float((s[normal] > thr).mean()),
                   dr_all=float((s[y_attack == 1] > thr).mean()),
                   dr_novel=float((s[is_novel] > thr).mean()),
                   dr_known=float((s[(y_attack == 1) & ~is_novel] > thr).mean()))
        for c in ("DoS", "Probe", "R2L", "U2R"):
            row[f"dr_{c}"] = float((s[category == c] > thr).mean())
        rows.append(row)
    return rows
