"""
explain.py — why did FORESIGHT raise this warning?

A security analyst will not act on a black box.  Two complementary views:

1. TEMPORAL ATTENTION (built into FORESIGHT-Net)
   The attention layer assigns a weight to each of the last W connections.
   High weight = "this connection influenced the forecast".  We check that
   the model really looks at the reconnaissance traffic: in warning windows,
   the attention mass on scan connections is compared with their share.

2. GRADIENT x INPUT ATTRIBUTION
   d P(attack within 100) / d input, multiplied by the input, summed over the
   window.  Tells us WHICH FEATURES (e.g. rerror_rate, dst_host_diff_srv_rate,
   service) drive the warning.  Categorical embeddings are summed per field.
"""
import numpy as np
import torch

from .data import CATEGORICAL, NUMERIC
from .models import ConnectionEncoder

EVIDENCE_NAMES = ["detector P(Normal)", "detector P(DoS)", "detector P(Probe)",
                  "detector P(R2L)", "detector P(U2R)", "novelty (autoencoder)"]
FEATURES = list(NUMERIC) + list(CATEGORICAL)            # pure-DL model


def feature_names(n_num: int):
    """Names of the input features, in the order the encoder sees them."""
    return list(NUMERIC) + EVIDENCE_NAMES[:n_num - len(NUMERIC)] + list(CATEGORICAL)


def _groups(n_num: int):
    """Column slices of the raw input that belong to each named feature."""
    sl, pos = [], 0
    for _ in range(n_num):
        sl.append(slice(pos, pos + 1))
        pos += 1
    for d in ConnectionEncoder.EMB_DIMS:
        sl.append(slice(pos, pos + d))
        pos += d
    return sl


def attribution(model, num, cat, horizon_idx):
    """Gradient x input per (window step, feature) -> [B, W, n_features]."""
    model.eval()
    raw = model.encoder.raw_inputs(num, cat).detach().requires_grad_(True)
    # cuDNN's LSTM kernel only supports backward in training mode, so use
    # the native implementation for this (small) explanation batch.
    with torch.backends.cudnn.flags(enabled=False):
        out = model(num, cat, raw=raw)
        out["prob"][:, horizon_idx].sum().backward()
    gi = (raw.grad * raw).detach()
    groups = _groups(model.encoder.n_num)
    return torch.stack([gi[..., s].sum(-1) for s in groups], -1).cpu().numpy()


def explain_window(model, num, cat, horizon_idx):
    """Full explanation of ONE window (tensors shaped [1, W, ...])."""
    with torch.no_grad():
        out = model(num, cat)
    att = out["attn"][0].cpu().numpy() if out["attn"] is not None else None
    a = attribution(model, num, cat, horizon_idx)[0]          # [W, F]
    feat = np.abs(a).sum(0)
    return dict(prob=out["prob"][0].cpu().numpy(),
                type_prob=torch.softmax(out["type_logits"], -1)[0].cpu().numpy(),
                attention=att,
                step_importance=np.abs(a).sum(1),
                feature_importance=feat / max(feat.sum(), 1e-12))


def global_feature_importance(model, ws, steps, horizon_idx, batch=256):
    """Average normalised |gradient x input| per feature over many windows."""
    tot = 0.0
    for i in range(0, len(steps), batch):
        num, cat, *_ = ws.batch(steps[i:i + batch])
        a = np.abs(attribution(model, num, cat, horizon_idx)).sum(1)   # [B, F]
        tot = tot + (a / np.maximum(a.sum(1, keepdims=True), 1e-12)).sum(0)
    return tot / max(len(steps), 1)


def attention_focus(attn_rows, cat_windows, probe_id=2):
    """How much more attention scan connections get than their share.

    attn_rows   : [N, W] attention weights of N warning windows
    cat_windows : [N, W] true category of each connection in those windows
    Returns (mean attention mass on probes, mean probe share, ratio).
    """
    is_probe = cat_windows == probe_id
    mass = (attn_rows * is_probe).sum(1)
    share = is_probe.mean(1)
    keep = share > 0
    m, s = float(mass[keep].mean()), float(share[keep].mean())
    return m, s, m / max(s, 1e-12)
