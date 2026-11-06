"""
baselines.py — the non-deep-learning systems FORESIGHT is compared against.

1. REACTIVE IDS  (what Phase 1 of this project built, and what most IDS do)
   A per-connection XGBoost classifier.  Its "warning score" at time t is
   the probability that the CURRENT connection is an attack.  By design it
   can only fire once the attack has already arrived — lead time ≈ 0.

2. PROBE-COUNT RULE  (a typical SOC correlation rule, e.g. Snort/Suricata
   threshold alerts)  "raise an alarm if many connections in the last W were
   scans".  Score = expected number of probe connections in the window
   (sum of per-connection P(probe)).  Strong, simple, explainable — and the
   baseline a reviewer would ask about first.

3. WINDOW-STATISTICS XGBOOST  (classical ML with hand-crafted temporal
   features)  mean / std / max of every feature over the window, the average
   per-connection class probabilities, plus the latest connection.  It sees
   the same window as the deep models but not the ORDER of events.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from xgboost import XGBClassifier

from .simulator import Stream
from .utils import get_device

XGB_DEVICE = "cuda" if get_device().type == "cuda" else "cpu"


def _xgb(**kw):
    base = dict(tree_method="hist", device=XGB_DEVICE, random_state=42,
                n_jobs=-1)
    base.update(kw)
    return XGBClassifier(**base)


# --------------------------------------------------------------- helpers
def rolling_sum(x: np.ndarray, W: int) -> np.ndarray:
    """Sum over the window ending at each step (zeros before step W-1)."""
    cs = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x]), axis=0)
    out = np.zeros_like(x, dtype=np.float64)
    out[W - 1:] = cs[W:] - cs[:-W]
    return out


def rolling_mean(x, W):
    return rolling_sum(x, W) / W


# --------------------------------------------------------------- 1. reactive
class RecordClassifier:
    """Per-connection 5-class XGBoost (Normal/DoS/Probe/R2L/U2R)."""

    def __init__(self, encoder):
        self.enc = encoder
        self.model = _xgb(n_estimators=300, max_depth=8, learning_rate=0.1,
                          eval_metric="mlogloss")

    def fit(self, pool_df):
        num, cat = self.enc.transform(pool_df)
        y = pool_df["cat_id"].values
        cnt = np.bincount(y, minlength=5).astype(float)
        w = 1 / np.sqrt(cnt)
        w = w / w.mean()                     # same weighting as the DL models
        self.model.fit(self.enc.onehot(num, cat), y, sample_weight=w[y])
        return self

    def proba_records(self, num, cat):
        return self.model.predict_proba(self.enc.onehot(num, cat))

    def proba(self, stream: Stream):
        return self.proba_records(stream.num, stream.cat)


def out_of_fold_proba(encoder, pool_df, n_folds: int = 5, seed: int = 0):
    """Per-record class probabilities where every record is scored by a
    model that did NOT see it in training (k-fold stacking).

    The deep model is trained on these, so during training it sees evidence
    with realistic errors — exactly like on new traffic.  Feeding in-sample
    probabilities (near-perfect on training records) would teach it to trust
    the detector blindly."""
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros((len(pool_df), 5), np.float32)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(pool_df, pool_df["category"]):
        rc = RecordClassifier(encoder).fit(pool_df.iloc[tr_idx])
        num, cat = encoder.transform(pool_df.iloc[te_idx])
        oof[te_idx] = rc.proba_records(num, cat)
    return oof


def reactive_score(rec_proba: np.ndarray) -> np.ndarray:
    """P(current connection is DoS, R2L or U2R)."""
    return rec_proba[:, [1, 3, 4]].sum(1)


# --------------------------------------------------------------- 2. rule
def probe_rule_score(rec_proba: np.ndarray, W: int) -> np.ndarray:
    """Expected number of scan connections among the last W."""
    return rolling_sum(rec_proba[:, 2], W)


# --------------------------------------------------------------- 3. window ML
def window_features(stream: Stream, rec_proba: np.ndarray, W: int):
    """Order-free summary of each window: [T, 38*3 + 5 + 5 + 2 + 38]
    (raw-feature statistics, detector probabilities over the whole window
    and the last 16 connections, novelty over both, latest connection)."""
    num = stream.num.astype(np.float64)
    mean = rolling_mean(num, W)
    sq = rolling_mean(num ** 2, W)
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    mx = np.zeros_like(num)
    mx[W - 1:] = sliding_window_view(num, W, axis=0).max(-1)
    p_all = rolling_mean(rec_proba, W)
    p_recent = rolling_mean(rec_proba, 16)
    nov = stream.novelty[:, None].astype(np.float64)
    n_all, n_recent = rolling_mean(nov, W), rolling_mean(nov, 16)
    return np.concatenate([mean, std, mx, p_all, p_recent, n_all, n_recent, num],
                          1).astype(np.float32)


class WindowXGB:
    """Gradient boosting on window statistics -> P(attack within horizon)."""

    def __init__(self, W: int, horizon_idx: int):
        self.W, self.k = W, horizon_idx
        self.model = _xgb(n_estimators=400, max_depth=6, learning_rate=0.08,
                          subsample=0.9, colsample_bytree=0.8,
                          eval_metric="logloss")

    def fit(self, stream: Stream, rec_proba, stride: int = 2,
            site_stream: Stream = None, site_proba=None):
        """Train on the training stream.  With `site_stream` (the unlabelled
        burn-in of the new network) its windows are added as "no attack
        coming" examples — the same site adaptation the deep models get."""
        X = window_features(stream, rec_proba, self.W)
        idx = np.arange(self.W - 1, len(stream), stride)
        Xs, ys = [X[idx]], [stream.y_h[idx, self.k].astype(int)]
        if site_stream is not None:
            Xa = window_features(site_stream, site_proba, self.W)
            ia = np.arange(self.W - 1, len(site_stream), stride)
            Xs.append(Xa[ia])
            ys.append(np.zeros(len(ia), int))
        self.model.fit(np.concatenate(Xs), np.concatenate(ys))
        return self

    def score(self, stream: Stream, rec_proba):
        X = window_features(stream, rec_proba, self.W)
        s = np.zeros(len(stream), np.float32)
        s[self.W - 1:] = self.model.predict_proba(X[self.W - 1:])[:, 1]
        return s
