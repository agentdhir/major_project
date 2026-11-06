"""
inference.py — load the trained FORESIGHT system and use it on traffic.

    from foresight.inference import Foresight
    fs = Foresight.load()                          # reads outputs/models/
    stream = fs.simulate("recon_dos", stealth=1.0, seed=7)   # demo traffic
    out = fs.analyse(stream)                       # forecasts for every step
    why = fs.explain(stream, t=850)                # why did it warn at t?

Every prediction at step t uses ONLY connections t-63 … t (the model never
sees the future), so analysing a whole stream at once is equivalent to
running it live connection by connection — just faster.

Alert levels
------------
    CLEAR    forecast below the site-calibrated threshold
    WARNING  P(attack within the next 100 connections) above threshold
             -> "an attack is being prepared against us"
    ATTACK   the current connection itself is classified as DoS/R2L/U2R
"""
from dataclasses import dataclass

import numpy as np
import torch

from . import config as C
from .data import FeatureEncoder, load_pools
from .baselines import RecordClassifier
from .explain import explain_window, feature_names
from .simulator import CampaignSimulator, featurize
from .train import load_checkpoint, predict
from .utils import get_device, load_json
from .windows import WindowSet
from .zero_day import AutoEncoder, EmpiricalCDF, anomaly_score

SCENARIOS = {
    "recon_dos": "Reconnaissance -> Denial of Service",
    "recon_r2l": "Reconnaissance -> Remote break-in (R2L)",
    "recon_r2l_u2r": "Full kill chain: recon -> break-in -> root (U2R)",
    "decoy_scan": "Harmless scan (no attack follows)",
    "stealth_r2l": "Stealth break-in (no reconnaissance)",
    "flash_dos": "Flash flood (no reconnaissance)",
}
FTYPES = C.FORECAST_TYPES          # None / DoS / R2L / U2R


@dataclass
class Analysis:
    prob: np.ndarray          # [T, K] P(attack within each horizon)
    warn: np.ndarray          # [T]    P(attack within alarm horizon)
    type_prob: np.ndarray     # [T, 4] which attack is coming
    det_prob: np.ndarray      # [T, 5] what the current connection is
    attn: np.ndarray          # [T, W] attention over the window
    anomaly: np.ndarray       # [T]    zero-day guard percentile (0..1)
    level: np.ndarray         # [T]    0 clear, 1 warning, 2 attack
    eta: np.ndarray           # [T]    horizon index of the ETA (-1 = none)
    threshold: float


class Foresight:
    """The deployed system: forecaster + zero-day guard + alert logic."""

    def __init__(self, model, ck, wcfg, encoder, detector, autoencoder, ae_cdf,
                 pools, device):
        self.model, self.ck, self.wcfg = model, ck, wcfg
        self.detector = detector
        self.use_evidence = bool(ck["model_cfg"].get("use_evidence", False))
        self.features = feature_names(ck["n_num"])
        self.enc, self.ae, self.ae_cdf = encoder, autoencoder, ae_cdf
        self.pools, self.device = pools, device
        self.threshold = float(ck["threshold"]["site"])
        self.K = wcfg.alarm_horizon_idx

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, model_dir=None, device=None):
        model_dir = model_dir or C.MODEL_DIR
        device = device or get_device()
        model, ck, wcfg = load_checkpoint(model_dir / "foresight_net.pt", device)
        enc = FeatureEncoder.load(model_dir / "encoder.pkl")
        pools = load_pools()
        from xgboost import XGBClassifier
        det = RecordClassifier(enc)
        det.model = XGBClassifier()
        det.model.load_model(model_dir / "record_xgb.json")
        ae = AutoEncoder(len(enc.onehot_names)).to(device)
        ae.load_state_dict(torch.load(model_dir / "autoencoder.pt",
                                      map_location=device))
        ae.eval()
        cdf = EmpiricalCDF(np.load(model_dir / "autoencoder_val_normal_scores.npy"))
        return cls(model, ck, wcfg, enc, det, ae, cdf, pools, device)

    # ------------------------------------------------------------- traffic
    def simulate(self, scenario: str, stealth: float = 1.0, seed: int = 0,
                 pool: str = "test", lead_in: int = 400, tail: int = 250):
        """Generate demo traffic: one campaign of `scenario` hidden inside
        normal traffic, built from REAL KDDTest+ records."""
        df = self.pools[pool]
        sim = CampaignSimulator(df, C.SimConfig(stealth_factor=stealth), seed)
        stream = sim.generate_single(scenario, lead_in=lead_in, tail=tail)
        stream = featurize(stream, df, self.enc, self.wcfg)
        stream.evidence = self.detector.proba(stream).astype(np.float32)
        stream.novelty = self.ae_cdf(anomaly_score(
            self.ae, self.enc.onehot(stream.num, stream.cat))).astype(np.float32)
        return stream

    def source_records(self, stream, pool: str = "test"):
        """The raw NSL-KDD rows behind a stream (for display)."""
        return self.pools[pool].iloc[stream.rows]

    # ------------------------------------------------------------- analysis
    def analyse(self, stream) -> Analysis:
        ws = WindowSet(stream, self.wcfg.window, self.device, self.use_evidence)
        out = predict(self.model, ws, with_attn=True)
        warn = out["prob"][:, self.K]
        # ATTACK level needs sustained evidence: >= 3 of the last 5
        # connections classified as DoS/R2L/U2R (a single odd connection
        # should not page the analyst).
        imp = (out["det_prob"][:, [1, 3, 4]].sum(1) > 0.5).astype(int)
        cs = np.cumsum(np.r_[0, imp])
        recent = cs[1:] - cs[np.maximum(np.arange(1, len(cs)) - 5, 0)]
        attack_now = recent >= 3
        level = np.where(attack_now, 2, np.where(warn > self.threshold, 1, 0))
        level[: self.wcfg.window - 1] = 0
        over = out["prob"] >= 0.5
        # ETA = first horizon whose probability passes 50 %; a warning that
        # is not yet that confident falls back to the alarm horizon itself.
        eta = np.where(over.any(1), over.argmax(1),
                       np.where(level >= 1, self.K, -1))
        anomaly = stream.novelty if stream.novelty is not None else np.zeros(len(stream))
        return Analysis(prob=out["prob"], warn=warn, type_prob=out["type_prob"],
                        det_prob=out["det_prob"], attn=out["attn"].astype(np.float32),
                        anomaly=anomaly, level=level, eta=eta,
                        threshold=self.threshold)

    def explain(self, stream, t: int, top: int = 8):
        """Attention + feature attribution for the window ending at step t."""
        ws = WindowSet(stream, self.wcfg.window, self.device, self.use_evidence)
        num, cat, *_ = ws.batch(np.array([max(t, self.wcfg.window - 1)]))
        e = explain_window(self.model, num, cat, self.K)
        order = np.argsort(-e["feature_importance"])[:top]
        e["top_features"] = [(self.features[i], float(e["feature_importance"][i]))
                             for i in order]
        return e

    # ------------------------------------------------------------- helpers
    def eta_text(self, eta_idx: int) -> str:
        if eta_idx < 0:
            return "—"
        return f"≤ {self.wcfg.horizons[eta_idx]} conns"

    @staticmethod
    def attack_type_text(type_prob_row) -> str:
        k = int(np.argmax(type_prob_row[1:])) + 1
        return f"{FTYPES[k]} ({type_prob_row[k] / max(type_prob_row[1:].sum(), 1e-9):.0%})"


def deployed_info():
    """Summary of the deployed model (seed, thresholds) if available."""
    f = C.MODEL_DIR / "deployed.json"
    return load_json(f) if f.exists() else {}
