"""
windows.py — turns a stream into sliding-window mini-batches, on the GPU.

At time t the model may only look at the PAST: connections t-W+1 ... t.
Instead of materialising every window (272k x 64 x 41 floats ≈ 2.8 GB) we
keep the stream once on the device and build each batch by index arithmetic:

    idx[b, j] = t_b - (W - 1) + j        (j = 0 .. W-1)

so a batch is just `num[idx]`, which is fast and memory-light.
"""
import numpy as np
import torch

from .simulator import Stream


N_RAW = 38         # NSL-KDD numeric features
N_EVIDENCE = 6     # 5 detector class probabilities + 1 novelty score


def evidence_features(p: np.ndarray, novelty: np.ndarray) -> np.ndarray:
    """Per-connection evidence channels appended to the raw features.

    * detector probabilities, log-scaled to ~[0, 1]: log() gives resolution
      where it matters (P(scan) = 0.001 vs 0.05 is a big difference);
    * novelty percentile from the zero-day autoencoder (0..1): how unusual
      the connection is compared with known-normal traffic."""
    lp = np.log(p + 1e-4) / 9.21 + 1.0
    return np.concatenate([lp, novelty[:, None]], 1).astype(np.float32)


class WindowSet:
    """Device-resident stream that serves (window, targets) batches.

    With use_evidence=True every connection's numeric vector is extended by
    the per-connection detector's 5 class probabilities (hybrid model)."""

    def __init__(self, stream: Stream, window: int, device,
                 use_evidence: bool = False):
        self.W = window
        self.device = device
        num = stream.num
        if use_evidence:
            if stream.evidence is None or stream.novelty is None:
                raise ValueError("stream has no evidence/novelty attached")
            num = np.concatenate(
                [num, evidence_features(stream.evidence, stream.novelty)], 1)
        self.num = torch.from_numpy(np.ascontiguousarray(num)).to(device)
        self.cat = torch.from_numpy(stream.cat).to(device)
        self.y_h = torch.from_numpy(stream.y_h).to(device)
        self.y_type = torch.from_numpy(stream.y_type).to(device)
        self.y_det = torch.from_numpy(stream.cat_id).to(device)
        # Every step that has a full window of history is a valid example.
        self.steps = np.arange(window - 1, len(stream))
        self._offs = torch.arange(-window + 1, 1, device=device)

    def __len__(self):
        return len(self.steps)

    def batch(self, t: np.ndarray):
        """Inputs and targets for the windows ending at the steps `t`."""
        t = torch.as_tensor(t, device=self.device, dtype=torch.long)
        idx = t[:, None] + self._offs[None, :]          # [B, W]
        return (self.num[idx], self.cat[idx],
                self.y_h[t], self.y_type[t], self.y_det[t])

    def iterate(self, batch_size: int, shuffle: bool, rng=None,
                fraction: float = 1.0):
        """Yield batches over the valid steps (optionally shuffled).

        `fraction` < 1 visits a random subset each epoch.  Neighbouring
        windows overlap in W-1 of their W connections, so half of them per
        epoch carries almost all of the information at half the cost.
        """
        steps = self.steps
        if shuffle:
            rng = rng or np.random.default_rng()
            steps = rng.permutation(steps)[:int(len(steps) * fraction)]
        for i in range(0, len(steps), batch_size):
            yield steps[i:i + batch_size]
