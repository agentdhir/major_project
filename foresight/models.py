"""
models.py — the deep-learning architectures.

All models share the same skeleton so that comparisons are fair:

    window of W connections
        │
    ConnectionEncoder        (embeds each connection: numeric + categorical)
        │
    temporal BACKBONE        (MLP | CNN | LSTM | GRU | Transformer | FORESIGHT)
        │
    ForecastHeads            (three tasks, trained jointly)
        ├── early-warning : P(attack within 10 / 25 / 50 / 100 connections)
        ├── attack type   : which attack is coming (None/DoS/R2L/U2R)
        └── detection     : what is the CURRENT connection (5 classes)

FORESIGHT-Net (the proposed model) backbone:
    causal dilated CNN  → spots local bursts (e.g. a sweep of scans)
    LSTM                → remembers the slow escalation over the window
    temporal attention  → focuses on the few connections that matter and
                          gives a human-readable explanation of each alarm

Survival-style early-warning head
---------------------------------
For horizons h1 < h2 < ... < hK the head outputs one *hazard* logit per
interval (h_{k-1}, h_k].  The probability that the attack has NOT arrived by
h_k is the product of "no attack in each interval so far":

    S_k = Π_{j<=k} (1 - σ(z_j)),     P(attack within h_k) = 1 - S_k

so the forecast is guaranteed to be monotone (an attack "within 25" is never
less likely than "within 10") and an ETA can be read off directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ---------------------------------------------------------------- building blocks
class ConnectionEncoder(nn.Module):
    """Embeds one connection record into a d_model-dimensional vector.

    Categorical fields get learned embeddings, so the network discovers that
    e.g. service=telnet and service=ftp behave similarly, instead of treating
    the 70 services as unrelated one-hot columns.
    """
    EMB_DIMS = (4, 16, 6)            # protocol_type, service, flag

    def __init__(self, n_num, cat_sizes, d_model, dropout):
        super().__init__()
        self.n_num = n_num
        self.embs = nn.ModuleList(nn.Embedding(n, d)
                                  for n, d in zip(cat_sizes, self.EMB_DIMS))
        self.proj = nn.Sequential(
            nn.Linear(n_num + sum(self.EMB_DIMS), d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))

    def raw_inputs(self, num, cat):
        """Numeric features + categorical embeddings, before projection.
        Exposed so explain.py can take gradients w.r.t. every input feature."""
        e = [emb(cat[..., j]) for j, emb in enumerate(self.embs)]
        return torch.cat([num] + e, dim=-1)

    def forward(self, num, cat, raw=None):             # [B,W,F], [B,W,3]
        if raw is None:
            raw = self.raw_inputs(num, cat)
        return self.proj(raw)                          # [B,W,d]


class CausalConvBlock(nn.Module):
    """Residual 1-D convolution that only looks backwards in time.

    Left-padding by (k-1)*dilation means the output at step t depends on
    steps <= t only — no peeking at the future inside the window.
    """

    def __init__(self, ch, kernel=5, dilation=1, dropout=0.2):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                              # [B,W,C]
        y = F.pad(x.transpose(1, 2), (self.pad, 0))
        y = self.conv(y).transpose(1, 2)
        return self.norm(x + self.drop(F.gelu(y)))


class TemporalAttention(nn.Module):
    """Additive (Bahdanau) attention pooling over the time axis.

    Returns a weighted summary of the window and the weights themselves;
    the weights are shown to the analyst as "the connections that triggered
    this warning".
    """

    def __init__(self, dim, att_dim=64):
        super().__init__()
        self.proj = nn.Linear(dim, att_dim)
        self.score = nn.Linear(att_dim, 1, bias=False)

    def forward(self, h):                              # [B,W,D]
        a = torch.softmax(self.score(torch.tanh(self.proj(h))).squeeze(-1), -1)
        return (a.unsqueeze(-1) * h).sum(1), a          # [B,D], [B,W]


class ForecastHeads(nn.Module):
    """The three task heads (see module docstring)."""

    def __init__(self, sum_dim, last_dim, n_horizons, dropout,
                 n_types=4, n_classes=5):
        super().__init__()

        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, 64), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(64, o))
        self.warn_head = mlp(sum_dim, n_horizons)   # hazard logits
        self.type_head = mlp(sum_dim, n_types)
        self.det_head = mlp(last_dim, n_classes)

    def forward(self, summary, last):
        z = self.warn_head(summary)
        # log S_k = Σ_{j<=k} log(1 - σ(z_j)) = cumsum(logsigmoid(-z))
        log_surv = torch.cumsum(F.logsigmoid(-z), dim=-1)
        return dict(log_surv=log_surv, prob=1 - torch.exp(log_surv),
                    type_logits=self.type_head(summary),
                    det_logits=self.det_head(last))


# ---------------------------------------------------------------- full model
class SequenceForecaster(nn.Module):
    """Encoder + configurable temporal backbone + multi-task heads."""

    def __init__(self, n_num, cat_sizes, window, n_horizons, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, h, p = cfg.d_model, cfg.hidden, cfg.dropout
        self.encoder = ConnectionEncoder(n_num, cat_sizes, d, p)
        b = cfg.backbone
        if b == "mlp":                       # no notion of order beyond position
            self.body = nn.Sequential(nn.Flatten(), nn.Linear(window * d, 256),
                                      nn.GELU(), nn.Dropout(p), nn.Linear(256, h),
                                      nn.GELU())
            sum_dim = last_dim = h
        elif b == "cnn":                     # local temporal patterns only
            self.body = nn.Sequential(*[CausalConvBlock(d, 5, dl, p)
                                        for dl in (1, 2, 4, 8)])
            sum_dim, last_dim = 3 * d, d
        elif b in ("lstm", "gru"):           # classic recurrent baseline
            rnn = nn.LSTM if b == "lstm" else nn.GRU
            self.body = rnn(d, h, cfg.n_layers, batch_first=True,
                            dropout=p if cfg.n_layers > 1 else 0)
            sum_dim = last_dim = h
        elif b == "transformer":             # self-attention baseline
            self.pos = nn.Parameter(torch.zeros(1, window, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, p,
                                               batch_first=True, norm_first=True,
                                               activation="gelu")
            self.body = nn.TransformerEncoder(layer, cfg.n_layers,
                                              enable_nested_tensor=False)
            sum_dim, last_dim = 2 * d, d
        elif b == "foresight":               # PROPOSED: CNN -> LSTM -> attention
            self.conv = (nn.Sequential(*[CausalConvBlock(d, 5, dl, p)
                                         for dl in (1, 2, 4)])
                         if cfg.use_conv else nn.Identity())
            self.body = nn.LSTM(d, h, cfg.n_layers, batch_first=True,
                                dropout=p if cfg.n_layers > 1 else 0)
            self.attn = TemporalAttention(h) if cfg.use_attention else None
            sum_dim, last_dim = (2 * h if cfg.use_attention else h), h
        else:
            raise ValueError(f"unknown backbone {b}")
        self.heads = ForecastHeads(sum_dim, last_dim, n_horizons, p)

    def forward(self, num, cat, raw=None):
        x = self.encoder(num, cat, raw)               # [B,W,d]
        b, attn = self.cfg.backbone, None
        if b == "mlp":
            summary = last = self.body(x)
        elif b == "cnn":
            y = self.body(x)
            last = y[:, -1]
            summary = torch.cat([last, y.mean(1), y.amax(1)], -1)
        elif b in ("lstm", "gru"):
            y, _ = self.body(x)
            summary = last = y[:, -1]
        elif b == "transformer":
            y = self.body(x + self.pos)
            last = y[:, -1]
            summary = torch.cat([last, y.mean(1)], -1)
        else:  # foresight
            y, _ = self.body(self.conv(x))
            last = y[:, -1]
            if self.attn is not None:
                ctx, attn = self.attn(y)
                summary = torch.cat([ctx, last], -1)
            else:
                summary = last
        out = self.heads(summary, last)
        out["attn"] = attn
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- loss
def _log1mexp(x):
    """Numerically stable log(1 - exp(x)) for x < 0."""
    x = x.clamp(max=-1e-7)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)),
                       torch.log1p(-torch.exp(x)))


def multitask_loss(out, y_h, y_type, y_det, w_warn, w_type, w_det,
                   type_weight=None, det_weight=None, multitask=True):
    """Weighted sum of the three task losses.

    warning  : binary cross-entropy on every horizon, computed in log-space
               from the survival function for numerical stability
    type     : class-weighted cross-entropy (attacks are rarer than "None")
    detection: class-weighted cross-entropy (U2R/R2L are very rare)
    """
    log_surv = out["log_surv"]
    bce = -(y_h * _log1mexp(log_surv) + (1 - y_h) * log_surv).mean()
    loss = w_warn * bce
    parts = {"warn": bce.item()}
    if multitask:
        lt = F.cross_entropy(out["type_logits"], y_type, weight=type_weight)
        ld = F.cross_entropy(out["det_logits"], y_det, weight=det_weight)
        loss = loss + w_type * lt + w_det * ld
        parts.update(type=lt.item(), det=ld.item())
    return loss, parts
