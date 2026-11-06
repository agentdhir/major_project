"""
train.py — training loop for every deep model (shared, so comparisons are fair).

Recipe
------
* AdamW optimiser + one-cycle learning-rate schedule
* gradient clipping (stabilises LSTM training)
* class-weighted losses for the rare classes
* model selection / early stopping on VALIDATION PR-AUC of the early-warning
  target (the validation stream is built from KDDTrain+ records that were
  held out — the test stream is never touched during training)
"""
import copy
import time
from dataclasses import asdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .config import ModelConfig, TrainConfig, WindowConfig
from .models import SequenceForecaster, count_params, multitask_loss
from .utils import get_device, log, set_seed
from .windows import N_RAW, WindowSet


def class_weights(labels: np.ndarray, n: int, device):
    """Inverse-square-root frequency weights, normalised to mean 1.

    Full inverse frequency would weight a U2R connection ~800x a normal one
    and make training unstable; the square root is a common compromise.
    """
    cnt = np.bincount(labels, minlength=n).astype(float) + 1.0
    w = 1.0 / np.sqrt(cnt)
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def augment(num, cat, tcfg: TrainConfig):
    """Training-time robustness augmentation.

    * token dropout: each categorical value is replaced by the <unknown> id
      with probability p, so the model cannot hinge a decision on a single
      service name (and learns what to do with services it has never seen);
    * feature noise: small Gaussian jitter on the standardised numerics;
    * modality dropout (hybrid models only): for a random 25 % of windows
      the detector's class probabilities are replaced by a confident
      "normal" verdict — exactly what happens when an attacker uses a tool
      the detector has never seen — and for another 25 % the raw features
      are hidden.  The model learns to forecast from whichever evidence
      channels remain (in particular from the novelty score).
    """
    if tcfg.modality_dropout > 0 and num.shape[-1] > N_RAW:
        u = torch.rand(num.shape[0], device=num.device)
        fooled = u < tcfg.modality_dropout
        blind = (u >= tcfg.modality_dropout) & (u < 2 * tcfg.modality_dropout)
        num = num.clone()
        normal_verdict = torch.tensor([1.0, 0, 0, 0, 0], device=num.device)
        num[fooled, :, N_RAW:N_RAW + 5] = normal_verdict
        num[blind, :, :N_RAW] = 0.0
    if tcfg.token_dropout > 0:
        drop = torch.rand(cat.shape, device=cat.device) < tcfg.token_dropout
        cat = cat.masked_fill(drop, 0)
    if tcfg.feature_noise > 0:
        num = num + tcfg.feature_noise * torch.randn_like(num)
    return num, cat


@torch.no_grad()
def predict(model, ws: WindowSet, batch_size=4096, with_attn=False):
    """Run the model over every valid step of a stream.

    Returns full-length arrays (steps without a full window are zero), so
    index t of the output always refers to stream step t.
    """
    model.eval()
    T = ws.num.shape[0]
    K = ws.y_h.shape[1]
    out = dict(prob=np.zeros((T, K), np.float32),
               type_prob=np.zeros((T, 4), np.float32),
               det_prob=np.zeros((T, 5), np.float32))
    if with_attn:
        out["attn"] = np.zeros((T, ws.W), np.float16)
    for t in ws.iterate(batch_size, shuffle=False):
        num, cat, *_ = ws.batch(t)
        o = model(num, cat)
        out["prob"][t] = o["prob"].float().cpu().numpy()
        out["type_prob"][t] = torch.softmax(o["type_logits"], -1).cpu().numpy()
        out["det_prob"][t] = torch.softmax(o["det_logits"], -1).cpu().numpy()
        if with_attn and o["attn"] is not None:
            out["attn"][t] = o["attn"].cpu().numpy().astype(np.float16)
    return out


def val_pr_auc(prob, stream, wcfg: WindowConfig):
    """Validation score used for early stopping (alarm horizon, masked)."""
    k = wcfg.alarm_horizon_idx
    m = stream.forecast_mask.copy()
    m[:wcfg.window - 1] = False
    return float(average_precision_score(stream.y_h[m, k], prob[m, k]))


def train_model(train_stream, val_stream, n_num, cat_sizes,
                mcfg: ModelConfig, tcfg: TrainConfig = None,
                wcfg: WindowConfig = None, seed: int = 0, verbose=True):
    """Train one model; returns (best_model, history, info)."""
    tcfg = tcfg or TrainConfig()
    wcfg = wcfg or WindowConfig()
    set_seed(seed)
    dev = get_device()
    tr = WindowSet(train_stream, wcfg.window, dev, mcfg.use_evidence)
    va = WindowSet(val_stream, wcfg.window, dev, mcfg.use_evidence)
    n_num = tr.num.shape[1]                 # 38 raw (+5 evidence if hybrid)
    model = SequenceForecaster(n_num, cat_sizes, wcfg.window,
                               len(wcfg.horizons), mcfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    steps_per_epoch = int(np.ceil(int(len(tr) * tcfg.train_fraction)
                                  / tcfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tcfg.lr, total_steps=tcfg.max_epochs * steps_per_epoch,
        pct_start=0.15)
    type_w = class_weights(train_stream.y_type, 4, dev)
    det_w = class_weights(train_stream.cat_id, 5, dev)
    rng = np.random.default_rng(seed)

    best, best_state, bad, history = -1.0, None, 0, []
    t_start = time.time()
    for epoch in range(1, tcfg.max_epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for t in tr.iterate(tcfg.batch_size, shuffle=True, rng=rng,
                            fraction=tcfg.train_fraction):
            num, cat, y_h, y_type, y_det = tr.batch(t)
            num, cat = augment(num, cat, tcfg)
            out = model(num, cat)
            loss, _ = multitask_loss(out, y_h, y_type, y_det,
                                     tcfg.w_warning, tcfg.w_type, tcfg.w_detect,
                                     type_w, det_w, mcfg.multitask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        vp = predict(model, va)
        score = val_pr_auc(vp["prob"], val_stream, wcfg)
        history.append(dict(epoch=epoch, train_loss=tot / nb, val_pr_auc=score,
                            seconds=time.time() - t0))
        if verbose:
            log(f"  [{mcfg.backbone}{'' if mcfg.multitask else '-st'} s{seed}] "
                f"epoch {epoch:2d}  loss {tot / nb:.4f}  val PR-AUC {score:.4f}  "
                f"({time.time() - t0:.1f}s)")
        if score > best + 1e-4:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tcfg.patience:
                break
    model.load_state_dict(best_state)
    info = dict(params=count_params(model), best_val_pr_auc=best,
                epochs_run=len(history), train_seconds=time.time() - t_start,
                model_cfg=asdict(mcfg), n_num=n_num)
    return model, history, info


def adapt_to_site(model, adapt_stream, train_stream, mcfg: ModelConfig,
                  tcfg: TrainConfig = None, wcfg: WindowConfig = None,
                  steps: int = 300, lr: float = 2e-4, seed: int = 0):
    """Site adaptation: learn what NORMAL looks like on the new network.

    Every network has its own "normal" (other services, other hosts, other
    traffic volumes).  During the attack-free burn-in we fine-tune briefly:
      * burn-in windows  -> target "no attack within any horizon"
        (only the early-warning head; no labels of any kind are needed);
      * replayed training windows -> the full multi-task loss, so the model
        does not forget what attacks look like.
    Small learning rate, a few hundred steps.  Returns a NEW model.
    """
    tcfg = tcfg or TrainConfig()
    wcfg = wcfg or WindowConfig()
    dev = next(model.parameters()).device
    rng = np.random.default_rng(seed + 1000)
    model = copy.deepcopy(model)
    tr = WindowSet(train_stream, wcfg.window, dev, mcfg.use_evidence)
    ad = WindowSet(adapt_stream, wcfg.window, dev, mcfg.use_evidence)
    type_w = class_weights(train_stream.y_type, 4, dev)
    det_w = class_weights(train_stream.cat_id, 5, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=tcfg.weight_decay)
    B = tcfg.batch_size // 2
    model.train()
    for _ in range(steps):
        num, cat, y_h, y_type, y_det = tr.batch(rng.choice(tr.steps, B))
        num, cat = augment(num, cat, tcfg)
        replay, _ = multitask_loss(model(num, cat), y_h, y_type, y_det,
                                   tcfg.w_warning, tcfg.w_type, tcfg.w_detect,
                                   type_w, det_w, mcfg.multitask)
        num, cat, *_ = ad.batch(rng.choice(ad.steps, B))
        num, cat = augment(num, cat, tcfg)
        # BCE with target 0 on every horizon  =  -log S_k
        site = -model(num, cat)["log_surv"].mean()
        loss = replay + tcfg.w_warning * site
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
    model.eval()
    return model


def save_checkpoint(path, model, mcfg, wcfg, n_num, cat_sizes, threshold,
                    extra=None):
    """Everything needed to rebuild the model for inference."""
    torch.save(dict(state_dict=model.state_dict(), model_cfg=asdict(mcfg),
                    window_cfg=asdict(wcfg), n_num=n_num, cat_sizes=cat_sizes,
                    threshold=threshold, extra=extra or {}), path)


def load_checkpoint(path, device=None):
    """Rebuild a trained SequenceForecaster from `save_checkpoint` output."""
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    wd = ck["window_cfg"]
    wcfg = WindowConfig(**{**wd, "horizons": tuple(wd["horizons"])})
    mcfg = ModelConfig(**ck["model_cfg"])
    model = SequenceForecaster(ck["n_num"], ck["cat_sizes"], wcfg.window,
                               len(wcfg.horizons), mcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck, wcfg
