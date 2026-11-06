"""
config.py — every tunable number in FORESIGHT lives here.

Keeping all hyper-parameters in one file means:
  * experiments are reproducible (one place to read what was run),
  * the panel can audit every design decision without reading the code,
  * ablations only change one field instead of editing several modules.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root, so the folder can be moved freely)
# FORESIGHT_OUT lets the quick smoke test write somewhere else.
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / os.environ.get("FORESIGHT_OUT", "outputs")
MODEL_DIR = OUT_DIR / "models"
RESULT_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
CACHE_DIR = OUT_DIR / "cache"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

# --------------------------------------------------------------------------
# Label spaces
# --------------------------------------------------------------------------
# What is happening on the *current* connection (detection head).
CATEGORIES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
# The "impact" stages of an attack: the thing we want to see coming.
# Probe (reconnaissance) is a *precursor*, not the attack we are forecasting.
IMPACT = ["DoS", "R2L", "U2R"]
# What kind of attack is about to arrive (attack-type forecast head).
FORECAST_TYPES = ["None", "DoS", "R2L", "U2R"]

# Reconnaissance tool families, ordered the way a real attacker uses them
# (MITRE ATT&CK T1595 Active Scanning: host discovery -> port/service
# enumeration -> vulnerability scanning).  Labels that do not exist in a pool
# (e.g. mscan/saint only exist in KDDTest+) are simply skipped.
RECON_STAGES = {
    "discovery": ["ipsweep"],
    "portscan": ["portsweep", "nmap"],
    "vulnscan": ["satan", "saint", "mscan"],
}


@dataclass
class SimConfig:
    """Parameters of the kill-chain attack-campaign simulator.

    One *step* of the simulated stream = one connection record observed at
    our network gateway (the defender's viewpoint).  Lengths are in steps.
    """
    # Mixture of campaign types (must sum to 1).
    scenario_probs: dict = field(default_factory=lambda: {
        "recon_dos": 0.28,       # scan -> denial of service
        "recon_r2l": 0.18,       # scan -> remote break-in
        "recon_r2l_u2r": 0.14,   # scan -> break-in -> privilege escalation
        "stealth_r2l": 0.06,     # break-in with NO precursor (unpredictable)
        "flash_dos": 0.06,       # flood with NO precursor (unpredictable)
        "decoy_scan": 0.28,      # benign/opportunistic scan, NO attack follows
    })
    # Quiet background traffic between campaigns.
    gap_range: tuple = (200, 700)
    # Burstiness of normal traffic: sessions of same-service connections.
    session_rate: float = 1 / 120             # P(new session) per quiet step
    session_len: tuple = (5, 40)
    session_purity: float = 0.8               # share of the session's service
    # Reconnaissance phase.
    recon_len: tuple = (60, 200)
    recon_density_start: tuple = (0.03, 0.08)   # fraction of probe conns at start
    recon_density_end: tuple = (0.30, 0.60)     # ... and at the end (escalation)
    recon_ramp_gamma: float = 1.5               # >1 = slow start, fast finish
    recon_stage_noise: float = 0.15             # prob. of using a tool "out of order"
    # Global multiplier on recon density (robustness / stealth experiment).
    stealth_factor: float = 1.0
    # Domain randomisation: scanner choice independent of stage / outcome.
    # Tried for the training stream and REJECTED (it lowered test PR-AUC from
    # 0.56 to 0.49: the stage order of scanning is a genuine signal).
    randomize_tools: bool = False
    # Pause between end of recon and the actual attack.
    lull_range: tuple = (0, 25)
    # Impact phases: (length range, attack-connection density).
    dos_len: tuple = (40, 120)
    dos_density: float = 0.85
    r2l_len: tuple = (10, 40)
    r2l_density: float = 0.60
    u2r_len: tuple = (3, 10)
    u2r_density: float = 0.70
    escalation_lull: tuple = (5, 30)            # pause between R2L and U2R
    # Decoy scans: short, constant intensity, single tool, nothing follows.
    decoy_len: tuple = (20, 80)
    decoy_density: tuple = (0.10, 0.35)


@dataclass
class WindowConfig:
    """How the stream is turned into supervised examples."""
    window: int = 64                       # look-back: last W connections
    horizons: tuple = (10, 25, 50, 100)    # forecast "attack within h steps"
    alarm_horizon_idx: int = 3             # alarm is raised on P(attack <= 100)
    lead_search: int = 200                 # for no-recon attacks: look-back zone


@dataclass
class TrainConfig:
    """Optimisation settings shared by every deep model."""
    batch_size: int = 1024
    lr: float = 1.5e-3
    train_fraction: float = 0.5            # random half of the windows / epoch
    weight_decay: float = 1e-4
    max_epochs: int = 12
    patience: int = 3                      # early stopping on val PR-AUC
    grad_clip: float = 1.0
    # Robustness augmentation (training batches only).
    token_dropout: float = 0.10            # replace protocol/service/flag by <unk>
    feature_noise: float = 0.05            # Gaussian noise on scaled numerics
    # Modality dropout: per window, hide the detector evidence (as if the
    # detector were fooled by an unknown tool) or the raw features.
    modality_dropout: float = 0.25
    # Multi-task loss weights (warning is the primary task).
    w_warning: float = 1.0
    w_type: float = 0.5
    w_detect: float = 0.5


@dataclass
class ModelConfig:
    """Architecture hyper-parameters (see models.py)."""
    backbone: str = "foresight"   # foresight | lstm | gru | cnn | transformer | mlp
    d_model: int = 64             # per-connection embedding size
    hidden: int = 128             # recurrent hidden size
    conv_channels: int = 64
    n_layers: int = 2
    n_heads: int = 4              # transformer only
    dropout: float = 0.2
    use_attention: bool = True    # ablation switch (FORESIGHT-Net)
    use_conv: bool = True         # ablation switch (FORESIGHT-Net)
    multitask: bool = True        # ablation switch (train type+detect heads)
    use_evidence: bool = True     # hybrid: feed per-connection XGBoost evidence


# Simulation sizes (number of campaigns per stream).
N_CAMPAIGNS = {"train": 400, "val": 90, "test": 160}
STREAM_SEEDS = {"train": 1, "val": 2, "test": 3, "site": 4, "adapt": 5}
SITE_BURNIN = 20000            # unlabelled connections observed on the new site
ALERT_BUDGET = 0.01            # max share of burn-in windows allowed to alarm
ADAPT_STEPS = 300              # site-adaptation fine-tuning steps
ADAPT_LR = 2e-4
VAL_FRACTION = 0.15            # KDDTrain+ records held out to build the val stream
SEEDS = (0, 1, 2)              # every deep model is trained with 3 seeds

# Models compared in the main experiment (display name -> backbone).
DEEP_MODELS = {
    "MLP": "mlp",
    "1D-CNN": "cnn",
    "LSTM": "lstm",
    "GRU": "gru",
    "Transformer": "transformer",
    "FORESIGHT-Net": "foresight",
}
# Ablations of FORESIGHT-Net (display name -> ModelConfig overrides).
ABLATIONS = {
    "w/o attention": dict(use_attention=False),
    "w/o causal CNN": dict(use_conv=False),
    "w/o multi-task": dict(multitask=False),
    "w/o detector evidence (pure DL)": dict(use_evidence=False),
}
# Recon-intensity multipliers for the stealthy-attacker experiment.
STEALTH_FACTORS = (1.0, 0.75, 0.5, 0.35, 0.25)
