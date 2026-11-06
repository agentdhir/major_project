"""
data.py — loading NSL-KDD and turning raw connection records into tensors.

NSL-KDD (Tavallaee et al., 2009) is the cleaned version of KDD-Cup-99.  Each
row is ONE network connection summarised by 41 features (duration, bytes,
protocol, service, flag, error rates, host-based traffic statistics, ...).

We keep the *official* split:
    KDDTrain+  -> used to build the training and validation streams
    KDDTest+   -> used ONLY to build the test stream
KDDTest+ contains 17 attack types that never appear in KDDTrain+ (e.g. mscan,
saint, apache2, snmpguess), so the test stream measures generalisation to
unseen attacks, not memorisation.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CATEGORIES, TEST_FILE, TRAIN_FILE, VAL_FRACTION

# 41 features + attack label + difficulty score (as published).
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL + ["label", "difficulty"]]

# Standard 5-category taxonomy of the 39 NSL-KDD attack names.
ATTACK_MAP = {
    "normal": "Normal",
    # Denial of Service — exhaust a resource
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # Probe — reconnaissance / scanning
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # Remote-to-Local — gain access from outside
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # User-to-Root — escalate to administrator
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def load_split(path: Path) -> pd.DataFrame:
    """Read one NSL-KDD file and attach the 5-way attack category."""
    df = pd.read_csv(path, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    df["category"] = df["label"].map(ATTACK_MAP)
    if df["category"].isna().any():
        unknown = df.loc[df["category"].isna(), "label"].unique()
        raise ValueError(f"Unmapped attack labels: {unknown}")
    df["cat_id"] = df["category"].map({c: i for i, c in enumerate(CATEGORIES)})
    return df.reset_index(drop=True)


def load_pools(seed: int = 42, site_normal_frac: float = 0.40,
               site_probe_frac: float = 0.06):
    """Return the five disjoint record pools used to build the streams.

    train : KDDTrain+ minus the validation share     -> training stream
    val   : stratified 15 % of KDDTrain+             -> validation stream
    adapt, site : 40 % of KDDTest+ NORMAL records (+6 % of its scans as
            background radiation), split in two halves.  Together they play
            the role of an unlabelled, attack-free "burn-in" period on the
            deployment network:
              adapt -> the model learns what normal looks like here
                       (train.adapt_to_site)
              site  -> the alarm threshold is set (metrics.site_threshold)
            No attack labels of the deployment network are used.
    test  : the rest of KDDTest+                     -> test stream
    No record appears in two pools.
    """
    train_full = load_split(TRAIN_FILE)
    test_full = load_split(TEST_FILE)
    train_pool, val_pool = train_test_split(
        train_full, test_size=VAL_FRACTION, random_state=seed,
        stratify=train_full["category"])
    rng = np.random.default_rng(seed)
    site_idx = []
    for cat, frac in (("Normal", site_normal_frac), ("Probe", site_probe_frac)):
        idx = np.flatnonzero(test_full["category"].values == cat)
        site_idx.append(rng.choice(idx, int(len(idx) * frac), replace=False))
    site_idx = rng.permutation(np.concatenate(site_idx))
    half = len(site_idx) // 2
    adapt_mask = np.zeros(len(test_full), bool)
    calib_mask = np.zeros(len(test_full), bool)
    adapt_mask[site_idx[:half]] = True
    calib_mask[site_idx[half:]] = True
    return dict(train=train_pool.reset_index(drop=True),
                val=val_pool.reset_index(drop=True),
                adapt=test_full[adapt_mask].reset_index(drop=True),
                site=test_full[calib_mask].reset_index(drop=True),
                test=test_full[~(adapt_mask | calib_mask)].reset_index(drop=True))


class FeatureEncoder:
    """Converts raw records into (numeric, categorical) model inputs.

    * Heavy-tailed counters (bytes, durations, counts) get log(1+x) so a
      1 GB transfer does not dwarf everything else.
    * Numeric features are standardised with TRAIN statistics only
      (no information from the test set leaks into preprocessing).
    * protocol / service / flag become integer ids for learned embeddings.
      Id 0 is reserved for "unseen in training" (KDDTest+ has new services).
    """

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        self.numeric = list(NUMERIC)
        self.log_cols = [c for c in self.numeric if df[c].max() > 1]
        x = self._log(df[self.numeric].astype(np.float64))
        self.mean = x.mean(0).values
        self.std = x.std(0).replace(0, 1).values
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].unique()))}
                      for c in CATEGORICAL}
        return self

    def _log(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        x[self.log_cols] = np.log1p(x[self.log_cols].clip(lower=0))
        return x

    def transform(self, df: pd.DataFrame):
        """-> num float32 [N, 38], cat int64 [N, 3]"""
        x = self._log(df[self.numeric].astype(np.float64)).values
        num = np.clip((x - self.mean) / self.std, -10, 10).astype(np.float32)
        cat = np.stack([df[c].map(self.vocab[c]).fillna(0).astype(np.int64).values
                        for c in CATEGORICAL], axis=1)
        return num, cat

    @property
    def cat_sizes(self):
        """Embedding table sizes (+1 for the 'unknown' id 0)."""
        return [len(self.vocab[c]) + 1 for c in CATEGORICAL]

    def onehot(self, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
        """Flat one-hot view for classical models (XGBoost, autoencoder)."""
        blocks = [num]
        for j, size in enumerate(self.cat_sizes):
            oh = np.zeros((len(cat), size), dtype=np.float32)
            oh[np.arange(len(cat)), cat[:, j]] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, axis=1)

    @property
    def onehot_names(self):
        names = list(self.numeric)
        for c in CATEGORICAL:
            names += [f"{c}=<unk>"] + [f"{c}={v}" for v in self.vocab[c]]
        return names

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "FeatureEncoder":
        with open(path, "rb") as f:
            return pickle.load(f)
