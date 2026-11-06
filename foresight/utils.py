"""
utils.py — small helpers shared by every module (seeding, JSON I/O, logging).
"""
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .config import CACHE_DIR, FIG_DIR, MODEL_DIR, RESULT_DIR


def ensure_dirs():
    """Create the output folders if they do not exist yet."""
    for d in (MODEL_DIR, RESULT_DIR, FIG_DIR, CACHE_DIR,
              MODEL_DIR / "runs", CACHE_DIR / "scores", RESULT_DIR / "runs"):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    """Make numpy / python / torch deterministic for a given seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU when available (training is ~20x faster), else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: Path):
    """Write a dict to pretty-printed JSON (numpy types converted)."""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, default=_default))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def log(msg: str):
    """Timestamped console print (flushes so background logs stay current)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def timer(name: str):
    """`with timer('train'):` prints how long a block took."""
    t0 = time.time()
    yield
    log(f"{name} took {time.time() - t0:.1f}s")
