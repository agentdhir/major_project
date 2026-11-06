"""
simulator.py — Kill-chain attack-campaign stream simulator.

WHY THIS EXISTS
---------------
Real attacks are not single events.  Following the Lockheed-Martin Cyber Kill
Chain and MITRE ATT&CK, an intruder first performs RECONNAISSANCE (host
discovery, port scanning, vulnerability scanning) and only later launches the
IMPACT stage (denial of service, remote break-in, privilege escalation).
Reconnaissance traffic is therefore a *precursor*: if a model can recognise
the precursor pattern it can raise an alarm BEFORE the damaging attack lands.

NSL-KDD records are real connections but they are shuffled — the dataset has
no timestamps.  To study forecasting we therefore *replay* real NSL-KDD
records in the order a multi-stage campaign would produce them, as seen from
our own gateway (the defender's viewpoint).  Every connection in the stream is
a genuine NSL-KDD record; only the ORDER is simulated.

WHAT MAKES THE TASK NON-TRIVIAL (no "if probe then alarm" shortcut)
-----------------------------------------------------------------
* Recon is "low-and-slow": it starts at 3-8 % probe density, hidden inside
  normal traffic, and only escalates towards the end.
* DECOY scans (28 % of campaigns): opportunistic scans that are NOT followed
  by an attack.  A naive probe counter raises false alarms on these.
* Internet background radiation: isolated stray probes in normal traffic.
* NO-PRECURSOR attacks (12 %): stealth break-ins / flash floods with no recon.
  These are fundamentally unpredictable and are reported separately.
* The TEST stream is built only from KDDTest+, whose scanning tools (mscan,
  saint) and many attack types never occur in training.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (CATEGORIES, IMPACT, RECON_STAGES, SimConfig,
                     WindowConfig)

PHASES = {0: "background", 1: "recon", 2: "attack", 3: "decoy", 4: "lull"}
CAT_ID = {c: i for i, c in enumerate(CATEGORIES)}
IMPACT_IDS = [CAT_ID[c] for c in IMPACT]
# category id -> attack-type-forecast id (None=0, DoS=1, R2L=2, U2R=3)
CAT_TO_FTYPE = {CAT_ID["DoS"]: 1, CAT_ID["R2L"]: 2, CAT_ID["U2R"]: 3}

# Share of the recon phase spent in each stage (discovery, portscan, vulnscan).
# DoS attackers mostly need live hosts; break-in attackers hunt for
# vulnerable services, so they spend longer vulnerability scanning.
RECON_PROFILE = {
    "recon_dos": (0.50, 0.40, 0.10),
    "recon_r2l": (0.25, 0.35, 0.40),
    "recon_r2l_u2r": (0.25, 0.35, 0.40),
}
BACKGROUND_PROBE_RATE = 0.004   # stray scans in normal traffic ("radiation")


class RecordPool:
    """Draws real record indices by attack label or category.

    Sampling is WITHOUT replacement (cycling through a reshuffled pool when it
    is exhausted) so that as many distinct real records as possible appear.

    Normal traffic is generated with realistic BURSTINESS: real traffic is
    self-similar (Leland et al., 1994) — users and applications open sessions
    that produce runs of similar connections (a backup job, a web page with
    many objects, a mail batch).  Without this, a burst of e.g. 'private'
    service connections would only ever be seen during attacks, and the model
    would learn the shortcut "many private connections = reconnaissance".
    """

    def __init__(self, df: pd.DataFrame, rng: np.random.Generator,
                 cfg: SimConfig = None):
        cfg = cfg or SimConfig()
        self.rng = rng
        self.by_label = {k: v.values for k, v in df.groupby("label").groups.items()}
        self.by_cat = {k: v.values for k, v in df.groupby("category").groups.items()}
        self._queues = {}
        # Sessions: pick a service (sqrt-frequency weighted so rarer services
        # also form sessions) and emit mostly connections of that service.
        normal = df[df.category == "Normal"]
        groups = normal.groupby("service").groups
        svc = {k: v.values for k, v in groups.items() if len(v) >= 20}
        self.normal_by_svc = svc
        self.svc_names = list(svc)
        w = np.sqrt(np.array([len(svc[k]) for k in self.svc_names], float))
        self.svc_w = w / w.sum()
        self.session_rate = cfg.session_rate
        self.session_len = cfg.session_len
        self.session_purity = cfg.session_purity

    def _draw(self, key: str, pool: np.ndarray, n: int) -> np.ndarray:
        out, q = [], self._queues.get(key, np.empty(0, dtype=np.int64))
        while n > 0:
            if len(q) == 0:
                q = self.rng.permutation(pool)
            take, q = q[:n], q[n:]
            out.append(take)
            n -= len(take)
        self._queues[key] = q
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def normal(self, n):
        """`n` normal connections: i.i.d. traffic with embedded sessions."""
        rows = self._draw("cat:Normal", self.by_cat["Normal"], n)
        if self.session_rate <= 0 or n == 0:
            return rows
        pos = int(self.rng.geometric(self.session_rate)) - 1
        while pos < n:
            L = min(int(self.rng.integers(self.session_len[0],
                                          self.session_len[1] + 1)), n - pos)
            svc = self.svc_names[self.rng.choice(len(self.svc_names), p=self.svc_w)]
            sel = self.rng.random(L) < self.session_purity
            k = int(sel.sum())
            if k:
                seg = rows[pos:pos + L]
                seg[sel] = self._draw("svc:" + svc, self.normal_by_svc[svc], k)
            pos += L + int(self.rng.geometric(self.session_rate))
        return rows

    def label(self, lab, n):
        return self._draw("lab:" + lab, self.by_label[lab], n)

    def category(self, cat, n):
        return self._draw("cat:" + cat, self.by_cat[cat], n)


@dataclass
class Stream:
    """A simulated, time-ordered stream of connections plus its labels."""
    rows: np.ndarray          # index of the real record in the source pool
    phase: np.ndarray         # 0 bg, 1 recon, 2 attack, 3 decoy, 4 lull
    campaign: np.ndarray      # campaign id (-1 = background)
    cat_id: np.ndarray        # true category of each connection (0..4)
    label: np.ndarray         # original NSL-KDD attack name
    campaigns: pd.DataFrame   # one row per campaign (scenario, onset, ...)
    num: np.ndarray = None    # numeric features   [T, 38]
    cat: np.ndarray = None    # categorical ids    [T, 3]
    y_h: np.ndarray = None    # attack within each horizon  [T, K] (0/1)
    y_type: np.ndarray = None # type of the next attack     [T]
    dist: np.ndarray = None   # steps until next attack connection [T]
    forecast_mask: np.ndarray = None  # True = "no attack in progress"
    evidence: np.ndarray = None       # per-connection detector probs [T, 5]
    novelty: np.ndarray = None        # autoencoder novelty percentile [T]
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.rows)


class CampaignSimulator:
    """Generates a stream of background traffic with injected campaigns."""

    def __init__(self, pool_df: pd.DataFrame, cfg: SimConfig, seed: int):
        self.df = pool_df
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.pool = RecordPool(pool_df, self.rng, cfg)
        labels = set(pool_df["label"].unique())
        # Recon tools available in THIS pool (KDDTest+ has extra tools).
        self.stage_tools = {s: [t for t in tools if t in labels]
                            for s, tools in RECON_STAGES.items()}
        self.probe_labels = [l for l in pool_df.loc[pool_df.category == "Probe",
                                                    "label"].unique()]
        # Attack names per impact category, weighted by sqrt(frequency) so a
        # single dominant attack (e.g. neptune) does not drown out the rest.
        self.attack_choices = {}
        for cat in IMPACT + ["Probe"]:
            vc = pool_df.loc[pool_df.category == cat, "label"].value_counts()
            w = np.sqrt(vc.values.astype(float))
            self.attack_choices[cat] = (list(vc.index), w / w.sum())
        self._reset()

    # ------------------------------------------------------------------ utils
    def _reset(self):
        self._rows, self._phase, self._camp = [], [], []
        self.t = 0

    def _emit(self, rows, phase, camp):
        self._rows.append(np.asarray(rows, dtype=np.int64))
        self._phase.append(np.full(len(rows), phase, dtype=np.int8))
        self._camp.append(np.full(len(rows), camp, dtype=np.int32))
        self.t += len(rows)

    def _u(self, lo_hi):
        return self.rng.uniform(*lo_hi)

    def _i(self, lo_hi):
        return int(self.rng.integers(lo_hi[0], lo_hi[1] + 1))

    def _pick(self, cat):
        names, w = self.attack_choices[cat]
        return names[self.rng.choice(len(names), p=w)]

    def _mix(self, length, density, draw_attack, force_first=False):
        """`length` connections where each is an attack with prob `density`
        (density may be an array for time-varying intensity)."""
        is_att = self.rng.random(length) < density
        if force_first and length:
            is_att[0] = True
        rows = np.empty(length, dtype=np.int64)
        rows[~is_att] = self.pool.normal(int((~is_att).sum()))
        if is_att.any():
            rows[is_att] = draw_attack(is_att)
        return rows

    # ----------------------------------------------------------------- phases
    def _background(self, n):
        """Normal traffic with rare stray scans (background radiation)."""
        rows = self._mix(n, BACKGROUND_PROBE_RATE,
                         lambda m: self.pool.category("Probe", int(m.sum())))
        self._emit(rows, 0, -1)

    def _recon(self, scenario, camp):
        """Low-and-slow, escalating reconnaissance in three stages."""
        c = self.cfg
        L = self._i(c.recon_len)
        d0 = self._u(c.recon_density_start) * c.stealth_factor
        d1 = self._u(c.recon_density_end) * c.stealth_factor
        u = np.arange(L) / max(L - 1, 1)
        density = d0 + (d1 - d0) * u ** c.recon_ramp_gamma
        f1, f2, _ = RECON_PROFILE[scenario]
        stage = np.where(u < f1, 0, np.where(u < f1 + f2, 1, 2))
        # The attacker uses one tool per stage for this campaign.  With
        # domain randomisation (training stream only) the tools are drawn
        # at random, so WHICH scanner is used carries no information and the
        # model must learn tool-agnostic precursors (intensity, escalation,
        # persistence) - the only kind that transfers to unseen scanners.
        tools = []
        for s in ("discovery", "portscan", "vulnscan"):
            avail = (self.probe_labels if c.randomize_tools
                     else (self.stage_tools[s] or self.probe_labels))
            tools.append(avail[self.rng.integers(len(avail))])

        def draw(mask):
            st = stage[mask].copy()
            noisy = self.rng.random(len(st)) < c.recon_stage_noise
            st[noisy] = self.rng.integers(0, 3, int(noisy.sum()))
            out = np.empty(len(st), dtype=np.int64)
            for k in range(3):
                sel = st == k
                if sel.any():
                    out[sel] = self.pool.label(tools[k], int(sel.sum()))
            return out

        start = self.t
        self._emit(self._mix(L, density, draw), 1, camp)
        return start, L

    def _impact(self, cat, camp):
        """The damaging stage itself (DoS flood / R2L break-in / U2R)."""
        c = self.cfg
        L, dens = {"DoS": (self._i(c.dos_len), c.dos_density),
                   "R2L": (self._i(c.r2l_len), c.r2l_density),
                   "U2R": (self._i(c.u2r_len), c.u2r_density)}[cat]
        name = self._pick(cat)
        # U2R has very few distinct records, so draw from the whole category.
        draw = (lambda m: self.pool.category(cat, int(m.sum()))) if cat == "U2R" \
            else (lambda m: self.pool.label(name, int(m.sum())))
        start = self.t
        self._emit(self._mix(L, dens, draw, force_first=True), 2, camp)
        return start

    def _lull(self, rng_range, camp):
        n = self._i(rng_range)
        if n:
            self._emit(self.pool.normal(n), 4, camp)

    def _decoy(self, camp):
        """Opportunistic scan with constant intensity and nothing after it."""
        c = self.cfg
        L, dens = self._i(c.decoy_len), self._u(c.decoy_density)
        tool = (self.probe_labels[self.rng.integers(len(self.probe_labels))]
                if c.randomize_tools else self._pick("Probe"))
        start = self.t
        self._emit(self._mix(L, dens, lambda m: self.pool.label(tool, int(m.sum()))),
                   3, camp)
        return start, L

    # --------------------------------------------------------------- campaign
    def _campaign(self, cid, scenario):
        c = self.cfg
        rec = dict(id=cid, scenario=scenario, start=self.t, recon_start=-1,
                   recon_len=0, onset=-1, end=-1, attack_type="None",
                   has_recon=False, predictable=False)
        if scenario == "decoy_scan":
            s, L = self._decoy(cid)
            rec.update(recon_start=s, recon_len=L, end=self.t - 1)
            return rec
        if scenario.startswith("recon_"):
            s, L = self._recon(scenario, cid)
            rec.update(recon_start=s, recon_len=L, has_recon=True,
                       predictable=True)
            self._lull(c.lull_range, cid)
        first = "DoS" if scenario in ("recon_dos", "flash_dos") else "R2L"
        rec["onset"] = self._impact(first, cid)
        rec["attack_type"] = first
        escalate = scenario == "recon_r2l_u2r" or (
            scenario == "stealth_r2l" and self.rng.random() < 0.5)
        if escalate:
            self._lull(c.escalation_lull, cid)
            self._impact("U2R", cid)
        rec["end"] = self.t - 1
        return rec

    def generate(self, n_campaigns: int, lead_in: int = 300) -> Stream:
        """Build a full stream: lead-in, then campaigns separated by gaps.
        With n_campaigns=0 this produces pure background traffic (used for
        the unlabelled site burn-in period)."""
        self._reset()
        names = list(self.cfg.scenario_probs)
        probs = np.array([self.cfg.scenario_probs[n] for n in names])
        self._background(lead_in)
        camps = []
        for cid in range(n_campaigns):
            scenario = names[self.rng.choice(len(names), p=probs / probs.sum())]
            camps.append(self._campaign(cid, scenario))
            self._background(self._i(self.cfg.gap_range))
        return self._finish(camps)

    def generate_single(self, scenario: str, lead_in: int = 400,
                        tail: int = 250) -> Stream:
        """Background, ONE campaign of the chosen type, background.
        Used by the live dashboard so the panel can pick the attack."""
        self._reset()
        self._background(lead_in)
        camps = [self._campaign(0, scenario)]
        self._background(tail)
        return self._finish(camps)

    def _finish(self, camps) -> Stream:
        rows = np.concatenate(self._rows)
        src = self.df.iloc[rows]
        cols = ["id", "scenario", "start", "recon_start", "recon_len", "onset",
                "end", "attack_type", "has_recon", "predictable"]
        return Stream(rows=rows, phase=np.concatenate(self._phase),
                      campaign=np.concatenate(self._camp),
                      cat_id=src["cat_id"].values.astype(np.int64),
                      label=src["label"].values.astype(str),
                      campaigns=pd.DataFrame(camps, columns=cols))


def attach_labels(stream: Stream, wcfg: WindowConfig) -> Stream:
    """Compute the forecasting targets for every time step t.

    dist[t]    = steps from t to the next attack connection (strictly after t)
    y_h[t, k]  = 1 if an attack connection occurs in (t, t + horizon_k]
    y_type[t]  = type of that next attack (None / DoS / R2L / U2R)
    forecast_mask[t] = True if no attack is already in progress at t.  The
                 forecasting metrics only use these steps, so a model cannot
                 score points by merely *detecting* an ongoing attack.
    """
    T = len(stream)
    is_imp = np.isin(stream.cat_id, IMPACT_IDS)
    idx = np.flatnonzero(is_imp)
    pos = np.searchsorted(idx, np.arange(T), side="right")
    has_next = pos < len(idx)
    if len(idx):
        nxt = np.where(has_next, idx[np.minimum(pos, len(idx) - 1)], T + 10 ** 6)
    else:                                   # e.g. the attack-free burn-in
        nxt = np.full(T, T + 10 ** 6)
    dist = nxt - np.arange(T)
    horizons = np.asarray(wcfg.horizons)
    stream.dist = dist
    stream.y_h = (dist[:, None] <= horizons[None, :]).astype(np.float32)
    ftype = np.zeros(T, dtype=np.int64)
    near = (dist <= horizons.max()) & has_next
    ftype[near] = [CAT_TO_FTYPE[c] for c in stream.cat_id[nxt[near]]]
    stream.y_type = ftype
    mask = np.ones(T, dtype=bool)
    for r in stream.campaigns.itertuples():
        if r.onset >= 0:
            mask[r.onset:r.end + 1] = False
    stream.forecast_mask = mask
    return stream


def featurize(stream: Stream, pool_df, encoder, wcfg=None) -> Stream:
    """Attach model inputs and forecasting targets to a raw stream."""
    stream.num, stream.cat = encoder.transform(pool_df.iloc[stream.rows])
    return attach_labels(stream, wcfg or WindowConfig())


def build_stream(pool_df, encoder, n_campaigns, seed, sim_cfg=None,
                 wcfg=None, lead_in=300) -> Stream:
    """Simulate a stream from a record pool and attach features + labels."""
    sim_cfg = sim_cfg or SimConfig()
    wcfg = wcfg or WindowConfig()
    stream = CampaignSimulator(pool_df, sim_cfg, seed).generate(n_campaigns,
                                                                lead_in)
    stream.num, stream.cat = encoder.transform(pool_df.iloc[stream.rows])
    stream = attach_labels(stream, wcfg)
    stream.meta = dict(seed=seed, n_campaigns=n_campaigns,
                       stealth_factor=sim_cfg.stealth_factor)
    return stream
