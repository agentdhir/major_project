"""
monitor.py — terminal live demo of FORESIGHT (backup for the web dashboard).

    python monitor.py                          # scan -> DoS campaign
    python monitor.py --scenario recon_r2l_u2r --stealth 0.5
    python monitor.py --scenario decoy_scan    # harmless scan: should stay calm

Streams real KDDTest+ connections past the model one by one and prints a
line whenever the threat level changes, plus a heartbeat every N connections.
At the end it reports how many connections of warning the defender had.
"""
import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from foresight.inference import SCENARIOS, Foresight  # noqa: E402

COL = {0: "\033[92m", 1: "\033[93m", 2: "\033[91m", "end": "\033[0m",
       "dim": "\033[90m", "b": "\033[1m"}
NAMES = {0: "CLEAR  ", 1: "WARNING", 2: "ATTACK "}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="recon_dos", choices=list(SCENARIOS))
    ap.add_argument("--stealth", type=float, default=1.0,
                    help="recon intensity multiplier (1 = normal, 0.25 = very stealthy)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--delay", type=float, default=0.01,
                    help="seconds between connections (0 = as fast as possible)")
    ap.add_argument("--heartbeat", type=int, default=50)
    a = ap.parse_args()

    if sys.platform == "win32":
        import os
        os.system("")                     # enable ANSI colours in Windows terminals
    print(f"{COL['b']}FORESIGHT live monitor{COL['end']} — scenario: "
          f"{SCENARIOS[a.scenario]} (stealth x{a.stealth})")
    print("Loading models...", flush=True)
    fs = Foresight.load()
    stream = fs.simulate(a.scenario, stealth=a.stealth, seed=a.seed)
    an = fs.analyse(stream)
    recs = fs.source_records(stream)
    camp = stream.campaigns.iloc[0]
    print(f"Alarm threshold (site-calibrated): {an.threshold:.3f}\n")

    W = fs.wcfg.window
    prev, first_warn = 0, None
    for t in range(W - 1, len(stream)):
        lvl = int(an.level[t])
        lab = recs["label"].iloc[t]
        if lvl != prev or t % a.heartbeat == 0:
            msg = (f"{COL[lvl]}[{NAMES[lvl]}]{COL['end']} conn {t:5d}  "
                   f"P(attack<=100) {an.warn[t]:5.1%}  ")
            if lvl >= 1:
                msg += (f"expected {fs.attack_type_text(an.type_prob[t])}, "
                        f"ETA {fs.eta_text(int(an.eta[t]))}  ")
            msg += f"{COL['dim']}(true: {lab}){COL['end']}"
            print(msg, flush=True)
        if lvl >= 1 and first_warn is None and (camp.onset < 0 or t < camp.onset) \
                and t >= max(camp.recon_start, 0):
            first_warn = t
        if camp.onset >= 0 and t == camp.onset:
            print(f"{COL['b']}{COL[2]}>>> first attack connection arrives "
                  f"({lab}) at conn {t}{COL['end']}", flush=True)
        prev = lvl
        if a.delay:
            time.sleep(a.delay)

    print("\n" + "=" * 70)
    if camp.onset < 0:
        print("No attack in this scenario (decoy scan).",
              "FORESIGHT raised a warning." if first_warn is not None
              else "FORESIGHT correctly stayed calm.")
    elif first_warn is not None:
        print(f"FORESIGHT warned {camp.onset - first_warn} connections BEFORE the "
              f"attack arrived.")
    else:
        print("No early warning for this campaign "
              "(attacks without reconnaissance cannot be predicted).")


if __name__ == "__main__":
    main()
