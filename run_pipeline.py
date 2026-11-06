"""
run_pipeline.py — one command to reproduce every result of FORESIGHT.

Usage
-----
    python run_pipeline.py                     # full run  (~1 h on an RTX 4060, plugged in)
    python run_pipeline.py --quick             # ~5 min smoke test, writes to outputs_quick/
    python run_pipeline.py --stage deep        # run a single stage
    python run_pipeline.py --stage figures     # regenerate the figures only

Stages: data, baselines, deep, ablation, robust, zeroday, explain, latency,
        summary, figures   (default: all, in that order)
Finished runs are cached in outputs/results/runs/, so an interrupted run can
simply be restarted and continues where it stopped.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ORDER = ["data", "baselines", "deep", "ablation", "robust", "zeroday",
         "explain", "latency", "summary", "figures"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="*", default=None,
                    help=f"one or more of {ORDER} (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny streams, 1 seed, 2 epochs (pipeline smoke test)")
    args = ap.parse_args()
    if args.quick:
        os.environ["FORESIGHT_OUT"] = "outputs_quick"
    # Import AFTER setting FORESIGHT_OUT so config picks up the right folder.
    from foresight.experiments import run
    from foresight import figures

    stages = args.stage or ORDER
    bad = [s for s in stages if s not in ORDER]
    if bad:
        sys.exit(f"unknown stage(s) {bad}; choose from {ORDER}")
    run([s for s in stages if s != "figures"], quick=args.quick)
    if "figures" in stages:
        figures.make_all(quick=args.quick)


if __name__ == "__main__":
    main()
