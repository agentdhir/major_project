"""
make_deck_data.py — collect every number the presentation needs into one JSON.

    python presentation/make_deck_data.py          # reads outputs/
    (set FORESIGHT_OUT=outputs_quick to build from the smoke-test results)

The slide builder (build_deck.js) reads only presentation/deck_data.json, so
the slides always show the results of the latest pipeline run.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from foresight import config as C  # noqa: E402

R = C.RESULT_DIR


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def runs(prefix):
    out = []
    for s in C.SEEDS:
        f = R / "runs" / f"{prefix}_s{s}.json"
        if f.exists():
            out.append(jload(f))
    return out


KEYS = ["roc_auc", "pr_auc", "f1", "ewr", "ewr_10", "lead_median",
        "false_alarms_per_10k", "alert_precision", "decoy_alarm_rate",
        "ewr_unpredictable"]


def agg(rs, setting="adapted"):
    d = {}
    for k in KEYS:
        v = np.array([r[setting][k] for r in rs], float)
        d[k] = float(v.mean())
        d[k + "_std"] = float(v.std())
    d["n_seeds"] = len(rs)
    return d


def main():
    out = {"out_dir": str(C.OUT_DIR.name)}
    stats = jload(R / "data_stats.json")
    out["data"] = dict(
        pools={k: v["records"] for k, v in stats["pools"].items()},
        streams={k: v["connections"] for k, v in stats["streams"].items()},
        campaigns={k: v["campaigns"] for k, v in stats["streams"].items()},
        novel_types=stats["novel_test_labels"],
        scenario_test=stats["streams"]["test"]["scenarios"],
    )
    base = {r["method"]: r for r in jload(R / "baselines.json")}
    methods = {}
    for name, r in base.items():
        methods[name] = agg([r])
    for name, bb in C.DEEP_MODELS.items():
        rs = runs(bb)
        if rs:
            methods[name] = agg(rs)
    out["methods"] = methods
    fr = runs("foresight")
    out["foresight_settings"] = {s: agg(fr, s) for s in ("zero_shot", "calibrated", "adapted")}
    out["rule_settings"] = {s: agg([base["Probe-count rule"]], s)
                            for s in ("zero_shot", "calibrated", "adapted")}
    # ablation
    abl = {"FORESIGHT-Net (full)": agg(fr)}
    for display in C.ABLATIONS:
        slug = "abl_" + display.replace("w/o ", "no_").replace(" ", "_").replace("-", "")
        rs = runs(slug)
        if rs:
            abl[display] = agg(rs)
    out["ablation"] = abl
    # what & when
    out["type_forecast"] = dict(
        accuracy=float(np.mean([r["type_forecast"]["accuracy"] for r in fr])),
        macro_f1=float(np.mean([r["type_forecast"]["macro_f1"] for r in fr])))
    out["eta"] = dict(
        exact=float(np.mean([r["eta"]["bin_accuracy"] for r in fr])),
        within_one=float(np.mean([r["eta"]["within_one_bin"] for r in fr])))
    dep = jload(C.MODEL_DIR / "deployed.json")
    dep_run = jload(R / "runs" / f"{dep['tag']}.json")
    out["deployed"] = dict(tag=dep["tag"], threshold=dep["thr_adapted"])
    out["detection"] = dict(record_xgb=jload(R / "detection_record_xgb.json"),
                            foresight=dep_run["detection"])
    out["training"] = dict(
        params=dep_run["info"]["params"],
        seconds_per_model=float(np.mean([r["info"]["train_seconds"] for r in fr])),
        epochs=float(np.mean([r["info"]["epochs_run"] for r in fr])),
        val_pr_auc=float(np.mean([r["info"]["best_val_pr_auc"] for r in fr])))
    # robustness
    f = R / "robustness.csv"
    if f.exists():
        df = pd.read_csv(f)
        out["robust"] = {m: {"stealth": d.sort_values("stealth", ascending=False).stealth.tolist(),
                             "ewr": d.sort_values("stealth", ascending=False).ewr_mean.tolist(),
                             "pr_auc": d.sort_values("stealth", ascending=False).pr_auc_mean.tolist(),
                             "lead": d.sort_values("stealth", ascending=False).lead_median_mean.tolist()}
                         for m, d in df.groupby("method")}
    zd = pd.read_csv(R / "zero_day.csv")
    out["zero_day"] = zd.to_dict("records")
    out["zero_day_n_novel"] = jload(R / "zero_day_curves.json")["n_novel"]
    out["latency"] = jload(R / "latency.json")
    ex = jload(R / "explain.json")
    out["explain"] = dict(attention_ratio=ex["attention_ratio"],
                          attention_mass=ex["attention_mass_on_scans"],
                          scan_share=ex["scan_share"], n_windows=ex["n_windows"],
                          top_features=list(ex["feature_importance"].items())[:8])
    bs = pd.read_csv(R / "budget_sensitivity.csv")
    out["budget"] = {m: d.sort_values("budget_pct").to_dict("records")
                     for m, d in bs.groupby("method")
                     if m in ("FORESIGHT-Net", "Probe-count rule", "Window-stats XGBoost")}
    # code size
    loc = 0
    for p in list((ROOT / "foresight").glob("*.py")) + [ROOT / "app" / "dashboard.py",
                                                        ROOT / "run_pipeline.py",
                                                        ROOT / "monitor.py"]:
        loc += sum(1 for _ in p.open(encoding="utf-8"))
    out["loc"] = loc
    # relative to presentation/, so the repo works on any machine
    out["fig_dir"] = os.path.relpath(C.FIG_DIR, Path(__file__).resolve().parent).replace("\\", "/")
    dst = Path(__file__).with_name("deck_data.json")
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dst)


if __name__ == "__main__":
    main()
