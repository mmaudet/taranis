"""Compare Tw=8 vs Tw=32 HGB-3ch on the same test set.

Chapter 20 experiment: does the model still work when we constrain it to
24 h of lookback, matching what a hiker's smartphone can realistically
carry in the sensor buffer? If yes, we ship Tw=8 as the production
model. If AUC collapses, we keep Tw=32 and warn users.

Also fits a fresh HGB-Tw8 and saves the calibrated thresholds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from taranis.models import Baseline3ch, Baseline3chHGB

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs" / "eval"
RUNS.mkdir(parents=True, exist_ok=True)


def calibrate_thresholds(y_val, s_val, target_o=0.70, target_r=0.30):
    _, r, thr = precision_recall_curve(y_val, s_val)
    r = r[:-1]
    idx_o = np.where(r >= target_o)[0]
    idx_r = np.where(r >= target_r)[0]
    thr_o = float(thr[idx_o[-1]]) if len(idx_o) else float(thr.min())
    thr_r = float(thr[idx_r[-1]]) if len(idx_r) else float(thr.max())
    if thr_r < thr_o:
        thr_r = thr_o
    return thr_o, thr_r


def denorm(X_norm, mean, std):
    return (X_norm * std + mean).astype(np.float32)


def bootstrap_auc(y, s, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return {
        "mean": float(np.mean(aucs)),
        "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
    }


def run_model(name, model_cls, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n== {name} ==")
    t0 = time.time()
    m = model_cls(step_minutes=180)
    m.fit(X_train, y_train)
    s_val = m.predict_proba(X_val)
    s_test = m.predict_proba(X_test)
    dt = time.time() - t0
    auc = roc_auc_score(y_test, s_test)
    ap = average_precision_score(y_test, s_test)
    print(f"  fit + predict in {dt:.1f} s  AUC={auc:.4f}  AP={ap:.4f}")
    boot = bootstrap_auc(y_test, s_test)
    print(f"  bootstrap: mean={boot['mean']:.4f}  IC95%=[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    thr_o, thr_r = calibrate_thresholds(y_val, s_val, 0.70, 0.30)
    print(f"  thresholds ORANGE={thr_o:.4f}  ROUGE={thr_r:.4f}")
    return {"auc": float(auc), "ap": float(ap), "boot": boot, "thr_o": thr_o, "thr_r": thr_r, "model": m}


def main():
    print("Loading Tw=8 dataset...")
    d = np.load(DATA / "real_combined_3ch_tw8_windows.npz", allow_pickle=True)
    mean, std = d["mean"], d["std"]
    X_train = denorm(d["X_train"].astype(np.float32), mean, std)
    X_val = denorm(d["X_val"].astype(np.float32), mean, std)
    X_test = denorm(d["X_test"].astype(np.float32), mean, std)
    y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]
    print(f"  train {len(y_train):,}, val {len(y_val):,}, test {len(y_test):,}")

    results = {}
    results["M0-3ch-Tw8"] = run_model("M0-3ch-Tw8 (LogReg)", Baseline3ch,
                                       X_train, y_train, X_val, y_val, X_test, y_test)
    results["HGB-3ch-Tw8"] = run_model("HGB-3ch-Tw8", Baseline3chHGB,
                                        X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n=== Comparison summary ===")
    print(f"{'Model':22s}  {'AUC':>7s}  {'IC95%':>18s}  {'AP':>6s}  {'thr_O':>6s}  {'thr_R':>6s}")
    for name, r in results.items():
        print(f"{name:22s}  {r['auc']:.4f}  [{r['boot']['ci_low']:.4f},{r['boot']['ci_high']:.4f}]  "
              f"{r['ap']:.4f}  {r['thr_o']:.4f}  {r['thr_r']:.4f}")

    # For reference, print the Tw=32 numbers from chapter 18
    print("\nReminder Tw=32 (chapter 18):")
    print("  M0-3ch      AUC=0.7399  ORANGE 85.9%  ROUGE 60.1%")
    print("  HGB-3ch     AUC=0.7734  ORANGE 90.7%  ROUGE 55.4%")

    out = RUNS / "baseline_3ch_tw8.json"
    out.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()},
        indent=2, default=str,
    ))
    print(f"\nSaved metrics: {out}")

    # Save the winning model's trees for the PWA export step
    hgb_meta = {
        "auc_test": results["HGB-3ch-Tw8"]["auc"],
        "orange_threshold": results["HGB-3ch-Tw8"]["thr_o"],
        "rouge_threshold": results["HGB-3ch-Tw8"]["thr_r"],
    }
    (RUNS / "baseline_3ch_tw8_hgb_meta.json").write_text(json.dumps(hgb_meta, indent=2))


if __name__ == "__main__":
    main()
