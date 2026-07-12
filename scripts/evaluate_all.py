"""Full downstream evaluation, M0 and M1, on all three datasets, with transfer.

Datasets:
- **synth**  : Tw=96, 10 min step. Original synthetic from step 1 bis.
- **synth3h**: Tw=32, 3h step. Same geometry as real, used for transfer.
- **real**   : Tw=32, 3h step. Meteo-France.

Encoders (checkpoints):
- runs/tsjepa_synth
- runs/tsjepa_synth3h
- runs/tsjepa_real

Evaluations performed:

1. **Within-dataset** M0 and M1 on the 3 datasets (6 rows).
2. **Cross-transfer** M1(synth3h): probe trained on real, tested on real.
3. **Inverse control** M1(real): probe trained on synth3h, tested on synth3h.

Outputs: `runs/eval_summary.json` and `docs/assets/step7_summary.png`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taranis.eval import (
    LinearProbe,
    classification_report,
    load_encoder_from_checkpoint,
    tune_threshold_on_val,
)
from taranis.models import BaselinePhysics

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
ASSETS = ROOT / "docs" / "assets"
OUT = RUNS / "eval_summary.json"

DATASETS = {
    "synth": {
        "path": DATA / "synthetic_windows.npz",
        "step_min": 10,
        "short_lag_min": 60,
        "long_lag_min": 180,
    },
    "synth3h": {
        "path": DATA / "synth3h_windows.npz",
        "step_min": 180,
        "short_lag_min": 180,
        "long_lag_min": 720,
    },
    "real": {
        "path": DATA / "real_windows.npz",
        "step_min": 180,
        "short_lag_min": 180,
        "long_lag_min": 720,
    },
}

ENCODERS = {
    "M1(synth)": RUNS / "tsjepa_synth",
    "M1(synth3h)": RUNS / "tsjepa_synth3h",
    "M1(real)": RUNS / "tsjepa_real",
}


def _load_dataset(name: str):
    z = np.load(DATASETS[name]["path"], allow_pickle=True)
    return {
        "X_train": z["X_train"].astype(np.float32),
        "y_train": z["y_train"],
        "X_val": z["X_val"].astype(np.float32),
        "y_val": z["y_val"],
        "X_test": z["X_test"].astype(np.float32),
        "y_test": z["y_test"],
        "mean": z["mean"],
        "std": z["std"],
    }


def _denorm(X, m, s):
    return (X * s + m).astype(np.float32)


def _score_and_report(y_val, score_val, y_test, score_test):
    thr = tune_threshold_on_val(y_val, score_val, criterion="f1")
    r = classification_report(y_test, score_test, threshold=thr)
    return {
        "auc": r.auc,
        "ap": r.average_precision,
        "threshold": r.threshold,
        "precision": r.precision,
        "recall": r.recall,
        "f1": r.f1,
        "prevalence": r.prevalence,
    }


def eval_baseline_m0(dataset_name: str):
    """M0: logistic regression on physics features, trained and tested on the same dataset."""
    d = _load_dataset(dataset_name)
    cfg = DATASETS[dataset_name]
    m, s = d["mean"], d["std"]
    X_train = _denorm(d["X_train"], m, s)
    X_val = _denorm(d["X_val"], m, s)
    X_test = _denorm(d["X_test"], m, s)

    model = BaselinePhysics(
        step_minutes=cfg["step_min"],
        short_lag_min=cfg["short_lag_min"],
        long_lag_min=cfg["long_lag_min"],
    ).fit(X_train, d["y_train"])
    score_val = model.predict_proba(X_val)
    score_test = model.predict_proba(X_test)
    return _score_and_report(d["y_val"], score_val, d["y_test"], score_test)


def eval_probe(encoder_ckpt: Path, dataset_name: str):
    """M1: downstream linear probe on a given encoder, applied to a given dataset."""
    enc = load_encoder_from_checkpoint(encoder_ckpt)
    d = _load_dataset(dataset_name)
    # geometry check
    if enc.config.Tw != d["X_train"].shape[1]:
        return {"skipped": True, "reason": f"Tw mismatch: enc={enc.config.Tw}, data={d['X_train'].shape[1]}"}
    probe = LinearProbe(enc).fit(d["X_train"], d["y_train"])
    score_val = probe.predict_proba(d["X_val"])
    score_test = probe.predict_proba(d["X_test"])
    return _score_and_report(d["y_val"], score_val, d["y_test"], score_test)


def build_all_rows():
    rows = []
    for ds in DATASETS:
        rep = eval_baseline_m0(ds)
        print(f"  M0            @ {ds:8s}  →  AUC={rep['auc']:.3f} AP={rep['ap']:.3f} F1={rep['f1']:.3f}")
        rows.append(dict(model="M0 (baseline physique)", encoder="-", train_dataset=ds, test_dataset=ds, **rep))

    for enc_name, enc_path in ENCODERS.items():
        for ds in DATASETS:
            rep = eval_probe(enc_path, ds)
            if rep.get("skipped"):
                print(f"  {enc_name:14s} @ {ds:8s}  →  skip ({rep['reason']})")
                continue
            print(f"  {enc_name:14s} @ {ds:8s}  →  AUC={rep['auc']:.3f} AP={rep['ap']:.3f} F1={rep['f1']:.3f}")
            rows.append(dict(model=f"{enc_name} + sonde", encoder=enc_name, train_dataset=ds, test_dataset=ds, **rep))
    return rows


def figure_summary(rows):
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AUC heatmap
    ax = axes[0]
    pivot = df.pivot_table(index="model", columns="test_dataset", values="auc")
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("AUC test par modèle × dataset")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < 0.75 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="AUC")

    # F1 heatmap
    ax = axes[1]
    pivot_f1 = df.pivot_table(index="model", columns="test_dataset", values="f1")
    im = ax.imshow(pivot_f1.values, cmap="cividis", vmin=0.0, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(pivot_f1.columns)))
    ax.set_xticklabels(pivot_f1.columns)
    ax.set_yticks(range(len(pivot_f1.index)))
    ax.set_yticklabels(pivot_f1.index)
    ax.set_title("F1 test par modèle × dataset")
    for i in range(pivot_f1.shape[0]):
        for j in range(pivot_f1.shape[1]):
            v = pivot_f1.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < 0.4 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="F1")

    fig.tight_layout()
    fig.savefig(ASSETS / "step7_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== Évaluations ===\n")
    rows = build_all_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nRécapitulatif sauvé : {OUT}")

    figure_summary(rows)
    print(f"Heatmap : {ASSETS / 'step7_summary.png'}")

    df = pd.DataFrame(rows)
    print("\nTableau final (AUC, AP, F1) :")
    print(df[["model", "test_dataset", "auc", "ap", "f1", "precision", "recall", "prevalence"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
