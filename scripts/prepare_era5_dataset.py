"""Prepare the pre-training dataset from ERA5 NetCDF files.

Input : `.nc` files produced by `scripts/fetch_era5.py`.
Output: `data/era5_pretrain_windows.npz`.

**Usage**: without labels. ERA5 has no storm ground truth, only
precipitation. We keep `storm_active` and `storm_onset` built as a heavy
rain proxy, but **we do not use them** in JEPA pre-training, which is
self-supervised by nature. Those labels only matter if one wants to train
a downstream probe directly on ERA5, which is not the protocol used here
(we prefer fine-tuning the probe on SYNOP).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from taranis.data import (
    CANAUX,
    channel_stats,
    chronological_split,
    make_windows,
    normalize,
)
from taranis.data.era5 import load_era5_files
from taranis.data.meteofrance import build_storm_labels, resample_regular

ROOT = Path(__file__).resolve().parent.parent
ERA5_DIR = ROOT / "data" / "era5"
OUT = ROOT / "data" / "era5_pretrain_windows.npz"

# Native ERA5 frequency: 1 h. Teaching choice: keep the native 1 h step.
Tw = 48   # 48 h of context, 2 days
H = 12    # 12 h of anticipation
STRIDE = 1
STEP_MIN = 60


def main():
    files = sorted(ERA5_DIR.glob("*.nc"))
    if not files:
        raise SystemExit(
            f"Aucun fichier ERA5 dans {ERA5_DIR}.\n"
            "Lancer d'abord : uv run python scripts/fetch_era5.py"
        )
    print(f"Lecture de {len(files)} fichiers NetCDF...")
    points = load_era5_files(files)
    print(f"  → {len(points)} points de grille")

    all_splits = {"train": [], "val": [], "test": []}
    for key, raw in points.items():
        d = resample_regular(raw, freq="1h", max_gap_hours=3)
        d = build_storm_labels(d, seuil_mm=2.0)
        d = d.dropna(subset=list(CANAUX)).reset_index(drop=True)
        if len(d) < Tw + H:
            continue
        ds = make_windows(d, Tw=Tw, H=H, stride=STRIDE, label_col="storm_onset")
        # per-point chronological split
        train, val, test = chronological_split(ds, ratios=(0.7, 0.15, 0.15))
        all_splits["train"].append(train)
        all_splits["val"].append(val)
        all_splits["test"].append(test)
        print(f"  {key} : {len(ds):,} fenêtres")

    def _concat(datasets):
        X = np.concatenate([d.X for d in datasets], axis=0)
        y = np.concatenate([d.y for d in datasets], axis=0)
        ts = np.concatenate([d.timestamps for d in datasets], axis=0)
        return X, y, ts

    X_train, y_train, ts_train = _concat(all_splits["train"])
    X_val, y_val, ts_val = _concat(all_splits["val"])
    X_test, y_test, ts_test = _concat(all_splits["test"])

    print(f"\nSplits : train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")
    mean, std = channel_stats(X_train)
    for c, m, s in zip(CANAUX, mean, std, strict=True):
        print(f"  {c:10s}  mean={m:8.2f}  std={s:6.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        X_train=normalize(X_train, mean, std),
        y_train=y_train,
        ts_train=ts_train.astype("datetime64[ns]").astype(np.int64),
        X_val=normalize(X_val, mean, std),
        y_val=y_val,
        ts_val=ts_val.astype("datetime64[ns]").astype(np.int64),
        X_test=normalize(X_test, mean, std),
        y_test=y_test,
        ts_test=ts_test.astype("datetime64[ns]").astype(np.int64),
        mean=mean,
        std=std,
        canaux=np.array(CANAUX),
        Tw=np.int64(Tw),
        H=np.int64(H),
        step_minutes=np.int64(STEP_MIN),
        source=f"era5, {len(points)} grid points",
    )
    print(f"\nOK : {OUT}, {OUT.stat().st_size / 1_000_000:.1f} Mo")


if __name__ == "__main__":
    main()
