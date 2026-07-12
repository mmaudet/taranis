"""Short-horizon SYNOP dataset, 6h variant (Track B).

- Tw = 8 SYNOP steps = 24 h of context.
- H  = 2 SYNOP steps = 6 h of anticipation.

Trade-off between H=1 (3h, too rare) and H=8 (24h, less accurate in
anticipation). Positive prevalence rises from ~0.04 % (H=1) to
~0.07-0.10 % (H=2), which makes calibration workable.

Output: `data/real_ww_h2_windows.npz`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from taranis.data import (
    CANAUX_MF_RICH,
    channel_stats,
    make_windows,
    normalize,
    prepare_station,
    read_synop_csv,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CSV = DATA / "mf" / "synop_full.csv"
OUT = DATA / "real_ww_h2_windows.npz"

Tw = 8
H = 2
STRIDE = 1
STEP_MIN = 180

SPLITS = {
    "train": ("2010-01-01", "2022-01-01"),
    "val": ("2022-01-01", "2024-01-01"),
    "test": ("2024-01-01", "2026-01-01"),
}


def _split_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = (df["timestamp"] >= start) & (df["timestamp"] < end)
    return df[m].reset_index(drop=True)


def main():
    if not CSV.exists():
        raise SystemExit(f"Fichier manquant : {CSV}")

    print(f"Lecture : {CSV}")
    stations = read_synop_csv(CSV)
    print(f"  → {len(stations)} stations")

    all_splits = {"train": [], "val": [], "test": []}
    total_onsets = 0

    for i, (_sid, raw) in enumerate(stations.items(), 1):
        d = prepare_station(raw, freq="3h", label_source="ww",
                            ww_window_before=1, ww_window_after=1)
        total_onsets += int(d["storm_onset"].sum())

        for split, (t_start, t_end) in SPLITS.items():
            sub = _split_by_date(d, t_start, t_end)
            sub = sub.dropna(subset=list(CANAUX_MF_RICH)).reset_index(drop=True)
            if len(sub) < Tw + H:
                continue
            ds = make_windows(
                sub, Tw=Tw, H=H, stride=STRIDE,
                label_col="storm_onset",
                canaux=CANAUX_MF_RICH,
            )
            all_splits[split].append(ds)

        if i % 10 == 0 or i == len(stations):
            n_train = sum(len(d) for d in all_splits["train"])
            print(f"  [{i}/{len(stations)}]  fenêtres train : {n_train:,}  onsets : {total_onsets:,}")

    def _concat(datasets):
        X = np.concatenate([d.X for d in datasets], axis=0)
        y = np.concatenate([d.y for d in datasets], axis=0)
        ts = np.concatenate([d.timestamps for d in datasets], axis=0)
        return X, y, ts

    X_train, y_train, ts_train = _concat(all_splits["train"])
    X_val, y_val, ts_val = _concat(all_splits["val"])
    X_test, y_test, ts_test = _concat(all_splits["test"])

    print(f"\nSplits : train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")
    print(f"Forme X_train : {X_train.shape}")
    print(
        "Prévalence positive : "
        f"train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}"
    )

    mean, std = channel_stats(X_train)
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
        canaux=np.array(CANAUX_MF_RICH),
        Tw=np.int64(Tw),
        H=np.int64(H),
        step_minutes=np.int64(STEP_MIN),
        source=(
            "meteo-france synop, opendatasoft, "
            f"{len(stations)} stations, 2010-2026, "
            "labels WMO 4677, Tw=8 (24h), H=2 (6h) for portable sensor"
        ),
    )
    print(f"OK : {OUT}, {OUT.stat().st_size / 1_000_000:.1f} Mo")


if __name__ == "__main__":
    main()
