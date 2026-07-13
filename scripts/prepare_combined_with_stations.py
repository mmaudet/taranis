"""Same as prepare_real_combined but also stores per-window station_id.

Needed for axes 3 (leave-one-station) and 4 (cross-regime) of the
rigorous evaluation, which slice the test set by station.

Output: `data/real_combined_stations_windows.npz` with an extra
`station_train`, `station_val`, `station_test` array of length N holding
the source station id (string, 5 chars).
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
OUT = DATA / "real_combined_stations_windows.npz"

Tw = 32
H = 8
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
        raise SystemExit(f"missing: {CSV}")
    print(f"Reading: {CSV}")
    stations = read_synop_csv(CSV)
    print(f"  -> {len(stations)} stations")

    splits: dict[str, list] = {"train": [], "val": [], "test": []}

    for i, (sid, raw) in enumerate(stations.items(), 1):
        d = prepare_station(
            raw, freq="3h", label_source="combined",
            rain_seuil_mm=5.0, ww_window_before=1, ww_window_after=1,
        )
        for split, (t_start, t_end) in SPLITS.items():
            sub = _split_by_date(d, t_start, t_end)
            sub = sub.dropna(subset=list(CANAUX_MF_RICH)).reset_index(drop=True)
            if len(sub) < Tw + H:
                continue
            ds = make_windows(
                sub, Tw=Tw, H=H, stride=STRIDE,
                label_col="storm_onset", canaux=CANAUX_MF_RICH,
            )
            splits[split].append((sid, ds))
        if i % 10 == 0 or i == len(stations):
            n_train = sum(len(d.y) for _, d in splits["train"])
            print(f"  [{i}/{len(stations)}]  train windows: {n_train:,}")

    def _concat(entries):
        X = np.concatenate([d.X for _, d in entries], axis=0)
        y = np.concatenate([d.y for _, d in entries], axis=0)
        ts = np.concatenate([d.timestamps for _, d in entries], axis=0)
        sids = np.concatenate([
            np.full(len(d.y), sid, dtype="U5") for sid, d in entries
        ], axis=0)
        return X, y, ts, sids

    X_train, y_train, ts_train, sid_train = _concat(splits["train"])
    X_val, y_val, ts_val, sid_val = _concat(splits["val"])
    X_test, y_test, ts_test, sid_test = _concat(splits["test"])

    print(f"\nSplits: train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")
    print(f"Unique station_ids in test: {len(np.unique(sid_test))}")
    print(
        f"Prevalence: train={y_train.mean():.4f}, "
        f"val={y_val.mean():.4f}, test={y_test.mean():.4f}"
    )

    mean, std = channel_stats(X_train)
    np.savez_compressed(
        OUT,
        X_train=normalize(X_train, mean, std),
        y_train=y_train,
        ts_train=ts_train.astype("datetime64[ns]").astype(np.int64),
        station_train=sid_train,
        X_val=normalize(X_val, mean, std),
        y_val=y_val,
        ts_val=ts_val.astype("datetime64[ns]").astype(np.int64),
        station_val=sid_val,
        X_test=normalize(X_test, mean, std),
        y_test=y_test,
        ts_test=ts_test.astype("datetime64[ns]").astype(np.int64),
        station_test=sid_test,
        mean=mean, std=std,
        canaux=np.array(CANAUX_MF_RICH),
        Tw=np.int64(Tw), H=np.int64(H), step_minutes=np.int64(STEP_MIN),
        source="meteo-france synop, combined labels, station_id per window",
    )
    print(f"OK: {OUT}, {OUT.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
