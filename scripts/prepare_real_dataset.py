"""Prepare the training dataset from the Meteo-France data.

Input: `data/mf/synop_2020_2024_4stations.csv`, downloaded from Opendatasoft.

Output:
- `data/real_windows.npz`: windows X and labels y for each split.

Design choices:

- Time step: 3 hours (native SYNOP), no artificial subsampling.
- Window `Tw = 32` steps, i.e. 96 hours or 4 days of context, versus 16
  hours in the synthetic case. We see a wider daily seasonality.
- Horizon `H = 8` steps, i.e. 24 hours of anticipation, versus 8 h in the
  synthetic case.
- Label: "heavy rain" defined as `rr1 > 2 mm/h` in the horizon. A
  convenient proxy, not a real convective storm.
- Chronological split: train = 2020-2022, val = 2023, test = 2024. Whole
  years are easier to read than 70/15/15.
- One **station** = joint train/val/test for that station. We then
  concatenate the 4 stations to form the final dataset. The split remains
  chronological within each station.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from taranis.data import (
    CANAUX,
    channel_stats,
    make_windows,
    normalize,
    prepare_station,
    read_synop_csv,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MF_CSV = DATA / "mf" / "synop_2020_2024_4stations.csv"

Tw = 32   # 4 days at 3h step
H = 8     # 24 hour horizon
STRIDE = 1
STEP_MIN = 180
RAIN_SEUIL_MM = 2.0

SPLITS = {
    "train": ("2020-01-01", "2023-01-01"),
    "val": ("2023-01-01", "2024-01-01"),
    "test": ("2024-01-01", "2025-01-01"),
}


def split_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = (df["timestamp"] >= start) & (df["timestamp"] < end)
    return df[m].reset_index(drop=True)


def main():
    print(f"Lecture : {MF_CSV}")
    stations = read_synop_csv(MF_CSV)
    print(f"  → {len(stations)} stations : {list(stations.keys())}")

    all_splits = {"train": [], "val": [], "test": []}
    station_stats = []

    for sid, raw in stations.items():
        # resampling + labels
        d = prepare_station(raw, freq="3h", rain_seuil_mm=RAIN_SEUIL_MM)
        n_on = int(d["storm_onset"].sum())
        pct_act = 100 * d["storm_active"].mean()
        station_stats.append(
            {
                "id": sid,
                "nom": raw["station_name"].iloc[0],
                "alt_m": raw["altitude_m"].iloc[0],
                "n_pas": len(d),
                "onsets": n_on,
                "pct_act": round(pct_act, 2),
            }
        )

        for split, (t_start, t_end) in SPLITS.items():
            sub = split_by_date(d, t_start, t_end)
            # drop rows that still have a NaN on any input channel
            sub = sub.dropna(subset=list(CANAUX)).reset_index(drop=True)
            if len(sub) < Tw + H:
                continue
            ds = make_windows(sub, Tw=Tw, H=H, stride=STRIDE, label_col="storm_onset")
            all_splits[split].append(ds)

    print("\nStations préparées :")
    print(pd.DataFrame(station_stats).to_string(index=False))

    # per-split concatenation
    def _concat(datasets):
        if not datasets:
            raise RuntimeError("aucun exemple, dataset vide")
        X = np.concatenate([d.X for d in datasets], axis=0)
        y = np.concatenate([d.y for d in datasets], axis=0)
        ts = np.concatenate([d.timestamps for d in datasets], axis=0)
        return X, y, ts

    X_train, y_train, ts_train = _concat(all_splits["train"])
    X_val, y_val, ts_val = _concat(all_splits["val"])
    X_test, y_test, ts_test = _concat(all_splits["test"])

    print(
        f"\nSplits : train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
    )
    print(
        "Prévalence positive : "
        f"train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}"
    )

    mean, std = channel_stats(X_train)
    print("\nStats train par canal :")
    for c, m, s in zip(CANAUX, mean, std, strict=True):
        print(f"  {c:10s}  mean={m:8.2f}  std={s:6.2f}")

    out = DATA / "real_windows.npz"
    np.savez_compressed(
        out,
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
        source="meteo-france synop, opendatasoft, 4 stations, 2020-2024",
    )
    print(f"\nFenêtres sauvegardées : {out}")


if __name__ == "__main__":
    main()
