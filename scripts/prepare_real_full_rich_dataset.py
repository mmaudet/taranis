"""Prepare the enriched Meteo-France dataset (5 channels instead of 4).

Step 9. Adds `wind_gust` (10-min gust) as a 5th channel. This channel is
a strong indicator of convective storms; the physics baseline overshoots
here because it does not exploit it.

- Window structure: `(N, Tw, 5)`.
- Channel order fixed by `CANAUX_MF_RICH`.
- The M0 baseline is left unchanged: it reads the first 4 channels and
  ignores the rest. Deliberate, to keep the M0 vs M1 comparison clean.
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
OUT = DATA / "real_full_rich_windows.npz"

Tw = 32
H = 8
STRIDE = 1
STEP_MIN = 180
RAIN_SEUIL_MM = 2.0

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
        raise SystemExit(
            f"Fichier manquant : {CSV}\n"
            "Lancer d'abord : uv run python scripts/fetch_meteofrance_full.py"
        )

    print(f"Lecture : {CSV}")
    stations = read_synop_csv(CSV)
    print(f"  → {len(stations)} stations")

    all_splits = {"train": [], "val": [], "test": []}

    for i, (_sid, raw) in enumerate(stations.items(), 1):
        d = prepare_station(raw, freq="3h", rain_seuil_mm=RAIN_SEUIL_MM)
        # missing gust: fall back to wind speed (guaranteed by resample_regular)
        for split, (t_start, t_end) in SPLITS.items():
            sub = _split_by_date(d, t_start, t_end)
            sub = sub.dropna(subset=list(CANAUX_MF_RICH)).reset_index(drop=True)
            if len(sub) < Tw + H:
                continue
            ds = make_windows(
                sub,
                Tw=Tw,
                H=H,
                stride=STRIDE,
                label_col="storm_onset",
                canaux=CANAUX_MF_RICH,
            )
            all_splits[split].append(ds)

        if i % 10 == 0 or i == len(stations):
            n_train = sum(len(d) for d in all_splits["train"])
            print(f"  [{i}/{len(stations)}]  cumul fenêtres train : {n_train:,}")

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
        f"train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}"
    )

    mean, std = channel_stats(X_train)
    print("\nStats train par canal :")
    for c, m, s in zip(CANAUX_MF_RICH, mean, std, strict=True):
        print(f"  {c:14s}  mean={m:8.2f}  std={s:6.2f}")

    print(f"\nÉcriture {OUT}...")
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
            f"{len(stations)} stations, 2010-2026, enriched channels"
        ),
    )
    print(f"OK : {OUT}, {OUT.stat().st_size / 1_000_000:.1f} Mo")


if __name__ == "__main__":
    main()
