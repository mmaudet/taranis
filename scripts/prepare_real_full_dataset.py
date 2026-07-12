"""Prépare le grand dataset Météo-France, 62 stations sur 16 ans.

Entrée  : `data/mf/synop_full.csv` (téléchargé par `fetch_meteofrance_full.py`).
Sortie  : `data/real_full_windows.npz` (fenêtres normalisées).

Volumes attendus, ordre de grandeur :

- ~2.7 M records bruts,
- ~2.6 M fenêtres après rééchantillonnage 3h et fenêtrage Tw=32, stride=1.

C'est de quoi entraîner sérieusement TS-JEPA sur GPU.
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
CSV = DATA / "mf" / "synop_full.csv"
OUT = DATA / "real_full_windows.npz"

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
    station_summaries = []

    for i, (sid, raw) in enumerate(stations.items(), 1):
        d = prepare_station(raw, freq="3h", rain_seuil_mm=RAIN_SEUIL_MM)
        n_on = int(d["storm_onset"].sum())
        pct_act = 100 * d["storm_active"].mean()
        station_summaries.append(
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
            sub = _split_by_date(d, t_start, t_end)
            sub = sub.dropna(subset=list(CANAUX)).reset_index(drop=True)
            if len(sub) < Tw + H:
                continue
            ds = make_windows(sub, Tw=Tw, H=H, stride=STRIDE, label_col="storm_onset")
            all_splits[split].append(ds)

        if i % 10 == 0 or i == len(stations):
            n_train = sum(len(d) for d in all_splits["train"])
            print(f"  [{i}/{len(stations)}]  cumul fenêtres train : {n_train:,}")

    print(f"\n{len(station_summaries)} stations traitées")

    def _concat(datasets):
        X = np.concatenate([d.X for d in datasets], axis=0)
        y = np.concatenate([d.y for d in datasets], axis=0)
        ts = np.concatenate([d.timestamps for d in datasets], axis=0)
        return X, y, ts

    X_train, y_train, ts_train = _concat(all_splits["train"])
    X_val, y_val, ts_val = _concat(all_splits["val"])
    X_test, y_test, ts_test = _concat(all_splits["test"])

    print(f"\nSplits : train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")
    print(
        "Prévalence positive : "
        f"train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}"
    )

    mean, std = channel_stats(X_train)
    print("\nStats train par canal :")
    for c, m, s in zip(CANAUX, mean, std, strict=True):
        print(f"  {c:10s}  mean={m:8.2f}  std={s:6.2f}")

    print(f"\nÉcriture {OUT} (peut prendre 30 s)...")
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
        source=f"meteo-france synop, opendatasoft, {len(station_summaries)} stations, 2010-2026",
    )
    size_mo = OUT.stat().st_size / 1_000_000
    print(f"OK : {OUT}, {size_mo:.1f} Mo")


if __name__ == "__main__":
    main()
