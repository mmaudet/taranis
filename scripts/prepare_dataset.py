"""Prépare le dataset synthétique d'entraînement.

Un an de mesures simulées à 10 minutes, découpé chronologiquement en
train / val / test 70 / 15 / 15. On sauvegarde :

- `data/synthetic_raw.csv` : la série brute (7 colonnes),
- `data/synthetic_windows.npz` : les fenêtres X et labels y, avec l'index de
  split et les stats de normalisation calculées sur le train.

Fenêtrage par défaut : Tw = 96 pas (16 h), H = 48 pas (8 h). Reproductible via
la seed fixée.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from taranis.data import (
    CANAUX,
    channel_stats,
    chronological_split,
    generate,
    make_windows,
    normalize,
)

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(parents=True, exist_ok=True)


def main(
    days: float = 365.0,
    step_minutes: int = 10,
    storms_per_day: float = 0.4,
    seed: int = 0,
    Tw: int = 96,
    H: int = 48,
    stride: int = 1,
):
    print(f"Génération : {days:.0f} jours à {step_minutes} min, ~{storms_per_day} orages/jour")
    df = generate(
        days=days,
        step_minutes=step_minutes,
        storms_per_day=storms_per_day,
        seed=seed,
    )
    n_onsets = int(df["storm_onset"].sum())
    n_active_pct = 100 * df["storm_active"].mean()
    print(f"  → {len(df)} pas, {n_onsets} orages, {n_active_pct:.2f} % du temps actif")

    raw_path = DATA / "synthetic_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"  → série brute : {raw_path}")

    print(f"\nFenêtrage : Tw={Tw}, H={H}, stride={stride}")
    ds = make_windows(df, Tw=Tw, H=H, stride=stride)
    print(f"  → {len(ds)} fenêtres, forme X={ds.X.shape}")

    train, val, test = chronological_split(ds, ratios=(0.7, 0.15, 0.15))
    print(
        f"  → split : train={len(train)}, val={len(val)}, test={len(test)}"
    )
    print(
        "  → prévalence positive (y=1) : "
        f"train={train.y.mean():.3f}, val={val.y.mean():.3f}, test={test.y.mean():.3f}"
    )

    mean, std = channel_stats(train.X)
    print("  → stats train par canal :")
    for c, m, s in zip(CANAUX, mean, std, strict=True):
        print(f"      {c:10s}  mean={m:8.2f}  std={s:6.2f}")

    win_path = DATA / "synthetic_windows.npz"
    np.savez_compressed(
        win_path,
        X_train=normalize(train.X, mean, std),
        y_train=train.y,
        ts_train=train.timestamps.astype("datetime64[ns]").astype(np.int64),
        X_val=normalize(val.X, mean, std),
        y_val=val.y,
        ts_val=val.timestamps.astype("datetime64[ns]").astype(np.int64),
        X_test=normalize(test.X, mean, std),
        y_test=test.y,
        ts_test=test.timestamps.astype("datetime64[ns]").astype(np.int64),
        mean=mean,
        std=std,
        canaux=np.array(CANAUX),
        Tw=np.int64(Tw),
        H=np.int64(H),
    )
    print(f"\nFenêtres sauvegardées : {win_path}")


if __name__ == "__main__":
    main()
