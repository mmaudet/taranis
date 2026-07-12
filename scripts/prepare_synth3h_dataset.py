"""Génère un dataset synthétique à pas 3h, aligné avec la géométrie du réel.

Il sert **exclusivement** au sim-to-real de l'étape 7 : pour comparer un
encodeur pré-entraîné sur synthétique à un encodeur pré-entraîné sur réel,
il faut que les deux acceptent des fenêtres de même forme.

Sortie :

- `data/synth3h_raw.csv`     : série brute au pas 3 h.
- `data/synth3h_windows.npz` : fenêtres Tw=32, H=8, split chronologique.
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
from taranis.data.synthetic import StormProfile

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def main(
    days: float = 365.0 * 5,      # 5 ans, pour la même volumétrie que le réel
    step_minutes: int = 180,      # 3 h, aligné SYNOP
    storms_per_day: float = 0.4,
    seed: int = 0,
    Tw: int = 32,
    H: int = 8,
    stride: int = 1,
):
    print(f"Génération : {days:.0f} jours à {step_minutes} min, ~{storms_per_day} orages/jour")
    # profil réajusté : l'approche doit rester visible avec un pas si grossier
    profile = StormProfile(
        approach_min=8 * 60,   # 8 h d'approche, environ 3 pas visibles avant l'onset
        active_min=3 * 60,     # 3 h d'orage actif, 1 pas
        recovery_min=9 * 60,   # 9 h de récupération, 3 pas
    )
    df = generate(
        days=days,
        step_minutes=step_minutes,
        storms_per_day=storms_per_day,
        seed=seed,
        profile=profile,
    )
    print(f"  → {len(df)} pas, {int(df['storm_onset'].sum())} orages")

    raw = DATA / "synth3h_raw.csv"
    df.to_csv(raw, index=False)
    print(f"  → série brute : {raw}")

    ds = make_windows(df, Tw=Tw, H=H, stride=stride)
    train, val, test = chronological_split(ds, ratios=(0.7, 0.15, 0.15))
    print(
        f"\nSplits : train={len(train)}, val={len(val)}, test={len(test)}"
    )
    print(
        "Prévalence positive : "
        f"train={train.y.mean():.3f}, val={val.y.mean():.3f}, test={test.y.mean():.3f}"
    )

    mean, std = channel_stats(train.X)
    for c, m, s in zip(CANAUX, mean, std, strict=True):
        print(f"  {c:10s}  mean={m:8.2f}  std={s:6.2f}")

    out = DATA / "synth3h_windows.npz"
    np.savez_compressed(
        out,
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
        step_minutes=np.int64(step_minutes),
    )
    print(f"\nFenêtres sauvegardées : {out}")


if __name__ == "__main__":
    main()
