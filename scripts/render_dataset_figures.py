"""Figures descriptives du dataset synthétique pour l'étape 1b.

Trois figures :

- step1b_timeline.png    : timeline des orages sur les 365 jours simulés.
- step1b_splits.png      : répartition des fenêtres train/val/test + prévalence.
- step1b_distributions.png : histogrammes par canal (train uniquement).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
DATA = ROOT / "data"
ASSETS.mkdir(parents=True, exist_ok=True)


def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA / "synthetic_raw.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _load_windows() -> dict:
    z = np.load(DATA / "synthetic_windows.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def figure_timeline(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.plot(df["timestamp"], df["pressure"], color="#1A73E8", linewidth=0.5, alpha=0.7)
    onsets = df.loc[df["storm_onset"], "timestamp"]
    for t in onsets:
        ax.axvline(t, color="#B00020", linestyle="-", alpha=0.5, linewidth=0.6)
    ax.set_ylabel("Pression (hPa)")
    ax.set_xlabel("Jour")
    ax.set_title(f"Timeline des 365 jours simulés, {len(onsets)} orages marqués en rouge")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "step1b_timeline.png", dpi=140)
    plt.close(fig)


def figure_splits(z: dict):
    train_n, val_n, test_n = len(z["y_train"]), len(z["y_val"]), len(z["y_test"])
    train_pos = int(z["y_train"].sum())
    val_pos = int(z["y_val"].sum())
    test_pos = int(z["y_test"].sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Bar chart des tailles
    ax = axes[0]
    labels = ["Train", "Val", "Test"]
    positives = [train_pos, val_pos, test_pos]
    negatives = [train_n - train_pos, val_n - val_pos, test_n - test_pos]
    x = np.arange(len(labels))
    width = 0.6
    ax.bar(x, negatives, width, label="Négatifs (y=0)", color="#B0BEC5")
    ax.bar(x, positives, width, bottom=negatives, label="Positifs (y=1)", color="#B00020")
    for i, (n, p) in enumerate(zip(negatives, positives, strict=True)):
        ax.text(i, n / 2, f"{n:,}", ha="center", va="center", fontsize=9, color="black")
        ax.text(i, n + p / 2, f"{p:,}", ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Nombre de fenêtres")
    ax.set_title("Répartition des fenêtres par split")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Prévalence
    ax = axes[1]
    prevalences = [train_pos / train_n, val_pos / val_n, test_pos / test_n]
    ax.bar(x, prevalences, width, color="#B00020", alpha=0.7)
    for i, p in enumerate(prevalences):
        ax.text(i, p + 0.005, f"{p:.1%}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prévalence positive")
    ax.set_title("Prévalence de la classe orage par split")
    ax.set_ylim(0, max(prevalences) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / "step1b_splits.png", dpi=140)
    plt.close(fig)


def figure_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    canaux = [
        ("pressure", "Pression (hPa)", "#1A73E8"),
        ("temp", "Température (°C)", "#B00020"),
        ("humidity", "Humidité (%)", "#137333"),
        ("wind", "Vent (m/s)", "#E37400"),
    ]
    for ax, (col, label, color) in zip(axes, canaux, strict=True):
        ax.hist(df[col].values, bins=60, color=color, alpha=0.75)
        ax.set_title(label)
        ax.set_ylabel("Fréquence") if col == "pressure" else None
        ax.grid(alpha=0.3)
    fig.suptitle("Distribution des mesures brutes sur 365 jours", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "step1b_distributions.png", dpi=140)
    plt.close(fig)


def main():
    df = _load_raw()
    z = _load_windows()
    figure_timeline(df)
    figure_splits(z)
    figure_distributions(df)
    print(f"Figures écrites dans {ASSETS}")


if __name__ == "__main__":
    main()
