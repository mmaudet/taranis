"""Génère les figures de l'étape 1 du carnet.

Sortie : docs/assets/step1_*.png. À exécuter avec `uv run python scripts/render_step1_figures.py`.
Reproductible via la seed fixée. On garde volontairement matplotlib brut,
sans style tiers, pour rester lisible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from taranis.data.synthetic import generate

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _plot_multivarie(df, onset_idx, title, filename, before_hours=6, after_hours=6):
    step_min = int((df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() // 60)
    n_before = before_hours * 60 // step_min
    n_after = after_hours * 60 // step_min

    start = max(0, onset_idx - n_before)
    end = min(len(df), onset_idx + n_after)
    sub = df.iloc[start:end].reset_index(drop=True)
    onset_local = onset_idx - start

    fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    canaux = [
        ("pressure", "Pression (hPa)", "tab:blue"),
        ("temp", "Température (°C)", "tab:red"),
        ("humidity", "Humidité (%)", "tab:green"),
        ("wind", "Vent (m/s)", "tab:orange"),
    ]
    for ax, (col, label, color) in zip(axes, canaux, strict=True):
        ax.plot(sub["timestamp"], sub[col], color=color)
        ax.axvline(sub["timestamp"].iloc[onset_local], color="black", linestyle="--", alpha=0.6)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Temps")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ASSETS / filename, dpi=120)
    plt.close(fig)


def _plot_vue_ensemble(df, filename):
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    canaux = [
        ("pressure", "Pression (hPa)", "tab:blue"),
        ("temp", "Température (°C)", "tab:red"),
        ("humidity", "Humidité (%)", "tab:green"),
        ("wind", "Vent (m/s)", "tab:orange"),
    ]
    for ax, (col, label, color) in zip(axes, canaux, strict=True):
        ax.plot(df["timestamp"], df[col], color=color, linewidth=0.8)
        onsets = df.loc[df["storm_onset"], "timestamp"]
        for t in onsets:
            ax.axvline(t, color="black", linestyle=":", alpha=0.3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_title("Vue d'ensemble, 10 jours (pointillés = onsets d'orage)")
    axes[-1].set_xlabel("Temps")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ASSETS / filename, dpi=120)
    plt.close(fig)


def main():
    df = generate(days=10.0, step_minutes=10, storms_per_day=0.5, seed=42)
    _plot_vue_ensemble(df, "step1_vue_ensemble.png")

    # premier onset avec assez de contexte autour
    onsets = np.flatnonzero(df["storm_onset"].values)
    print(f"{len(onsets)} orages simulés sur 10 jours")
    for idx in onsets:
        step_min = 10
        if idx > 6 * 60 // step_min and idx + 6 * 60 // step_min < len(df):
            _plot_multivarie(
                df,
                onset_idx=int(idx),
                title="Signature d'un orage synthétique, +/- 6 heures autour de l'onset",
                filename="step1_zoom_orage.png",
            )
            break

    print(f"Figures écrites dans {ASSETS}")


if __name__ == "__main__":
    main()
