"""Figures comparatives synthétique vs réel pour l'étape 1 ter.

Trois figures :

- step1c_stations_map.png  : positionnement schématique des 4 stations sur
                              un fond simplifié + altitudes.
- step1c_timeseries.png    : un échantillon de série temporelle sur une
                              semaine, pour chaque station, canaux superposés.
- step1c_comparison.png    : histogrammes comparant les distributions
                              synthétique vs réel pour les 4 canaux.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from taranis.data import prepare_station, read_synop_csv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def figure_stations_map(stations):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    coords = {
        "07471": ("Le Puy", 833, 45.08, 3.76),
        "07481": ("Lyon-St Exupéry", 235, 45.72, 5.09),
        "07558": ("Millau", 712, 44.12, 3.02),
        "07591": ("Embrun", 871, 44.57, 6.50),
    }
    lats = [c[2] for c in coords.values()]
    lons = [c[3] for c in coords.values()]
    alts = [c[1] for c in coords.values()]
    noms = [c[0] for c in coords.values()]

    sizes = [80 + a * 0.4 for a in alts]
    ax.scatter(lons, lats, s=sizes, c=alts, cmap="terrain", edgecolors="black")
    for lon, lat, nom, alt in zip(lons, lats, noms, alts, strict=True):
        ax.annotate(f"{nom}\n{alt} m", (lon, lat), xytext=(8, 5),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Les 4 stations SYNOP retenues (sud-est de la France)")
    ax.grid(alpha=0.3)
    ax.set_xlim(2, 7.5)
    ax.set_ylim(43.5, 46.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "step1c_stations_map.png", dpi=140)
    plt.close(fig)


def figure_timeseries(stations):
    """Une semaine (juillet 2023) pour chacune des 4 stations."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    labels_pretty = {
        "07471": "Le Puy (833 m)",
        "07481": "Lyon (235 m)",
        "07558": "Millau (712 m)",
        "07591": "Embrun (871 m)",
    }
    for ax, (sid, sub_raw) in zip(axes, stations.items(), strict=True):
        d = prepare_station(sub_raw, freq="3h")
        window = d[(d["timestamp"] >= "2023-07-10") & (d["timestamp"] < "2023-07-17")]
        ax2 = ax.twinx()
        ax.plot(window["timestamp"], window["pressure"], color="#1A73E8", label="Pression")
        ax.set_ylabel("Pression (hPa)")
        ax2.plot(window["timestamp"], window["rain_1h"], color="#B00020",
                 linestyle="-", drawstyle="steps-post", label="Pluie 1h")
        ax2.set_ylabel("Pluie 1h (mm)", color="#B00020")
        # marquer les onsets
        onsets = window[window["storm_onset"]]
        for t in onsets["timestamp"]:
            ax.axvline(t, color="#B00020", alpha=0.4, linewidth=0.8)
        ax.set_title(f"{labels_pretty[sid]}  |  semaine du 10 au 17 juillet 2023")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Temps")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ASSETS / "step1c_timeseries.png", dpi=140)
    plt.close(fig)


def figure_comparison():
    """Distributions synthétique vs réel, par canal."""
    synth_raw = pd.read_csv(DATA / "synthetic_raw.csv")

    stations = read_synop_csv(DATA / "mf" / "synop_2020_2024_4stations.csv")
    real_frames = []
    for sub in stations.values():
        d = prepare_station(sub, freq="3h")
        real_frames.append(d[["pressure", "temp", "humidity", "wind"]])
    real = pd.concat(real_frames, ignore_index=True).dropna()

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    canaux = [
        ("pressure", "Pression (hPa)", "#1A73E8"),
        ("temp", "Température (°C)", "#B00020"),
        ("humidity", "Humidité (%)", "#137333"),
        ("wind", "Vent (m/s)", "#E37400"),
    ]
    for ax, (col, label, _color) in zip(axes, canaux, strict=True):
        ax.hist(synth_raw[col], bins=60, color="#7CB342", alpha=0.55,
                label="Synthétique", density=True)
        ax.hist(real[col], bins=60, color="#455A64", alpha=0.55,
                label="Réel (4 stations)", density=True)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        if col == "pressure":
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Distribution des mesures brutes : synthétique vs réel", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "step1c_comparison.png", dpi=140)
    plt.close(fig)


def main():
    stations = read_synop_csv(DATA / "mf" / "synop_2020_2024_4stations.csv")
    figure_stations_map(stations)
    figure_timeseries(stations)
    figure_comparison()
    print(f"Figures écrites dans {ASSETS}")


if __name__ == "__main__":
    main()
