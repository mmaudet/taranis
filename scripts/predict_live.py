"""Prédit sur du **vrai** SYNOP live pour quelques stations.

Récupère les 5 derniers jours pour la station demandée, met à jour la
fenêtre glissante à Tw pas, appelle l'inférence, affiche le résultat en
ligne de commande + génère un graphique.

Usage :

    uv run python scripts/predict_live.py --station 07591 --output docs/assets/step12_live_embrun.png
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

from taranis.data.meteofrance import CANAUX_MF_RICH, prepare_station, read_synop_csv
from taranis.infer import Predictor

ROOT = Path(__file__).resolve().parent.parent

_OPENDS = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "donnees-synop-essentielles-omm/exports/csv"
)


def _fetch_max_date(station_id: str) -> datetime:
    """Retourne la date la plus récente disponible sur la station."""
    api = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "donnees-synop-essentielles-omm/records"
    )
    r = requests.get(
        api,
        params={
            "where": f"numer_sta='{station_id.zfill(5)}'",
            "select": "max(date) as fin",
            "limit": 1,
        },
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    fin = payload["results"][0]["fin"]
    return datetime.fromisoformat(fin.replace("Z", "+00:00"))


def fetch_synop(station_id: str, days: int = 6, end: datetime | None = None) -> str:
    """Télécharge les `days` derniers jours SYNOP jusqu'à `end` pour la station.

    Si `end` est None, on prend `max(date)` disponible (les données Opendatasoft
    sont rafraîchies mensuellement, l'instant courant peut être en avance).
    """
    if end is None:
        end = _fetch_max_date(station_id)
    start = end - timedelta(days=days)
    params = {
        "where": (
            f"numer_sta='{station_id.zfill(5)}' "
            f"AND date >= '{start:%Y-%m-%d}' "
            f"AND date <= '{(end + timedelta(days=1)):%Y-%m-%d}'"
        ),
        "select": (
            "numer_sta,nom,altitude,date,pres,pmer,tc,u,ff,dd,rr1,rr3,raf10,ww"
        ),
        "delimiter": ";",
    }
    r = requests.get(_OPENDS, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def _plot_live(d, prediction, station_name, altitude, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    canaux_plot = [
        ("pressure", "Pression (hPa)", "#1A73E8"),
        ("temp", "Température (°C)", "#B00020"),
        ("humidity", "Humidité (%)", "#137333"),
        ("wind_gust", "Rafale 10 min (m/s)", "#E37400"),
    ]
    for ax, (col, label, color) in zip(axes, canaux_plot, strict=True):
        ax.plot(d["timestamp"], d[col], color=color, marker="o", markersize=3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[0].set_title(
        f"{station_name} ({altitude:.0f} m)   |   "
        f"prédiction Taranis : {prediction.alert.value.upper()}   "
        f"(p={prediction.proba:.3f})"
    )
    axes[-1].set_xlabel("Temps")
    fig.autofmt_xdate()

    # bandeau d'alerte visible
    fig.subplots_adjust(top=0.90)
    band_color = prediction.alert.color
    fig.text(
        0.5, 0.94, prediction.alert.french.upper(),
        ha="center", va="center",
        fontsize=13, color="white", weight="bold",
        bbox={"facecolor": band_color, "edgecolor": "none", "boxstyle": "round,pad=0.6"},
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station", required=True, help="ID SYNOP à 5 chiffres, ex. 07591 (Embrun)")
    p.add_argument("--probe", default=str(ROOT / "runs" / "probe" / "ww_rich"))
    p.add_argument("--output", default=None, help="Chemin PNG optionnel")
    p.add_argument(
        "--end-date", default=None,
        help="Date de fin de la fenêtre à évaluer (YYYY-MM-DD). Par défaut, plus récente dispo.",
    )
    args = p.parse_args()

    print(f"Chargement du prédicteur : {args.probe}")
    pred = Predictor(args.probe)
    print(f"  Canaux attendus : {pred.canaux}, Tw={pred.config.Tw}")

    end = None
    if args.end_date:
        end = datetime.fromisoformat(args.end_date).replace(tzinfo=UTC)

    print(f"\nTéléchargement SYNOP pour {args.station} (fin={end or 'auto'})...")
    csv_text = fetch_synop(args.station, days=6, end=end)
    stations = read_synop_csv(StringIO(csv_text))
    if not stations:
        raise SystemExit(f"aucune donnée pour {args.station}")

    (_sid, raw), = stations.items()
    print(f"  Station : {raw['station_name'].iloc[0]}")
    print(f"  {len(raw)} records reçus, dernier : {raw['timestamp'].max()}")

    d = prepare_station(raw, freq="3h")
    d_full = d.dropna(subset=list(CANAUX_MF_RICH)).reset_index(drop=True)
    if len(d_full) < pred.config.Tw:
        raise SystemExit(f"pas assez de mesures ({len(d_full)} pas)")

    window = d_full.iloc[-pred.config.Tw :]
    X = window[list(CANAUX_MF_RICH)].to_numpy(dtype=np.float32)
    prediction = pred.predict_from_raw(X)

    print(f"\n=== Prédiction Taranis pour {raw['station_name'].iloc[0]} ===")
    print(f"  Probabilité orage 24h : {prediction.proba:.4f}")
    print(f"  Alerte              : {prediction.alert.value.upper()}   ({prediction.alert.french})")
    print(f"  Seuil de référence  : {prediction.threshold:.4f}")
    print(f"  Fenêtre utilisée    : {window['timestamp'].iloc[0]} -> {window['timestamp'].iloc[-1]}")
    print()
    print(prediction.disclaimer)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        _plot_live(window, prediction, raw["station_name"].iloc[0],
                   raw["altitude_m"].iloc[0], out)
        print(f"\nGraphique : {out}")


if __name__ == "__main__":
    main()
