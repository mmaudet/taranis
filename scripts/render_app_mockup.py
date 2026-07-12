"""Rend un mockup d'écran d'application mobile pour Taranis.

C'est une projection UI/UX à partir des données réelles d'un cas d'orage.
Pas un vrai front, juste une visualisation pour discuter la charte et la
lisibilité en montagne.

Sortie : `docs/assets/step12_mobile_mockup.png`, une image au ratio d'un
smartphone, avec :

- en-tête station + heure,
- gros bandeau d'alerte coloré (VERT / ORANGE / ROUGE),
- valeur de la probabilité + libellé,
- petit sparkline des 4 derniers jours par canal,
- disclaimer permanent en pied.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import matplotlib.patches as mpatches
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


def _fetch(station_id: str, end: datetime, days: int = 6) -> str:
    from datetime import timedelta

    start = end - timedelta(days=days)
    r = requests.get(
        _OPENDS,
        params={
            "where": (
                f"numer_sta='{station_id.zfill(5)}' "
                f"AND date >= '{start:%Y-%m-%d}' "
                f"AND date <= '{(end + timedelta(days=1)):%Y-%m-%d}'"
            ),
            "select": (
                "numer_sta,nom,altitude,date,pres,pmer,tc,u,ff,dd,rr1,rr3,raf10,ww"
            ),
            "delimiter": ";",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.text


def render(station_id: str, end: datetime, out_path: Path):
    pred = Predictor(ROOT / "runs" / "probe" / "ww_rich")
    csv_text = _fetch(station_id, end)
    stations = read_synop_csv(StringIO(csv_text))
    (_sid, raw), = stations.items()
    d = prepare_station(raw, freq="3h")
    d = d.dropna(subset=list(CANAUX_MF_RICH)).reset_index(drop=True)
    window = d.iloc[-pred.config.Tw :]
    X = window[list(CANAUX_MF_RICH)].to_numpy(dtype=np.float32)
    prediction = pred.predict_from_raw(X)

    station_nom = raw["station_name"].iloc[0]
    altitude = raw["altitude_m"].iloc[0]
    now = window["timestamp"].iloc[-1]

    # Palette du bandeau d'alerte
    color = prediction.alert.color
    text_light = "white"

    # Silhouette mobile, tout en figure-coords pour un layout stable
    fig = plt.figure(figsize=(4.5, 9), facecolor="#F5F5F5")

    # cadre écran (dessin sur une axe pleine page)
    frame_ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    frame_ax.set_xlim(0, 1)
    frame_ax.set_ylim(0, 1)
    frame_ax.axis("off")
    frame_ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.01), 0.96, 0.98,
        boxstyle="round,pad=0.005,rounding_size=0.03",
        linewidth=1.2, edgecolor="#BDBDBD", facecolor="white",
    ))
    # header
    frame_ax.text(0.5, 0.955, "TARANIS", ha="center", fontsize=15,
                  weight="bold", color="#333333")
    frame_ax.text(0.5, 0.928, f"{station_nom}  ·  {altitude:.0f} m",
                  ha="center", fontsize=10, color="#616161")
    frame_ax.text(0.5, 0.905, now.strftime("%d %b %Y, %H:%M UTC"),
                  ha="center", fontsize=9, color="#9E9E9E")

    # bandeau d'alerte
    frame_ax.add_patch(mpatches.FancyBboxPatch(
        (0.06, 0.72), 0.88, 0.16,
        boxstyle="round,pad=0.005,rounding_size=0.04",
        linewidth=0, facecolor=color,
    ))
    frame_ax.text(0.5, 0.845, prediction.alert.french.upper(),
                  ha="center", va="center", fontsize=17,
                  color=text_light, weight="bold")
    frame_ax.text(0.5, 0.79, f"p(orage 24h)  =  {prediction.proba:.2f}",
                  ha="center", va="center", fontsize=11, color=text_light)
    frame_ax.text(0.5, 0.755, f"(seuil de référence {prediction.threshold:.2f})",
                  ha="center", va="center", fontsize=8, color=text_light)

    # panneau de mesures : 4 lignes, chaque ligne = label+valeur + sparkline
    canaux_plot = [
        ("pressure", "Pression", "hPa", "#1A73E8"),
        ("temp", "Température", "°C", "#B00020"),
        ("humidity", "Humidité", "%", "#137333"),
        ("wind_gust", "Rafale 10 min", "m/s", "#E37400"),
    ]
    # bornes du bloc mesures en fig-coords
    top = 0.68
    bottom = 0.14
    n = len(canaux_plot)
    row_h = (top - bottom) / n

    for i, (col, label, unit, ccol) in enumerate(canaux_plot):
        y0 = top - (i + 1) * row_h
        # label + valeur à gauche
        cur = window[col].iloc[-1]
        frame_ax.text(0.08, y0 + row_h * 0.60, label,
                      fontsize=9, color="#424242")
        frame_ax.text(0.08, y0 + row_h * 0.25, f"{cur:.1f} {unit}",
                      fontsize=13, weight="bold", color="#212121")

        # sparkline sur une axe dédiée à droite
        spark = fig.add_axes((0.42, y0 + row_h * 0.15, 0.50, row_h * 0.70))
        spark.plot(window[col].values, color=ccol, linewidth=1.6)
        spark.margins(x=0.02, y=0.10)
        spark.axis("off")

    # disclaimer permanent
    frame_ax.text(
        0.5, 0.06,
        "Aide à la décision, pas une garantie.\n"
        "Ne remplace pas les bulletins officiels\n"
        "ni le jugement de l'utilisateur.",
        ha="center", fontsize=7, color="#757575", style="italic",
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    end = datetime.fromisoformat(args.end_date).replace(tzinfo=UTC)
    prediction = render(args.station, end, Path(args.output))
    print(f"Mockup écrit : {args.output}")
    print(f"  Alerte : {prediction.alert.value.upper()}, p={prediction.proba:.3f}")


if __name__ == "__main__":
    main()
