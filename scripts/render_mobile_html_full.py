"""Generate a self-contained mobile HTML page with many real stations.

Difference with `render_mobile_html.py`: instead of 3 fixed scenarios,
we fetch the **most recent available** measurements for a broad set of
stations (mountain, plain, coast), compute the TS-JEPA prediction for
each, and package everything into a single self-contained HTML file.

Output: `taranis_app_full.html`. Openable on mobile without network.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import requests

from taranis.data.meteofrance import CANAUX_MF_RICH, prepare_station, read_synop_csv
from taranis.infer import Predictor

ROOT = Path(__file__).resolve().parent.parent

_OPENDS_EXPORT = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "donnees-synop-essentielles-omm/exports/csv"
)
_OPENDS_RECORDS = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "donnees-synop-essentielles-omm/records"
)

# Broad and diverse selection: altitude, climate type, exposure.
# Each station is paired with a region for the label.
STATIONS = [
    ("07591", "Alpes du Sud"),
    ("07471", "Massif Central"),
    ("07558", "Causses"),
    ("07481", "Vallée du Rhône"),
    ("07690", "Côte d'Azur"),
    ("07747", "Roussillon"),
    ("07510", "Aquitaine"),
    ("07110", "Bretagne"),
    ("07037", "Normandie"),
    ("07149", "Île-de-France"),
    ("07072", "Champagne"),
    ("07190", "Alsace"),
    ("07761", "Corse-du-Sud"),
    ("07790", "Haute-Corse"),
    ("07630", "Sud-Ouest"),
    ("07299", "Alsace-plaine"),
]

# Notable historical cases: real storm onsets confirmed by the WMO ww
# code in SYNOP, where the model outputs an ORANGE or RED prediction.
# Format: (station_id, window_end_date, region label)
HISTORIQUES = [
    ("07690", datetime(2025, 9, 10, tzinfo=UTC), "Nice · orage 10 sept 2025"),
    ("07690", datetime(2025, 9, 11, tzinfo=UTC), "Nice · orage 11 sept 2025"),
    ("07761", datetime(2025, 8, 20, tzinfo=UTC), "Ajaccio · 20 août 2025"),
    ("07481", datetime(2025, 7, 13, tzinfo=UTC), "Lyon · canicule 13 juillet 2025"),
    ("07072", datetime(2025, 9, 11, tzinfo=UTC), "Reims · 11 septembre 2025"),
]


def _fetch_max_date(station_id: str) -> datetime:
    r = requests.get(
        _OPENDS_RECORDS,
        params={
            "where": f"numer_sta='{station_id.zfill(5)}'",
            "select": "max(date) as fin",
            "limit": 1,
        },
        timeout=30,
    )
    r.raise_for_status()
    fin = r.json()["results"][0]["fin"]
    return datetime.fromisoformat(fin.replace("Z", "+00:00"))


def _fetch_window(station_id: str, end: datetime, days: int = 6) -> str:
    start = end - timedelta(days=days)
    r = requests.get(
        _OPENDS_EXPORT,
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


def _sparkline_svg(values, color, width=280, height=44, pad=3) -> str:
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-6)
    step_x = (width - 2 * pad) / max(len(values) - 1, 1)
    pts = " ".join(
        f"{pad + i * step_x:.1f},"
        f"{height - pad - (v - lo) / span * (height - 2 * pad):.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="none" width="100%" height="{height}">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
        f"</svg>"
    )


def build_screen(pred: Predictor, station_id: str, region: str,
                  end: datetime | None = None):
    if end is None:
        end = _fetch_max_date(station_id)
    csv_text = _fetch_window(station_id, end)
    stations = read_synop_csv(StringIO(csv_text))
    if not stations:
        return None
    (_sid, raw), = stations.items()
    d = prepare_station(raw, freq="3h")
    d = d.dropna(subset=list(CANAUX_MF_RICH)).reset_index(drop=True)
    if len(d) < pred.config.Tw:
        return None
    window = d.iloc[-pred.config.Tw :].reset_index(drop=True)
    X = window[list(CANAUX_MF_RICH)].to_numpy(dtype=np.float32)
    prediction = pred.predict_from_raw(X)

    canaux_meta = [
        ("pressure", "Pression", "hPa", "#1A73E8"),
        ("temp", "Température", "°C", "#B00020"),
        ("humidity", "Humidité", "%", "#137333"),
        ("wind_gust", "Rafale 10 min", "m/s", "#E37400"),
    ]
    canaux = []
    for col, name, unit, color in canaux_meta:
        vals = window[col].tolist()
        canaux.append({
            "name": name,
            "unit": unit,
            "color": color,
            "current": vals[-1],
            "sparkline": _sparkline_svg(vals, color),
        })

    return {
        "station_id": station_id,
        "station": raw["station_name"].iloc[0],
        "region": region,
        "altitude": float(raw["altitude_m"].iloc[0]),
        "timestamp": window["timestamp"].iloc[-1].strftime("%d %b %Y, %H:%M UTC"),
        "alert": prediction.alert.value,
        "alert_label": prediction.alert.french,
        "alert_color": prediction.alert.color,
        "proba": prediction.proba,
        "threshold": prediction.threshold,
        "canaux": canaux,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#455A64">
<title>Taranis · Nowcasting orage</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; background: #ECEFF1;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                             "Helvetica Neue", Arial, sans-serif; color: #212121; }
  .app { max-width: 480px; margin: 0 auto; min-height: 100vh; position: relative;
         padding: env(safe-area-inset-top, 0) 0 env(safe-area-inset-bottom, 0); }
  header { padding: 16px 20px 4px; text-align: center; }
  header h1 { margin: 0; font-size: 22px; letter-spacing: 4px; font-weight: 700;
              color: #263238; }
  header .subtitle { margin-top: 4px; font-size: 11px; color: #90A4AE; }

  .selector { display: flex; overflow-x: auto; gap: 6px; padding: 10px 12px;
              scrollbar-width: none; }
  .selector::-webkit-scrollbar { display: none; }
  .chip { flex: 0 0 auto; background: #fff; border: 1px solid #CFD8DC;
          color: #455A64; padding: 6px 12px; border-radius: 999px; font-size: 12px;
          cursor: pointer; transition: transform .15s, border-color .15s; }
  .chip.active { background: #263238; border-color: #263238; color: white;
                 transform: scale(1.03); }
  .chip .badge { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                 margin-right: 6px; vertical-align: middle; }

  .track { display: flex; overflow-x: auto; scroll-snap-type: x mandatory;
           -webkit-overflow-scrolling: touch; scrollbar-width: none;
           padding: 6px 0 0; scroll-behavior: smooth; }
  .track::-webkit-scrollbar { display: none; }
  .card { flex: 0 0 100%; scroll-snap-align: center; padding: 6px 16px 24px; }

  .screen { background: #fff; border-radius: 20px; padding: 20px 22px 22px;
            box-shadow: 0 6px 22px rgba(0,0,0,.08); }
  .station { text-align: center; margin: 0 0 2px; font-size: 16px; color: #263238;
             font-weight: 600; }
  .region { text-align: center; margin: 0 0 4px; font-size: 11px; color: #78909C;
            text-transform: uppercase; letter-spacing: 1px; }
  .stamp { text-align: center; margin: 0 0 16px; font-size: 11px; color: #B0BEC5; }

  .alert-banner { border-radius: 18px; padding: 16px 12px 14px; color: white;
                   text-align: center; margin-bottom: 18px; }
  .alert-title { font-size: 18px; font-weight: 800; letter-spacing: 1px; }
  .alert-proba { margin-top: 5px; font-size: 12px; opacity: .95; }
  .alert-seuil { margin-top: 1px; font-size: 10px; opacity: .8; }

  .row { display: grid; grid-template-columns: 110px 1fr; align-items: center;
         gap: 10px; padding: 9px 0; border-bottom: 1px dashed #ECEFF1; }
  .row:last-child { border-bottom: none; }
  .label { font-size: 11px; color: #78909C; }
  .value { font-size: 15px; font-weight: 700; color: #212121; margin-top: 2px; }

  .footer { text-align: center; font-size: 10px; color: #90A4AE; padding: 12px 24px 24px;
            font-style: italic; }
  .hint { text-align: center; font-size: 11px; color: #B0BEC5; padding: 4px 20px; }

  @media (min-width: 640px) {
    header h1 { font-size: 26px; }
    .screen { padding: 24px 28px 26px; }
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>TARANIS</h1>
    <div class="subtitle">Nowcasting orage · TS-JEPA + sonde WMO</div>
  </header>

  <div class="selector" id="selector">__CHIPS__</div>
  <div class="hint">Balaie les cartes ← → ou touche une pastille</div>
  <div class="track" id="track">__CARDS__</div>

  <div class="footer">
    Aide à la décision, pas une garantie.<br>
    Ne remplace pas les bulletins officiels ni le jugement de l'utilisateur.<br>
    En montagne, un faux négatif peut être mortel.
  </div>
</div>

<script>
  const track = document.getElementById('track');
  const cards = Array.from(track.querySelectorAll('.card'));
  const chips = Array.from(document.querySelectorAll('.chip'));

  function setActive(i) {
    chips.forEach((c, k) => c.classList.toggle('active', k === i));
  }
  chips.forEach((c, i) => {
    c.addEventListener('click', () => {
      cards[i].scrollIntoView({behavior: 'smooth', block: 'nearest', inline: 'center'});
      setActive(i);
    });
  });
  track.addEventListener('scroll', () => {
    const w = track.clientWidth;
    setActive(Math.round(track.scrollLeft / w));
  }, { passive: true });
</script>
</body>
</html>
"""


def _chip(screen: dict) -> str:
    return (
        f'<div class="chip">'
        f'<span class="badge" style="background:{screen["alert_color"]}"></span>'
        f'{screen["station"]}'
        f'</div>'
    )


def _card(screen: dict) -> str:
    rows = "".join(
        f'<div class="row">'
        f'<div><div class="label">{c["name"]}</div>'
        f'<div class="value">{c["current"]:.1f} {c["unit"]}</div></div>'
        f'<div>{c["sparkline"]}</div>'
        f'</div>'
        for c in screen["canaux"]
    )
    return (
        '<div class="card">'
        '<div class="screen">'
        f'<div class="station">{screen["station"]}</div>'
        f'<div class="region">{screen["region"]} · {screen["altitude"]:.0f} m</div>'
        f'<div class="stamp">{screen["timestamp"]}</div>'
        f'<div class="alert-banner" style="background:{screen["alert_color"]}">'
        f'<div class="alert-title">{screen["alert_label"].upper()}</div>'
        f'<div class="alert-proba">p(orage 24 h) = {screen["proba"]:.2f}</div>'
        f'<div class="alert-seuil">(seuil {screen["threshold"]:.2f})</div>'
        f'</div>'
        f'{rows}'
        '</div></div>'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "taranis_app_full.html"))
    args = p.parse_args()

    pred = Predictor(ROOT / "runs" / "probe" / "ww_rich")

    print(f"Récupération de {len(STATIONS)} stations, dernière fenêtre disponible...")
    screens_now = []
    for sid, region in STATIONS:
        try:
            s = build_screen(pred, sid, region)
        except Exception as e:  # pragma: no cover - depends on network
            print(f"  ✗ {sid} ({region}) : {e}")
            continue
        if s:
            screens_now.append(s)
            print(f"  ✓ {s['station']:16s}  p={s['proba']:.3f}  {s['alert'].upper()}")

    print(f"\nRécupération de {len(HISTORIQUES)} cas historiques d'orage...")
    screens_hist = []
    for sid, dt, region in HISTORIQUES:
        try:
            s = build_screen(pred, sid, region, end=dt)
        except Exception as e:  # pragma: no cover - depends on network
            print(f"  ✗ {sid} ({region}) : {e}")
            continue
        if s:
            screens_hist.append(s)
            print(f"  ✓ {s['station']:16s}  p={s['proba']:.3f}  {s['alert'].upper()}")

    # final order: historical storm cases (visual hook) first, then the
    # current state sorted by decreasing altitude.
    screens_hist.sort(key=lambda s: -s["proba"])
    screens_now.sort(key=lambda s: -s["altitude"])
    screens = screens_hist + screens_now

    html = HTML_TEMPLATE
    html = html.replace("__CHIPS__", "".join(_chip(s) for s in screens))
    html = html.replace("__CARDS__", "".join(_card(s) for s in screens))
    Path(args.out).write_text(html, encoding="utf-8")

    summary = [{"id": s["station_id"], "nom": s["station"], "alerte": s["alert"],
                "proba": round(s["proba"], 3)} for s in screens]
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nÉcrit : {args.out}")


if __name__ == "__main__":
    main()
