"""Generate a self-contained mobile-first HTML page with 3 navigable Taranis screens.

Output: `taranis_app.html` (or the given path). Openable in any mobile
browser, no network connection required.

Contents:
- inline mobile-first CSS,
- JS navigation by swipe and buttons,
- three screens with real historical SYNOP measurements,
- SVG sparklines rendered on the Python side for reproducibility,
- real probabilities computed by the Predictor.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

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


def _sparkline_svg(values, color, width=280, height=48, pad=4) -> str:
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-6)
    n = len(values)
    step_x = (width - 2 * pad) / max(n - 1, 1)
    pts = []
    for i, v in enumerate(values):
        x = pad + i * step_x
        y = height - pad - (v - lo) / span * (height - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    path = " ".join(pts)
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="none" width="100%" height="{height}">'
        f'<polyline points="{path}" fill="none" stroke="{color}" '
        f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
        f"</svg>"
    )


def build_screen(station_id: str, end: datetime, label: str, pred: Predictor) -> dict:
    import numpy as np

    csv_text = _fetch(station_id, end)
    stations = read_synop_csv(StringIO(csv_text))
    (_sid, raw), = stations.items()
    d = prepare_station(raw, freq="3h")
    d = d.dropna(subset=list(CANAUX_MF_RICH)).reset_index(drop=True)
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
        "id": label,
        "station": raw["station_name"].iloc[0],
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
<title>Taranis · Démo mobile</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; background: #ECEFF1;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                             "Helvetica Neue", Arial, sans-serif; color: #212121; }
  .app { max-width: 480px; margin: 0 auto; min-height: 100vh; position: relative;
         padding: env(safe-area-inset-top, 0) 0 env(safe-area-inset-bottom, 0); }
  header { padding: 18px 20px 4px; text-align: center; }
  header h1 { margin: 0; font-size: 22px; letter-spacing: 4px; font-weight: 700;
              color: #263238; }
  header .subtitle { margin-top: 4px; font-size: 12px; color: #90A4AE; }

  .track { display: flex; overflow-x: auto; scroll-snap-type: x mandatory;
           -webkit-overflow-scrolling: touch; scrollbar-width: none; padding: 12px 0 0; }
  .track::-webkit-scrollbar { display: none; }
  .card { flex: 0 0 100%; scroll-snap-align: center; padding: 8px 16px 24px; }

  .screen { background: #fff; border-radius: 20px; padding: 22px 22px 24px;
            box-shadow: 0 6px 22px rgba(0,0,0,.08); }
  .station { text-align: center; margin: 0 0 4px; font-size: 14px; color: #607D8B; }
  .altitude { text-align: center; margin: 0 0 2px; font-size: 12px; color: #90A4AE; }
  .stamp { text-align: center; margin: 0 0 16px; font-size: 11px; color: #B0BEC5; }

  .alert-banner { border-radius: 18px; padding: 18px 12px 16px; color: white;
                   text-align: center; margin-bottom: 20px; }
  .alert-title { font-size: 20px; font-weight: 800; letter-spacing: 1px; }
  .alert-proba { margin-top: 6px; font-size: 13px; opacity: .95; }
  .alert-seuil { margin-top: 2px; font-size: 10px; opacity: .8; }

  .row { display: grid; grid-template-columns: 110px 1fr; align-items: center;
         gap: 10px; padding: 10px 0; border-bottom: 1px dashed #ECEFF1; }
  .row:last-child { border-bottom: none; }
  .label { font-size: 12px; color: #78909C; }
  .value { font-size: 16px; font-weight: 700; color: #212121; margin-top: 2px; }
  .spark { width: 100%; height: 48px; }

  .footer { text-align: center; font-size: 10px; color: #90A4AE; padding: 12px 24px 24px;
            font-style: italic; }
  .dots { display: flex; justify-content: center; gap: 10px; padding: 8px 0 4px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #CFD8DC;
         transition: transform .18s, background .18s; }
  .dot.active { background: #455A64; transform: scale(1.3); }
  .nav-hint { text-align: center; font-size: 11px; color: #B0BEC5; margin-top: 4px;
              padding: 0 20px; }
  @media (min-width: 640px) {
    header h1 { font-size: 26px; }
    .screen { padding: 24px 28px 28px; }
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>TARANIS</h1>
    <div class="subtitle">Démo · Nowcasting orage</div>
  </header>
  <div class="dots" id="dots"></div>
  <div class="nav-hint">Balaie de gauche à droite ← pour changer d'exemple →</div>

  <div class="track" id="track">__CARDS__</div>

  <div class="footer">
    Aide à la décision, pas une garantie.<br>
    Ne remplace pas les bulletins officiels ni le jugement de l'utilisateur.<br>
    En montagne, un faux négatif peut être mortel.
  </div>
</div>

<script>
  const track = document.getElementById('track');
  const cards = track.querySelectorAll('.card');
  const dotsHost = document.getElementById('dots');
  cards.forEach((_, i) => {
    const d = document.createElement('div');
    d.className = 'dot' + (i === 0 ? ' active' : '');
    dotsHost.appendChild(d);
  });
  const dots = dotsHost.querySelectorAll('.dot');
  track.addEventListener('scroll', () => {
    const w = track.clientWidth;
    const i = Math.round(track.scrollLeft / w);
    dots.forEach((d, k) => d.classList.toggle('active', k === i));
  }, { passive: true });
</script>
</body>
</html>
"""


def _card_html(screen: dict) -> str:
    rows = "".join(
        f'<div class="row">'
        f'<div><div class="label">{c["name"]}</div>'
        f'<div class="value">{c["current"]:.1f} {c["unit"]}</div></div>'
        f'<div class="spark">{c["sparkline"]}</div>'
        f'</div>'
        for c in screen["canaux"]
    )
    return (
        '<div class="card">'
        '<div class="screen">'
        f'<div class="station">{screen["station"]}</div>'
        f'<div class="altitude">altitude {screen["altitude"]:.0f} m</div>'
        f'<div class="stamp">{screen["timestamp"]}</div>'
        f'<div class="alert-banner" style="background:{screen["alert_color"]}">'
        f'  <div class="alert-title">{screen["alert_label"].upper()}</div>'
        f'  <div class="alert-proba">p(orage 24 h) = {screen["proba"]:.2f}</div>'
        f'  <div class="alert-seuil">(seuil de référence {screen["threshold"]:.2f})</div>'
        f'</div>'
        f'{rows}'
        '</div>'
        '</div>'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "taranis_app.html"))
    args = p.parse_args()

    pred = Predictor(ROOT / "runs" / "probe" / "ww_rich")

    scenarios = [
        ("07591", datetime(2026, 1, 15, tzinfo=UTC), "vert"),
        ("07690", datetime(2025, 9, 11, tzinfo=UTC), "orange"),
        ("07690", datetime(2025, 9, 10, tzinfo=UTC), "rouge"),
    ]
    print("Construction des 3 écrans, chaque appel télécharge quelques jours de SYNOP...")
    screens = [build_screen(sid, dt, lbl, pred) for sid, dt, lbl in scenarios]

    html = HTML_TEMPLATE.replace("__CARDS__", "".join(_card_html(s) for s in screens))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Écrit : {args.out}")
    print("\nÉcrans :")
    for s in screens:
        print(f"  {s['id'].upper():7s}  {s['station']:15s}  p={s['proba']:.3f}  {s['timestamp']}")
    print("\nOuvrir avec n'importe quel navigateur (mobile ou desktop).")
    print("Debug JSON :")
    print(json.dumps({s['id']: {'proba': s['proba'], 'alert': s['alert']} for s in screens}, indent=2))


if __name__ == "__main__":
    main()
