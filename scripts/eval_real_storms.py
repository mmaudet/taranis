"""Quantitative evaluation of the native short probe on 15 real storms.

Selects 15 historical storm events (WMO ww code in CODES_ORAGE_WMO)
spread across several stations, seasons and regions. For each, goes back
5 days and forward 2 days, applies the probe on every sliding window,
and measures:

- first RED before onset: lead time in hours.
- first ORANGE before onset: lead time in hours.
- number of alerts around the event (persistence).

Output: summary table in the terminal + PNG figure.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from taranis.data.meteofrance import (
    CANAUX_MF_RICH,
    CODES_ORAGE_WMO,
    prepare_station,
    read_synop_csv,
)
from taranis.infer import Predictor

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "mf" / "synop_full.csv"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def find_storm_events(csv_path: Path, year_start=2024, n_target=15):
    """Select a diverse panel of historical storms from the SYNOP CSV.

    Selection criteria:
    - WMO ww code in CODES_ORAGE_WMO (17, 29, 91-99).
    - year >= year_start (test set).
    - station diversity (at most one per station when possible).
    - season diversity (summer, autumn, spring).
    """
    df = pd.read_csv(csv_path, sep=";", usecols=["numer_sta", "nom", "altitude", "date", "ww"],
                      dtype={"numer_sta": str}, low_memory=False)
    df["numer_sta"] = df["numer_sta"].str.zfill(5)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    storms = df[df["ww"].isin(CODES_ORAGE_WMO)].copy()
    storms = storms[storms["year"] >= year_start]

    # diverse pick: at most 3 per station, varied seasons
    storms = storms.sort_values("date").reset_index(drop=True)
    picked = []
    seen_stations = {}
    for _, row in storms.iterrows():
        sid = row["numer_sta"]
        if seen_stations.get(sid, 0) >= 2:
            continue
        picked.append(row)
        seen_stations[sid] = seen_stations.get(sid, 0) + 1
        if len(picked) >= n_target:
            break
    return pd.DataFrame(picked)


def fetch_around(station_id: str, event_date: datetime, days_before=5, days_after=2) -> str:
    """Download SYNOP data around an event."""
    url = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "donnees-synop-essentielles-omm/exports/csv"
    )
    start = event_date - timedelta(days=days_before)
    end = event_date + timedelta(days=days_after)
    r = requests.get(url, params={
        "where": (
            f"numer_sta='{station_id.zfill(5)}' "
            f"AND date >= '{start:%Y-%m-%d}' "
            f"AND date <= '{end:%Y-%m-%d}'"
        ),
        "select": ("numer_sta,nom,altitude,date,pres,pmer,tc,u,ff,dd,rr1,rr3,raf10,ww"),
        "delimiter": ";",
    }, timeout=60)
    r.raise_for_status()
    return r.text


def eval_one_event(pred: Predictor, station_id: str, station_name: str,
                    event_date: datetime) -> dict:
    """Timeline around an event with alerts and measured lead times."""
    csv_text = fetch_around(station_id, event_date)
    stations = read_synop_csv(StringIO(csv_text))
    if not stations:
        return None
    (_sid, raw), = stations.items()
    d = prepare_station(raw, freq="3h").dropna(subset=list(CANAUX_MF_RICH)).reset_index(drop=True)
    if len(d) < pred.config.Tw + 1:
        return None

    rows = []
    for i in range(pred.config.Tw, len(d) + 1):
        end_ts = d["timestamp"].iloc[i - 1]
        win = d.iloc[i - pred.config.Tw : i][list(CANAUX_MF_RICH)].to_numpy(dtype=np.float32)
        p = pred.predict_from_raw(win)
        rows.append({"ts": end_ts, "proba": p.proba, "alert": p.alert.value})
    tl = pd.DataFrame(rows)

    # lead time: last time the probe was RED or ORANGE strictly BEFORE event_date
    before = tl[tl["ts"] < event_date]
    def _lead(alert):
        m = before[before["alert"] == alert]
        if m.empty:
            return None
        first_ts = m["ts"].iloc[0]
        return (event_date - first_ts).total_seconds() / 3600.0

    return {
        "station_id": station_id,
        "station_name": station_name,
        "event_date": event_date,
        "n_fenetres": len(tl),
        "n_vert": int((tl["alert"] == "vert").sum()),
        "n_orange": int((tl["alert"] == "orange").sum()),
        "n_rouge": int((tl["alert"] == "rouge").sum()),
        "p_max": float(tl["proba"].max()),
        "p_moy": float(tl["proba"].mean()),
        "lead_orange_h": _lead("orange"),
        "lead_rouge_h": _lead("rouge"),
        "timeline": tl,
    }


def plot_evals(results, out_path):
    ok = [r for r in results if r]
    n = len(ok)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.4 * rows), sharey=True)
    axes = axes.flatten() if n > 1 else [axes]

    for ax, r in zip(axes, ok, strict=False):
        tl = r["timeline"]
        colors = tl["alert"].map({"vert": "#137333", "orange": "#E37400", "rouge": "#B00020"})
        ax.bar(tl["ts"], tl["proba"], width=0.11, color=colors, edgecolor="none")
        # thresholds are drawn from the first result's probe implicitly
        ax.axvline(r["event_date"], color="black", linestyle=":", linewidth=1.2)
        lead_r = f"{r['lead_rouge_h']:.0f}h" if r["lead_rouge_h"] else "--"
        lead_o = f"{r['lead_orange_h']:.0f}h" if r["lead_orange_h"] else "--"
        ax.set_title(f"{r['station_name']} · {r['event_date']:%Y-%m-%d %H:%M}\n"
                     f"préavis ORANGE {lead_o} / ROUGE {lead_r}",
                     fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis="x", labelsize=7, rotation=25)
    for ax in axes[len(ok):]:
        ax.set_visible(False)
    fig.suptitle("Sonde native short (Tw=8, H=6h) sur 15 orages réels 2024-2025",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="runs/probe/ww_short_native")
    ap.add_argument("--tag", default="short")
    args = ap.parse_args()
    pred = Predictor(ROOT / args.probe)
    print(f"Sonde : {args.probe}, Tw={pred.config.Tw}")
    print(f"Seuils : ORANGE={pred.orange_threshold:.3f}  ROUGE={pred.rouge_threshold:.3f}")

    print(f"\nPêche des orages dans {CSV}...")
    events = find_storm_events(CSV, year_start=2024, n_target=15)
    print(f"{len(events)} événements sélectionnés :")
    for _, e in events.iterrows():
        print(f"  {e['date']:%Y-%m-%d %H:%M}  station {e['numer_sta']} ({e['nom']}), ww={int(e['ww'])}")

    print("\nÉvaluation timeline par timeline (~30 s de réseau)...")
    results = []
    for _, e in events.iterrows():
        r = eval_one_event(pred, e["numer_sta"], e["nom"], e["date"].to_pydatetime())
        if r:
            results.append(r)
            lr = f"{r['lead_rouge_h']:>5.1f}h" if r['lead_rouge_h'] is not None else "  -- "
            lo = f"{r['lead_orange_h']:>5.1f}h" if r['lead_orange_h'] is not None else "  -- "
            print(f"  ✓ {r['station_name']:20s} {r['event_date']:%Y-%m-%d %H:%M}  "
                  f"lead_R={lr}  lead_O={lo}  pmax={r['p_max']:.3f}")

    # global stats
    print("\n=== Statistiques globales ===")
    lead_r = [r["lead_rouge_h"] for r in results if r["lead_rouge_h"] is not None]
    lead_o = [r["lead_orange_h"] for r in results if r["lead_orange_h"] is not None]
    detected_r = sum(1 for r in results if r["lead_rouge_h"] is not None)
    detected_o = sum(1 for r in results if r["lead_orange_h"] is not None)
    if lead_r:
        print(f"  ROUGE avant onset : {detected_r}/{len(results)} événements  "
              f"préavis médian {np.median(lead_r):.1f} h (min {min(lead_r):.1f}, max {max(lead_r):.1f})")
    else:
        print(f"  ROUGE avant onset : 0/{len(results)} événements")
    if lead_o:
        print(f"  ORANGE avant onset : {detected_o}/{len(results)} événements  "
              f"préavis médian {np.median(lead_o):.1f} h (min {min(lead_o):.1f}, max {max(lead_o):.1f})")
    else:
        print(f"  ORANGE avant onset : 0/{len(results)} événements")

    out_png = ASSETS / f"step13_real_storms_{args.tag}.png"
    plot_evals(results, out_png)
    print(f"\nFigure : {out_png}")


if __name__ == "__main__":
    main()
