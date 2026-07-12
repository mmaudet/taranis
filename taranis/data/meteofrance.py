"""Real-data loader for Meteo-France SYNOP format.

Source: Opendatasoft portal "donnees-synop-essentielles-omm", which hosts
a copy of the official Meteo-France SYNOP data, refreshed continuously and
accessible without a key.

- Typical frequency: 3 hours (sometimes hourly for recent stations).
- Channels used: `pres` (pressure, Pa), `tc` (temperature, deg C), `u`
  (relative humidity, %), `ff` (wind speed, m/s).
- Proxy label: `rr1` (rainfall over the last hour, mm) is used to build a
  heavy-rain event label in the horizon.

Unlike the synthetic data, the real data exhibit:

- sporadic **missing values** (NaN),
- **heterogeneous frequency** across stations,
- **strong seasonality** tied to the actual calendar,
- **atypical regimes** that the synthetic generator does not model.

The loader returns one DataFrame per station, aligned on a regular time
grid (default resampling at 3h), with missing values interpolated or flagged
according to the chosen strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# "Base" channels used by baseline M0 and TS-JEPA v1
CANAUX_MF = ("pressure", "temp", "humidity", "wind")

# "Rich" channels for step 9: we add the 10-min gust, very informative for
# a pre-storm situation. Baseline M0 does not use them (it reads only the
# first 4 channels), but TS-JEPA sees them.
CANAUX_MF_RICH = ("pressure", "temp", "humidity", "wind", "wind_gust")

_COLONNES_BRUT = {
    "pres": "pressure_pa",   # Pa
    "pmer": "pressure_mer_pa",  # Pa, sea level
    "tc": "temp",            # deg C
    "u": "humidity",         # %
    "ff": "wind",            # m/s
    "rr1": "rain_1h",        # mm
    "rr3": "rain_3h",        # mm
    "raf10": "wind_gust",    # m/s, 10-min gust
    "dd": "wind_dir",        # degrees
    "ww": "weather_code",    # WMO code 4677 (present weather)
    "date": "timestamp",
    "numer_sta": "station_id",
    "nom": "station_name",
    "altitude": "altitude_m",
}

# WMO 4677 codes corresponding to an observed storm, strictly speaking
# https://library.wmo.int/records/item/35713 (table 4677)
# - 17: storm without precipitation at station
# - 29: storm within the last hour, with or without precipitation
# - 91 to 94: light to heavy rain with a recent storm
# - 95 to 99: storm at observation time (95 light, 97 heavy, 99 hail)
CODES_ORAGE_WMO = (17, 29, 91, 92, 93, 94, 95, 96, 97, 98, 99)


@dataclass(frozen=True)
class StationInfo:
    id: str
    nom: str
    altitude_m: float


def read_synop_csv(
    path: str | Path,
    stations: tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read a downloaded SYNOP CSV and return one DataFrame per station.

    Expected input format: semicolon-separated CSV with the raw Opendatasoft
    columns (`numer_sta`, `nom`, `altitude`, `date`, `pres`, `tc`, `u`, `ff`, ...).

    Returns
    -------
    dict[station_id -> DataFrame], each DataFrame chronologically indexed,
    with renamed columns and the `pressure` channel in hPa.
    """
    df = pd.read_csv(path, sep=";", dtype={"numer_sta": str})
    df["numer_sta"] = df["numer_sta"].str.zfill(5)  # preserve leading zero
    df = df.rename(columns=_COLONNES_BRUT)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    # pressures Pa -> hPa, optional columns
    df["pressure"] = df["pressure_pa"] / 100.0
    df = df.drop(columns=["pressure_pa"])
    if "pressure_mer_pa" in df.columns:
        df["pressure_mer"] = df["pressure_mer_pa"] / 100.0
        df = df.drop(columns=["pressure_mer_pa"])

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    if stations is not None:
        df = df[df["station_id"].astype(str).isin(stations)]

    out = {}
    for sid, sub in df.groupby("station_id"):
        sub = sub.reset_index(drop=True)
        out[str(sid)] = sub
    return out


def resample_regular(
    df: pd.DataFrame,
    freq: str = "3h",
    max_gap_hours: int = 6,
) -> pd.DataFrame:
    """Resample a station DataFrame onto a regular grid.

    - Aligns on the `freq` grid (default 3h, aligned on UTC hours 00, 03, ...).
    - Missing measurements are linearly interpolated for small gaps
      (up to `max_gap_hours`). Beyond that they remain NaN and will be
      filtered out at the windowing stage.

    Columns kept: timestamp, pressure, temp, humidity, wind, rain_1h.
    """
    # columns to keep if available (the file may not contain everything)
    optional = ["pressure_mer", "wind_gust", "rain_3h", "wind_dir", "weather_code"]
    keep = ["timestamp", "pressure", "temp", "humidity", "wind", "rain_1h"]
    for c in optional:
        if c in df.columns:
            keep.append(c)
    d = df[keep].copy()
    d = d.set_index("timestamp").sort_index()
    # deduplicate identical timestamps (rare but present on some stations)
    d = d[~d.index.duplicated(keep="first")]
    # regular grid
    idx = pd.date_range(
        start=d.index.min().floor(freq),
        end=d.index.max().ceil(freq),
        freq=freq,
    )
    d = d.reindex(idx)
    # limited interpolation on continuous physical channels
    limit = max(1, max_gap_hours // int(freq.rstrip("h")))
    continu = [c for c in ["pressure", "temp", "humidity", "wind",
                           "pressure_mer", "wind_gust"] if c in d.columns]
    d[continu] = d[continu].interpolate(method="time", limit=limit, limit_area="inside")
    # missing gusts are somewhat suspicious; as a fallback we keep the wind
    if "wind_gust" in d.columns:
        d["wind_gust"] = d["wind_gust"].fillna(d["wind"])
    # rain_1h: missing values are left as NaN (we do not want to invent rain)
    d = d.reset_index().rename(columns={"index": "timestamp"})
    return d


def build_storm_labels(
    df: pd.DataFrame,
    seuil_mm: float = 2.0,
    duration_steps: int = 1,
) -> pd.DataFrame:
    """Label heavy-rain events as a storm proxy.

    An event is considered active when hourly rainfall exceeds `seuil_mm`.
    `storm_onset` marks the first time step of the event, `storm_active`
    covers `duration_steps` time steps starting from the onset (to represent
    the event's duration).

    This label is a proxy: we do not claim to identify convective storms
    strictly speaking. Heavy hourly rainfall is a convenient marker that is
    available everywhere.
    """
    d = df.copy()
    rr = d["rain_1h"].fillna(0.0)
    is_strong = rr > seuil_mm
    # onset: transition non-strong -> strong
    d["storm_active"] = is_strong.values
    onset = is_strong & ~is_strong.shift(1, fill_value=False)
    d["storm_onset"] = onset.values

    if duration_steps > 1:
        # extend storm_active over `duration_steps` steps after each onset
        active = np.zeros(len(d), dtype=bool)
        onset_idx = np.flatnonzero(onset.values)
        for i in onset_idx:
            active[i : min(len(d), i + duration_steps)] = True
        d["storm_active"] = active
    return d


def build_storm_labels_from_ww(
    df: pd.DataFrame,
    window_before: int = 1,
    window_after: int = 1,
) -> pd.DataFrame:
    """Storm label from the WMO 4677 **present weather code**.

    A storm is observed at instant `t` when `weather_code` is in
    `CODES_ORAGE_WMO`. This is a point observation; we then **extend** the
    active window by `window_before` steps before and `window_after` steps
    after to represent the typical event duration (about 3 hours at the
    SYNOP 3h step).

    The onset is the **first** step of a group of consecutive storm
    observations, optionally extended.

    Cleaner than the rain proxy but rarer: observers only report a storm at
    the station at the exact observation moment, so short daytime storms
    passing between two SYNOP tops can be missed.
    """
    d = df.copy()
    if "weather_code" not in d.columns:
        raise ValueError(
            "Colonne 'weather_code' absente. "
            "Régénérer le dataset brut en incluant la colonne 'ww' du SYNOP."
        )
    ww = d["weather_code"]
    is_storm = ww.isin(CODES_ORAGE_WMO).fillna(False)

    # time extension: broaden each storm observation
    active = np.zeros(len(d), dtype=bool)
    for i in np.flatnonzero(is_storm.values):
        lo = max(0, i - window_before)
        hi = min(len(d), i + window_after + 1)
        active[lo:hi] = True
    d["storm_active"] = active
    # onset: transition inactive -> active
    onset = np.zeros(len(d), dtype=bool)
    onset[1:] = active[1:] & ~active[:-1]
    onset[0] = active[0]
    d["storm_onset"] = onset
    return d


def build_storm_labels_combined(
    df: pd.DataFrame,
    rain_seuil_mm: float = 5.0,
    window_before: int = 1,
    window_after: int = 1,
) -> pd.DataFrame:
    """Union of WMO storm code labels and heavy-rain labels.

    An event is active if (a) the WMO present-weather code is in
    `CODES_ORAGE_WMO`, OR (b) hourly rain exceeds `rain_seuil_mm`.

    The union catches short daytime storms that fall between two SYNOP tops
    (the ww code is instantaneous, and the observer is not always at the
    station within 3 hours). The rain threshold here is **stricter** than
    the plain rain proxy (5 mm instead of 2), to avoid polluting with plain
    stratiform rain.

    The `window_before/window_after` extension applies around each storm
    observation.
    """
    d = df.copy()
    if "weather_code" not in d.columns:
        raise ValueError(
            "Colonne 'weather_code' absente. "
            "Régénérer le dataset brut en incluant la colonne 'ww' du SYNOP."
        )
    ww = d["weather_code"]
    is_ww_storm = ww.isin(CODES_ORAGE_WMO).fillna(False)
    rr = d["rain_1h"].fillna(0.0)
    is_heavy_rain = rr > rain_seuil_mm
    is_event = is_ww_storm | is_heavy_rain

    active = np.zeros(len(d), dtype=bool)
    for i in np.flatnonzero(is_event.values):
        lo = max(0, i - window_before)
        hi = min(len(d), i + window_after + 1)
        active[lo:hi] = True
    d["storm_active"] = active
    onset = np.zeros(len(d), dtype=bool)
    onset[1:] = active[1:] & ~active[:-1]
    onset[0] = active[0]
    d["storm_onset"] = onset
    return d


def prepare_station(
    raw_df: pd.DataFrame,
    freq: str = "3h",
    rain_seuil_mm: float = 2.0,
    label_source: str = "rain",
    ww_window_before: int = 1,
    ww_window_after: int = 1,
) -> pd.DataFrame:
    """Full per-station pipeline: resample + label.

    `label_source`:
      - `"rain"` (default): heavy-rain proxy, `rain_1h > rain_seuil_mm`.
      - `"ww"`            : true WMO 4677 storm codes (17, 29, 91-99).
      - `"combined"`      : union of "ww" and hourly rain > 5 mm.

    Returns a DataFrame ready to be windowed via
    `taranis.data.windows.make_windows`.
    """
    d = resample_regular(raw_df, freq=freq)
    if label_source == "rain":
        d = build_storm_labels(d, seuil_mm=rain_seuil_mm, duration_steps=1)
    elif label_source == "ww":
        d = build_storm_labels_from_ww(
            d, window_before=ww_window_before, window_after=ww_window_after
        )
    elif label_source == "combined":
        d = build_storm_labels_combined(
            d,
            rain_seuil_mm=max(rain_seuil_mm, 5.0),
            window_before=ww_window_before,
            window_after=ww_window_after,
        )
    else:
        raise ValueError(f"label_source inconnu : {label_source}")
    return d


def summarize_stations(stations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Small summary table: covered duration, density, altitude, onsets."""
    rows = []
    for sid, sub in stations.items():
        rows.append(
            {
                "station_id": sid,
                "nom": sub["station_name"].iloc[0] if "station_name" in sub else sid,
                "altitude_m": (
                    sub["altitude_m"].iloc[0] if "altitude_m" in sub else np.nan
                ),
                "n_records": len(sub),
                "date_min": sub["timestamp"].min(),
                "date_max": sub["timestamp"].max(),
            }
        )
    return pd.DataFrame(rows)
