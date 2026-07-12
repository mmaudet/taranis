"""Chargeur de données réelles Météo-France, format SYNOP.

Source : portail Opendatasoft "donnees-synop-essentielles-omm", qui héberge
une copie des données SYNOP officielles de Météo-France, rafraîchies en
continu et accessibles sans clé.

- Fréquence typique : 3 heures (parfois horaire pour certaines stations récentes).
- Canaux utilisés : `pres` (pression, Pa), `tc` (température, °C), `u`
  (humidité relative, %), `ff` (vitesse du vent, m/s).
- Label proxy : `rr1` (précipitations sur la dernière heure, mm) sert à
  construire une étiquette d'événement pluvieux fort dans l'horizon.

Contrairement au synthétique, les données réelles présentent :

- des **valeurs manquantes** (NaN) sporadiques,
- une **fréquence hétérogène** selon les stations,
- une **saisonnalité forte** liée au calendrier réel,
- des **régimes atypiques** que le générateur synthétique ne modélise pas.

Le chargeur retourne un DataFrame par station, aligné sur une grille de temps
régulière (rééchantillonnage à 3h par défaut), avec les manquants interpolés
ou marqués selon la stratégie choisie.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Canaux "de base" utilisés par la baseline M0 et le TS-JEPA v1
CANAUX_MF = ("pressure", "temp", "humidity", "wind")

# Canaux "enrichis" pour l'étape 9 : on ajoute la rafale à 10 min, très
# informative pour un pré-orage. La baseline M0 ne les utilise pas (elle lit
# les 4 premiers canaux seulement), mais TS-JEPA les voit.
CANAUX_MF_RICH = ("pressure", "temp", "humidity", "wind", "wind_gust")

_COLONNES_BRUT = {
    "pres": "pressure_pa",   # Pa
    "pmer": "pressure_mer_pa",  # Pa, niveau mer
    "tc": "temp",            # °C
    "u": "humidity",         # %
    "ff": "wind",            # m/s
    "rr1": "rain_1h",        # mm
    "rr3": "rain_3h",        # mm
    "raf10": "wind_gust",    # m/s, rafale sur 10 min
    "dd": "wind_dir",        # degrés
    "ww": "weather_code",    # code WMO 4677 (temps présent)
    "date": "timestamp",
    "numer_sta": "station_id",
    "nom": "station_name",
    "altitude": "altitude_m",
}

# codes WMO 4677 correspondant à un orage observé, au sens strict
# https://library.wmo.int/records/item/35713 (table 4677)
# - 17 : orage sans précipitation à la station
# - 29 : orage à la dernière heure, avec ou sans précipitation
# - 91 à 94 : pluie légère à forte avec orage récent
# - 95 à 99 : orage à l'observation (95 léger, 97 fort, 99 grêle)
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
    """Lit un CSV SYNOP téléchargé, retourne un DataFrame par station.

    Format d'entrée attendu : CSV point-virgule, colonnes brutes Opendatasoft
    (`numer_sta`, `nom`, `altitude`, `date`, `pres`, `tc`, `u`, `ff`, ...).

    Retour
    ------
    dict[station_id -> DataFrame], chaque DataFrame indexé chronologiquement,
    avec les colonnes renommées et le canal `pressure` en hPa.
    """
    df = pd.read_csv(path, sep=";", dtype={"numer_sta": str})
    df["numer_sta"] = df["numer_sta"].str.zfill(5)  # préserve le zéro de tête
    df = df.rename(columns=_COLONNES_BRUT)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    # pressions Pa -> hPa, colonnes optionnelles
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
    """Rééchantillonne un DataFrame station sur une grille régulière.

    - On aligne sur la grille `freq` (par défaut 3h, aligné sur les heures UTC 00, 03, ...).
    - Les mesures manquantes sont interpolées linéairement pour de petits trous
      (jusqu'à `max_gap_hours`). Au delà, elles restent NaN et seront filtrées
      côté fenêtrage.

    Colonnes conservées : timestamp, pressure, temp, humidity, wind, rain_1h.
    """
    # colonnes à garder si disponibles (le fichier peut ne pas tout contenir)
    optional = ["pressure_mer", "wind_gust", "rain_3h", "wind_dir", "weather_code"]
    keep = ["timestamp", "pressure", "temp", "humidity", "wind", "rain_1h"]
    for c in optional:
        if c in df.columns:
            keep.append(c)
    d = df[keep].copy()
    d = d.set_index("timestamp").sort_index()
    # dédoublonnage des timestamps identiques (rare mais présent sur certaines stations)
    d = d[~d.index.duplicated(keep="first")]
    # grille régulière
    idx = pd.date_range(
        start=d.index.min().floor(freq),
        end=d.index.max().ceil(freq),
        freq=freq,
    )
    d = d.reindex(idx)
    # interpolation limitée sur les canaux physiques continus
    limit = max(1, max_gap_hours // int(freq.rstrip("h")))
    continu = [c for c in ["pressure", "temp", "humidity", "wind",
                           "pressure_mer", "wind_gust"] if c in d.columns]
    d[continu] = d[continu].interpolate(method="time", limit=limit, limit_area="inside")
    # les rafales manquantes sont un peu suspectes ; à défaut, on garde le vent
    if "wind_gust" in d.columns:
        d["wind_gust"] = d["wind_gust"].fillna(d["wind"])
    # rain_1h : les manquants sont laissés NaN (on ne veut pas inventer de pluie)
    d = d.reset_index().rename(columns={"index": "timestamp"})
    return d


def build_storm_labels(
    df: pd.DataFrame,
    seuil_mm: float = 2.0,
    duration_steps: int = 1,
) -> pd.DataFrame:
    """Étiquette les événements pluvieux forts comme proxy d'orage.

    On considère qu'un événement est actif si les précipitations horaires
    dépassent `seuil_mm`. `storm_onset` marque le premier pas de temps de
    l'événement, `storm_active` couvre `duration_steps` pas de temps à
    partir de l'onset (pour représenter la durée de l'événement).

    Cette étiquette est un proxy, on ne prétend pas identifier des orages
    convectifs au sens strict. Les précipitations fortes horaires sont un
    marqueur pratique et disponible partout.
    """
    d = df.copy()
    rr = d["rain_1h"].fillna(0.0)
    is_strong = rr > seuil_mm
    # onset : transition non-strong -> strong
    d["storm_active"] = is_strong.values
    onset = is_strong & ~is_strong.shift(1, fill_value=False)
    d["storm_onset"] = onset.values

    if duration_steps > 1:
        # étendre le storm_active sur `duration_steps` pas après chaque onset
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
    """Étiquette d'orage à partir du **code temps présent** WMO 4677.

    Un orage est observé à un instant `t` si `weather_code` figure dans
    `CODES_ORAGE_WMO`. C'est une observation ponctuelle, on **étend** ensuite
    la fenêtre active de `window_before` pas avant et `window_after` pas
    après pour représenter la durée typique de l'événement (environ 3 heures
    au pas SYNOP de 3h).

    L'onset est le **premier** instant d'un groupe d'observations orageuses
    consécutives, éventuellement étendues.

    Nettement plus honnête que le proxy pluie, mais plus rare : les observateurs
    n'annoncent un orage à la station qu'au moment précis de l'observation, et
    manquent les orages qui passent entre deux tops horaires.
    """
    d = df.copy()
    if "weather_code" not in d.columns:
        raise ValueError(
            "Colonne 'weather_code' absente. "
            "Régénérer le dataset brut en incluant la colonne 'ww' du SYNOP."
        )
    ww = d["weather_code"]
    is_storm = ww.isin(CODES_ORAGE_WMO).fillna(False)

    # extension temporelle : étendre chaque observation orageuse
    active = np.zeros(len(d), dtype=bool)
    for i in np.flatnonzero(is_storm.values):
        lo = max(0, i - window_before)
        hi = min(len(d), i + window_after + 1)
        active[lo:hi] = True
    d["storm_active"] = active
    # onset : transition inactive -> active
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
    """Pipeline complet pour une station : rééchantillonne + étiquette.

    `label_source` :
      - `"rain"` (défaut) : proxy pluie forte, `rain_1h > rain_seuil_mm`.
      - `"ww"`             : vrai code orage WMO 4677 (codes 17, 29, 91-99).

    Retourne un DataFrame prêt à fenêtrer via `taranis.data.windows.make_windows`.
    """
    d = resample_regular(raw_df, freq=freq)
    if label_source == "rain":
        d = build_storm_labels(d, seuil_mm=rain_seuil_mm, duration_steps=1)
    elif label_source == "ww":
        d = build_storm_labels_from_ww(
            d, window_before=ww_window_before, window_after=ww_window_after
        )
    else:
        raise ValueError(f"label_source inconnu : {label_source}")
    return d


def summarize_stations(stations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Un petit tableau récapitulatif : durée couverte, densité, altitude, onsets."""
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
