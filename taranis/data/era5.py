"""Chargeur ERA5 (Copernicus, ECMWF Reanalysis v5).

ERA5 est la réanalyse horaire globale de référence, publiée par le Copernicus
Climate Change Service (C3S) sur infrastructure européenne. Elle est physique
plutôt qu'observationnelle, homogène, sans manquants, et disponible depuis
1940.

**Prérequis utilisateur** :

1. Compte gratuit sur https://cds.climate.copernicus.eu/
2. Fichier `~/.cdsapirc` avec le token API (voir la page how-to-api)
3. Acceptation des conditions du dataset `reanalysis-era5-single-levels`
   (case à cocher, une fois pour toutes, sur la fiche dataset).

Nos conventions :

- On demande **quatre variables surface** correspondant à nos canaux :
    * `surface_pressure` (Pa)
    * `2m_temperature` (K)
    * `2m_dewpoint_temperature` (K) → sert à calculer l'humidité relative
    * `10m_u_component_of_wind` et `10m_v_component_of_wind` (m/s) → module = vent
- On récupère aussi `total_precipitation` (m, cumul horaire) pour l'étiquette
  proxy pluie forte.
- Résolution 0.25° x 0.25°.
- On récupère un ou plusieurs **points de grille** (lat, lon) et on retourne
  un DataFrame par point, format compatible avec `taranis.data.windows`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CANAUX_ERA5 = ("pressure", "temp", "humidity", "wind")

# Constantes pour l'humidité relative à partir de la point de rosée
# Formule de Magnus-Tetens
_A, _B = 17.625, 243.04


def relative_humidity_from_dewpoint(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Humidité relative en %, à partir de température et point de rosée en °C."""
    e_s = np.exp(_A * t_c / (_B + t_c))
    e = np.exp(_A * td_c / (_B + td_c))
    return np.clip(100.0 * e / e_s, 0.0, 100.0)


@dataclass(frozen=True)
class GridPoint:
    lat: float
    lon: float
    nom: str


def fetch_era5(
    grid_points: Iterable[GridPoint],
    years: Iterable[int],
    out_dir: Path,
    verbose: bool = True,
) -> list[Path]:
    """Télécharge un fichier ERA5 par (point, année), format NetCDF.

    Retourne la liste des chemins écrits.

    Nécessite `cdsapi` et un token valide dans `~/.cdsapirc`.
    """
    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit(
            "cdsapi n'est pas installé. Lancer :\n"
            "  uv add --group era5 cdsapi netCDF4 xarray"
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    client = cdsapi.Client(quiet=not verbose)

    for pt in grid_points:
        for year in years:
            path = out_dir / f"era5_{pt.nom}_{year}.nc"
            if path.exists():
                if verbose:
                    print(f"  [skip] {path.name} existe déjà")
                written.append(path)
                continue
            if verbose:
                print(f"  [fetch] {pt.nom} {year}")
            request = {
                "product_type": ["reanalysis"],
                "format": "netcdf",
                "variable": [
                    "surface_pressure",
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "total_precipitation",
                ],
                "year": [str(year)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                # bounding-box d'un unique pixel autour du point
                "area": [pt.lat + 0.125, pt.lon - 0.125, pt.lat - 0.125, pt.lon + 0.125],
            }
            client.retrieve("reanalysis-era5-single-levels", request, str(path))
            written.append(path)
    return written


def load_era5_files(paths: Iterable[Path]) -> dict[str, pd.DataFrame]:
    """Charge un ou plusieurs fichiers NetCDF ERA5, retourne un DataFrame par
    point de grille (identifié par sa position `lat/lon`).

    Colonnes de sortie : `timestamp, pressure (hPa), temp (°C), humidity (%),
    wind (m/s), rain_1h (mm), lat, lon`.
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise SystemExit("xarray requis pour lire les NetCDF ERA5") from e

    by_point: dict[str, list[pd.DataFrame]] = {}
    for p in paths:
        ds = xr.open_dataset(p)
        df = _dataset_to_frame(ds)
        key = f"{df['lat'].iloc[0]:.3f}_{df['lon'].iloc[0]:.3f}"
        by_point.setdefault(key, []).append(df)

    return {k: pd.concat(v).sort_values("timestamp").reset_index(drop=True)
            for k, v in by_point.items()}


def _dataset_to_frame(ds) -> pd.DataFrame:
    """Convertit un xarray.Dataset ERA5 en DataFrame Taranis."""
    lat = float(ds["latitude"].values.mean())
    lon = float(ds["longitude"].values.mean())
    t_k = ds["t2m"].values.squeeze()
    td_k = ds["d2m"].values.squeeze()
    p_pa = ds["sp"].values.squeeze()
    u = ds["u10"].values.squeeze()
    v = ds["v10"].values.squeeze()
    tp_m = ds["tp"].values.squeeze()  # cumul horaire, mètres
    times = pd.to_datetime(ds["time"].values)

    t_c = t_k - 273.15
    td_c = td_k - 273.15
    return pd.DataFrame(
        {
            "timestamp": times,
            "pressure": p_pa / 100.0,             # hPa
            "temp": t_c,
            "humidity": relative_humidity_from_dewpoint(t_c, td_c),
            "wind": np.hypot(u, v),
            "rain_1h": tp_m * 1000.0,             # m -> mm
            "lat": lat,
            "lon": lon,
        }
    )
