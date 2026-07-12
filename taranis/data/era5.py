"""ERA5 loader (Copernicus, ECMWF Reanalysis v5).

ERA5 is the reference hourly global reanalysis, published by the Copernicus
Climate Change Service (C3S) on European infrastructure. It is physical
rather than observational, homogeneous, without missing values, and
available since 1940.

**User prerequisites**:

1. Free account at https://cds.climate.copernicus.eu/
2. `~/.cdsapirc` file holding the API token (see the how-to-api page).
3. Accept the terms of the `reanalysis-era5-single-levels` dataset
   (checkbox on the dataset page, once and for all).

Our conventions:

- We request **four surface variables** matching our channels:
    * `surface_pressure` (Pa)
    * `2m_temperature` (K)
    * `2m_dewpoint_temperature` (K), used to compute relative humidity.
    * `10m_u_component_of_wind` and `10m_v_component_of_wind` (m/s); the
      modulus gives the wind speed.
- We also grab `total_precipitation` (m, hourly cumulative) for the heavy
  rain proxy label.
- Resolution 0.25 deg x 0.25 deg.
- We fetch one or several **grid points** (lat, lon) and return one
  DataFrame per point, compatible with `taranis.data.windows`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CANAUX_ERA5 = ("pressure", "temp", "humidity", "wind")

# Constants for relative humidity from dew point
# Magnus-Tetens formula
_A, _B = 17.625, 243.04


def relative_humidity_from_dewpoint(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Relative humidity in %, from temperature and dew point in deg C."""
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
    """Download one ERA5 file per (point, year, month) in NetCDF format.

    We split by MONTH to stay within the CDS cost limits (tightened at the
    end of 2025, ~120k items per request). Each file has ~700 hourly obs
    x 6 variables x 1 pixel = ~4200 items, well below the ceiling.

    Returns: list of written paths.

    Requires `cdsapi` and a valid token in `~/.cdsapirc`.
    """
    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit(
            "cdsapi n'est pas installé. Lancer :\n"
            "  uv sync --extra era5"
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    client = cdsapi.Client(quiet=not verbose)

    for pt in grid_points:
        for year in years:
            for month in range(1, 13):
                path = out_dir / f"era5_{pt.nom}_{year}_{month:02d}.nc"
                if path.exists():
                    if verbose:
                        print(f"  [skip] {path.name}")
                    written.append(path)
                    continue
                if verbose:
                    print(f"  [fetch] {pt.nom} {year}-{month:02d}")
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
                    "month": [f"{month:02d}"],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    # bounding box of a single pixel around the point
                    "area": [pt.lat + 0.125, pt.lon - 0.125, pt.lat - 0.125, pt.lon + 0.125],
                }
                client.retrieve("reanalysis-era5-single-levels", request, str(path))
                written.append(path)
    return written


def load_era5_files(paths: Iterable[Path]) -> dict[str, pd.DataFrame]:
    """Load one or several ERA5 NetCDF files, one DataFrame per grid point.

    Each grid point is identified by its `lat/lon` position.

    Output columns: `timestamp, pressure (hPa), temp (deg C), humidity (%),
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
    """Convert an ERA5 xarray.Dataset into a Taranis DataFrame."""
    lat = float(ds["latitude"].values.mean())
    lon = float(ds["longitude"].values.mean())
    t_k = ds["t2m"].values.squeeze()
    td_k = ds["d2m"].values.squeeze()
    p_pa = ds["sp"].values.squeeze()
    u = ds["u10"].values.squeeze()
    v = ds["v10"].values.squeeze()
    tp_m = ds["tp"].values.squeeze()  # hourly cumulative, meters
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
