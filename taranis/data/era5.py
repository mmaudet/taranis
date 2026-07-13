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

**Design choices (rev 2, chapter 21):**

- We fetch two datasets per (point, year):
    * `single-levels`: surface fields + storm-precursor indices
      (pressure, T, dewpoint, wind + gust, precip, CAPE)
    * `pressure-levels`: dynamics at 500 hPa and 850 hPa
      (T, u, v, geopotential) which drive synoptic forcing.
- Requests are **annual, not monthly**. That is 12x fewer CDS queue
  waits per year per point: 390 requests total for the full panel
  (13 points x 15 years x 2 datasets) instead of 2340.
- Resolution 0.25 deg x 0.25 deg on a single pixel around the point.
- Files land in two flavours:
    * `era5_sl_<pt>_<year>.nc` for single-levels
    * `era5_pl_<pt>_<year>.nc` for pressure-levels
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CANAUX_ERA5 = ("pressure", "temp", "humidity", "wind")

# Constants for relative humidity from dew point (Magnus-Tetens)
_A, _B = 17.625, 243.04

# Storm-precursor variables at the surface. Kept to 6 to stay within
# the CDS cost limit tightened end of 2025. Yearly + full 8-var request
# is now refused with "cost limits exceeded", so we stick with the
# original 6-var monthly schema. Adding CAPE + gust turned into a
# separate later pass if we ever manage to raise our quota.
_SINGLE_LEVELS_VARS = [
    "surface_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]

# Synoptic dynamics at altitude. 500 hPa carries steering flow + short-wave
# troughs; 850 hPa carries thermal advection that feeds convective set-up.
_PRESSURE_LEVELS_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]
_PRESSURE_LEVELS = ["500", "850"]


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


def _area(pt: GridPoint) -> list[float]:
    """CDS 'area' is [north, west, south, east]; +/- 0.125 deg around pt."""
    return [pt.lat + 0.125, pt.lon - 0.125, pt.lat - 0.125, pt.lon + 0.125]


def _month_request(pt: GridPoint, year: int, month: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "format": "netcdf",
        "variable": _SINGLE_LEVELS_VARS,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": _area(pt),
    }


def fetch_era5(
    grid_points: Iterable[GridPoint],
    years: Iterable[int],
    out_dir: Path,
    verbose: bool = True,
) -> list[Path]:
    """Download ERA5 single-levels NetCDFs, one per (point, year, month).

    Monthly split is required by the CDS cost policy tightened end of
    2025: anything larger (annual, or 3-month with the extra CAPE/gust
    vars) is refused with "cost limits exceeded". Existing files on
    disk are skipped so the loop can be resumed at any time.

    Pressure-levels dataset is available separately via
    fetch_era5_pressure_levels(); it uses fewer vars per request and
    can go annual.
    """
    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit(
            "cdsapi n'est pas installe. Lancer :\n"
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
                        print(f"  [skip]  {path.name}")
                    written.append(path)
                    continue
                if verbose:
                    print(f"  [fetch] {pt.nom} {year}-{month:02d}")
                req = _month_request(pt, year, month)
                client.retrieve("reanalysis-era5-single-levels", req, str(path))
                written.append(path)
    return written


def fetch_era5_pressure_levels(
    grid_points: Iterable[GridPoint],
    years: Iterable[int],
    out_dir: Path,
    verbose: bool = True,
) -> list[Path]:
    """Complementary annual pressure-levels download (500 + 850 hPa).

    Kept separate from the main fetch: 4 variables x 2 pressure levels
    x 8760 hours = 70k items per year per point sits at the edge of
    the current CDS cost policy. Optional, only for chapter 21 riches.
    """
    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit(
            "cdsapi n'est pas installe."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    client = cdsapi.Client(quiet=not verbose)

    for pt in grid_points:
        for year in years:
            path = out_dir / f"era5_pl_{pt.nom}_{year}.nc"
            if path.exists():
                if verbose:
                    print(f"  [skip pl]  {path.name}")
                written.append(path)
                continue
            if verbose:
                print(f"  [fetch pl] {pt.nom} {year}")
            req = {
                "product_type": ["reanalysis"],
                "format": "netcdf",
                "variable": _PRESSURE_LEVELS_VARS,
                "pressure_level": _PRESSURE_LEVELS,
                "year": [str(year)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": _area(pt),
            }
            client.retrieve("reanalysis-era5-pressure-levels", req, str(path))
            written.append(path)
    return written


def load_era5_files(paths: Iterable[Path]) -> dict[str, pd.DataFrame]:
    """Load one or several ERA5 NetCDF files (single-levels), one
    DataFrame per grid point.

    Pressure-levels files are recognised but produce additional columns
    only when the caller opts in via `load_era5_files(..., include_pl=True)`
    once implemented at chapter 21.  For chapter 20 backwards compat we
    keep only surface single-levels variables here.
    """
    try:
        import xarray as xr  # noqa: F401
    except ImportError as e:
        raise SystemExit("xarray requis pour lire les NetCDF ERA5") from e

    import xarray as xr

    by_point: dict[str, list[pd.DataFrame]] = {}
    for p in paths:
        p = Path(p)
        # Skip pressure-levels files in the legacy loader
        if p.name.startswith("era5_pl_"):
            continue
        ds = xr.open_dataset(p)
        df = _dataset_to_frame(ds)
        key = f"{df['lat'].iloc[0]:.3f}_{df['lon'].iloc[0]:.3f}"
        by_point.setdefault(key, []).append(df)

    return {k: pd.concat(v).sort_values("timestamp").reset_index(drop=True)
            for k, v in by_point.items()}


def _dataset_to_frame(ds) -> pd.DataFrame:
    """Convert an ERA5 single-levels xarray.Dataset into a Taranis DataFrame."""
    lat = float(ds["latitude"].values.mean())
    lon = float(ds["longitude"].values.mean())
    t_k = ds["t2m"].values.squeeze()
    td_k = ds["d2m"].values.squeeze()
    p_pa = ds["sp"].values.squeeze()
    u = ds["u10"].values.squeeze()
    v = ds["v10"].values.squeeze()
    tp_m = ds["tp"].values.squeeze()  # hourly cumulative, meters
    times = pd.to_datetime(ds["valid_time"].values if "valid_time" in ds
                           else ds["time"].values)

    t_c = t_k - 273.15
    td_c = td_k - 273.15
    frame = {
        "timestamp": times,
        "pressure": p_pa / 100.0,             # hPa
        "temp": t_c,
        "humidity": relative_humidity_from_dewpoint(t_c, td_c),
        "wind": np.hypot(u, v),
        "rain_1h": tp_m * 1000.0,             # m -> mm
        "lat": lat,
        "lon": lon,
    }
    # Optional new channels present in the rev2 schema
    if "i10fg" in ds:
        frame["wind_gust"] = ds["i10fg"].values.squeeze()
    if "cape" in ds:
        frame["cape"] = ds["cape"].values.squeeze()
    return pd.DataFrame(frame)
