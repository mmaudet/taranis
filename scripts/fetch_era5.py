"""Download ERA5 for a panel of grid points suitable for the project.

**Requires a Copernicus CDS account**:

1. Create an account at https://cds.climate.copernicus.eu/
2. Fetch the API token from https://cds.climate.copernicus.eu/how-to-api
3. Place the token in `~/.cdsapirc`
4. On the `reanalysis-era5-single-levels` dataset page, accept the terms
   of use.

Then simply run:

    uv run python scripts/fetch_era5.py

Expected volumes, default configuration:

- 10 points x 15 years x 24 obs/day x 365 days = ~1.3 M observations.
- One NetCDF file per (point, year), a few dozen MB total.
- CDS downloads are slow (queue); expect **several hours** the first time.

Once downloaded, `scripts/prepare_era5_dataset.py` (to be written) takes
over to build the windows.
"""

from __future__ import annotations

from pathlib import Path

from taranis.data.era5 import GridPoint, fetch_era5

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "era5"

# 13 grid points, diversified panel for a mountain-hiker model.
#
# Plains and cities (5): Lyon, Bordeaux, Paris, Marseille, Nice.
#   Useful contrast: the model must distinguish mountain-specific dynamics
#   from plain and coastal ones.
#
# Mid-mountain 1000-1500 m (3): Chamonix valley, St-Girons (central
#   Pyrenees), Puy-de-Sancy (Massif Central).
#
# High altitude > 2000 m (5): Mont-Blanc, Vanoise, Ecrins, Mercantour,
#   Vignemale. At this altitude the ERA5 surface pressure hovers around
#   750-800 hPa and convective regimes are fundamentally different.
GRID_POINTS = [
    # Plains and cities
    GridPoint(lat=45.72, lon=5.09, nom="lyon"),
    GridPoint(lat=44.83, lon=-0.58, nom="bordeaux"),
    GridPoint(lat=48.85, lon=2.35, nom="paris"),
    GridPoint(lat=43.30, lon=5.40, nom="marseille"),
    GridPoint(lat=43.70, lon=7.27, nom="nice"),
    # Mid-mountain
    GridPoint(lat=45.93, lon=6.87, nom="chamonix_vallee"),
    GridPoint(lat=42.98, lon=1.13, nom="st_girons"),
    GridPoint(lat=45.53, lon=2.81, nom="puy_de_sancy"),
    # High altitude > 2000 m
    GridPoint(lat=45.85, lon=6.87, nom="mont_blanc"),
    GridPoint(lat=45.35, lon=6.85, nom="vanoise"),
    GridPoint(lat=44.95, lon=6.35, nom="ecrins"),
    GridPoint(lat=44.15, lon=7.15, nom="mercantour"),
    GridPoint(lat=42.80, lon=0.05, nom="vignemale"),
]

YEARS = list(range(2010, 2025))  # 15 years, aligned with SYNOP


def main():
    n_files = len(GRID_POINTS) * len(YEARS) * 12
    print(f"Téléchargement ERA5 vers {OUT}")
    print(f"{len(GRID_POINTS)} points × {len(YEARS)} ans × 12 mois = {n_files} fichiers")
    print("Découpage mensuel pour rester sous les limites de coût CDS.")
    print("La queue CDS est plus rapide la nuit et le week-end.")
    print("Les fichiers déjà présents sont conservés, on peut reprendre à tout moment.\n")

    written = fetch_era5(GRID_POINTS, YEARS, OUT)
    print(f"\n{len(written)} fichiers présents dans {OUT}")


if __name__ == "__main__":
    main()
