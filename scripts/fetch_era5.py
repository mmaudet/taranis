"""Télécharge ERA5 pour un panel de points de grille adaptés au projet.

**Nécessite un compte CDS Copernicus** :

1. Créer un compte sur https://cds.climate.copernicus.eu/
2. Récupérer le token API sur https://cds.climate.copernicus.eu/how-to-api
3. Placer le token dans `~/.cdsapirc`
4. Sur la fiche du dataset `reanalysis-era5-single-levels`, accepter les
   conditions d'usage.

Ensuite, lancer simplement :

    uv run python scripts/fetch_era5.py

Volumes attendus, en configuration par défaut :

- 10 points × 15 ans × 24 obs/jour × 365 jours = **~1.3 M observations**
- Un fichier NetCDF par (point, année), quelques dizaines de Mo au total.
- Téléchargement CDS lent (queue), compter **plusieurs heures** la première fois.

Après téléchargement, `scripts/prepare_era5_dataset.py` (à écrire) prendra le
relais pour construire les fenêtres.
"""

from __future__ import annotations

from pathlib import Path

from taranis.data.era5 import GridPoint, fetch_era5

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "era5"

# 10 points de grille, choisis pour balayer une variété de régimes :
# - Alpes (Grenoble, Embrun, Chamonix),
# - Pyrénées (Tarbes, Ariège),
# - Sud méditerranéen (Nice, Montpellier, Marseille),
# - Plaines (Lyon, Bordeaux, Paris).
GRID_POINTS = [
    GridPoint(lat=45.19, lon=5.72, nom="grenoble"),
    GridPoint(lat=44.57, lon=6.50, nom="embrun"),
    GridPoint(lat=45.92, lon=6.87, nom="chamonix"),
    GridPoint(lat=43.24, lon=0.08, nom="tarbes"),
    GridPoint(lat=42.99, lon=1.10, nom="foix"),
    GridPoint(lat=43.70, lon=7.27, nom="nice"),
    GridPoint(lat=43.61, lon=3.88, nom="montpellier"),
    GridPoint(lat=43.30, lon=5.40, nom="marseille"),
    GridPoint(lat=44.84, lon=-0.58, nom="bordeaux"),
    GridPoint(lat=48.85, lon=2.35, nom="paris"),
]

YEARS = list(range(2010, 2025))  # 15 ans, aligné SYNOP


def main():
    print(f"Téléchargement ERA5 vers {OUT}")
    print(f"{len(GRID_POINTS)} points, {len(YEARS)} ans → {len(GRID_POINTS) * len(YEARS)} fichiers")
    print("Chaque fichier est mis en file d'attente côté CDS, comptez plusieurs")
    print("heures la première fois (téléchargements séquentiels).\n")

    written = fetch_era5(GRID_POINTS, YEARS, OUT)
    print(f"\n{len(written)} fichiers présents dans {OUT}")


if __name__ == "__main__":
    main()
