"""Smoke test: one point (Lyon), one year (2024), both datasets.

Used to validate that the annual schema stays within CDS cost limits
and returns readable NetCDFs, before committing the 195-year x 13-point
full download.
"""

from __future__ import annotations

from pathlib import Path

from taranis.data.era5 import GridPoint, fetch_era5

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "era5"


def main():
    pts = [GridPoint(lat=45.72, lon=5.09, nom="lyon")]
    years = [2024]
    print("Test fetch: 1 point (lyon) x 1 year (2024) x 2 datasets")
    print("Expect: era5_sl_lyon_2024.nc  +  era5_pl_lyon_2024.nc")
    print()
    written = fetch_era5(pts, years, OUT)
    print(f"\n{len(written)} files:")
    for p in written:
        if p.exists():
            print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")
        else:
            print(f"  {p.name}  MISSING")


if __name__ == "__main__":
    main()
