"""Download the entire SYNOP essential WMO dataset, 62 stations, 2010 to today.

Output: `data/mf/synop_full.csv` (~80 MB, ~2.7 million records).

A single CSV export request filtered on the period. No key required,
Etalab license, French-sovereign hosting.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "mf" / "synop_full.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Broad selection: all mainland + overseas stations, whole period.
URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/donnees-synop-essentielles-omm/exports/csv"
WHERE = "date >= '2010-01-01' AND date < '2026-01-01'"
SELECT = (
    "numer_sta,nom,altitude,latitude,longitude,nom_dept,nom_reg,"
    "date,pres,pmer,tc,u,ff,dd,rr1,rr3,rr6,raf10,rafper,ww"
)


def main():
    print(f"Téléchargement vers {OUT}")
    print("Cela prend une à quelques minutes selon le réseau.")
    cmd = [
        "curl",
        "-#",
        "-G",
        URL,
        "--data-urlencode",
        f"where={WHERE}",
        "--data-urlencode",
        f"select={SELECT}",
        "--data-urlencode",
        "delimiter=;",
        "-o",
        str(OUT),
    ]
    subprocess.run(cmd, check=True)
    size_mo = OUT.stat().st_size / 1_000_000
    n_lines = sum(1 for _ in OUT.open())
    print(f"\nOK : {size_mo:.1f} Mo, {n_lines:,} lignes (avec en-tête)")


if __name__ == "__main__":
    main()
