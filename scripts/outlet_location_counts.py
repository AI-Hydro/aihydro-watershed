"""Count ``location_class`` (aihydro_watershed.delineation.us_location) over a list of points.

Usage: python -s -E scripts/outlet_location_counts.py FILE.csv [--population CONUS_S] [--lat lat] [--lon lon]
Reads the CSV; never copies it. With --population it keeps rows whose ``population`` column matches.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout, not an installed copy
from collections import Counter

from aihydro_watershed.delineation.us_location import classify_outlet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--population")
    ap.add_argument("--lat", default="lat")
    ap.add_argument("--lon", default="lon")
    a = ap.parse_args()
    counts: Counter = Counter()
    with open(a.csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if a.population and row.get("population") != a.population:
                continue
            try:
                lat, lon = float(row[a.lat]), float(row[a.lon])
            except (KeyError, ValueError):
                lat = lon = None
            counts[classify_outlet(lat, lon)] += 1
    print(dict(counts), "n =", sum(counts.values()))


if __name__ == "__main__":
    main()
