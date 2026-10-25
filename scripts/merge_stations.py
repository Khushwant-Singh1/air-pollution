"""
Merge individual station CSV files in data/stations/ into a unified long-format CSV.

Schema:
- Timestamp: hourly datetime (2021-06-04 00:00:00 to 2025-12-31 23:00:00)
- Station: station identifier (Sector_3B, Shahjahan_Garden, Rohta, Sanjay_Palace, Manoharpur, Shastripuram)
- Pollutants & Met columns: PM2.5, PM10, NO2, AT, RH, etc.
"""
from pathlib import Path
import pandas as pd

# Resolve project root dynamically so this script can be executed from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIONS_DIR = PROJECT_ROOT / "data" / "stations"
OUTPUT_PATH = PROJECT_ROOT / "data" / "agra_stations_long.csv"
OUTPUT_CANONICAL = PROJECT_ROOT / "data" / "Agra_Stations_Long_2021_2025.csv"

STATION_NAME_MAP = {
    "Manoharpur": "Manoharpur",
    "Rohta": "Rohta",
    "Sanjay_Palace": "Sanjay_Palace",
    "Sector_3B_Avas_Vikas_Colony": "Sector_3B",
    "Shahjahan_Garden": "Shahjahan_Garden",
    "Shastripuram": "Shastripuram",
}


def merge_stations():
    files = sorted(STATIONS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No station CSV files found in {STATIONS_DIR}")

    dfs = []
    for f in files:
        raw_name = f.stem.split("__")[0]
        station_name = STATION_NAME_MAP.get(raw_name, raw_name)

        df = pd.read_csv(f)
        df.insert(1, "Station", station_name)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    # Sort chronologically by timestamp, then station
    merged = merged.sort_values(by=["Timestamp", "Station"]).reset_index(drop=True)

    # Save to both standard naming conventions
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    merged.to_csv(OUTPUT_CANONICAL, index=False)

    print(f"Successfully merged {len(files)} station files into:")
    print(f"  - {OUTPUT_PATH}")
    print(f"  - {OUTPUT_CANONICAL}")
    print(f"Total Rows: {len(merged):,} (40,128 hours x {len(files)} stations)")
    print(f"Total Columns: {len(merged.columns)}")
    print(f"Stations: {list(merged['Station'].unique())}")
    return merged


if __name__ == "__main__":
    merge_stations()
