"""Download hourly weather from Open-Meteo Archive API for each ERCOT zone.

Iterates the representative (lat, lon) points in
  data/processed/ercot_locations_with_coords.csv
and fetches 2019-01-01 -> 2026-04-11 in one-year chunks per coordinate,
then expands so every weather_zone (Location) gets its own rows.

Timestamps are returned by the API as naive local time in America/Chicago
(== US/Central, matching the ERCOT prices dataset).

Output: data/raw/ercot_weather_2019_2026.parquet
"""
from pathlib import Path
import time
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "weather_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2019-01-01"
END_DATE = "2026-04-11"
TIMEZONE = "America/Chicago"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
]
SLEEP_SECONDS = 1.0
OUT_PATH = RAW_DIR / "ercot_weather_2019_2026.parquet"


def slugify(city: str) -> str:
    return city.lower().replace(",", "").replace(" ", "_")


def year_chunks(start: str, end: str):
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    for year in range(start_dt.year, end_dt.year + 1):
        cs = max(pd.Timestamp(f"{year}-01-01"), start_dt)
        ce = min(pd.Timestamp(f"{year}-12-31"), end_dt)
        yield year, cs.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")


def fetch_chunk(lat: float, lon: float, slug: str, year: int, start: str, end: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{slug}_{year}.parquet"
    if cache_path.exists():
        print(f"  [{slug} {year}] cached")
        return pd.read_parquet(cache_path)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
        "temperature_unit": "fahrenheit",
    }
    print(f"  [{slug} {year}] fetching {start} -> {end}")
    r = requests.get(API_URL, params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df.to_parquet(cache_path, index=False)
    time.sleep(SLEEP_SECONDS)
    return df


def fetch_coord(lat: float, lon: float, city: str) -> pd.DataFrame:
    slug = slugify(city)
    frames = [
        fetch_chunk(lat, lon, slug, year, cs, ce)
        for year, cs, ce in year_chunks(START_DATE, END_DATE)
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates("time").reset_index(drop=True)


def main():
    locs = pd.read_csv(PROCESSED_DIR / "ercot_locations_with_coords.csv")
    print(f"Zones to weatherize: {len(locs)}")
    print(locs[["Location", "representative_city", "latitude", "longitude"]].to_string(index=False))

    # Dedupe on (lat, lon) so e.g. Houston is fetched once for HB_HOUSTON + LZ_HOUSTON.
    unique_coords = (
        locs[["latitude", "longitude", "representative_city"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"\nUnique (lat, lon) points: {len(unique_coords)}")

    weather_by_coord = {}
    for row in unique_coords.itertuples(index=False):
        print(f"\n=== {row.representative_city} ({row.latitude}, {row.longitude}) ===")
        weather_by_coord[(row.latitude, row.longitude)] = fetch_coord(
            row.latitude, row.longitude, row.representative_city
        )

    frames = []
    for loc_row in locs.itertuples(index=False):
        df = weather_by_coord[(loc_row.latitude, loc_row.longitude)].copy()
        df.insert(0, "weather_zone", loc_row.Location)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")
    print(f"Date range: {combined['time'].min()} -> {combined['time'].max()}")
    print(f"\nNull counts:\n{combined.isnull().sum()}")
    print(f"\nSample rows:\n{combined.head()}")

    sample_zone = "LZ_HOUSTON" if "LZ_HOUSTON" in combined["weather_zone"].unique() else combined["weather_zone"].iloc[0]
    feb_2021 = combined[
        (combined["weather_zone"] == sample_zone)
        & (combined["time"].dt.year == 2021)
        & (combined["time"].dt.month == 2)
    ]
    print(f"\n=== Feb 2021 sample for {sample_zone} (Winter Storm Uri) ===")
    print(f"Rows: {len(feb_2021):,}")
    print(feb_2021.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
