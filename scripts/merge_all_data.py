"""Merge ERCOT DAM SPP, weather, and Henry Hub gas prices into one hourly dataset.

Datetime handling: SPP timestamps are tz-aware (US/Central); weather timestamps
are tz-naive local (America/Chicago == US/Central). Both describe the same wall
clock, so we strip the SPP tz to a naive local datetime and join on that.

Output: data/processed/ercot_merged_dataset.parquet
"""
from pathlib import Path
import sys
import pandas as pd
import holidays

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SPP_PATH = PROCESSED_DIR / "ercot_dam_spp_all_years.parquet"
WEATHER_PATH = RAW_DIR / "ercot_weather_2019_2026.parquet"
GAS_PATH = RAW_DIR / "henry_hub_gas_prices_2019_2026.parquet"
OUT_PATH = PROCESSED_DIR / "ercot_merged_dataset.parquet"

# Price Location -> weather_zone label used when pulling weather.
# Zones that share a representative city (e.g., LZ_HOUSTON + HB_HOUSTON)
# still have distinct rows in the weather file because pull_weather_data.py
# expanded one coordinate fetch into per-Location rows. So this is effectively
# an identity map, but we spell it out to make the join contract explicit.
LOCATION_TO_WEATHER_ZONE = {
    "LZ_AEN": "LZ_AEN",
    "LZ_CPS": "LZ_CPS",
    "LZ_HOUSTON": "LZ_HOUSTON",
    "LZ_LCRA": "LZ_LCRA",
    "LZ_NORTH": "LZ_NORTH",
    "LZ_SOUTH": "LZ_SOUTH",
    "HB_BUSAVG": "HB_BUSAVG",
    "HB_HOUSTON": "HB_HOUSTON",
    "HB_HUBAVG": "HB_HUBAVG",
    "HB_NORTH": "HB_NORTH",
    "HB_PAN": "HB_PAN",
    "HB_SOUTH": "HB_SOUTH",
}


def load_and_inspect():
    print("=" * 70)
    print("A. LOAD & INSPECT")
    print("=" * 70)

    spp = pd.read_parquet(SPP_PATH)
    print(f"\nSPP: {SPP_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  shape={spp.shape}  columns={list(spp.columns)}")
    print(f"  unique Location ({spp['Location'].nunique()}): {sorted(spp['Location'].unique())}")

    weather = pd.read_parquet(WEATHER_PATH)
    print(f"\nWeather: {WEATHER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  shape={weather.shape}  columns={list(weather.columns)}")
    print(f"  unique weather_zone ({weather['weather_zone'].nunique()}): {sorted(weather['weather_zone'].unique())}")

    gas = pd.read_parquet(GAS_PATH)
    print(f"\nGas: {GAS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  shape={gas.shape}  columns={list(gas.columns)}")
    print(f"  date range: {gas['date'].min().date()} -> {gas['date'].max().date()}")

    return spp, weather, gas


def validate_mapping(spp: pd.DataFrame, weather: pd.DataFrame):
    print("\n" + "=" * 70)
    print("B. VALIDATE Location -> weather_zone MAPPING")
    print("=" * 70)
    for loc, wz in LOCATION_TO_WEATHER_ZONE.items():
        print(f"  {loc:12s} -> {wz}")

    price_locs = set(spp["Location"].unique())
    mapped_locs = set(LOCATION_TO_WEATHER_ZONE.keys())
    weather_zones = set(weather["weather_zone"].unique())

    unmapped = price_locs - mapped_locs
    missing_weather = {wz for wz in LOCATION_TO_WEATHER_ZONE.values() if wz not in weather_zones}

    if unmapped:
        print(f"\nERROR: price Locations with no mapping entry: {sorted(unmapped)}")
        raise ValueError(f"Unmapped price Locations: {sorted(unmapped)}")
    if missing_weather:
        print(f"\nERROR: mapping targets missing from weather file: {sorted(missing_weather)}")
        raise ValueError(f"Missing weather_zone rows: {sorted(missing_weather)}")

    print("\nAll 12 price Locations have a matching weather_zone.")


def merge_weather(spp: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("C. MERGE WEATHER ONTO PRICES")
    print("=" * 70)
    original_rows = len(spp)

    spp = spp.copy()
    # Canonical join key: tz-naive local (America/Chicago) hourly timestamp.
    spp["datetime"] = spp["Interval Start"].dt.tz_localize(None)
    spp["weather_zone_key"] = spp["Location"].map(LOCATION_TO_WEATHER_ZONE)

    weather = weather.rename(columns={"time": "datetime", "weather_zone": "weather_zone_key"})

    merged = spp.merge(
        weather,
        on=["datetime", "weather_zone_key"],
        how="left",
        validate="many_to_one",
    )

    print(f"  pre-merge rows: {original_rows:,}")
    print(f"  post-merge rows: {len(merged):,}")
    assert len(merged) == original_rows, "Row count changed during weather merge"

    weather_cols = [c for c in weather.columns if c not in ("datetime", "weather_zone_key")]
    unmatched = merged[weather_cols[0]].isna().sum()
    print(f"  rows with no weather match (NaN {weather_cols[0]}): {unmatched:,}")
    return merged


def merge_gas(df: pd.DataFrame, gas: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("D. MERGE GAS PRICES")
    print("=" * 70)
    original_rows = len(df)

    df["date"] = df["datetime"].dt.normalize()
    gas = gas.copy()
    gas["date"] = pd.to_datetime(gas["date"]).dt.normalize()

    merged = df.merge(gas, on="date", how="left", validate="many_to_one")
    print(f"  rows: {len(merged):,} (unchanged: {len(merged) == original_rows})")
    print(f"  null gas rows: {merged['gas_price_henry_hub'].isna().sum():,}")
    return merged


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("E. CALENDAR FEATURES")
    print("=" * 70)
    dt = df["datetime"]
    df["hour"] = dt.dt.hour.astype("int16")
    df["day_of_week"] = dt.dt.dayofweek.astype("int16")
    df["month"] = dt.dt.month.astype("int16")
    df["day_of_year"] = dt.dt.dayofyear.astype("int16")
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    years = range(dt.dt.year.min(), dt.dt.year.max() + 1)
    us_holidays = holidays.US(years=years)
    df["is_holiday"] = df["date"].dt.date.map(lambda d: d in us_holidays)

    print(f"  years covered: {list(years)}")
    print(f"  unique holidays flagged: {df.loc[df['is_holiday'], 'date'].nunique()}")
    print(f"  weekend rows: {df['is_weekend'].sum():,}  holiday rows: {df['is_holiday'].sum():,}")
    return df


def main():
    spp, weather, gas = load_and_inspect()
    validate_mapping(spp, weather)
    merged = merge_weather(spp, weather)
    merged = merge_gas(merged, gas)
    merged = add_calendar_features(merged)

    merged = merged.sort_values(["datetime", "Location"]).reset_index(drop=True)
    merged.to_parquet(OUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("F. FINAL OUTPUT")
    print("=" * 70)
    print(f"Saved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Shape: {merged.shape}")
    print(f"Date range: {merged['datetime'].min()} -> {merged['datetime'].max()}")
    print(f"\nColumns & dtypes:")
    for col, dtype in merged.dtypes.items():
        print(f"  {col:30s} {dtype}")
    print(f"\nNull counts:")
    nulls = merged.isnull().sum()
    for col, n in nulls.items():
        print(f"  {col:30s} {n:,}")

    print(f"\nSPP summary:")
    print(merged["SPP"].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

    print("\n--- Sample: July 2023, hour 15 (summer peak) ---")
    summer = merged[
        (merged["datetime"].dt.year == 2023)
        & (merged["datetime"].dt.month == 7)
        & (merged["datetime"].dt.hour == 15)
    ].head(10)
    cols_show = ["datetime", "Location", "SPP", "temperature_2m", "gas_price_henry_hub", "is_weekend", "is_holiday"]
    print(summer[cols_show].to_string(index=False))

    print("\n--- Sample: February 2021 (Winter Storm Uri) ---")
    uri = merged[
        (merged["datetime"].dt.year == 2021)
        & (merged["datetime"].dt.month == 2)
        & (merged["Location"] == "LZ_HOUSTON")
    ].iloc[80:90]
    print(uri[cols_show].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nMERGE ASSERTION FAILED: {e}", file=sys.stderr)
        raise
