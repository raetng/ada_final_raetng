"""Build the modeling matrix for spike prediction.

Loads the merged ERCOT dataset, applies the $100/MWh binary spike threshold
chosen in notebook 01, and assembles the predictor set whose relevance was
established in notebook 03 (weather + gas) plus the calendar features added
during the merge. Saves a single parquet with `datetime`, all features, and
the binary `is_spike` target so the modeling scripts can load it directly.

Restricted to HB_HOUSTON: notebooks 02 and 03 framed the EDA at this hub, so
the feature-relevance findings only justify modeling here for now. Other hubs
can be added by widening LOCATIONS once we've validated the pipeline on one.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ercot_merged_dataset.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data.parquet"

SPIKE_THRESHOLD = 100.0
LOCATION = "HB_HOUSTON"

WEATHER_FEATURES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dewpoint_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_100m",
    "shortwave_radiation",
    "direct_normal_irradiance",
]
FUEL_FEATURES = ["gas_price_henry_hub"]
CALENDAR_FEATURES = ["hour", "day_of_week", "month", "day_of_year", "is_weekend", "is_holiday"]
FEATURES = WEATHER_FEATURES + FUEL_FEATURES + CALENDAR_FEATURES
TARGET = "is_spike"


def main():
    print("=" * 70)
    print("LOAD MERGED DATASET")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    print(f"  shape: {df.shape}")
    print(f"  date range: {df['datetime'].min()} -> {df['datetime'].max()}")

    sub = df[df["Location"] == LOCATION].copy()
    sub = sub.sort_values("datetime").reset_index(drop=True)
    print(f"\n  filtered to Location={LOCATION}: {sub.shape}")

    sub[TARGET] = (sub["SPP"] > SPIKE_THRESHOLD).astype("int8")

    # Booleans -> int8 so the parquet stays compact and sklearn accepts the matrix.
    for col in ("is_weekend", "is_holiday"):
        sub[col] = sub[col].astype("int8")

    keep = ["datetime"] + FEATURES + [TARGET]
    out = sub[keep].copy()

    pre = len(out)
    out = out.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    dropped = pre - len(out)
    print(f"\n  dropped rows with NaN in features/target: {dropped:,}")

    out.to_parquet(OUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("FEATURE LIST")
    print("=" * 70)
    for f in FEATURES:
        print(f"  {f}")

    spike_rate = out[TARGET].mean()
    n_spikes = int(out[TARGET].sum())
    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(f"  X shape: ({len(out):,}, {len(FEATURES)})")
    print(f"  y shape: ({len(out):,},)")
    print(f"  spike threshold: ${SPIKE_THRESHOLD:.0f}/MWh")
    print(f"  spike count: {n_spikes:,} / {len(out):,}")
    print(f"  spike rate:  {spike_rate:.4f}  ({spike_rate * 100:.2f}%)")
    print(f"\n  saved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
