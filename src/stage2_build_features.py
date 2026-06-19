"""Build the STAGE 2 magnitude-regression matrix (HB_HOUSTON only).

STAGE 2 is the second leg of the two-stage spike-prediction setup. Stage 1
asks "is this hour a spike?" — stage 2 asks "given that it IS a spike, how
big?". We therefore filter to rows with SPP > $100/MWh (the same threshold
used by stage 1's binary target) and regress log(SPP) on the same predictor
set stage 1 used.

This script does NOT overwrite any stage-1 artifact. Output goes to a new
parquet under data/processed/ with the `stage2_` prefix so it cannot be
confused with the stage-1 modeling matrix.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ercot_merged_dataset.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "stage2_modeling_data_houston.parquet"

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
TARGET = "log_spp"


def main():
    print("=" * 70)
    print("STAGE 2: BUILD MAGNITUDE-REGRESSION DATASET (HB_HOUSTON)")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    print(f"  merged shape: {df.shape}")
    print(f"  date range: {df['datetime'].min()} -> {df['datetime'].max()}")

    sub = df[df["Location"] == LOCATION].copy()
    sub = sub.sort_values("datetime").reset_index(drop=True)
    print(f"\n  filtered to Location={LOCATION}: {sub.shape}")

    spike_mask = sub["SPP"] > SPIKE_THRESHOLD
    sub = sub.loc[spike_mask].reset_index(drop=True)
    print(f"  spike-only subset (SPP > ${SPIKE_THRESHOLD:.0f}): {sub.shape}")

    sub[TARGET] = np.log(sub["SPP"].astype(float))

    for col in ("is_weekend", "is_holiday"):
        sub[col] = sub[col].astype("int8")

    keep = ["datetime", "SPP"] + FEATURES + [TARGET]
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

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(f"  X shape: ({len(out):,}, {len(FEATURES)})")
    print(f"  y target = log(SPP), SPP > ${SPIKE_THRESHOLD:.0f}")
    print(f"  SPP range:    ${out['SPP'].min():.2f}  ->  ${out['SPP'].max():.2f}")
    print(f"  SPP median:   ${out['SPP'].median():.2f}")
    print(f"  log_spp range: {out[TARGET].min():.3f}  ->  {out[TARGET].max():.3f}")
    print(f"  log_spp mean: {out[TARGET].mean():.3f}  std: {out[TARGET].std():.3f}")
    print(f"\n  saved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
