"""Build the multi-hub modeling matrix for spike prediction.

Multi-hub analog of build_features.py. Includes the four real ERCOT trading
hubs — HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_PAN — and excludes HB_BUSAVG and
HB_HUBAVG (which are computed system-wide averages of the per-hub prices,
not independent settlement points; including them would duplicate signal).
Load zones (LZ_*) are also excluded per the brief.

Location is encoded as one-hot dummies (drop_first=True, HB_HOUSTON as the
reference category) so the training scripts can consume the matrix without
needing any categorical-handling logic. Rows are sorted by (datetime,
Location) so that the grouped time-series splitter in evaluation.py can
align fold boundaries to timestamp groups.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ercot_merged_dataset.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data_multihub.parquet"

SPIKE_THRESHOLD = 100.0
HUBS = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_PAN"]
REFERENCE_HUB = "HB_HOUSTON"

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
NUMERIC_FEATURES = WEATHER_FEATURES + FUEL_FEATURES + CALENDAR_FEATURES
TARGET = "is_spike"


def main():
    print("=" * 70)
    print("LOAD MERGED DATASET")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    print(f"  shape: {df.shape}")
    print(f"  date range: {df['datetime'].min()} -> {df['datetime'].max()}")

    sub = df[df["Location"].isin(HUBS)].copy()
    sub = sub.sort_values(["datetime", "Location"]).reset_index(drop=True)
    print(f"\n  filtered to {len(HUBS)} hubs: {HUBS}")
    print(f"  rows per hub:")
    for loc, n in sub["Location"].value_counts().items():
        print(f"    {loc:12s}  {n:,}")

    sub[TARGET] = (sub["SPP"] > SPIKE_THRESHOLD).astype("int8")

    for col in ("is_weekend", "is_holiday"):
        sub[col] = sub[col].astype("int8")

    # One-hot encode Location with HB_HOUSTON as the reference category.
    location_dummies = pd.get_dummies(sub["Location"], prefix="loc", dtype="int8")
    ref_col = f"loc_{REFERENCE_HUB}"
    if ref_col in location_dummies.columns:
        location_dummies = location_dummies.drop(columns=[ref_col])
    location_cols = list(location_dummies.columns)
    print(f"\n  one-hot location columns (reference = {REFERENCE_HUB}):")
    for c in location_cols:
        print(f"    {c}")

    features = NUMERIC_FEATURES + location_cols
    out = pd.concat(
        [sub[["datetime", "Location"] + NUMERIC_FEATURES], location_dummies, sub[[TARGET]]],
        axis=1,
    )

    pre = len(out)
    out = out.dropna(subset=NUMERIC_FEATURES + [TARGET]).reset_index(drop=True)
    dropped = pre - len(out)
    print(f"\n  dropped rows with NaN in features/target: {dropped:,}")

    out.to_parquet(OUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("FEATURE LIST")
    print("=" * 70)
    for f in features:
        print(f"  {f}")

    spike_rate = out[TARGET].mean()
    n_spikes = int(out[TARGET].sum())
    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(f"  X shape: ({len(out):,}, {len(features)})")
    print(f"  y shape: ({len(out):,},)")
    print(f"  spike threshold: ${SPIKE_THRESHOLD:.0f}/MWh")
    print(f"  spike count: {n_spikes:,} / {len(out):,}")
    print(f"  spike rate:  {spike_rate:.4f}  ({spike_rate * 100:.2f}%)")
    print(f"\n  per-hub spike rate:")
    for loc, grp in out.groupby("Location", observed=True):
        print(f"    {loc:12s}  n={len(grp):,}  rate={grp[TARGET].mean():.4f}")
    print(f"\n  saved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
