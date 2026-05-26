"""Build the STAGE 2 magnitude-regression matrix (multi-hub).

Multi-hub analog of stage2_build_features.py. Same target (log of SPP > $100)
but pooled across the four real HB_* hubs: HB_HOUSTON, HB_NORTH, HB_SOUTH,
HB_PAN. HB_BUSAVG and HB_HUBAVG are excluded (system-wide aggregates, not
independent settlement points — see multihub_findings.md).

Location is one-hot encoded with HB_HOUSTON as the reference category so the
matrix is directly consumable by the same OLS / linear-model code that runs
on the Houston-only stage-2 set.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ercot_merged_dataset.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "stage2_modeling_data_multihub.parquet"

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
TARGET = "log_spp"


def main():
    print("=" * 70)
    print("STAGE 2: BUILD MAGNITUDE-REGRESSION DATASET (MULTI-HUB)")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    print(f"  merged shape: {df.shape}")

    sub = df[df["Location"].isin(HUBS)].copy()
    sub = sub.sort_values(["datetime", "Location"]).reset_index(drop=True)
    print(f"\n  filtered to {len(HUBS)} hubs: {HUBS}")

    spike_mask = sub["SPP"] > SPIKE_THRESHOLD
    sub = sub.loc[spike_mask].reset_index(drop=True)
    print(f"  spike-only subset (SPP > ${SPIKE_THRESHOLD:.0f}): {sub.shape}")
    print(f"  spike rows per hub:")
    for loc in sorted(sub["Location"].unique()):
        n = (sub["Location"] == loc).sum()
        print(f"    {loc:12s}  {n:,}")

    sub[TARGET] = np.log(sub["SPP"].astype(float))

    for col in ("is_weekend", "is_holiday"):
        sub[col] = sub[col].astype("int8")

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
        [sub[["datetime", "Location", "SPP"] + NUMERIC_FEATURES], location_dummies, sub[[TARGET]]],
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

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(f"  X shape: ({len(out):,}, {len(features)})")
    print(f"  y target = log(SPP), SPP > ${SPIKE_THRESHOLD:.0f}")
    print(f"  SPP range:    ${out['SPP'].min():.2f}  ->  ${out['SPP'].max():.2f}")
    print(f"  SPP median:   ${out['SPP'].median():.2f}")
    print(f"  log_spp mean: {out[TARGET].mean():.3f}  std: {out[TARGET].std():.3f}")
    print(f"\n  per-hub log_spp summary:")
    for loc, grp in out.groupby("Location", observed=True):
        print(f"    {loc:12s}  n={len(grp):,}  mean={grp[TARGET].mean():.3f}  median={grp[TARGET].median():.3f}")
    print(f"\n  saved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
