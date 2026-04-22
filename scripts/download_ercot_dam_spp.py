"""Download ERCOT Day-Ahead Market Settlement Point Prices (2019-present).

Saves one parquet per year to data/raw/, then combines and inspects.
"""
from pathlib import Path
import pandas as pd
from gridstatus import Ercot

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2019
END_YEAR = 2026  # current year; gridstatus will return partial YTD
TEXAS_TZ = "US/Central"
TIME_COLS = ["Time", "Interval Start", "Interval End"]


def to_texas_local(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all timestamp columns are tz-aware in US/Central (Texas local)."""
    for col in TIME_COLS:
        if col not in df.columns:
            continue
        s = pd.to_datetime(df[col])
        if s.dt.tz is None:
            s = s.dt.tz_localize("UTC").dt.tz_convert(TEXAS_TZ)
        else:
            s = s.dt.tz_convert(TEXAS_TZ)
        df[col] = s
    return df


def main():
    iso = Ercot()
    frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        out_path = RAW_DIR / f"ercot_dam_spp_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] cached -> {out_path.name}")
            df = pd.read_parquet(out_path)
        else:
            print(f"[{year}] downloading...")
            df = iso.get_dam_spp(year)
            df.to_parquet(out_path, index=False)
            print(f"[{year}] saved {len(df):,} rows -> {out_path.name}")
        df = to_texas_local(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = to_texas_local(combined)

    print("\n=== Combined DataFrame ===")
    print(f"Shape: {combined.shape}")
    print(f"\nColumns: {list(combined.columns)}")
    print(f"\ndtypes:\n{combined.dtypes}")

    date_col = next(
        (c for c in combined.columns if "time" in c.lower() or "date" in c.lower()),
        combined.columns[0],
    )
    print(f"\nDate column detected: {date_col!r}")
    print(f"Date range: {combined[date_col].min()}  ->  {combined[date_col].max()}")

    print(f"\nNull counts:\n{combined.isnull().sum()}")

    print("\n=== Sample rows (head) ===")
    print(combined.head())

    print("\n=== February 2021 sample (winter storm Uri, Texas local time) ===")
    ts = combined[date_col]
    feb_2021 = combined[(ts.dt.year == 2021) & (ts.dt.month == 2)]
    print(f"Feb 2021 rows: {len(feb_2021):,}")
    print(f"Feb 2021 range: {feb_2021[date_col].min()}  ->  {feb_2021[date_col].max()}")
    print(feb_2021.head(10))

    return combined


if __name__ == "__main__":
    main()
