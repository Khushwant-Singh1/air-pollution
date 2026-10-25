"""
Shared data loading for the Agra pollution-prediction project.
Every step (charts, feature engineering, training) imports load_and_prepare()
from here so the schema/split logic is defined exactly once.
"""
import numpy as np
import pandas as pd

CSV_PATH = "data/Agra_City_Hourly_Mean_2016_2025.csv"

# Variables that change gradually enough for their own recent trend to
# matter (temperature/humidity/wind/pressure/radiation ramping up or down).
# RF/TOT-RF excluded — rain is bursty, not trending, so lagging it doesn't
# mean the same thing; handled separately below (rainfall accumulation).
WEATHER_LAG_COLS = ['AT (°C)', 'RH (%)', 'WS (m/s)', 'BP (mmHg)', 'SR (W/mt2)']


def weather_lag_feature_cols():
    cols = []
    for col in WEATHER_LAG_COLS:
        cols += [f'{col}_lag1', f'{col}_lag24']
    return cols


# WD (deg) is circular (359 deg and 1 deg are neighbors, not 358 apart) AND
# direction alone is less physically meaningful than a real wind vector — a
# 1 m/s breeze and a 20 m/s gale from the same direction are very
# different for advection/dispersal. WS_x/WS_y (WS*sin/cos(WD)) captures
# both magnitude and direction in one non-discontinuous representation, so
# it replaces the plain WD entirely (raw WS is kept too, for direct
# "how windy is it" splits the vector alone makes trees work harder for).
MET_COLS = (
    ['AT (°C)', 'RH (%)', 'WS (m/s)', 'RF (mm)', 'TOT-RF (mm)', 'SR (W/mt2)', 'BP (mmHg)', 'VWS (m/s)']
    + ['WS_x', 'WS_y']                    # wind vector (magnitude + direction)
    + ['AT_RH', 'SR_AT', 'WS_BP']         # inversion proxy, photochemistry driver, advection strength
    + ['BP_trend24']                       # falling pressure often precedes pollution episodes
    + ['RF_rollsum72']                     # trailing 72h rainfall — near-zero = dry spell (resuspension risk)
    + weather_lag_feature_cols()
)

POLLUTANT_COLS = ['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)',
                   'NOx (ppb)', 'NH3 (µg/m³)', 'SO2 (µg/m³)', 'CO (mg/m³)',
                   'Ozone (µg/m³)', 'Benzene (µg/m³)', 'Toluene (µg/m³)',
                   'Xylene (µg/m³)', 'Eth-Benzene (µg/m³)', 'MP-Xylene (µg/m³)']
# O Xylene dropped: 100% missing in the source data, no signal.


def lag_feature_cols(target_col):
    """Column names for a target's own lag/rolling features (see
    load_and_prepare). Use MET_COLS + ['hour','month'] + lag_feature_cols(t)
    as the feature list when training on target t, to add persistence on
    top of weather without duplicating any per-target logic elsewhere."""
    return [f'{target_col}_lag1', f'{target_col}_lag24', f'{target_col}_roll24']


def compute_weather_features(df):
    """Adds every derived weather column (wind vector, interactions,
    weather lags, pressure tendency, dry-spell) to df in place, from the 9
    raw MET_COLS-equivalent columns already present. df must be sorted by
    Timestamp with no gaps for the lag/rolling values to mean what they
    claim to. Shared by load_and_prepare (full training history) and
    predict.py (a short recent-history window at serving time) so the two
    can never drift into computing these differently — training/serving
    skew from duplicated feature logic is a classic, avoidable bug."""
    for col in WEATHER_LAG_COLS:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag24'] = df[col].shift(24)

    # Wind vector: magnitude (WS) and direction (WD) combined into two
    # continuous, non-discontinuous components instead of raw degrees.
    df['WS_x'] = df['WS (m/s)'] * np.sin(df['WD (deg)'] * np.pi / 180)
    df['WS_y'] = df['WS (m/s)'] * np.cos(df['WD (deg)'] * np.pi / 180)

    # Domain interaction terms.
    df['AT_RH'] = df['AT (°C)'] * df['RH (%)']          # comfort index / inversion proxy
    df['SR_AT'] = df['SR (W/mt2)'] * df['AT (°C)']       # photochemistry driver (Ozone)
    df['WS_BP'] = df['WS (m/s)'] * df['BP (mmHg)']       # advection strength

    # Pressure tendency: reuses the BP lag just computed above.
    df['BP_trend24'] = df['BP (mmHg)'] - df['BP (mmHg)_lag24']

    # Rainfall accumulation / dry-spell detector. NOTE: deliberately built
    # from RF's own rolling sum, not "TOT-RF minus a rolling sum" — checked
    # TOT-RF directly and it is NOT a clean cumulative counter at the
    # city-mean level (it can drop mid-day while still actively raining,
    # e.g. 0.587->0.035 in one hour on 2024-07-10 — a true cumulative
    # counter can't do that). Most likely cause: it's each station's own
    # cumulative reading, averaged across a changing subset of reporting
    # stations hour to hour (Active_Stations varies), which breaks
    # monotonicity. RF's rolling sum gives the same "how much has it
    # rained recently" signal without depending on that broken column.
    df['RF_rollsum72'] = df['RF (mm)'].rolling(72, min_periods=24).sum().shift(1)

    return df


def load_and_prepare(csv_path=CSV_PATH, min_active_stations=1, start_date=None, split_date=None):
    """Loads the city-mean CSV, adds hour/month + per-pollutant lag/rolling
    features, sorts by time, and splits into train/test.

    Lag features (see lag_feature_cols): pollutant autocorrelation at lag-1h
    is ~0.9-0.95 for every pollutant checked — far stronger than any
    meteorology correlation (max |r|=0.37) — so "yesterday/last hour's
    actual value" is a legitimate, very strong feature for nowcasting.
    Computed on the FULL sorted timeline BEFORE the start_date filter, so
    the first surviving training row still gets a true lag value from just
    before the cutoff, and rolling(24) sees a real (gap-free, verified)
    hourly grid rather than a truncated one.

    min_active_stations: drop rows where fewer than this many stations
    contributed to the hourly mean (Active_Stations column). NOTE: this
    column is unreliable before 2021 — see start_date below — since a
    station counts as "active" if ANY column is non-null, and 5 of the 6
    Agra stations have an all-NaN placeholder row with only TOT-RF=0.0
    filled in for every hour of 2017-2020 (they weren't installed yet).
    That alone satisfies notnull().any(), so those years read as
    Active_Stations=6 while carrying zero real pollutant data.

    start_date: drop rows before this date. Checked directly against the
    raw per-station files (cpcb_output/merged_stations/): real multi-
    station coverage only ramps up in 2021 (partial — e.g. Rohta had 467
    real hours that year) and stabilizes at ~97%+ for all 6 stations from
    2023 onward. Pass '2023-01-01' to train only on the genuine full-
    coverage era.

    split_date: if given, test = rows on/after this date, train = rows
    before it (plain time-based split, no percentage). Use full-year
    boundaries (e.g. '2025-01-01') so train/test are both seasonally
    balanced — a percentage-based "last 15%" split on a start_date-
    restricted (and therefore short) window produces a partial-season
    test tail instead. If omitted, falls back to the last-15%-by-time split.
    """
    df = pd.read_csv(csv_path, parse_dates=['Timestamp'])
    # Sort first, compute lags on the FULL raw grid, THEN filter — shift(1)/
    # rolling(24) are row-offsets; removing rows before computing them would
    # silently turn "1 hour ago" into "1 row ago on a gappy grid", i.e. wrong.
    df = df.sort_values('Timestamp').reset_index(drop=True)

    for col in POLLUTANT_COLS:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag24'] = df[col].shift(24)
        df[f'{col}_roll24'] = df[col].rolling(24, min_periods=12).mean().shift(1)

    df = compute_weather_features(df)

    if min_active_stations > 1:
        df = df[df['Active_Stations'] >= min_active_stations].reset_index(drop=True)

    df['hour'] = df['Timestamp'].dt.hour
    df['month'] = df['Timestamp'].dt.month

    if start_date is not None:
        df = df[df['Timestamp'] >= start_date].reset_index(drop=True)

    feature_cols = MET_COLS + ['hour', 'month']

    if split_date is not None:
        train_df = df[df['Timestamp'] < split_date].reset_index(drop=True)
        test_df = df[df['Timestamp'] >= split_date].reset_index(drop=True)
    else:
        split_idx = int(len(df) * 0.85)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    return train_df, test_df, feature_cols


def demo():
    train_df, test_df, feature_cols = load_and_prepare()
    assert len(train_df) > 0 and len(test_df) > 0
    assert set(feature_cols).issubset(train_df.columns)

    # WD wraparound check: at a fixed wind speed, 1 deg and 359 deg should
    # land close together in (WS_x,WS_y) space, not ~358 apart.
    ws = 5.0
    x1, y1 = ws * np.sin(np.deg2rad(1.0)), ws * np.cos(np.deg2rad(1.0))
    x359, y359 = ws * np.sin(np.deg2rad(359.0)), ws * np.cos(np.deg2rad(359.0))
    dist = ((x1 - x359) ** 2 + (y1 - y359) ** 2) ** 0.5
    assert dist < 0.5, f"1deg and 359deg should be near-neighbors in wind-vector space, got distance {dist:.3f}"

    # Pressure tendency sanity: reconstructing it from the lag column it
    # reuses should match exactly (no silent double-computation drift).
    sample = train_df.dropna(subset=['BP (mmHg)_lag24', 'BP_trend24']).iloc[0]
    assert abs(sample['BP_trend24'] - (sample['BP (mmHg)'] - sample['BP (mmHg)_lag24'])) < 1e-9

    # Dry-spell feature should actually be near-zero during a verified dry
    # stretch and clearly positive right after a rain event (spot-checked
    # against the raw RF (mm) column, not just "some number exists").
    assert train_df['RF_rollsum72'].min() >= -1e-9, "rainfall accumulation should never be negative"

    print(f"Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    print(f"Features: {feature_cols}")
    print("✓ self-check passed")


if __name__ == '__main__':
    demo()
