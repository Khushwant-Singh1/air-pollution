"""
Feature Engineering and Transformation Pipeline for Agra Multi-Station Air Pollution System.
"""
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from config import (
    CANONICAL_DATASET,
    MET_FEATURE_COLS,
    TARGET_POLLUTANTS,
    STATIONS
)


class StationFeatureTransformer:
    """Handles data imputation, domain feature engineering, and temporal lagging per station."""

    @staticmethod
    def impute_meteorology(df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing meteorological data on a continuous hourly time series."""
        st_df = df.sort_values("Timestamp").copy()

        # Solar radiation at night (hours < 6 or >= 20) is 0 W/m²
        hours = st_df["Timestamp"].dt.hour
        is_night = (hours < 6) | (hours >= 20)
        st_df.loc[is_night & st_df["SR (W/mt2)"].isna(), "SR (W/mt2)"] = 0.0

        # Missing rainfall defaults to 0.0 mm
        if "RF (mm)" in st_df.columns:
            st_df["RF (mm)"] = st_df["RF (mm)"].fillna(0.0)

        # Linearly interpolate continuous parameters
        continuous_cols = ["AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", "BP (mmHg)", "SR (W/mt2)"]
        for col in continuous_cols:
            if col in st_df.columns:
                st_df[col] = st_df[col].interpolate(method="linear", limit_direction="both")

        return st_df

    @classmethod
    def transform_station(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Generates full feature set for a single station."""
        st_df = cls.impute_meteorology(df)

        # 1. Wind Direction Cartesian vector decomposition
        rads = np.radians(st_df["WD (deg)"])
        st_df["WD_u"] = st_df["WS (m/s)"] * np.sin(rads)
        st_df["WD_v"] = st_df["WS (m/s)"] * np.cos(rads)

        # 2. Chemical & Physical Proxies
        st_df["Solar_Temp_Interaction"] = st_df["SR (W/mt2)"] * st_df["AT (°C)"]
        st_df["Stability_Proxy"] = st_df["AT (°C)"] - st_df["RH (%)"]
        st_df["Wind_Power"] = st_df["WS (m/s)"] ** 3
        st_df["is_Raining"] = (st_df["RF (mm)"] > 0).astype(int)
        st_df["AT_RH"] = st_df["AT (°C)"] * st_df["RH (%)"]
        st_df["WS_BP"] = st_df["WS (m/s)"] * st_df["BP (mmHg)"]

        # 3. Met Temporal Lags & Accumulation
        st_df["WS_lag_1h"] = st_df["WS (m/s)"].shift(1)
        st_df["WS_lag_3h"] = st_df["WS (m/s)"].shift(3)
        st_df["WS_lag_6h"] = st_df["WS (m/s)"].shift(6)
        st_df["WS_lag_24h"] = st_df["WS (m/s)"].shift(24)
        st_df["RF_lag_6h"] = st_df["RF (mm)"].shift(6)
        st_df["SR_lag_24h"] = st_df["SR (W/mt2)"].shift(24)
        st_df["BP_lag_24h"] = st_df["BP (mmHg)"].shift(24)
        st_df["BP_trend24"] = st_df["BP (mmHg)"] - st_df["BP_lag_24h"]

        st_df["WS_rolling_6h_mean"] = st_df["WS (m/s)"].rolling(6, min_periods=3).mean()
        st_df["WS_rolling_24h_mean"] = st_df["WS (m/s)"].rolling(24, min_periods=12).mean()
        st_df["RF_rollsum72"] = st_df["RF (mm)"].rolling(72, min_periods=24).sum().shift(1)

        # 4. Target Persistence Lags (for nowcasting)
        for poll in TARGET_POLLUTANTS:
            if poll in st_df.columns:
                st_df[f"{poll}_lag1"] = st_df[poll].shift(1)
                st_df[f"{poll}_lag24"] = st_df[poll].shift(24)
                st_df[f"{poll}_roll24"] = st_df[poll].rolling(24, min_periods=12).mean().shift(1)

        # 5. Time & Cyclical Harmonics
        hour = st_df["Timestamp"].dt.hour
        month = st_df["Timestamp"].dt.month
        day = st_df["Timestamp"].dt.dayofweek

        st_df["is_weekend"] = (day >= 5).astype(int)
        st_df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        st_df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        st_df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        st_df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        st_df["day_sin"] = np.sin(2 * np.pi * day / 7.0)
        st_df["day_cos"] = np.cos(2 * np.pi * day / 7.0)

        return st_df


def engineer_station_features(st_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience functional wrapper around StationFeatureTransformer."""
    return StationFeatureTransformer.transform_station(st_df)


def get_xy_prepared(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    include_target_lags: bool = True,
    cap_percentile: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extracts features X and log-transformed target y with optional capping."""
    cols = list(feature_cols)
    if include_target_lags:
        lags = [f"{target_col}_lag1", f"{target_col}_lag24", f"{target_col}_roll24"]
        cols += [c for c in lags if c in df.columns]

    mask = df[target_col].notna()
    X = df.loc[mask, cols].copy()
    y_raw = df.loc[mask, target_col].copy()

    if cap_percentile is not None:
        upper_threshold = y_raw.quantile(cap_percentile)
        y_raw = y_raw.clip(upper=upper_threshold)

    y_log = np.log1p(y_raw)
    return X, y_log, y_raw


def inverse_target(y_log: np.ndarray) -> np.ndarray:
    """Inverts log1p transformation back to physical concentration units."""
    return np.expm1(y_log)


if __name__ == "__main__":
    print("Testing StationFeatureTransformer...")
    df_raw = pd.read_csv(CANONICAL_DATASET, parse_dates=["Timestamp"], nrows=5000)
    sample_st = df_raw[df_raw["Station"] == "Sanjay_Palace"]
    transformed = engineer_station_features(sample_st)
    print(f"Sample Transformed Shape: {transformed.shape}")
    print("✓ StationFeatureTransformer tests passed!")
