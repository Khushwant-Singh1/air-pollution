"""
PyTorch Dataset and 24-Hour Sliding Window Sequence Pipeline for Agra Air Quality.
"""
from typing import Tuple, List, Optional
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    SCALER_PATH,
    SEQUENCE_FEATURE_COLS,
    TARGET_POLLUTANTS,
    STATIONS,
    LSTMHyperparameters
)


def preprocess_station_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses a single station's hourly timeseries for sequence modeling."""
    st_df = df.sort_values("Timestamp").copy()

    # 1. Impute met data
    hours = st_df["Timestamp"].dt.hour
    is_night = (hours < 6) | (hours >= 20)
    st_df.loc[is_night & st_df["SR (W/mt2)"].isna(), "SR (W/mt2)"] = 0.0
    if "RF (mm)" in st_df.columns:
        st_df["RF (mm)"] = st_df["RF (mm)"].fillna(0.0)

    for col in ["AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", "BP (mmHg)", "SR (W/mt2)"]:
        if col in st_df.columns:
            st_df[col] = st_df[col].interpolate(method="linear", limit_direction="both")

    # 2. Wind direction Cartesian vectors
    rads = np.radians(st_df["WD (deg)"])
    st_df["WD_u"] = st_df["WS (m/s)"] * np.sin(rads)
    st_df["WD_v"] = st_df["WS (m/s)"] * np.cos(rads)

    # 3. Chemical/physical interactions
    st_df["Solar_Temp_Interaction"] = st_df["SR (W/mt2)"] * st_df["AT (°C)"]
    st_df["Stability_Proxy"] = st_df["AT (°C)"] - st_df["RH (%)"]
    st_df["Wind_Power"] = st_df["WS (m/s)"] ** 3
    st_df["is_Raining"] = (st_df["RF (mm)"] > 0).astype(int)
    st_df["AT_RH"] = st_df["AT (°C)"] * st_df["RH (%)"]
    st_df["WS_BP"] = st_df["WS (m/s)"] * st_df["BP (mmHg)"]
    st_df["BP_lag_24h"] = st_df["BP (mmHg)"].shift(24).interpolate(method="linear", limit_direction="both")
    st_df["BP_trend24"] = st_df["BP (mmHg)"] - st_df["BP_lag_24h"]
    st_df["RF_rollsum72"] = st_df["RF (mm)"].rolling(72, min_periods=24).sum().shift(1).fillna(0.0)

    # 4. Time harmonics
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

    # 5. Log-transformed pollutant historical features (with forward-fill for feature sequences)
    st_df["PM2.5_log"] = np.log1p(st_df["PM2.5 (µg/m³)"].ffill().bfill())
    st_df["PM10_log"] = np.log1p(st_df["PM10 (µg/m³)"].ffill().bfill())
    st_df["Ozone_log"] = np.log1p(st_df["Ozone (µg/m³)"].ffill().bfill())

    return st_df


class AgraSequenceDataset(Dataset):
    """Sliding Window PyTorch Dataset generating (Batch, 24, Num_Features) -> (Batch, 3) pairs."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray, masks: np.ndarray, metadata: pd.DataFrame):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)  # 1 if ground truth exists, 0 if NaN
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx], self.masks[idx]


def build_sliding_windows(
    st_df: pd.DataFrame,
    scaler: Optional[StandardScaler],
    seq_length: int = 24,
    is_fit_scaler: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, StandardScaler]:
    """Generates 24-hour lookback sequences and next-hour multi-pollutant targets."""
    features_raw = st_df[SEQUENCE_FEATURE_COLS].values

    if is_fit_scaler:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_raw)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when is_fit_scaler=False.")
        scaled_features = scaler.transform(features_raw)

    raw_targets = st_df[TARGET_POLLUTANTS].values
    timestamps = st_df["Timestamp"].values
    stations = st_df["Station"].values

    sequences = []
    targets = []
    masks = []
    meta_records = []

    n_rows = len(st_df)
    for i in range(seq_length, n_rows):
        # Lookback sequence from [i - seq_length : i]
        seq_window = scaled_features[i - seq_length : i]
        target_vals = raw_targets[i]

        mask = (~np.isnan(target_vals)).astype(float)
        # If all 3 targets are NaN, skip this prediction step
        if mask.sum() == 0:
            continue

        # Replace NaNs in target with 0.0 for masked loss
        clean_target = np.where(np.isnan(target_vals), 0.0, np.log1p(np.maximum(0.0, target_vals)))

        sequences.append(seq_window)
        targets.append(clean_target)
        masks.append(mask)
        meta_records.append({
            "Timestamp": timestamps[i],
            "Station": stations[i],
            "Actual_PM2.5": target_vals[0],
            "Actual_PM10": target_vals[1],
            "Actual_Ozone": target_vals[2]
        })

    return (
        np.array(sequences, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        np.array(masks, dtype=np.float32),
        pd.DataFrame(meta_records),
        scaler
    )


def create_dataloaders(
    config: LSTMHyperparameters = LSTMHyperparameters()
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, StandardScaler]:
    """Prepares Train and Test PyTorch DataLoaders across all 6 Agra monitoring stations."""
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])

    train_seqs, train_tgts, train_masks, train_meta = [], [], [], []
    test_seqs, test_tgts, test_masks, test_meta = [], [], [], []

    # First pass: Fit scaler on all training data (2023-2024)
    train_slices = []
    for station in STATIONS:
        st_data = df_raw[df_raw["Station"] == station].copy()
        st_eng = preprocess_station_timeseries(st_data)
        st_tr = st_eng[(st_eng["Timestamp"] >= config.start_train_date) & (st_eng["Timestamp"] < config.split_date)]
        train_slices.append(st_tr[SEQUENCE_FEATURE_COLS].values)

    all_train_features = np.vstack(train_slices)
    scaler = StandardScaler().fit(all_train_features)
    joblib.dump(scaler, SCALER_PATH)

    # Second pass: Build sliding windows per station
    for station in STATIONS:
        st_data = df_raw[df_raw["Station"] == station].copy()
        st_eng = preprocess_station_timeseries(st_data)

        # Split station timeseries
        st_tr = st_eng[(st_eng["Timestamp"] >= config.start_train_date) & (st_eng["Timestamp"] < config.split_date)]
        st_te = st_eng[st_eng["Timestamp"] >= config.split_date]

        # Train windows
        s_tr, t_tr, m_tr, meta_tr, _ = build_sliding_windows(st_tr, scaler, config.seq_length)
        train_seqs.append(s_tr)
        train_tgts.append(t_tr)
        train_masks.append(m_tr)
        train_meta.append(meta_tr)

        # Test windows
        s_te, t_te, m_te, meta_te, _ = build_sliding_windows(st_te, scaler, config.seq_length)
        test_seqs.append(s_te)
        test_tgts.append(t_te)
        test_masks.append(m_te)
        test_meta.append(meta_te)

    X_train = np.concatenate(train_seqs, axis=0)
    y_train = np.concatenate(train_tgts, axis=0)
    m_train = np.concatenate(train_masks, axis=0)

    X_test = np.concatenate(test_seqs, axis=0)
    y_test = np.concatenate(test_tgts, axis=0)
    m_test = np.concatenate(test_masks, axis=0)
    df_test_meta = pd.concat(test_meta, ignore_index=True)

    train_dataset = AgraSequenceDataset(X_train, y_train, m_train, pd.concat(train_meta, ignore_index=True))
    test_dataset = AgraSequenceDataset(X_test, y_test, m_test, df_test_meta)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader, df_test_meta, scaler


if __name__ == "__main__":
    print("Testing sequence dataset construction...")
    train_dl, test_dl, meta, sc = create_dataloaders()
    sample_seq, sample_tgt, sample_mask = next(iter(train_dl))
    print(f"Batch Sequence Shape: {sample_seq.shape} (Batch, 24_Hours, Features)")
    print(f"Batch Target Shape:   {sample_tgt.shape} (Batch, 3_Pollutants)")
    print("✓ AgraSequenceDataset verified successfully!")
