"""
Configuration for Deep Learning Sequence Models (LSTM) for Agra Air Quality Forecasting.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


DL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DL_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "Agra_Stations_Long_2021_2025.csv"
CHECKPOINTS_DIR = DL_DIR / "checkpoints"
SCALER_PATH = DL_DIR / "scaler.joblib"
RESULTS_CSV = DL_DIR / "lstm_evaluation_results.csv"

TARGET_POLLUTANTS: List[str] = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "Ozone (µg/m³)"
]

STATIONS: List[str] = [
    "Manoharpur",
    "Rohta",
    "Sanjay_Palace",
    "Sector_3B",
    "Shahjahan_Garden",
    "Shastripuram"
]

# Features to feed into the 24-hour sliding sequence
SEQUENCE_FEATURE_COLS: List[str] = [
    # Meteorological measurements
    "AT (°C)", "RH (%)", "WS (m/s)", "RF (mm)", "SR (W/mt2)", "BP (mmHg)",
    # Wind vector components
    "WD_u", "WD_v",
    # Physical/Chemical interaction proxies
    "Solar_Temp_Interaction", "Stability_Proxy", "Wind_Power", "is_Raining",
    "AT_RH", "WS_BP", "BP_trend24", "RF_rollsum72",
    # Harmonic time encodings
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos", "is_weekend",
    # Pollutant sequence history (in lookback window)
    "PM2.5_log", "PM10_log", "Ozone_log"
]


@dataclass
class LSTMHyperparameters:
    seq_length: int = 24           # 24-hour lookback window
    hidden_dim: int = 64           # Compact hidden units (reduces overfitting)
    num_layers: int = 1            # 1-layer GRU/LSTM
    cell_type: str = "gru"         # GRU has fewer parameters than LSTM
    dropout: float = 0.30          # Higher dropout probability
    noise_std: float = 0.02        # Input Gaussian noise jitter during training
    bidirectional: bool = False    # Unidirectional for causal forecasting
    batch_size: int = 128          # Batch size for training
    learning_rate: float = 0.0004  # Smaller initial learning rate
    weight_decay: float = 1e-3     # Stronger L2 weight regularization
    max_epochs: int = 30           # Max training epochs
    patience: int = 6              # Early stopping patience
    loss_fn: str = "huber"         # "huber" or "mse"
    start_train_date: str = "2023-01-01"
    split_date: str = "2025-01-01"
