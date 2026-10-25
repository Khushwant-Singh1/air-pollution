"""
Centralized Configuration for Agra Air Pollution ML System.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple


# --- Directory and File Paths ---
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_STATIONS_DIR = DATA_DIR / "stations"
CANONICAL_DATASET = DATA_DIR / "Agra_Stations_Long_2021_2025.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "station_models"
CHARTS_DIR = PROJECT_ROOT / "charts"
RESULTS_CSV = PROJECT_ROOT / "station_wise_engineered_results.csv"
CITY_FORECAST_CSV = PROJECT_ROOT / "city_aggregated_forecast.csv"


# --- Stations & Target Pollutants ---
STATIONS: List[str] = [
    "Manoharpur",
    "Rohta",
    "Sanjay_Palace",
    "Sector_3B",
    "Shahjahan_Garden",
    "Shastripuram"
]

TARGET_POLLUTANTS: List[str] = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "Ozone (µg/m³)"
]


# --- Feature Set Definitions ---
MET_FEATURE_COLS: List[str] = [
    # Base Meteorological
    "AT (°C)", "RH (%)", "WS (m/s)", "RF (mm)", "SR (W/mt2)", "BP (mmHg)",
    # Wind Direction Vector Components
    "WD_u", "WD_v",
    # Physical & Chemical Interaction Proxies
    "Solar_Temp_Interaction", "Stability_Proxy", "Wind_Power", "is_Raining",
    "AT_RH", "WS_BP", "BP_trend24",
    # Temporal Met Lags & Rolling Accumulation
    "WS_lag_1h", "WS_lag_3h", "WS_lag_6h", "WS_lag_24h",
    "RF_lag_6h", "SR_lag_24h",
    "WS_rolling_6h_mean", "WS_rolling_24h_mean", "RF_rollsum72",
    # Time & Cyclical Harmonics
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos", "is_weekend"
]


# --- Training & Split Configuration ---
@dataclass
class ModelConfig:
    start_train_date: str = "2023-01-01"
    split_date: str = "2025-01-01"
    min_train_samples: int = 100
    min_test_samples: int = 20
    cap_percentile: float = None  # e.g., 0.995 or None
    
    # HistGradientBoosting Hyperparameters
    learning_rate: float = 0.08
    max_iter: int = 150
    max_leaf_nodes: int = 31
    random_state: int = 42


# --- CPCB National Air Quality Index (NAQI) Breakpoints ---
# (Conc_Low, Conc_High, AQI_Low, AQI_High)
CPCB_BREAKPOINTS = {
    "PM2.5": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 380.0, 401, 500),
    ],
    "PM10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 510.0, 401, 500),
    ],
    "Ozone": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 848.0, 401, 500),
    ],
}
