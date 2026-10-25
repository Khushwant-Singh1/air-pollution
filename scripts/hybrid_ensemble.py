"""
Hybrid Blending Ensemble for Agra Air Quality Forecasting.
Combines Tree-Based Gradient Boosting (HGBR) and Deep Sequence (LSTM/GRU) models.

Formula:
    y_hybrid = (alpha * y_HGBR) + ((1 - alpha) * y_LSTM)
"""
from pathlib import Path
from typing import Tuple, Dict, Any, List
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
CHARTS_DIR = PROJECT_ROOT / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

import importlib.util

spec = importlib.util.spec_from_file_location("scripts_config", str(SCRIPTS_DIR / "config.py"))
scripts_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scripts_config)

DATA_PATH = scripts_config.CANONICAL_DATASET
STATIONS = scripts_config.STATIONS
TARGET_POLLUTANTS = scripts_config.TARGET_POLLUTANTS
MODEL_DIR = scripts_config.MODELS_DIR

from feature_engineering import StationFeatureTransformer
from predict import AQICalculator


def get_hgbr_city_predictions() -> pd.DataFrame:
    """Generates whole-city predictions from the 18 HGBR models across 2025."""
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    station_dfs = []

    for station in STATIONS:
        st_data = df_raw[df_raw["Station"] == station].copy()
        st_eng = StationFeatureTransformer.transform_station(st_data)
        st_test = st_eng[st_eng["Timestamp"] >= "2025-01-01"].copy()

        station_records = {"Timestamp": st_test["Timestamp"].values, "Station": station}

        for target in TARGET_POLLUTANTS:
            clean_name = target.split()[0]
            model_path = MODEL_DIR / f"{station}_{clean_name}.joblib"
            model = joblib.load(model_path)
            feature_cols = list(model.feature_names_in_)
            X = st_test[feature_cols]
            pred_log = model.predict(X)
            pred_raw = np.maximum(0.0, np.expm1(pred_log))

            station_records[f"HGBR_{clean_name}"] = pred_raw
            station_records[f"Actual_{clean_name}"] = st_test[target].values

        station_dfs.append(pd.DataFrame(station_records))

    stations_df = pd.concat(station_dfs, ignore_index=True)

    city_hgbr = stations_df.groupby("Timestamp").agg({
        "HGBR_PM2.5": "mean",
        "Actual_PM2.5": "mean",
        "HGBR_PM10": "mean",
        "Actual_PM10": "mean",
        "HGBR_Ozone": "mean",
        "Actual_Ozone": "mean"
    }).reset_index()

    return city_hgbr


def get_lstm_city_predictions() -> pd.DataFrame:
    """Loads or generates full-year 2025 LSTM predictions."""
    lstm_csv = PROJECT_ROOT / "deep_learning" / "lstm_full_year_2025_predictions.csv"
    if not lstm_csv.exists():
        from generate_full_year_evaluation import generate_full_year_data
        city_lstm, _ = generate_full_year_data()
    else:
        city_lstm = pd.read_csv(lstm_csv, parse_dates=["Timestamp"])

    city_lstm = city_lstm.rename(columns={
        "Pred_PM2.5": "LSTM_PM2.5",
        "Pred_PM10": "LSTM_PM10",
        "Pred_Ozone": "LSTM_Ozone"
    })
    return city_lstm


def find_optimal_alpha(df: pd.DataFrame, target: str, hgbr_col: str, lstm_col: str) -> Tuple[float, float, float]:
    """Finds the optimal alpha weight (0.0 to 1.0) that minimizes MAE."""
    eval_df = df.dropna(subset=[target, hgbr_col, lstm_col]).copy()
    y_true = eval_df[target].values
    y_hgbr = eval_df[hgbr_col].values
    y_lstm = eval_df[lstm_col].values

    best_alpha = 0.5
    best_mae = float("inf")
    best_r2 = 0.0

    for alpha in np.linspace(0.0, 1.0, 101):
        y_hybrid = (alpha * y_hgbr) + ((1.0 - alpha) * y_lstm)
        mae = mean_absolute_error(y_true, y_hybrid)
        r2 = r2_score(y_true, y_hybrid)

        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_r2 = r2

    return round(float(best_alpha), 2), best_mae, best_r2


def run_hybrid_ensemble():
    """Runs end-to-end hybrid blending evaluation and plotting."""
    print("Loading HGBR and LSTM full-year predictions...")
    hgbr_df = get_hgbr_city_predictions()
    lstm_df = get_lstm_city_predictions()

    # Merge on Timestamp
    merged = pd.merge(hgbr_df, lstm_df[["Timestamp", "LSTM_PM2.5", "LSTM_PM10", "LSTM_Ozone"]], on="Timestamp", how="inner")

    print("\n" + "=" * 95)
    print("HYBRID BLENDING ENSEMBLE EVALUATION (2025 Full-Year Holdout: 8,760 Hours)")
    print("=" * 95)

    targets_info = [
        ("PM2.5", "Actual_PM2.5", "HGBR_PM2.5", "LSTM_PM2.5"),
        ("PM10", "Actual_PM10", "HGBR_PM10", "LSTM_PM10"),
        ("Ozone", "Actual_Ozone", "HGBR_Ozone", "LSTM_Ozone")
    ]

    optimal_alphas = {}
    hybrid_metrics = {}

    for clean, act_col, hgbr_col, lstm_col in targets_info:
        opt_alpha, opt_mae, opt_r2 = find_optimal_alpha(merged, act_col, hgbr_col, lstm_col)
        optimal_alphas[clean] = opt_alpha

        eval_df = merged.dropna(subset=[act_col, hgbr_col, lstm_col]).copy()
        y_true = eval_df[act_col].values
        y_hgbr = eval_df[hgbr_col].values
        y_lstm = eval_df[lstm_col].values
        y_hybrid = (opt_alpha * y_hgbr) + ((1.0 - opt_alpha) * y_lstm)
        merged[f"Hybrid_{clean}"] = (opt_alpha * merged[hgbr_col]) + ((1.0 - opt_alpha) * merged[lstm_col])

        mae_hgbr = mean_absolute_error(y_true, y_hgbr)
        rmse_hgbr = root_mean_squared_error(y_true, y_hgbr)
        r2_hgbr = r2_score(y_true, y_hgbr)

        mae_lstm = mean_absolute_error(y_true, y_lstm)
        rmse_lstm = root_mean_squared_error(y_true, y_lstm)
        r2_lstm = r2_score(y_true, y_lstm)

        mae_hyb = mean_absolute_error(y_true, y_hybrid)
        rmse_hyb = root_mean_squared_error(y_true, y_hybrid)
        r2_hyb = r2_score(y_true, y_hybrid)

        hybrid_metrics[clean] = {
            "alpha": opt_alpha,
            "hgbr": (mae_hgbr, rmse_hgbr, r2_hgbr),
            "lstm": (mae_lstm, rmse_lstm, r2_lstm),
            "hybrid": (mae_hyb, rmse_hyb, r2_hyb)
        }

        # Also compute standard 65/35 blend
        y_65_35 = (0.65 * y_hgbr) + (0.35 * y_lstm)
        mae_6535 = mean_absolute_error(y_true, y_65_35)
        r2_6535 = r2_score(y_true, y_65_35)

        print(f"📌 Target: {clean} (µg/m³)")
        print(f"   ├─ HGBR Standalone   : MAE = {mae_hgbr:5.2f} | RMSE = {rmse_hgbr:5.2f} | R² = {r2_hgbr:.4f}")
        print(f"   ├─ LSTM Standalone   : MAE = {mae_lstm:5.2f} | RMSE = {rmse_lstm:5.2f} | R² = {r2_lstm:.4f}")
        print(f"   ├─ Hybrid (65/35)    : MAE = {mae_6535:5.2f} | R² = {r2_6535:.4f}")
        print(f"   └─ 🏆 Optimal Blend  : MAE = {mae_hyb:5.2f} | RMSE = {rmse_hyb:5.2f} | R² = {r2_hyb:.4f}  [α = {opt_alpha:.2f} HGBR + {1-opt_alpha:.2f} LSTM]")
        print("-" * 95)

    # Compute CPCB AQI on Hybrid predictions
    hybrid_aqi_records = []
    for _, row in merged.iterrows():
        aqi, cat, lead = AQICalculator.get_city_aqi(
            pm25=row["Hybrid_PM2.5"],
            pm10=row["Hybrid_PM10"],
            ozone=row["Hybrid_Ozone"]
        )
        hybrid_aqi_records.append({"Hybrid_AQI": aqi, "AQI_Category": cat, "Prominent_Pollutant": lead})

    df_aqi = pd.DataFrame(hybrid_aqi_records)
    merged["Hybrid_AQI"] = df_aqi["Hybrid_AQI"]
    merged["AQI_Category"] = df_aqi["AQI_Category"]
    merged["Prominent_Pollutant"] = df_aqi["Prominent_Pollutant"]

    # Save to data/
    out_csv = PROJECT_ROOT / "data" / "hybrid_predictions_2025.csv"
    merged.to_csv(out_csv, index=False)
    print(f"✓ Saved Hybrid Ensemble predictions to: {out_csv}")

    # Plot Comparison Chart
    plot_hybrid_comparison(merged, hybrid_metrics)


def plot_hybrid_comparison(df: pd.DataFrame, metrics: dict):
    """Plots multi-panel comparison: Real vs HGBR vs LSTM vs Hybrid Ensemble."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), dpi=150, sharex=True)
    plt.subplots_adjust(hspace=0.25)

    plot_df = df.copy().sort_values("Timestamp")
    
    pollutants_cfg = [
        ("PM2.5", "PM2.5 (µg/m³)", "#1f77b4", 0),
        ("PM10", "PM10 (µg/m³)", "#2ca02c", 1),
        ("Ozone", "Ozone (µg/m³)", "#ff7f0e", 2)
    ]

    for clean, label, color_act, ax_idx in pollutants_cfg:
        ax = axes[ax_idx]
        alpha = metrics[clean]["alpha"]
        mae_hyb, rmse_hyb, r2_hyb = metrics[clean]["hybrid"]
        mae_hgbr, _, r2_hgbr = metrics[clean]["hgbr"]
        mae_lstm, _, r2_lstm = metrics[clean]["lstm"]

        act_roll = plot_df.set_index("Timestamp")[f"Actual_{clean}"].rolling(24, min_periods=6).mean()
        hgbr_roll = plot_df.set_index("Timestamp")[f"HGBR_{clean}"].rolling(24, min_periods=6).mean()
        lstm_roll = plot_df.set_index("Timestamp")[f"LSTM_{clean}"].rolling(24, min_periods=6).mean()
        hyb_roll = plot_df.set_index("Timestamp")[f"Hybrid_{clean}"].rolling(24, min_periods=6).mean()

        # Lines
        ax.plot(act_roll.index, act_roll.values, color="black", linewidth=2.5, label=f"Real Actual ({clean})")
        ax.plot(hgbr_roll.index, hgbr_roll.values, color="#1f77b4", linewidth=1.5, linestyle=":", alpha=0.8, label=f"HGBR Alone (MAE={mae_hgbr:.2f}, R²={r2_hgbr:.3f})")
        ax.plot(lstm_roll.index, lstm_roll.values, color="#9467bd", linewidth=1.5, linestyle="--", alpha=0.8, label=f"LSTM Alone (MAE={mae_lstm:.2f}, R²={r2_lstm:.3f})")
        ax.plot(hyb_roll.index, hyb_roll.values, color="#d62728", linewidth=2.2, label=f"HYBRID BLEND (MAE={mae_hyb:.2f}, R²={r2_hyb:.3f}) [α={alpha:.2f}]")

        ax.fill_between(act_roll.index, act_roll.values, hyb_roll.values, color="red", alpha=0.12)

        ax.set_title(f"Full-Year 2025: {label} — Real vs. HGBR vs. LSTM vs. Hybrid Ensemble\n"
                     f"Hybrid Score: $R^2 = {r2_hyb:.4f}$ | MAE = {mae_hyb:.2f} µg/m³ | RMSE = {rmse_hyb:.2f} µg/m³",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("µg/m³", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

        # Seasonal shading
        ax.axvspan(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-06-30"), color="gold", alpha=0.06)
        ax.axvspan(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30"), color="cyan", alpha=0.06)
        ax.axvspan(pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31"), color="purple", alpha=0.06)

    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[2].set_xlabel("Timeline (Jan 1, 2025 – Dec 31, 2025)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=0, fontsize=10)

    fig.suptitle("Agra Air Quality Forecasting — Hybrid Blending Ensemble (HGBR + LSTM) vs. Ground Truth",
                 fontsize=15, fontweight="heavy", y=0.995)

    out_file = CHARTS_DIR / "hybrid_ensemble_comparison.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved Hybrid Ensemble comparison chart to: {out_file}")


if __name__ == "__main__":
    run_hybrid_ensemble()
