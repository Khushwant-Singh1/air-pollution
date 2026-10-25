"""
Generates Full-Year (Jan 1, 2025 - Dec 31, 2025) Continuous LSTM Predictions & Plots.
"""
from pathlib import Path
from typing import Tuple, List, Dict
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import joblib

DL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DL_DIR.parent
CHARTS_DIR = PROJECT_ROOT / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DL_DIR))
from config import DATA_PATH, CHECKPOINTS_DIR, SCALER_PATH, SEQUENCE_FEATURE_COLS, STATIONS, LSTMHyperparameters
from dataset import preprocess_station_timeseries
from model import AgraMultiOutputLSTM


def generate_full_year_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates predictions across all 8,760 hours of 2025."""
    scaler = joblib.load(SCALER_PATH)
    ckpt_path = CHECKPOINTS_DIR / "best_lstm_model.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", LSTMHyperparameters())
    model = AgraMultiOutputLSTM(
        input_dim=len(SEQUENCE_FEATURE_COLS),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        cell_type=getattr(config, "cell_type", "gru"),
        noise_std=0.0
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df_raw = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    station_dfs = []

    for station in STATIONS:
        st_data = df_raw[df_raw["Station"] == station].copy()
        st_eng = preprocess_station_timeseries(st_data)
        scaled = scaler.transform(st_eng[SEQUENCE_FEATURE_COLS].values)

        st_test = st_eng[st_eng["Timestamp"] >= "2025-01-01"].copy()
        n_test = len(st_test)
        lookback = 24
        start_idx = len(st_eng) - n_test

        st_records = []
        for i in range(start_idx, len(st_eng)):
            seq_window = scaled[i - lookback : i]
            seq_tensor = torch.tensor(seq_window, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred_log = model(seq_tensor).numpy()[0]
                pred_raw = np.maximum(0.0, np.expm1(pred_log))

            ts = st_eng["Timestamp"].iloc[i]
            st_records.append({
                "Timestamp": ts,
                "Station": station,
                "Pred_PM2.5": pred_raw[0],
                "Actual_PM2.5": st_eng["PM2.5 (µg/m³)"].iloc[i],
                "Pred_PM10": pred_raw[1],
                "Actual_PM10": st_eng["PM10 (µg/m³)"].iloc[i],
                "Pred_Ozone": pred_raw[2],
                "Actual_Ozone": st_eng["Ozone (µg/m³)"].iloc[i]
            })

        station_dfs.append(pd.DataFrame(st_records))

    stations_df = pd.concat(station_dfs, ignore_index=True)

    city_df = stations_df.groupby("Timestamp").agg({
        "Pred_PM2.5": "mean",
        "Actual_PM2.5": "mean",
        "Pred_PM10": "mean",
        "Actual_PM10": "mean",
        "Pred_Ozone": "mean",
        "Actual_Ozone": "mean"
    }).reset_index()

    # Save to CSV
    csv_out = DL_DIR / "lstm_full_year_2025_predictions.csv"
    city_df.to_csv(csv_out, index=False)
    print(f"✓ Saved full-year 2025 predictions to: {csv_out}")

    return city_df, stations_df


def plot_full_year_timeseries(city_df: pd.DataFrame):
    """Plots full-year continuous real vs predicted time series across all 12 months of 2025."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), dpi=150, sharex=True)
    plt.subplots_adjust(hspace=0.22)

    # 24-hour rolling averages for visual clarity across 8,760 hourly points + raw backdrop
    city_plot = city_df.copy().sort_values("Timestamp")

    pollutant_specs = [
        ("PM2.5", "PM2.5 Concentration (µg/m³)", "#1f77b4", "#e41a1c", 0),
        ("PM10", "PM10 Concentration (µg/m³)", "#2ca02c", "#984ea3", 1),
        ("Ozone", "Ozone Concentration (µg/m³)", "#ff7f0e", "#a65628", 2)
    ]

    for clean, label, c_act, c_pred, ax_idx in pollutant_specs:
        ax = axes[ax_idx]
        act_col = f"Actual_{clean}"
        pred_col = f"Pred_{clean}"

        valid = city_plot.dropna(subset=[act_col, pred_col])
        act_24h = valid.set_index("Timestamp")[act_col].rolling(24, min_periods=6).mean()
        pred_24h = valid.set_index("Timestamp")[pred_col].rolling(24, min_periods=6).mean()

        # 1. Raw hourly points (subtle alpha)
        ax.plot(valid["Timestamp"], valid[act_col], color=c_act, alpha=0.18, linewidth=0.8, label="Real (Hourly Raw)")
        
        # 2. 24-Hour Smoothed Trends
        ax.plot(act_24h.index, act_24h.values, color=c_act, linewidth=2.2, label="Real (24h Daily Average)")
        ax.plot(pred_24h.index, pred_24h.values, color=c_pred, linewidth=2.0, linestyle="--", label="LSTM Predicted (24h Daily Average)")

        # Fill difference
        ax.fill_between(act_24h.index, act_24h.values, pred_24h.values, color="gray", alpha=0.2)

        # Full year stats
        r2 = np.corrcoef(valid[act_col], valid[pred_col])[0, 1] ** 2
        mae = np.mean(np.abs(valid[act_col] - valid[pred_col]))
        rmse = np.sqrt(np.mean((valid[act_col] - valid[pred_col]) ** 2))

        ax.set_title(f"Full-Year 2025 Tracking: {label} (Jan 1, 2025 – Dec 31, 2025)\n"
                     f"Full-Year Metrics: $R^2 = {r2:.4f}$ | MAE = {mae:.2f} µg/m³ | RMSE = {rmse:.2f} µg/m³ (N = {len(valid):,} Hours)",
                     fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("µg/m³", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

        # Add seasonal background shading
        # Summer (Mar-Jun)
        ax.axvspan(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-06-30"), color="gold", alpha=0.07)
        # Monsoon (Jul-Sep)
        ax.axvspan(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30"), color="cyan", alpha=0.07)
        # Winter Smog (Oct-Dec)
        ax.axvspan(pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31"), color="purple", alpha=0.06)

    # Format X-axis for full year
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[2].set_xlabel("Timeline (Full Year 2025)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=0, fontsize=10)

    fig.suptitle("Deep Learning Sequence Model (LSTM/GRU) — Full-Year 2025 Real vs. Predicted Tracking across Agra",
                 fontsize=15, fontweight="heavy", y=0.995)

    out_file = CHARTS_DIR / "lstm_full_year_2025_real_vs_predicted.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved Full-Year 2025 plot to: {out_file}")


def print_seasonal_breakdown(city_df: pd.DataFrame):
    """Calculates and prints quarterly seasonal statistics."""
    df = city_df.copy()
    df["Month"] = df["Timestamp"].dt.month
    
    seasons = {
        "Winter (Jan-Feb)": [1, 2],
        "Summer / Pre-Monsoon (Mar-Jun)": [3, 4, 5, 6],
        "Monsoon Washout (Jul-Sep)": [7, 8, 9],
        "Post-Monsoon & Winter Smog (Oct-Dec)": [10, 11, 12]
    }

    print("\n" + "=" * 90)
    print("FULL-YEAR 2025 SEASONAL ACCURACY BREAKDOWN (LSTM / GRU SEQUENCE MODEL)")
    print("=" * 90)
    print(f"{'Season / Window':<36} | {'PM2.5 MAE':<12} | {'PM10 MAE':<12} | {'Ozone MAE':<12} | {'Avg AQI Level'}")
    print("-" * 90)

    for s_name, months in seasons.items():
        sub = df[df["Month"].isin(months)].dropna()
        pm25_mae = np.mean(np.abs(sub["Actual_PM2.5"] - sub["Pred_PM2.5"]))
        pm10_mae = np.mean(np.abs(sub["Actual_PM10"] - sub["Pred_PM10"]))
        o3_mae = np.mean(np.abs(sub["Actual_Ozone"] - sub["Pred_Ozone"]))
        avg_pm10 = sub["Actual_PM10"].mean()

        print(f"{s_name:<36} | {pm25_mae:>6.2f} µg/m³ | {pm10_mae:>6.2f} µg/m³ | {o3_mae:>6.2f} µg/m³ | {avg_pm10:>6.1f} µg/m³")

    print("=" * 90)


if __name__ == "__main__":
    city_df, stations_df = generate_full_year_data()
    plot_full_year_timeseries(city_df)
    print_seasonal_breakdown(city_df)
