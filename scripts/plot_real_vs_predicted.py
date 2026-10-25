"""
Diagnostic Plotting Module for Agra Air Quality Machine Learning System.

Generates:
1. charts/real_vs_predicted_city.png:
   - Overlay time series of actual vs predicted concentrations.
   - Parity scatter plots with 1:1 line and R², MAE, RMSE.
   - Residual error distribution histograms.
2. charts/real_vs_predicted_stations.png:
   - Station-by-station tracking across all 6 monitoring stations.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import joblib

from config import (
    CANONICAL_DATASET,
    CHARTS_DIR,
    MODELS_DIR,
    MET_FEATURE_COLS,
    TARGET_POLLUTANTS,
    STATIONS
)
from feature_engineering import engineer_station_features, inverse_target


def generate_evaluation_data(test_start_date: str = "2025-01-01") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates ground truth and predictions for the holdout test period."""
    df_raw = pd.read_csv(CANONICAL_DATASET, parse_dates=["Timestamp"])
    station_dfs = []

    for station in STATIONS:
        st_data = df_raw[df_raw["Station"] == station].copy()
        st_eng = engineer_station_features(st_data)
        st_test = st_eng[st_eng["Timestamp"] >= test_start_date].copy()

        st_res = pd.DataFrame({"Timestamp": st_test["Timestamp"].values, "Station": station})

        for target in TARGET_POLLUTANTS:
            clean = target.split(" ")[0]
            model_path = MODELS_DIR / f"{station}_{clean}.joblib"
            if not model_path.exists():
                continue

            model = joblib.load(model_path)
            fcols = MET_FEATURE_COLS + [f"{target}_lag1", f"{target}_lag24", f"{target}_roll24"]
            X = st_test[fcols]

            pred_log = model.predict(X)
            pred_raw = np.maximum(0.0, inverse_target(pred_log))

            st_res[f"Pred_{clean}"] = pred_raw
            st_res[f"Actual_{clean}"] = st_test[target].values

        station_dfs.append(st_res)

    pooled_stations = pd.concat(station_dfs, ignore_index=True)

    city_aggregated = pooled_stations.groupby("Timestamp").agg({
        "Pred_PM2.5": "mean",
        "Actual_PM2.5": "mean",
        "Pred_PM10": "mean",
        "Actual_PM10": "mean",
        "Pred_Ozone": "mean",
        "Actual_Ozone": "mean"
    }).reset_index()

    return city_aggregated, pooled_stations


def plot_citywide_evaluation(city_df: pd.DataFrame) -> Path:
    """Generates a 3x3 diagnostic evaluation figure for whole-city predictions."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 14), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Focus window: 3-week winter peak episode in November 2025
    zoom_window = city_df[
        (city_df["Timestamp"] >= "2025-11-01") & (city_df["Timestamp"] <= "2025-11-21")
    ].copy()

    plot_configs = [
        ("PM2.5", "PM2.5 (µg/m³)", "#1f77b4", "#d62728"),
        ("PM10", "PM10 (µg/m³)", "#2ca02c", "#9467bd"),
        ("Ozone", "Ozone (µg/m³)", "#ff7f0e", "#8c564b")
    ]

    for idx, (clean, label, color_act, color_pred) in enumerate(plot_configs):
        # 1. Time Series Panel
        ax1 = fig.add_subplot(3, 3, idx + 1)
        ax1.plot(zoom_window["Timestamp"], zoom_window[f"Actual_{clean}"],
                 label="Real (Actual)", color=color_act, linewidth=1.8, alpha=0.85)
        ax1.plot(zoom_window["Timestamp"], zoom_window[f"Pred_{clean}"],
                 label="Predicted", color=color_pred, linewidth=1.6, linestyle="--", alpha=0.95)
        ax1.fill_between(zoom_window["Timestamp"], zoom_window[f"Actual_{clean}"], zoom_window[f"Pred_{clean}"],
                         color="gray", alpha=0.15, label="Residual Gap")
        ax1.set_title(f"A. {label} - Real vs Predicted (Nov 2025 Episode)", fontsize=11, fontweight="bold")
        ax1.set_ylabel(f"Concentration ({label})")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.tick_params(axis="x", rotation=30)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # 2. Parity Scatter Panel
        ax2 = fig.add_subplot(3, 3, idx + 4)
        valid = city_df.dropna(subset=[f"Actual_{clean}", f"Pred_{clean}"])
        actual = valid[f"Actual_{clean}"]
        pred = valid[f"Pred_{clean}"]

        r2 = np.corrcoef(actual, pred)[0, 1] ** 2
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))

        ax2.scatter(actual, pred, color=color_act, alpha=0.25, s=10, edgecolors="none")
        max_val = max(actual.quantile(0.999), pred.quantile(0.999))
        ax2.plot([0, max_val], [0, max_val], "k--", linewidth=1.5, label="1:1 Perfect Parity")

        z = np.polyfit(actual, pred, 1)
        p = np.poly1d(z)
        ax2.plot(np.linspace(0, max_val, 100), p(np.linspace(0, max_val, 100)),
                 color="crimson", linewidth=1.5, linestyle="-", label=f"Fit (slope={z[0]:.2f})")

        ax2.set_title(f"B. {label} Parity (Full 2025: 8,760 hrs)\n$R^2={r2:.4f}$ | MAE={mae:.2f} | RMSE={rmse:.2f}",
                      fontsize=10.5, fontweight="bold")
        ax2.set_xlabel(f"Real Concentration ({label})")
        ax2.set_ylabel(f"Predicted Concentration ({label})")
        ax2.set_xlim(0, max_val)
        ax2.set_ylim(0, max_val)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

        # 3. Residual Distribution Panel
        ax3 = fig.add_subplot(3, 3, idx + 7)
        residuals = pred - actual
        p01, p99 = residuals.quantile(0.01), residuals.quantile(0.99)
        trimmed = residuals[(residuals >= p01) & (residuals <= p99)]

        ax3.hist(trimmed, bins=45, color=color_pred, alpha=0.7, edgecolor="black", linewidth=0.5, density=True)
        ax3.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
        ax3.axvline(residuals.mean(), color="blue", linestyle="-", linewidth=1.2, label=f"Mean Error ({residuals.mean():+.2f})")

        ax3.set_title(f"C. {label} Prediction Error Distribution\nMean: {residuals.mean():+.2f} | Std: {residuals.std():.2f}",
                      fontsize=10.5, fontweight="bold")
        ax3.set_xlabel("Residual (Predicted - Real)")
        ax3.set_ylabel("Probability Density")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig.suptitle("Agra Air Quality: Real vs. Predicted Model Performance (PM2.5, PM10, Ozone)",
                 fontsize=15, fontweight="heavy", y=0.995)

    out_file = CHARTS_DIR / "real_vs_predicted_city.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_station_breakdown(stations_df: pd.DataFrame) -> Path:
    """Generates a 3x2 grid showing Real vs Predicted PM2.5 tracking across all stations."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=150, sharex=True)
    axes = axes.flatten()

    ts_sample = stations_df[
        (stations_df["Timestamp"] >= "2025-11-05") & (stations_df["Timestamp"] <= "2025-11-19")
    ].copy()

    for idx, station in enumerate(STATIONS):
        ax = axes[idx]
        st_data = ts_sample[ts_sample["Station"] == station]

        ax.plot(st_data["Timestamp"], st_data["Actual_PM2.5"], label="Real PM2.5", color="#1f77b4", linewidth=1.6)
        ax.plot(st_data["Timestamp"], st_data["Pred_PM2.5"], label="Predicted PM2.5", color="#d62728", linestyle="--", linewidth=1.5)

        ax.set_title(f"Station: {station.replace('_', ' ')}", fontsize=11, fontweight="bold")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Station-Level Real vs. Predicted PM2.5 Tracking (Nov 2025 Winter Holdout)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = CHARTS_DIR / "real_vs_predicted_stations.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def main():
    print("Generating predictions and plotting diagnostic graphs...")
    city_df, stations_df = generate_evaluation_data()
    city_chart = plot_citywide_evaluation(city_df)
    station_chart = plot_station_breakdown(stations_df)
    print(f"✓ Saved city plot: {city_chart}")
    print(f"✓ Saved station plot: {station_chart}")


if __name__ == "__main__":
    main()
