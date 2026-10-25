"""
Deep Learning Inference CLI for Agra Multi-Output LSTM Sequence Predictor.

Usage:
    python3 deep_learning/predict.py --station Sanjay_Palace --hours 6
    python3 deep_learning/predict.py --city --hours 12
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

from config import (
    DATA_PATH,
    CHECKPOINTS_DIR,
    SCALER_PATH,
    SEQUENCE_FEATURE_COLS,
    TARGET_POLLUTANTS,
    STATIONS,
    LSTMHyperparameters
)
from dataset import preprocess_station_timeseries
from model import AgraMultiOutputLSTM


class LSTMPredictor:
    """Runs sequence inference using the trained multi-output LSTM model."""

    def __init__(self):
        self.device = torch.device("cpu")

        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run training first.")
        self.scaler = joblib.load(SCALER_PATH)

        checkpoint_path = CHECKPOINTS_DIR / "best_lstm_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Run training first.")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        input_dim = len(SEQUENCE_FEATURE_COLS)
        config = checkpoint.get("config", LSTMHyperparameters())

        self.model = AgraMultiOutputLSTM(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            cell_type=getattr(config, "cell_type", "gru"),
            noise_std=0.0
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_station_sequence(self, station_name: str, n_recent_hours: int = 12) -> pd.DataFrame:
        """Predicts PM2.5, PM10, and Ozone for a specific station using 24h lookback windows."""
        df_raw = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
        st_data = df_raw[df_raw["Station"] == station_name].copy()

        if st_data.empty:
            raise ValueError(f"No records found for station: {station_name}")

        st_eng = preprocess_station_timeseries(st_data)
        scaled_features = self.scaler.transform(st_eng[SEQUENCE_FEATURE_COLS].values)

        n_rows = len(st_eng)
        lookback = 24

        records = []
        # Predict for the last n_recent_hours
        start_idx = max(lookback, n_rows - n_recent_hours)

        for i in range(start_idx, n_rows):
            seq_window = scaled_features[i - lookback : i]
            seq_tensor = torch.tensor(seq_window, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_log = self.model(seq_tensor).cpu().numpy()[0]
                pred_raw = np.maximum(0.0, np.expm1(pred_log))

            ts = st_eng["Timestamp"].iloc[i]
            rec = {
                "Timestamp": ts,
                "Station": station_name,
                "Pred_PM2.5": round(float(pred_raw[0]), 2),
                "Actual_PM2.5": round(float(st_eng["PM2.5 (µg/m³)"].iloc[i]), 2) if pd.notna(st_eng["PM2.5 (µg/m³)"].iloc[i]) else np.nan,
                "Pred_PM10": round(float(pred_raw[1]), 2),
                "Actual_PM10": round(float(st_eng["PM10 (µg/m³)"].iloc[i]), 2) if pd.notna(st_eng["PM10 (µg/m³)"].iloc[i]) else np.nan,
                "Pred_Ozone": round(float(pred_raw[2]), 2),
                "Actual_Ozone": round(float(st_eng["Ozone (µg/m³)"].iloc[i]), 2) if pd.notna(st_eng["Ozone (µg/m³)"].iloc[i]) else np.nan,
            }
            records.append(rec)

        return pd.DataFrame(records)

    def predict_city(self, n_recent_hours: int = 12) -> pd.DataFrame:
        """Aggregates LSTM predictions across all 6 stations for whole-city forecasting."""
        station_dfs = []
        for station in STATIONS:
            try:
                st_df = self.predict_station_sequence(station, n_recent_hours=n_recent_hours)
                station_dfs.append(st_df)
            except Exception as e:
                print(f"Warning: Failed prediction for {station}: {e}")

        pooled = pd.concat(station_dfs, ignore_index=True)

        city_records = []
        for ts, group in pooled.groupby("Timestamp"):
            row = {
                "Timestamp": ts,
                "Active_Stations": len(group["Station"].unique()),
                "City_Mean_PM2.5": round(group["Pred_PM2.5"].mean(), 2),
                "Actual_Mean_PM2.5": round(group["Actual_PM2.5"].mean(), 2),
                "City_Mean_PM10": round(group["Pred_PM10"].mean(), 2),
                "Actual_Mean_PM10": round(group["Actual_PM10"].mean(), 2),
                "City_Mean_Ozone": round(group["Pred_Ozone"].mean(), 2),
                "Actual_Mean_Ozone": round(group["Actual_Ozone"].mean(), 2),
            }
            city_records.append(row)

        return pd.DataFrame(city_records).sort_values("Timestamp").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Agra LSTM Sequence Predictor")
    parser.add_argument("--station", type=str, default=None, help="Station to predict for.")
    parser.add_argument("--city", action="store_true", help="Predict whole-city aggregated air quality.")
    parser.add_argument("--hours", type=int, default=8, help="Number of recent hours to predict.")
    args = parser.parse_args()

    predictor = LSTMPredictor()

    if args.city or not args.station:
        city_df = predictor.predict_city(n_recent_hours=args.hours)
        print("\n" + "=" * 95)
        print(f"DEEP LEARNING LSTM WHOLE-CITY FORECAST (Last {args.hours} Hours)")
        print("=" * 95)
        print(city_df.to_string(index=False))
        print("=" * 95)
    else:
        st_df = predictor.predict_station_sequence(args.station, n_recent_hours=args.hours)
        print(f"\n--- LSTM Predictions for {args.station} (Last {args.hours} Hours) ---")
        print(st_df.to_string(index=False))


if __name__ == "__main__":
    main()
