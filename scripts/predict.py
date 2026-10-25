"""
Unified Inference & AQI Predictor for Agra Air Quality System.

Provides high-level APIs and CLI for:
1. Station-Specific Forecasts (PM2.5, PM10, Ozone).
2. Whole-City Spatial Ensemble Predictions.
3. Indian CPCB National AQI (NAQI) Calculation and Health Categorization.

CLI Usage:
    python3 scripts/predict.py --station Sanjay_Palace --hours 12
    python3 scripts/predict.py --city --hours 24
    python3 scripts/predict.py --all --hours 6
"""
import argparse
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from config import (
    CANONICAL_DATASET,
    MODELS_DIR,
    MET_FEATURE_COLS,
    TARGET_POLLUTANTS,
    STATIONS,
    CPCB_BREAKPOINTS,
    CITY_FORECAST_CSV
)
from feature_engineering import engineer_station_features, inverse_target


class AQICalculator:
    """Calculates official Indian CPCB National Air Quality Index (NAQI) sub-indices and categories."""

    @staticmethod
    def calculate_sub_index(conc: float, pollutant_key: str) -> float:
        """Interpolates CPCB piecewise linear AQI sub-index for a concentration value."""
        if pd.isna(conc) or conc < 0:
            return np.nan

        breakpoints = CPCB_BREAKPOINTS.get(pollutant_key, [])
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= conc <= c_high:
                return i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)

        # Truncate above max scale
        if breakpoints and conc > breakpoints[-1][1]:
            last = breakpoints[-1]
            return min(500.0, last[3] + (conc - last[1]))

        return np.nan

    @classmethod
    def get_city_aqi(cls, pm25: float, pm10: float, ozone: float) -> Tuple[Optional[int], str, str]:
        """Calculates overall City AQI, Health Category, and Prominent Pollutant."""
        sub_indices = {
            "PM2.5": cls.calculate_sub_index(pm25, "PM2.5"),
            "PM10": cls.calculate_sub_index(pm10, "PM10"),
            "Ozone": cls.calculate_sub_index(ozone, "Ozone"),
        }
        valid_sub = {k: v for k, v in sub_indices.items() if not pd.isna(v)}

        if not valid_sub:
            return None, "Unknown", "None"

        prominent = max(valid_sub, key=valid_sub.get)
        overall_aqi = int(round(valid_sub[prominent]))

        # Categorize
        if overall_aqi <= 50:
            category = "Good"
        elif overall_aqi <= 100:
            category = "Satisfactory"
        elif overall_aqi <= 200:
            category = "Moderate"
        elif overall_aqi <= 300:
            category = "Poor"
        elif overall_aqi <= 400:
            category = "Very Poor"
        else:
            category = "Severe"

        return overall_aqi, category, prominent


class AgraPollutionPredictor:
    """Inference engine for station and whole-city pollution prediction."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir

    def predict_station(
        self,
        station_name: str,
        df: Optional[pd.DataFrame] = None,
        n_recent_hours: int = 24
    ) -> pd.DataFrame:
        """Generates PM2.5, PM10, and Ozone forecasts for a specific station."""
        if df is None:
            df_raw = pd.read_csv(CANONICAL_DATASET, parse_dates=["Timestamp"])
            st_data = df_raw[df_raw["Station"] == station_name].copy()
        else:
            st_data = df[df["Station"] == station_name].copy()

        if st_data.empty:
            raise ValueError(f"No records found for station: {station_name}")

        st_eng = engineer_station_features(st_data)
        recent = st_eng.tail(n_recent_hours).copy()

        preds_dict = {
            "Timestamp": recent["Timestamp"].values,
            "Station": station_name
        }

        for target in TARGET_POLLUTANTS:
            clean_name = target.split(" ")[0]
            model_path = self.models_dir / f"{station_name}_{clean_name}.joblib"
            if not model_path.exists():
                continue

            model = joblib.load(model_path)
            fcols = MET_FEATURE_COLS + [f"{target}_lag1", f"{target}_lag24", f"{target}_roll24"]
            X = recent[fcols]

            pred_log = model.predict(X)
            pred_raw = np.maximum(0.0, inverse_target(pred_log))

            preds_dict[f"Pred_{clean_name}"] = np.round(pred_raw, 2)
            if target in recent.columns:
                preds_dict[f"Actual_{clean_name}"] = np.round(recent[target].values, 2)

        return pd.DataFrame(preds_dict)

    def predict_city(self, n_recent_hours: int = 24) -> pd.DataFrame:
        """Aggregates all 6 station forecasts into a whole-city forecast and AQI."""
        df_raw = pd.read_csv(CANONICAL_DATASET, parse_dates=["Timestamp"])

        station_dfs = []
        for station in STATIONS:
            try:
                st_pred = self.predict_station(station, df=df_raw, n_recent_hours=n_recent_hours)
                station_dfs.append(st_pred)
            except Exception as e:
                print(f"Warning: Failed prediction for {station}: {e}")

        if not station_dfs:
            raise RuntimeError("Could not generate predictions for any station.")

        pooled = pd.concat(station_dfs, ignore_index=True)

        # Aggregate across stations for each timestamp
        records = []
        for ts, group in pooled.groupby("Timestamp"):
            row = {
                "Timestamp": ts,
                "Active_Stations": len(group["Station"].unique())
            }

            for clean in ["PM2.5", "PM10", "Ozone"]:
                pred_col = f"Pred_{clean}"
                if pred_col in group.columns:
                    row[f"City_Mean_{clean}"] = round(group[pred_col].mean(), 2)
                    row[f"City_Min_{clean}"] = round(group[pred_col].min(), 2)
                    row[f"City_Max_{clean}"] = round(group[pred_col].max(), 2)

                act_col = f"Actual_{clean}"
                if act_col in group.columns and group[act_col].notna().any():
                    row[f"Actual_Mean_{clean}"] = round(group[act_col].mean(), 2)

            aqi, cat, prom = AQICalculator.get_city_aqi(
                row.get("City_Mean_PM2.5", np.nan),
                row.get("City_Mean_PM10", np.nan),
                row.get("City_Mean_Ozone", np.nan)
            )

            row["City_AQI"] = aqi
            row["AQI_Category"] = cat
            row["Prominent_Pollutant"] = prom

            records.append(row)

        city_df = pd.DataFrame(records).sort_values("Timestamp").reset_index(drop=True)
        city_df.to_csv(CITY_FORECAST_CSV, index=False)
        return city_df


def main():
    parser = argparse.ArgumentParser(description="Agra Pollution & AQI Predictor")
    parser.add_argument("--station", type=str, default=None, help="Specific station to predict for.")
    parser.add_argument("--city", action="store_true", help="Generate whole-city aggregated forecast and AQI.")
    parser.add_argument("--all", action="store_true", help="Generate forecasts for all individual stations.")
    parser.add_argument("--hours", type=int, default=12, help="Number of recent hours to predict/display.")
    args = parser.parse_args()

    predictor = AgraPollutionPredictor()

    if args.city or (not args.station and not args.all):
        city_df = predictor.predict_city(n_recent_hours=args.hours)
        print("\n" + "=" * 105)
        print(f"AGRA WHOLE-CITY POLLUTION FORECAST & AQI (Last {args.hours} Hours)")
        print("=" * 105)
        cols = [
            "Timestamp", "Active_Stations",
            "City_Mean_PM2.5", "Actual_Mean_PM2.5",
            "City_Mean_PM10", "Actual_Mean_PM10",
            "City_Mean_Ozone", "City_AQI", "AQI_Category", "Prominent_Pollutant"
        ]
        print(city_df[[c for c in cols if c in city_df.columns]].to_string(index=False))
        print("=" * 105)
        print(f"Saved to: {CITY_FORECAST_CSV}")

    elif args.all:
        for st in STATIONS:
            print(f"\n--- Station: {st} (Last {args.hours} Hours) ---")
            st_df = predictor.predict_station(st, n_recent_hours=args.hours)
            print(st_df.to_string(index=False))

    elif args.station:
        st_df = predictor.predict_station(args.station, n_recent_hours=args.hours)
        print(f"\n--- Predictions for {args.station} (Last {args.hours} Hours) ---")
        print(st_df.to_string(index=False))


if __name__ == "__main__":
    main()
