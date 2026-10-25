"""
Station-Wise Model Trainer for Agra Air Pollution System.

Fits dedicated HistGradientBoostingRegressor models for (Station, Pollutant) pairs.
"""
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from config import (
    CANONICAL_DATASET,
    MODELS_DIR,
    RESULTS_CSV,
    MET_FEATURE_COLS,
    TARGET_POLLUTANTS,
    STATIONS,
    ModelConfig
)
from feature_engineering import engineer_station_features, get_xy_prepared, inverse_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Trainer")


class StationModelTrainer:
    """Orchestrates model training, evaluation, and persistence per station."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates evaluation metrics in real units."""
        test_mean = y_true.mean()
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rel_rmse = (rmse / test_mean) * 100 if test_mean > 0 else np.nan

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Relative_RMSE": rel_rmse,
            "Test_Mean": test_mean,
            "Test_Std": y_true.std()
        }

    def train_single_target(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str
    ) -> Tuple[Optional[HistGradientBoostingRegressor], Dict[str, Any]]:
        """Trains and scores a model for a specific station and target pollutant."""
        X_train, y_train_log, _ = get_xy_prepared(
            train_df, MET_FEATURE_COLS, target_col,
            include_target_lags=True, cap_percentile=self.config.cap_percentile
        )
        X_test, y_test_log, y_test_raw = get_xy_prepared(
            test_df, MET_FEATURE_COLS, target_col,
            include_target_lags=True, cap_percentile=None
        )

        if len(X_train) < self.config.min_train_samples or len(X_test) < self.config.min_test_samples:
            return None, {"skipped": True, "n_train": len(X_train), "n_test": len(X_test)}

        model = HistGradientBoostingRegressor(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            random_state=self.config.random_state
        )
        model.fit(X_train, y_train_log)

        preds_log = model.predict(X_test)
        preds_raw = np.maximum(0.0, inverse_target(preds_log))

        metrics = self.calculate_metrics(y_test_raw, preds_raw)
        metrics["skipped"] = False
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)

        return model, metrics

    def run(self) -> pd.DataFrame:
        """Executes full training pipeline across all stations and target pollutants."""
        logger.info(f"Loading canonical dataset from {CANONICAL_DATASET}...")
        df_raw = pd.read_csv(CANONICAL_DATASET, parse_dates=["Timestamp"])

        results_list = []

        for station in STATIONS:
            logger.info(f"Training models for station: {station}")
            st_raw = df_raw[df_raw["Station"] == station].copy()
            st_engineered = engineer_station_features(st_raw)

            st_train = st_engineered[
                (st_engineered["Timestamp"] >= self.config.start_train_date) &
                (st_engineered["Timestamp"] < self.config.split_date)
            ].copy()
            st_test = st_engineered[st_engineered["Timestamp"] >= self.config.split_date].copy()

            for target in TARGET_POLLUTANTS:
                model, metrics = self.train_single_target(st_train, st_test, target)

                if metrics["skipped"]:
                    logger.warning(f"  [SKIPPED] {station} - {target} (Train: {metrics['n_train']}, Test: {metrics['n_test']})")
                    continue

                clean_name = target.split(" ")[0]
                model_filename = f"{station}_{clean_name}.joblib"
                joblib.dump(model, MODELS_DIR / model_filename)

                logger.info(f"  [SAVED] {model_filename} | R² = {metrics['R2']:.4f} | MAE = {metrics['MAE']:.2f}")

                results_list.append({
                    "Station": station,
                    "Pollutant": target,
                    "Train Samples": metrics["n_train"],
                    "Test Samples": metrics["n_test"],
                    "MAE": round(metrics["MAE"], 2),
                    "RMSE": round(metrics["RMSE"], 2),
                    "R2 Score": round(metrics["R2"], 4),
                    "Relative RMSE": f"{metrics['Relative_RMSE']:.1f}%",
                    "Model File": model_filename
                })

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(RESULTS_CSV, index=False)
        logger.info(f"Summary metrics saved to {RESULTS_CSV}")
        return results_df


def main():
    trainer = StationModelTrainer()
    summary = trainer.run()
    print("\n" + "=" * 90)
    print("TRAINING SUMMARY (2025 Test Holdout)")
    print("=" * 90)
    print(summary.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
