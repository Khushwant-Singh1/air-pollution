"""
Master Orchestration Pipeline for Agra Air Quality Machine Learning System.

Usage:
    python3 scripts/pipeline.py --all           # Run full end-to-end pipeline
    python3 scripts/pipeline.py --step merge    # Merge raw station files
    python3 scripts/pipeline.py --step train    # Train all station models
    python3 scripts/pipeline.py --step evaluate # Plot diagnostic evaluation charts
    python3 scripts/pipeline.py --step predict  # Generate whole-city forecast
"""
import argparse
import sys
from pathlib import Path

from config import PROJECT_ROOT, SCRIPTS_DIR
from merge_stations import merge_stations
from train_station_models import StationModelTrainer
from plot_real_vs_predicted import main as run_evaluation_plots
from predict import AgraPollutionPredictor


def run_merge():
    print("\n[Step 1/4] Merging raw station CSVs into canonical long dataset...")
    merge_stations()


def run_train():
    print("\n[Step 2/4] Training station-specific models for PM2.5, PM10, and Ozone...")
    trainer = StationModelTrainer()
    trainer.run()


def run_evaluate():
    print("\n[Step 3/4] Generating evaluation and parity plots...")
    run_evaluation_plots()


def run_predict(hours: int = 12):
    print(f"\n[Step 4/4] Generating whole-city pollution and AQI forecast (last {hours} hours)...")
    predictor = AgraPollutionPredictor()
    predictor.predict_city(n_recent_hours=hours)


def main():
    parser = argparse.ArgumentParser(description="Agra Air Pollution ML Master Pipeline")
    parser.add_argument("--all", action="store_true", help="Run the complete end-to-end pipeline.")
    parser.add_argument("--step", type=str, choices=["merge", "train", "evaluate", "predict"],
                        help="Run a specific pipeline step.")
    parser.add_argument("--hours", type=int, default=12, help="Forecast window in hours.")
    args = parser.parse_args()

    if args.all or not args.step:
        run_merge()
        run_train()
        run_evaluate()
        run_predict(args.hours)
        print("\n" + "=" * 60)
        print("✓ Complete Agra ML pipeline executed successfully!")
        print("=" * 60)
    elif args.step == "merge":
        run_merge()
    elif args.step == "train":
        run_train()
    elif args.step == "evaluate":
        run_evaluate()
    elif args.step == "predict":
        run_predict(args.hours)


if __name__ == "__main__":
    main()
