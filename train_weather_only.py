"""
Trains and saves the pure weather-only model: 9 meteorological variables +
hour/month -> each of the 14 pollutants. No persistence/lag features — for
deployment scenarios where only a weather reading/forecast is available,
not the pollutant's own recent history.

This is the Step 3 configuration (see FINDINGS.md "Bottom line" — weaker
than the nowcasting model in train.py, but usable with weather alone).
Saved separately to models_weather_only/ so it doesn't clobber the
nowcasting models in models/.
"""
import time
import joblib
from pathlib import Path

from data_prep import load_and_prepare, POLLUTANT_COLS, MET_COLS
from train import train_one_pollutant

MODELS_DIR = Path("models_weather_only")
CALENDAR_COLS = ['hour', 'month']
FEATURE_COLS = MET_COLS + CALENDAR_COLS


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    # Same verified split as every other step here — see FINDINGS.md.
    train_df, test_df, _ = load_and_prepare(start_date='2023-01-01', split_date='2025-01-01')

    print(f"{'Pollutant':<22} {'n_train':>8} | {'HistGB R2':>10} | {'RF R2':>8} | Winner")
    print("-" * 60)

    rows = []
    start = time.time()
    for target_col in POLLUTANT_COLS:
        results, winner, model = train_one_pollutant(train_df, test_df, FEATURE_COLS, target_col)
        print(f"{target_col:<22} {results['n_train']:>8,} | {results['HistGB']['R2']:>10.2f} | "
              f"{results['RandomForest']['R2']:>8.2f} | {winner}")

        safe_name = target_col.split(' ')[0].replace('.', '').replace('/', '_')
        joblib.dump(model, MODELS_DIR / f"{safe_name}.joblib")
        rows.append({'target': target_col, 'winner': winner, **results})

    print(f"\nTrained {len(POLLUTANT_COLS)} weather-only models in {time.time() - start:.1f}s")
    print(f"Saved to {MODELS_DIR}/")
    return rows


def demo():
    rows = main()
    assert len(rows) == len(POLLUTANT_COLS)
    saved = list(MODELS_DIR.glob('*.joblib'))
    assert len(saved) == len(POLLUTANT_COLS)
    print("✓ self-check passed")


if __name__ == '__main__':
    demo()
