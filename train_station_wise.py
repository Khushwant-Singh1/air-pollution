"""
Station-wise variant of Step 3: train per (station, pollutant) instead of
per pollutant on the city mean. Same split, same models, same metrics —
reuses data_prep.load_and_prepare() and train.train_one_pollutant() as-is,
just pointed at each station's own CSV (identical column schema to the
city-mean file, verified in data/stations/).

Same start_date/split_date as the city-mean run — chosen once from the raw
per-station coverage check (see FINDINGS.md), applied uniformly here too.
Picking a different window per station now would be exactly the test-score
cherry-picking the last step deliberately avoided.
"""
from pathlib import Path
import pandas as pd

from data_prep import load_and_prepare, POLLUTANT_COLS
from features import get_xy
from train import train_one_pollutant  # reuse, don't duplicate

STATIONS_DIR = Path("data/stations")
MIN_TRAIN_ROWS = 100  # below this a fitted model is noise, not a result


def station_name_from_path(p):
    return p.stem.replace('_merged', '').replace('___', ' - ').replace('__', ', ').replace('_', ' ')


def main():
    station_files = sorted(STATIONS_DIR.glob("*.csv"))
    print(f"Training {len(station_files)} stations x {len(POLLUTANT_COLS)} pollutants "
          f"= {len(station_files) * len(POLLUTANT_COLS)} models\n")

    rows = []
    for station_file in station_files:
        station = station_name_from_path(station_file)
        train_df, test_df, feature_cols = load_and_prepare(
            csv_path=station_file, start_date='2023-01-01', split_date='2025-01-01')

        for target_col in POLLUTANT_COLS:
            X_train, _ = get_xy(train_df, feature_cols, target_col)
            X_test, _ = get_xy(test_df, feature_cols, target_col)
            if len(X_train) < MIN_TRAIN_ROWS or len(X_test) < 20:
                rows.append({'station': station, 'target': target_col, 'skipped': True,
                             'n_train': len(X_train), 'n_test': len(X_test)})
                continue

            results, winner, _ = train_one_pollutant(train_df, test_df, feature_cols, target_col)
            best = results[winner]
            rows.append({'station': station, 'target': target_col, 'skipped': False,
                         'n_train': results['n_train'], 'n_test': results['n_test'],
                         'winner': winner, 'R2': best['R2'], 'RMSE': best['RMSE'],
                         'RMSE_over_mean': best['RMSE_over_mean']})

    return pd.DataFrame(rows)


def demo():
    df = main()
    fitted = df[~df['skipped']]
    skipped = df[df['skipped']]
    print(f"Fitted {len(fitted)}, skipped {len(skipped)} (insufficient data) "
          f"of {len(df)} (station, pollutant) pairs\n")

    print("Skipped combos (insufficient data):")
    for _, r in skipped.iterrows():
        print(f"  {r['station']:<30} {r['target']:<22} n_train={r['n_train']}, n_test={r['n_test']}")

    print("\nMean R2 per pollutant, averaged across stations that fit:")
    summary = fitted.groupby('target')['R2'].agg(['mean', 'min', 'max', 'count']).round(2)
    print(summary.to_string())

    df.to_csv("station_wise_results.csv", index=False)
    print("\nFull results saved to station_wise_results.csv")

    assert len(fitted) > 0, "no station/pollutant pair had enough data to fit"
    print("\n✓ self-check passed")


if __name__ == '__main__':
    demo()
