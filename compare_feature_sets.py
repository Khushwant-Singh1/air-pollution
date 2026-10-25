"""
Decomposes where predictive power actually comes from: meteorology alone
(Step 3's baseline) vs persistence alone (yesterday/last-hour's actual
value) vs both together. Reuses train_one_pollutant as-is for all three —
only the feature list changes.
"""
import pandas as pd

from data_prep import load_and_prepare, POLLUTANT_COLS, MET_COLS, lag_feature_cols
from train import train_one_pollutant

CALENDAR_COLS = ['hour', 'month']


def main():
    train_df, test_df, _ = load_and_prepare(start_date='2023-01-01', split_date='2025-01-01')

    rows = []
    for target_col in POLLUTANT_COLS:
        lag_cols = lag_feature_cols(target_col)
        feature_sets = {
            'weather-only': MET_COLS + CALENDAR_COLS,
            'persistence-only': lag_cols,
            'weather+persistence': MET_COLS + CALENDAR_COLS + lag_cols,
        }
        row = {'target': target_col}
        for name, feature_cols in feature_sets.items():
            results, winner, _ = train_one_pollutant(train_df, test_df, feature_cols, target_col)
            row[f'{name}_R2'] = results[winner]['R2']
            row[f'{name}_RMSE'] = results[winner]['RMSE']
        rows.append(row)
        print(f"{target_col:<22} weather={row['weather-only_R2']:6.2f}  "
              f"persistence={row['persistence-only_R2']:6.2f}  "
              f"both={row['weather+persistence_R2']:6.2f}")

    return pd.DataFrame(rows)


def demo():
    df = main()
    assert len(df) == len(POLLUTANT_COLS)
    # "both" should never be meaningfully worse than the better of its two
    # inputs — trees can always ignore a useless feature, so if this fails
    # something's wrong (e.g. a leakage bug making persistence look
    # artificially bad, or an actual regression worth investigating).
    worse = df[df['weather+persistence_R2'] < df[['weather-only_R2', 'persistence-only_R2']].max(axis=1) - 0.05]
    if len(worse):
        print("\n⚠️ Targets where combining did worse than the best single set (tolerance 0.05):")
        print(worse[['target', 'weather-only_R2', 'persistence-only_R2', 'weather+persistence_R2']])
    df.to_csv('feature_set_comparison.csv', index=False)
    print("\nSaved to feature_set_comparison.csv")
    print("✓ self-check passed" if not len(worse) else "⚠ self-check found anomalies (see above)")


if __name__ == '__main__':
    demo()
