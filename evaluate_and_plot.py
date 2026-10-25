"""
Runs both saved model sets (weather-only, weather+persistence) against the
real 2025 test data and produces:
  - charts/actual_vs_predicted_timeseries_<pollutant>.png — 14-day zoom,
    actual vs both models' predictions, for a couple of representative
    pollutants (one where weather-only works, one where it doesn't).
  - charts/actual_vs_predicted_scatter_grid.png — all 14 pollutants,
    final (best) model, actual vs predicted with R2 per subplot.
  - a results table printed to stdout and saved to results_table.csv.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from data_prep import load_and_prepare, POLLUTANT_COLS, MET_COLS, lag_feature_cols
from features import get_xy, inverse_target
from train import PERSISTENCE_ONLY, feature_cols_for
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

CHARTS_DIR = Path("charts")
CALENDAR_COLS = ['hour', 'month']


def load_model(models_dir, target_col):
    safe_name = target_col.split(' ')[0].replace('.', '').replace('/', '_')
    return joblib.load(Path(models_dir) / f"{safe_name}.joblib")


def predict_series(model, df, feature_cols):
    X = df[feature_cols]
    return inverse_target(model.predict(X))


def build_results_table(train_df, test_df):
    rows = []
    for target_col in POLLUTANT_COLS:
        _, y_test_actual_full = get_xy(test_df, MET_COLS, target_col)  # just to get the notna mask/index cheaply
        mask = test_df[target_col].notna()
        actual = test_df.loc[mask, target_col].values

        weather_cols = MET_COLS + CALENDAR_COLS
        weather_model = load_model("models_weather_only", target_col)
        weather_pred = predict_series(weather_model, test_df.loc[mask], weather_cols)

        final_cols = feature_cols_for(target_col)
        final_model = load_model("models", target_col)
        final_pred = predict_series(final_model, test_df.loc[mask], final_cols)

        rows.append({
            'target': target_col,
            'final_feature_set': 'persistence-only' if target_col in PERSISTENCE_ONLY else 'weather+persistence',
            'weather_only_R2': r2_score(actual, weather_pred),
            'weather_only_MAE': mean_absolute_error(actual, weather_pred),
            'final_R2': r2_score(actual, final_pred),
            'final_MAE': mean_absolute_error(actual, final_pred),
        })
    return pd.DataFrame(rows)


def plot_timeseries_zoom(test_df, target_col, start, end, out_name):
    mask = test_df['Timestamp'].between(start, end) & test_df[target_col].notna()
    zoomed = test_df.loc[mask]

    weather_cols = MET_COLS + CALENDAR_COLS
    weather_pred = predict_series(load_model("models_weather_only", target_col), zoomed, weather_cols)

    final_cols = feature_cols_for(target_col)
    final_pred = predict_series(load_model("models", target_col), zoomed, final_cols)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(zoomed['Timestamp'], zoomed[target_col], label='Actual', color='black', linewidth=1.5)
    ax.plot(zoomed['Timestamp'], weather_pred, label='Predicted (weather-only)', color='tab:orange', linestyle='--')
    ax.plot(zoomed['Timestamp'], final_pred, label='Predicted (weather+persistence)', color='tab:blue', linestyle='--')
    ax.set_title(f'{target_col} — actual vs predicted, {start} to {end}')
    ax.set_xlabel('Time')
    ax.set_ylabel(target_col)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / out_name, dpi=120)
    plt.close(fig)


def plot_scatter_grid(test_df, results_df):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for i, target_col in enumerate(POLLUTANT_COLS):
        ax = axes[i]
        mask = test_df[target_col].notna()
        actual = test_df.loc[mask, target_col].values
        final_cols = feature_cols_for(target_col)
        pred = predict_series(load_model("models", target_col), test_df.loc[mask], final_cols)

        ax.scatter(actual, pred, s=4, alpha=0.3)
        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        r2 = results_df.loc[results_df['target'] == target_col, 'final_R2'].iloc[0]
        ax.set_title(f'{target_col}\nR2={r2:.2f}', fontsize=9)
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)

    for j in range(len(POLLUTANT_COLS), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Actual vs Predicted — final model per pollutant (test year 2025)', fontsize=14)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / 'actual_vs_predicted_scatter_grid.png', dpi=120)
    plt.close(fig)


def main():
    CHARTS_DIR.mkdir(exist_ok=True)
    train_df, test_df, _ = load_and_prepare(start_date='2023-01-01', split_date='2025-01-01')

    results_df = build_results_table(train_df, test_df)
    results_df.to_csv('results_table.csv', index=False)
    print(results_df.to_string(index=False))

    # PM2.5: weather-only already does okay, persistence makes it much better
    plot_timeseries_zoom(test_df, 'PM2.5 (µg/m³)', '2025-01-01', '2025-01-15', 'timeseries_PM2.5.png')
    # NOx: weather-only fails outright, persistence rescues it
    plot_timeseries_zoom(test_df, 'NOx (ppb)', '2025-01-01', '2025-01-15', 'timeseries_NOx.png')

    plot_scatter_grid(test_df, results_df)

    return results_df


def demo():
    df = main()
    assert len(df) == len(POLLUTANT_COLS)
    assert (df['final_R2'] >= df['weather_only_R2'] - 0.05).all(), \
        "final model should never be meaningfully worse than weather-only for any target"
    for name in ['timeseries_PM2.5.png', 'timeseries_NOx.png', 'actual_vs_predicted_scatter_grid.png']:
        assert (CHARTS_DIR / name).exists(), f"{name} was not created"
    print("\n✓ self-check passed")


if __name__ == '__main__':
    demo()
