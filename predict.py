"""
Deployment entry point for the weather-only model: recent hourly weather
history in, all 14 pollutant predictions out (for the LATEST hour in that
history).

Usage:
    from predict import predict
    import pandas as pd

    history = pd.DataFrame([
        {'Timestamp': '2025-06-12 14:00:00', 'AT (°C)': 34.1, 'RH (%)': 38.0, 'WS (m/s)': 1.8,
         'WD (deg)': 190.0, 'RF (mm)': 0.0, 'TOT-RF (mm)': 0.0, 'SR (W/mt2)': 700.0,
         'BP (mmHg)': 741.0, 'VWS (m/s)': -0.1},
        ... one row per hour, chronological, no gaps ...
        {'Timestamp': '2025-06-15 14:00:00', 'AT (°C)': 32.5, 'RH (%)': 45.0, 'WS (m/s)': 2.1,
         'WD (deg)': 180.0, 'RF (mm)': 0.0, 'TOT-RF (mm)': 0.0, 'SR (W/mt2)': 650.0,
         'BP (mmHg)': 740.0, 'VWS (m/s)': -0.1},
    ])
    predict(history)
    -> {'PM2.5 (µg/m³)': 41.2, 'PM10 (µg/m³)': 88.7, ...}  # prediction for the LAST row's hour

Needs history because several features are the pollutant's-own-weather's
recent trend, not just the current instant (see FINDINGS.md Step 6):
lag1/lag24 for AT/RH/WS/BP/SR, a 72h rainfall rolling sum (dry-spell
detector), and a 24h pressure tendency. Minimum ~25 hours of history for
lag24 to be non-NaN; 73+ hours (3 days) recommended so the rainfall
rolling sum isn't truncated. Shorter history still runs — the underlying
models handle NaN features natively — just with a weaker signal on
those specific columns, same tradeoff training itself makes early in a
series.

This is the weather-only model (see FINDINGS.md "Bottom line") — no
persistence/lag-of-the-pollutant-itself inputs. For the stronger
nowcasting model (needs the pollutant's own recent readings too, not just
weather), load from models/ instead of models_weather_only/ and add each
target's own lag_feature_cols.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from data_prep import POLLUTANT_COLS, MET_COLS, compute_weather_features

MODELS_DIR = Path("models_weather_only")
CALENDAR_COLS = ['hour', 'month']
MIN_HISTORY_HOURS = 25  # lag24 needs at least this many rows to be non-NaN

_models = {}  # loaded lazily, cached — 14 small joblib files, cheap to keep in memory


def _load(target_col):
    if target_col not in _models:
        safe_name = target_col.split(' ')[0].replace('.', '').replace('/', '_')
        _models[target_col] = joblib.load(MODELS_DIR / f"{safe_name}.joblib")
    return _models[target_col]


def predict(weather_history):
    """weather_history: DataFrame with columns Timestamp, AT (°C), RH (%),
    WS (m/s), WD (deg), RF (mm), TOT-RF (mm), SR (W/mt2), BP (mmHg),
    VWS (m/s) — raw readings, one row per hour, no gaps. Order doesn't
    matter (sorted here) but coverage must be contiguous hourly for the
    lag/rolling features to mean what they claim to. Returns predictions
    for the LAST (most recent) hour in the history."""
    if len(weather_history) < MIN_HISTORY_HOURS:
        raise ValueError(f"Need at least {MIN_HISTORY_HOURS} hours of history for lag24 features, "
                          f"got {len(weather_history)}")

    df = weather_history.sort_values('Timestamp').reset_index(drop=True)
    df = compute_weather_features(df)

    now = df.iloc[[-1]].copy()
    ts = pd.Timestamp(now.iloc[0]['Timestamp'])
    now['hour'] = ts.hour
    now['month'] = ts.month
    row = now[MET_COLS + CALENDAR_COLS]  # fixed column order, matches training

    predictions = {}
    for target_col in POLLUTANT_COLS:
        model = _load(target_col)
        pred_log = model.predict(row)[0]
        predictions[target_col] = float(np.expm1(pred_log))  # undo training's log1p
    return predictions


def demo():
    # Self-check against real data rather than synthetic: pull 100 real
    # consecutive hours from the actual dataset, run predict() on the raw
    # columns, and confirm it matches what load_and_prepare's own
    # precomputed features give the same model for that exact timestamp —
    # proving predict.py's feature computation and training's feature
    # computation haven't drifted apart (they share compute_weather_features,
    # but this catches it if that sharing ever breaks).
    from data_prep import load_and_prepare, CSV_PATH

    raw = pd.read_csv(CSV_PATH, parse_dates=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    RAW_COLS = ['Timestamp', 'AT (°C)', 'RH (%)', 'WS (m/s)', 'WD (deg)',
                'RF (mm)', 'TOT-RF (mm)', 'SR (W/mt2)', 'BP (mmHg)', 'VWS (m/s)']
    window = raw[RAW_COLS].iloc[-100:].reset_index(drop=True)
    target_ts = window.iloc[-1]['Timestamp']

    result = predict(window)
    print(f"Prediction for {target_ts}:")
    for pollutant, value in result.items():
        print(f"  {pollutant:<22} {value:8.2f}")

    train_df, test_df, feature_cols = load_and_prepare()
    full = pd.concat([train_df, test_df])
    reference_row = full[full['Timestamp'] == target_ts]
    assert len(reference_row) == 1, "target timestamp not found in load_and_prepare's output"
    model = _load('PM2.5 (µg/m³)')
    reference_pred = float(np.expm1(model.predict(reference_row[feature_cols])[0]))
    assert abs(result['PM2.5 (µg/m³)'] - reference_pred) < 0.01, \
        f"predict.py's own feature computation drifted from training's: {result['PM2.5 (µg/m³)']} vs {reference_pred}"

    assert len(result) == len(POLLUTANT_COLS)
    assert all(v >= 0 for v in result.values())
    print("\n✓ self-check passed (matches load_and_prepare's features exactly)")


if __name__ == '__main__':
    demo()
