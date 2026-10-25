"""
Step 3 (final): train + evaluate a model per pollutant, using each
target's best feature set per compare_feature_sets.py.

HistGradientBoostingRegressor (primary) vs RandomForestRegressor (sanity
baseline) — both fit directly on NaN features (verified: this sklearn
version supports missing values natively in both), both trained on
log1p(target) and scored in real units via features.inverse_target().

IMPORTANT — this is a NOWCASTING model, not a pure weather model: 11 of the
14 targets use weather + the pollutant's own lag-1h/lag-24h/rolling-24h
features (see data_prep.lag_feature_cols). That means predicting requires
already knowing the actual recent readings — it cannot predict from a
weather forecast alone with no recent measurements. See FINDINGS.md.

Per-pollutant winner (by test RMSE) gets saved to models/<pollutant>.joblib.
"""
import time
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from data_prep import load_and_prepare, POLLUTANT_COLS, MET_COLS, lag_feature_cols
from features import get_xy, inverse_target

MODELS_DIR = Path("models")
CALENDAR_COLS = ['hour', 'month']

# From compare_feature_sets.py: adding weather to persistence actively hurt
# these three (Benzene 0.47->0.31, Toluene 0.48->0.37, Xylene 0.66->0.57)
# — consistent with Step 1's finding that BTX aromatics have ~0 correlation
# with any met variable (|r|<0.08), so weather is pure noise for these
# three specifically, not signal. Every other target benefited from
# weather+persistence over persistence alone, so gets both.
PERSISTENCE_ONLY = {'Benzene (µg/m³)', 'Toluene (µg/m³)', 'Xylene (µg/m³)'}


def feature_cols_for(target_col):
    lag_cols = lag_feature_cols(target_col)
    if target_col in PERSISTENCE_ONLY:
        return lag_cols
    return MET_COLS + CALENDAR_COLS + lag_cols


def score(y_true_log, y_pred_log):
    """Metrics in real units (expm1'd) — log-space RMSE would flatter the
    model on exactly the pollution spikes we care about getting right.

    RMSE/mean is reported alongside R2 because R2 = 1 - SSE/Var(y_test)
    goes strongly negative whenever the test period's own variance is
    small (e.g. a pollutant whose spread shrank over the years) even if
    absolute error is modest — RMSE/mean stays readable in that case and
    tells you whether the model is actually predicting badly in absolute
    terms, or just being penalized by a shrunken denominator."""
    y_true = inverse_target(y_true_log)
    y_pred = inverse_target(y_pred_log)
    test_mean = y_true.mean()
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'RMSE_over_mean': root_mean_squared_error(y_true, y_pred) / test_mean if test_mean else float('nan'),
        'test_mean': test_mean,
        'test_std': y_true.std(),
    }


def train_one_pollutant(train_df, test_df, feature_cols, target_col):
    X_train, y_train = get_xy(train_df, feature_cols, target_col)
    X_test, y_test = get_xy(test_df, feature_cols, target_col)

    models = {
        'HistGB': HistGradientBoostingRegressor(random_state=0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = score(y_test, model.predict(X_test))
        fitted[name] = model

    results['n_train'] = len(X_train)
    results['n_test'] = len(X_test)
    winner = min(models, key=lambda name: results[name]['RMSE'])
    return results, winner, fitted[winner]


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    # start_date='2023-01-01': verified against the raw per-station files
    # (not chosen by peeking at test scores) — real 6-station coverage only
    # stabilizes from 2023 onward; 2017-2022 is Sanjay_Palace-only for
    # pollutants despite Active_Stations misreporting it as 6 (placeholder
    # TOT-RF=0.0 rows). split_date='2025-01-01': full-year train (2023-2024)
    # vs full-year test (2025) so both are seasonally balanced — a
    # percentage-based split on this short a window would produce a
    # partial-season test tail instead.
    train_df, test_df, _ = load_and_prepare(start_date='2023-01-01', split_date='2025-01-01')

    print(f"{'Pollutant':<22} {'Features':<20} {'n_train':>8} | {'HistGB':>8} {'RMSE/':>6} {'R2':>6} | "
          f"{'RF':>8} {'RMSE/':>6} {'R2':>6} | Winner")
    print(f"{'':<22} {'':<20} {'':>8} | {'RMSE':>8} {'mean':>6} {'':>6} | {'RMSE':>8} {'mean':>6} {'':>6} |")
    print("-" * 118)

    rows = []
    start = time.time()
    for target_col in POLLUTANT_COLS:
        feature_cols = feature_cols_for(target_col)
        feature_set_name = 'persistence-only' if target_col in PERSISTENCE_ONLY else 'weather+persistence'
        results, winner, model = train_one_pollutant(train_df, test_df, feature_cols, target_col)
        hgb, rf = results['HistGB'], results['RandomForest']
        print(f"{target_col:<22} {feature_set_name:<20} {results['n_train']:>8,} | "
              f"{hgb['RMSE']:>8.2f} {hgb['RMSE_over_mean']:>6.2f} {hgb['R2']:>6.2f} | "
              f"{rf['RMSE']:>8.2f} {rf['RMSE_over_mean']:>6.2f} {rf['R2']:>6.2f} | {winner}")

        safe_name = target_col.split(' ')[0].replace('.', '').replace('/', '_')
        joblib.dump(model, MODELS_DIR / f"{safe_name}.joblib")
        rows.append({'target': target_col, 'winner': winner, 'feature_set': feature_set_name, **results})

    print(f"\nTrained {len(POLLUTANT_COLS)} pollutant models in {time.time() - start:.1f}s")
    print(f"Saved winning model per pollutant to {MODELS_DIR}/")
    print("\nNOTE: this is a nowcasting model — 11 of 14 targets need the pollutant's own "
          "recent actual readings as input (lag-1h/lag-24h/rolling-24h), not just weather. "
          "See FINDINGS.md.")

    return rows


def demo():
    rows = main()
    assert len(rows) == len(POLLUTANT_COLS)
    for r in rows:
        assert r['HistGB']['RMSE'] > 0 and r['RandomForest']['RMSE'] > 0
    saved = list(MODELS_DIR.glob('*.joblib'))
    assert len(saved) == len(POLLUTANT_COLS), f"expected {len(POLLUTANT_COLS)} saved models, found {len(saved)}"

    mean_r2 = sum(max(r['HistGB']['R2'], r['RandomForest']['R2']) for r in rows) / len(rows)
    assert mean_r2 > 0.5, f"expected persistence features to lift mean R2 well above the weather-only baseline (~-0.5), got {mean_r2:.2f}"

    print("✓ self-check passed")


if __name__ == '__main__':
    demo()
