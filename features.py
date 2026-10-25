"""
Step 2: feature engineering for the Agra pollution-prediction project.

Per the EDA findings (see charts/*.py + the synthesis from Step 1):
- No imputer: HistGradientBoostingRegressor and RandomForestRegressor both
  take NaN features natively in this sklearn version. Feature columns keep
  their NaNs as-is.
- No StandardScaler: tree splits are invariant to feature scale, so scaling
  is a no-op for this model family. Skipped entirely.
- No sin/cos encoding for WD/hour/month: that helps linear/NN models, not
  trees (a tree just needs a couple of splits to handle wraparound). Kept
  as raw integers/degrees.
- log1p on every target: all 14 pollutants are right-skewed (skew 1.7-48).
  Train in log space, invert with expm1 before scoring/reporting.
- No outlier capping: checked, nothing looked like a sensor error.

So the only real transform left is per-target: drop rows where THIS target
is NaN (targets have very different missingness — dropping the union
across all 14 would gut PM10/NH3/Eth-Benzene/MP-Xylene, which only started
being measured in 2021), and log1p the target.
"""
import numpy as np


def get_xy(df, feature_cols, target_col, cap_percentile=None):
    """Rows where target_col is present; features unmodified (NaNs kept),
    target optionally capped at cap_percentile and log1p-transformed.
    Use inverse_target() to get back to real units."""
    mask = df[target_col].notna()
    X = df.loc[mask, feature_cols]
    y_raw = df.loc[mask, target_col]
    if cap_percentile is not None:
        y_raw = y_raw.clip(upper=y_raw.quantile(cap_percentile))
    y = np.log1p(y_raw)
    return X, y


def inverse_target(y_log):
    """Undo get_xy's log1p so predictions/metrics are in the pollutant's real units."""
    return np.expm1(y_log)


def demo():
    from data_prep import load_and_prepare, POLLUTANT_COLS

    train_df, test_df, feature_cols = load_and_prepare()

    # PM10 only has real readings from 2021 onward (sensor installed then) —
    # confirm get_xy's per-target dropna actually captures that, not the
    # full-history row count.
    X, y = get_xy(train_df, feature_cols, 'PM10 (µg/m³)')
    print(f"PM10: {len(X):,} usable training rows (of {len(train_df):,} total)")
    assert 20_000 < len(X) < 35_000, f"expected ~27k post-2021 rows, got {len(X)}"

    # round-trip check
    recovered = inverse_target(y)
    assert np.allclose(recovered, train_df.loc[X.index, 'PM10 (µg/m³)'], rtol=1e-6)

    # a target with near-full history (PM2.5, ~4.6% missing) should keep most rows
    X_pm25, _ = get_xy(train_df, feature_cols, 'PM2.5 (µg/m³)')
    assert len(X_pm25) > 0.9 * len(train_df)

    # every pollutant should produce a usable non-empty train+test split
    for col in POLLUTANT_COLS:
        X_tr, y_tr = get_xy(train_df, feature_cols, col)
        X_te, y_te = get_xy(test_df, feature_cols, col)
        assert len(X_tr) > 0 and len(X_te) > 0, f"{col} has no usable rows in train or test"

    print("✓ self-check passed")


if __name__ == '__main__':
    demo()
