"""
Checks whether the algorithm is actually the bottleneck, by comparing our
current non-linear tree models against a genuinely linear baseline
(LinearRegression) and a different non-linear approach (MLPRegressor, a
small neural net — already in sklearn, no new dependency) on the same
data/split/features.

If HistGB/RF already beat LinearRegression by a wide margin, that's direct
evidence the current models are already capturing the real non-linear
structure — the remaining error is a signal ceiling, not an algorithm
limitation. If MLPRegressor doesn't beat HistGB/RF either, that confirms
it's not "trees specifically" that's the limit, it's genuinely how much
these particular inputs can explain.
"""
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error

from data_prep import load_and_prepare
from features import get_xy, inverse_target
from train import feature_cols_for

# One from each bucket in FINDINGS.md: strong (PM2.5), weak-with-persistence
# (NO, the single weakest target at R2=0.18), persistence-only (Benzene).
TARGETS_TO_CHECK = ['PM2.5 (µg/m³)', 'NO (µg/m³)', 'Benzene (µg/m³)']


def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred_log = model.predict(X_test)
    y_test_real, pred_real = inverse_target(y_test), inverse_target(pred_log)
    return r2_score(y_test_real, pred_real), root_mean_squared_error(y_test_real, pred_real)


def main():
    train_df, test_df, _ = load_and_prepare(start_date='2023-01-01', split_date='2025-01-01')

    print(f"{'Target':<18} {'Algorithm':<20} {'R2':>7} {'RMSE':>8}  Non-linear?")
    print("-" * 70)

    for target_col in TARGETS_TO_CHECK:
        feature_cols = feature_cols_for(target_col)
        X_train, y_train = get_xy(train_df, feature_cols, target_col)
        X_test, y_test = get_xy(test_df, feature_cols, target_col)

        models = {
            'LinearRegression': (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LinearRegression()), 'No'),
            'HistGB (current)': (HistGradientBoostingRegressor(random_state=0), 'Yes'),
            'RandomForest (current)': (RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1), 'Yes'),
            'MLPRegressor (small NN)': (make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                                        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0)), 'Yes'),
        }

        for name, (model, is_nonlinear) in models.items():
            r2, rmse = evaluate(model, X_train, y_train, X_test, y_test)
            print(f"{target_col:<18} {name:<20} {r2:>7.2f} {rmse:>8.2f}  {is_nonlinear}")
        print()


if __name__ == '__main__':
    main()
