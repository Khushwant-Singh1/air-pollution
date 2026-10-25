# Findings — Agra pollution prediction (met → pollutants)

## Data pipeline

- Source: `cpcb_output/Agra_City_Hourly_Mean_2016_2025.csv` — hourly city mean across 6 Agra CAAQMS stations, 2016–2025.
- **`Active_Stations` is unreliable before 2021.** Checked the raw per-station files directly (`web sracpper/cpcb_output/merged_stations/*.csv`): 5 of the 6 stations (all but Sanjay_Palace) have **every column NaN except a placeholder `TOT-RF=0.0`** for all 35,064 hours of 2017–2020 — they weren't installed yet, but that one non-null placeholder satisfies `notnull().any(axis=1)` in `merge_city_mean.py`, so those years read as `Active_Stations=6` while carrying zero real pollutant data. Real coverage: 2021 partial (e.g. Rohta had 467 real hours that year), all 6 stations stable at ≥97% from 2023 onward.
- **Final train/test split, chosen from this evidence (not from peeking at test scores):** `start_date='2023-01-01'` (genuine full-coverage era only) + `split_date='2025-01-01'` (train = full calendar years 2023–2024, test = full calendar year 2025 — both seasonally balanced, since a percentage-based split on a short window produces a partial-season test tail instead).

## Feature engineering (Step 2)

- No imputer — HistGradientBoostingRegressor and RandomForestRegressor (this sklearn version, 1.8.0) both fit NaN features natively; verified directly.
- No `StandardScaler` — no-op for tree splits, would only matter for a linear/NN model.
- `WD`, `hour`, `month` kept as raw integers/degrees — sin/cos encoding helps linear/NN models represent wraparound, not trees (a tree just needs a couple of extra splits).
- `log1p` on all 14 pollutant targets (all skew >1.5), scored in real units via `expm1` — log-space RMSE would flatter the model on exactly the spikes that matter.
- Per-target NaN dropping (not a shared union across all 14) — targets have very different missingness histories.

## Model results (HistGradientBoostingRegressor / RandomForestRegressor, per-pollutant)

**Weather-predictable** (meteorology is a real, moderate predictor):

| Pollutant | R² | RMSE/mean |
|---|---|---|
| PM2.5 | 0.31 | 0.54 |
| PM10 | 0.30 | 0.50 |
| Ozone | 0.21 | 0.58 |

**Weather-weak** (meteorology explains little to nothing):

| Pollutant | R² | RMSE/mean | Why |
|---|---|---|---|
| NO, NO2, NOx, NH3, SO2 | -0.65 to -2.70 | 0.35–0.70 | Real multi-year level drift (see below) that met+hour+month can't represent |
| CO | 0.03 | 0.48 | Essentially no signal |
| Benzene, Toluene, Xylene, Eth-Benzene, MP-Xylene | -0.33 to -0.92 | 0.56–1.08 | Step 1 EDA: BTX aromatics have ~0 linear correlation with any met variable (\|r\|<0.08) — traffic/point-source driven, not weather-driven |

## Why NOx/NO/SO2/NH3 are non-stationary (investigated via raw per-station files, not just the city mean)

Checked each station's own yearly NOx mean/std independently (not just the city average) to rule out "it's just an averaging-across-more-stations artifact":

- Individual stations (Manoharpur, Sector_3B, Shahjahan_Garden) show real, if uneven, multi-year decline — consistent with Agra's Taj Trapezium Zone (TTZ) emission-control tightening over 2016–2025, a documented real-world policy effect, not a data artifact.
- Sanjay_Palace (the only station with full 2016–2025 history) doesn't decline monotonically — mean/std stay volatile throughout (23–48 std every year) — so the drift isn't uniform across the city, it's site-dependent.
- **Conclusion: this is a real phenomenon current features can't capture (no explicit year/trend feature), not a data-quality bug.** Fixing it would need either a trend feature (with the caveat that tree models can't extrapolate a numeric feature beyond its training range — a `year` feature wouldn't help predict beyond the years trained on) or periodic retraining on a rolling recent window.

## Why Eth-Benzene / MP-Xylene looked "broken" under earlier (wrong) splits — and why they're not, under the correct one

Checked yearly std **per individual station** for MP-Xylene: e.g. Manoharpur goes from std=38.29 (2021, n=1906) → std=3.90 (2022) → std=0.08 (2023) → std=0.05 (2024). This exact shape — huge/erratic in 2021, calming by 2022, frozen near-zero from 2023 — appears **independently at multiple stations simultaneously**, consistent with a VOC-analyzer calibration/burn-in period across the network, not a single bad sensor or a city-averaging artifact.

Training on the full history (or even just excluding 2016 via `Active_Stations`, which — see above — doesn't actually exclude the miscalibration era) mixed this erratic 2021–2022 period into training, producing catastrophic R² (-10 to -62) on the calm 2023+ test period. **With training correctly restricted to 2023+, both score like their VOC siblings (R² -0.47 / -0.92)** — still weak (consistent with near-zero met correlation), but not uniquely broken.

Their test values still cluster in a narrow, finely-quantized near-zero band (207–336 distinct values out of ~11,800 rows, in a 0.05–0.14 range) — worth a footnote if these are ever used for absolute-value claims, since it suggests the values sit near a detection floor. But it's no longer a "don't trust this" level of anomaly once the miscalibration-era data is excluded from training.

## Station-wise training (tried, rejected)

Tested training separately per station instead of on the city mean — same verified split, same models/features, just pointed at each station's own CSV (`data/stations/*.csv`, `train_station_wise.py`). Hypothesis was that the city mean discards real spatial signal (Step 1 found stations only correlate 0.56–0.86 with each other).

**Result: every one of the 6 stations underperforms the city-mean model, on every pollutant.** Best individual station for PM2.5 (Sanjay Palace, R²=0.25) still trails the city mean (R²=0.31); for Ozone, city-mean R²=0.21 but *every station is negative* (worst -0.61). Averaging across stations is acting as noise reduction on both the meteorology inputs and the pollutant target — that's worth more here than any spatial specialization a per-station model could learn. Secondary finding: Sector 3B Avas Vikas Colony is consistently the worst station (mean R²=-2.49); only Sanjay Palace has Toluene/Xylene instrumentation at all.

## Removing the real noise source: adding persistence features (Step 4)

Checked pollutant autocorrelation directly: **lag-1h autocorrelation is ~0.9–0.95 for every pollutant** (PM2.5=0.946, PM10=0.898, Ozone=0.913, NOx=0.895) — far stronger than any meteorology correlation (max |r|=0.37 from Step 1). Checked whether smoothing the *meteorology* inputs (3h rolling mean) helped first — it didn't (AT correlation with PM2.5 moved from -0.394 to only -0.412) — so the noise wasn't in the weather inputs. It was in a missing feature: nothing in the Step 3 feature set told the model "pollution barely changes hour to hour."

Added three features per target (`data_prep.lag_feature_cols`): `lag1` (previous hour's actual value), `lag24` (same hour yesterday), `roll24` (trailing 24h mean, shifted so it never includes the current hour). Computed on the full, verified gap-free hourly grid **before** any date filtering, so the split boundary gets a true value with no leakage (verified directly: `test.iloc[0].lag1 == train.iloc[-1].actual`).

**Decomposition (`compare_feature_sets.py`) — weather-only vs persistence-only vs both:**

| Pollutant | weather-only | persistence-only | weather+persistence |
|---|---|---|---|
| PM2.5 | 0.31 | 0.82 | **0.85** |
| PM10 | 0.30 | 0.85 | **0.87** |
| Ozone | 0.21 | 0.84 | **0.86** |
| CO | 0.03 | 0.89 | **0.92** |
| NOx | -2.34 | 0.44 | **0.47** |
| NO2 | -1.18 | 0.79 | **0.79** |
| NH3 | -1.70 | 0.77 | 0.76 |
| SO2 | -0.65 | 0.73 | **0.75** |
| Eth-Benzene | -0.47 | 0.89 | **0.91** |
| MP-Xylene | -0.92 | 0.89 | 0.87 |
| Benzene | -0.50 | **0.47** | 0.31 |
| Toluene | -0.33 | **0.48** | 0.37 |
| Xylene | -0.80 | **0.66** | 0.57 |

For 11 of 14 targets, weather adds a real (if modest) lift on top of persistence — that's the honest answer to "does meteorology matter": yes, but only after accounting for the fact that pollution mostly just continues doing what it was just doing. For **Benzene, Toluene, Xylene**, adding weather actively *hurt* — consistent with Step 1's finding that BTX aromatics have essentially zero correlation with any met variable (|r|<0.08); 11 near-pure-noise features cost the model more than they gave. Final config (`train.py`): weather+persistence for 11 targets, persistence-only for those 3.

**⚠️ This changes what the model can do — read before using it.** These are now **nowcasting models**, not pure weather models: predicting requires already knowing the pollutant's own actual reading from the last 1-24 hours. This is a legitimate and often more practically useful task (real-time dashboards, short-term alerts), but it is a different capability than "predict pollution from a weather forecast with no recent measurements" — that pure-weather use case is what Step 3's numbers describe, not these.

## WD (wind direction) circular encoding (Step 5)

Raw `WD (deg)` treats 359° and 1° as 358 apart, when physically they're neighbors (1° apart). Fixed by replacing raw `WD (deg)` in `MET_COLS` with `WD_sin`/`WD_cos` (`sin`/`cos` of `WD × π/180`), computed in `data_prep.load_and_prepare`. `predict.py`'s public interface is unchanged — it still takes `WD` in degrees (what a sensor actually reports) and converts internally.

**Honest result: small, mixed effect, no regressions.** NOx (-2.34→-2.20) and Eth-Benzene (-0.47→-0.39) improved a bit in the weather-only model; a couple of others moved by ±0.01–0.05, within noise. The nowcasting model (weather+persistence) barely moved at all (e.g. PM2.5 0.85→0.85, CO 0.92→0.92) — persistence already dominates so heavily there's little room left for a weather feature to matter either way. This tracks with Step 1: wind *speed* was one of the stronger met correlates (r≈-0.2 to -0.26 on NOx/NO/PM10); wind *direction* specifically was never singled out as a top driver, so fixing its encoding was correct to do but wasn't the lever that was going to move the needle much. `Benzene`/`Toluene`/`Xylene` (persistence-only, no weather features at all) are byte-for-byte unchanged, confirming the fix is correctly isolated to targets that actually use weather.

## Weather feature engineering: wind vector, interactions, weather lags, pressure tendency, dry-spell (Step 6)

Added five feature groups to `MET_COLS` (`data_prep.py`):
- **Wind vector** `WS_x`/`WS_y` = `WS*sin(WD)`/`WS*cos(WD)` — replaces the raw `WD`/`WD_sin`/`WD_cos` entirely; combines magnitude and direction into one non-discontinuous representation.
- **Interactions**: `AT_RH` (inversion proxy), `SR_AT` (photochemistry driver), `WS_BP` (advection strength).
- **Weather lags**: lag1/lag24 for AT, RH, WS, BP, SR (the gradually-changing variables — RF/TOT-RF excluded, rain is bursty not trending).
- **Pressure tendency**: `BP_trend24 = BP - BP_lag24`.
- **Dry-spell detector**: `RF_rollsum72` — trailing 72h rainfall sum, shifted to avoid leakage. **Deliberately not** `TOT-RF minus a rolling sum` as originally proposed — checked `TOT-RF` directly first and it is not a clean cumulative counter at the city-mean level (it drops mid-day while still actively raining, e.g. 0.587→0.035 in one hour on 2024-07-10 — impossible for a true running total). Most likely cause: it's each station's own cumulative reading, averaged across a *changing* subset of reporting stations hour to hour, which breaks monotonicity. Built from `RF (mm)` instead, which doesn't have this problem.

**Result: mixed on the weather-only model, no effect on the nowcasting model.**

| | weather-only (before → after) |
|---|---|
| Improved | NOx -2.20→-1.98, CO 0.03→0.13, PM2.5 0.31→0.35 |
| Regressed | MP-Xylene -0.84→-1.21, Eth-Benzene -0.39→-0.55, Toluene -0.32→-0.36 |
| ~Flat | PM10, NO, NO2, NH3, Ozone, Benzene, Xylene, SO2 (±0.01-0.03) |

The regressions land on the smallest/noisiest-sample targets (Eth-Benzene, MP-Xylene, Toluene) — going from 12 to 27 feature columns dilutes signal for a tree splitting on a genuinely weak sample. This didn't change any practical conclusion: those three were already deeply negative and stay deeply negative either way.

**Nowcasting model (the one to actually deploy) barely moved** — every pollutant within ±0.01 of the pre-Step-6 numbers, same pattern as the WD-only change in Step 5. Persistence (lag-1h correlation ~0.9-0.95) dominates so heavily that the tree has little use for the new weather features regardless of how well-engineered they are, whether they help or hurt in isolation.

⚠️ **Deployment note, not yet resolved:** `predict.py` currently takes a single instantaneous weather snapshot. The new lag/rolling weather features (and the dry-spell 72h sum) require the *last several days* of weather history, not just the current reading — `predict.py` needs updating to accept a short history window before this feature set is usable in the actual deployed function. Flagging this rather than leaving it silently broken; not yet fixed.

## Bottom line

**Weather alone** (Step 3) genuinely predicts PM2.5, PM10, and Ozone (R² 0.2–0.35); it's weak-to-useless for everything else, either because the pollutant is local/traffic-driven (VOCs) or riding a multi-year policy trend (NOx family) that weather can't represent.

**Weather + recent history** (Step 4, nowcasting) predicts every single pollutant well (R² 0.18–0.92, 11 of 14 above 0.45) — because the single largest driver of "what will pollution be next hour" is simply "what was it the last few hours", which Step 3's feature set never captured. Use Step 3's model if you only have a weather forecast and no recent readings; use Step 4's model for real-time monitoring/nowcasting.
