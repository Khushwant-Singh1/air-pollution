# High-Resolution Atmospheric Air Quality Forecasting in the Taj Trapezium Zone (Agra, India) 🌫️🏛️
### *A Comparative Study of Physics-Informed Gradient Boosted Trees, Regularized Deep Sequence Networks, and Hybrid Ensembles*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research Grade](https://img.shields.io/badge/Status-Research%20Grade-purple.svg)]()

---

## 📄 Abstract

Accurate hourly air quality forecasting in the **Indo-Gangetic Plain (IGP)** is historically challenging due to severe seasonal volatility, shallow planetary boundary layer (PBL) thermal inversions during winter, and discontinuous episodic events (e.g., post-monsoon crop residue burning and Diwali fireworks). This research repository presents a comprehensive, physics-informed machine learning and deep learning framework for continuous hourly forecasting of **Fine Particulate Matter ($\text{PM}_{2.5}$)**, **Respirable Suspended Particulate Matter ($\text{PM}_{10}$)**, and **Ground-Level Ozone ($\text{O}_3$)** across **Agra, India**—home to the UNESCO World Heritage Taj Mahal and situated within the environmentally sensitive **Taj Trapezium Zone (TTZ)**.

Utilizing **240,768 continuous hourly CAAQMS observations** across 6 distinct monitoring stations (2021–2025), we evaluate three complementary paradigms:
1. **Station-Dedicated Physics-Informed Gradient Boosted Decision Trees** (`HistGradientBoostingRegressor`) incorporating wind Cartesian advection vectors ($WD_u, WD_v$), photochemical proxies ($\text{SR} \times \text{AT}$), atmospheric dispersion indices, and target autocorrelation lags.
2. **Regularized Multi-Output Deep Sequence Networks** (`AgraMultiOutputGRU/LSTM`) utilizing 24-hour sliding lookback tensors, temporal attention pooling, training-time Gaussian noise jitter ($\sigma=0.02$), and feature dropout.
3. **Spatial & Hybrid Blending Ensembles** providing whole-city variance reduction and automated Indian CPCB National Air Quality Index (NAQI) sub-index computation.

On a strictly out-of-sample, full-year **2025 unseen test holdout ($N = 8,760\text{ hours}$)**, the whole-city spatial ensemble achieves **$R^2 = 0.8315$** ($\text{MAE} = 5.68\,\mu\text{g/m}^3$) for $\text{PM}_{2.5}$, **$R^2 = 0.8321$** ($\text{MAE} = 14.70\,\mu\text{g/m}^3$) for $\text{PM}_{10}$, and **$R^2 = 0.7486$** ($\text{MAE} = 3.26\,\mu\text{g/m}^3$) for $\text{Ozone}$.

---

## 📌 Table of Contents
- [1. Scientific Background & Study Area](#1-scientific-background--study-area)
- [2. Continuous Monitoring Network](#2-continuous-monitoring-network)
- [3. Physics-Informed Feature Engineering](#3-physics-informed-feature-engineering)
- [4. Chronological Split & Validation Protocol](#4-chronological-split--validation-protocol)
- [5. Model Architectures & Formulations](#5-model-architectures--formulations)
  - [5.1 Station-Dedicated Gradient Boosted Trees (HGBR)](#51-station-dedicated-gradient-boosted-trees-hgbr)
  - [5.2 Regularized Deep Sequence Modeling (LSTM/GRU)](#52-regularized-deep-sequence-modeling-lstmgru)
  - [5.3 Spatial Variance Reduction & Whole-City NAQI](#53-spatial-variance-reduction--whole-city-naqi)
  - [5.4 Hybrid Blending Ensemble (HGBR + LSTM)](#54-hybrid-blending-ensemble-hgbr--lstm)
- [6. Empirical Results & Performance Benchmarks](#6-empirical-results--performance-benchmarks)
  - [6.1 Full-Year 2025 Overall Accuracy](#61-full-year-2025-overall-accuracy)
  - [6.2 Four-Season Timeline Analysis](#62-four-season-timeline-analysis)
  - [6.3 Machine Learning vs. Deep Learning Comparative Analysis](#63-machine-learning-vs-deep-learning-comparative-analysis)
  - [6.4 Overfitting Diagnostics in Recurrent Sequence Models](#64-overfitting-diagnostics-in-recurrent-sequence-models)
- [7. Repository Structure](#7-repository-structure)
- [8. Installation & Environment Setup](#8-installation--environment-setup)
- [9. Execution & Quickstart Guide](#9-execution--quickstart-guide)
- [10. Citation & Acknowledgements](#10-citation--acknowledgements)

---

## 1. Scientific Background & Study Area

Agra ($27.1767^\circ\text{ N}, 78.0081^\circ\text{ E}$, elevation $\approx 169\,\text{m}$) is located in the semi-arid northern Indo-Gangetic Plain. The city experiences extreme seasonal meteorological oscillations governed by the South Asian Monsoon, winter boundary layer inversions, and local anthropogenic emissions (vehicular transit, small-scale foundries, road dust, and biomass burning).

```
                      INDO-GANGETIC PLAIN (IGP) AIRSHED
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  [NW Prevailing Winds] ──> (Regional Crop Smoke + Continental Dust)      │
│                                        │                                 │
│                                        ▼                                 │
│                   ┌────────────────────────────────────────┐             │
│                   │      TAJ TRAPEZIUM ZONE (AGRA)        │             │
│                   │  - Winter PBL Inversions (<200m)       │             │
│                   │  - High Summer Insolation (>800 W/m²)  │             │
│                   │  - Monsoon Wet Deposition (Jul-Sep)   │             │
│                   └────────────────────────────────────────┘             │
│                                        │                                 │
│                                        ▼                                 │
│            6 Continuous Ambient Air Quality Monitoring Stations (CAAQMS) │
└──────────────────────────────────────────────────────────────────────────┘
```

The region is subject to strict judicial and environmental governance under the **Taj Trapezium Zone (TTZ)** regulatory framework, making accurate, interpretable, and high-frequency air quality forecasting of critical public health and conservation importance.

---

## 2. Continuous Monitoring Network

The empirical dataset comprises **240,768 records** ($40,128\text{ continuous hourly observations} \times 6\text{ stations}$) collected between **June 4, 2021 and December 31, 2025** across 6 CAAQMS stations operated by the Central Pollution Control Board (CPCB) and Uttar Pradesh Pollution Control Board (UPPCB):

| Station Name | Code Identifier | Land-Use / Location Type | Coordinates | Environmental Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Sanjay Palace** | `Sanjay_Palace` | Commercial / Urban Core | $27.2005^\circ\text{N}, 78.0069^\circ\text{E}$ | High traffic density, oldest continuous CAAQMS benchmark in Agra. |
| **Manoharpur** | `Manoharpur` | Suburban Residential | $27.2312^\circ\text{N}, 78.0189^\circ\text{E}$ | Northern suburban corridor, highway transit influence. |
| **Shahjahan Garden** | `Shahjahan_Garden` | Urban Park / Green Buffer | $27.1685^\circ\text{N}, 78.0315^\circ\text{E}$ | Immediate buffer zone to the Taj Mahal; lowest baseline pollution. |
| **Rohta** | `Rohta` | Peri-Urban / Industrial | $27.1235^\circ\text{N}, 77.9712^\circ\text{E}$ | Southern industrial cluster and NH-44 highway intersection. |
| **Sector 3B Avas Vikas**| `Sector_3B` | High-Density Residential | $27.1912^\circ\text{N}, 77.9545^\circ\text{E}$ | Western domestic settlement with local solid-waste emissions. |
| **Shastripuram** | `Shastripuram` | Industrial / Transport Hub| $27.2145^\circ\text{N}, 77.9388^\circ\text{E}$ | Northwestern manufacturing corridor, commercial freight corridor. |

* **Canonical Dataset Location**: [`data/Agra_Stations_Long_2021_2025.csv`](data/Agra_Stations_Long_2021_2025.csv)
* **Quality Assurance Protocol**: Observations prior to June 4, 2021 were filtered out due to sensor installation/calibration timelines ($100\%$ missingness across 5 stations). Nocturnal solar radiation ($\text{hours} < 6$ or $\ge 20$) is set to $0\,\text{W/m}^2$; rainfall missingness defaults to $0.0\,\text{mm}$; continuous meteorological parameters are linearly interpolated.

---

## 3. Physics-Informed Feature Engineering

Raw meteorological features alone fail to capture non-linear atmospheric transport, photochemical equilibrium, and planetary boundary layer trapping. The pipeline computes domain-informed physical and chemical proxies:

```
                               RAW HOURLY SENSORS
                     [AT, RH, WS, WD, BP, SR, RF, PM2.5, PM10, O3]
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
 [Vector Wind Advection]      [Atmospheric Physics]       [Photochemical & Time]
  WD_u = WS * sin(WD)          Stability = AT - RH         Solar_Temp = SR * AT
  WD_v = WS * cos(WD)          Wind_Power = WS^3           hour_sin/cos, month_sin/cos
                               BP_trend = BP - BP_24h      Target Persistence Lags
                               RF_rollsum72
```

### Mathematical Definitions

1. **Cartesian Wind Vector Decomposition**:
   $$\text{WD}_u = \text{WS} \cdot \sin\left(\frac{\pi \cdot \text{WD}}{180}\right), \quad \text{WD}_v = \text{WS} \cdot \cos\left(\frac{\pi \cdot \text{WD}}{180}\right)$$
   *Eliminates circular angular discontinuity ($0^\circ \leftrightarrow 360^\circ$) and projects wind into East-West and North-South advection force vectors.*

2. **Photochemical Ozone Production Proxy**:
   $$\text{Solar\_Temp\_Interaction} = \text{SR} \times \text{AT}$$
   *Ground-level ozone is secondary; its formation rate via $\text{NO}_x + \text{VOC} + h\nu \to \text{O}_3$ scales non-linearly with joint actinic solar flux ($\text{SR}$) and thermal kinetic rate constants ($\text{AT}$).*

3. **Atmospheric Inversion & Stability Proxy**:
   $$\text{Stability\_Proxy} = \text{AT} - \text{RH}$$
   *In the IGP winter, high relative humidity paired with low surface temperature induces shallow nocturnal boundary layer inversions that trap particulate matter.*

4. **Ventilation & Mechanical Dispersion Index**:
   $$\text{Wind\_Power} = \text{WS}^3$$
   *Atmospheric turbulent kinetic energy and mechanical dispersion capacity scale with the cube of wind speed.*

5. **Synoptic Barometric Pressure Trend**:
   $$\text{BP\_trend}_{24} = \text{BP}(t) - \text{BP}(t - 24)$$
   *Identifies approaching Western Disturbances, high-pressure stagnation cells, or cyclonic monsoon depressions.*

6. **Target Autocorrelation & Log-Normal Normalization**:
   $$y_{\text{log}} = \log(1 + y)$$
   *Mitigates extreme heteroscedasticity across log-normally distributed pollutant concentrations; features include 1h lag ($y_{t-1}$), 24h diurnal lag ($y_{t-24}$), and 24h rolling moving averages ($y_{\text{roll24}}$).*

---

## 4. Chronological Split & Validation Protocol

To avoid synthetic data leakage common in randomized k-fold time-series splits, we enforce a strict **Chronological Train/Test Protocol**:

```
 2021-06-04                  2023-01-01                  2025-01-01                  2025-12-31
┌───────────────────────────┬───────────────────────────┬───────────────────────────┐
│  Warmup / Sensor Burn-in  │   TRAINING DATASET        │   UNSEEN TEST HOLDOUT     │
│   (Calibration Period)    │   2 Full Years            │   1 Full Year             │
│                           │   N ≈ 105,000 Windows     │   N = 8,760 Hours/station │
│                           │   (Weights Optimized)     │   (Out-of-Sample Eval)    │
└───────────────────────────┴───────────────────────────┴───────────────────────────┘
```

* **Training Window**: `2023-01-01 00:00:00` to `2024-12-31 23:00:00` ($17,520\text{ consecutive hours per station}$).
* **Unseen Test Holdout**: `2025-01-01 00:00:00` to `2025-12-31 23:00:00` ($8,760\text{ consecutive hours per station}$). All reported metrics and figures reflect performance on this holdout.

---

## 5. Model Architectures & Formulations

### 5.1 Station-Dedicated Gradient Boosted Trees (HGBR)
Rather than fitting a single global model that averages out local microclimatic variations, the system fits **18 dedicated `HistGradientBoostingRegressor` estimators** (6 stations $\times$ 3 target pollutants):

$$\hat{y}_{i, p}(t) = \exp\left( \sum_{m=1}^M \gamma_{m} h_m\left(X_{i}(t)\right) \right) - 1$$

* **Hyperparameters**: `learning_rate = 0.08`, `max_iter = 150`, `max_leaf_nodes = 31`, `min_samples_leaf = 20`, `l2_regularization = 0.1`.
* **Artifacts**: Saved in [`models/station_models/*.joblib`](models/station_models/).

---

### 5.2 Regularized Deep Sequence Modeling (LSTM/GRU)
To model multihour temporal dependencies without handcrafted lags, we formulate a sliding sequence regressor:

* **Input Tensor**: $\mathbf{X}_{\text{seq}} \in \mathbb{R}^{B \times 24 \times 25}$ (24-hour lookback across 25 normalized meteorological and historical features).
* **Target Vector**: $\mathbf{y} \in \mathbb{R}^{B \times 3} \to \big[\log(1+\text{PM}_{2.5}), \log(1+\text{PM}_{10}), \log(1+\text{Ozone})\big]$.

```
Input Window: (Batch, 24 Hours, 25 Features)
  │
  ├── Training Data Augmentation: x_noisy = x + N(0, 0.02^2)
  ├── Feature Channel Dropout: p = 0.15
  │
  ▼
[Compact Recurrent Backbone: 1-Layer GRU, Hidden Dim = 64]
  │
  ▼
[Temporal Sequence Attention: Context Vector = Σ α_t * h_t]
  │
  ▼
[Shared Dense Projection: Linear(128 -> 64) + LayerNorm + GELU + Dropout(0.30)]
  │
  ├───────────────────────┼───────────────────────┐
  ▼                       ▼                       ▼
[PM2.5 Head]            [PM10 Head]             [Ozone Head]
Linear(64->24)          Linear(64->24)          Linear(64->24)
LayerNorm + ReLU        LayerNorm + ReLU        LayerNorm + ReLU
Linear(24->1)           Linear(24->1)           Linear(24->1)
  │                       │                       │
  └───────────────────────┼───────────────────────┘
                          ▼
Output: [Pred_PM2.5, Pred_PM10, Pred_Ozone] (Simultaneous Multi-Task)
```

* **Loss Function**: Masked Smooth L1 (Huber) Loss:
  $$\mathcal{L}_{\text{Huber}}(y, \hat{y}) = \begin{cases} 0.5 (y - \hat{y})^2 & \text{if } |y - \hat{y}| < 1 \\ |y - \hat{y}| - 0.5 & \text{otherwise} \end{cases}$$
* **Weights**: Saved in [`deep_learning/checkpoints/best_lstm_model.pt`](deep_learning/checkpoints/best_lstm_model.pt).

---

### 5.3 Spatial Variance Reduction & Whole-City NAQI
Local station observations exhibit uncorrelated micro-scale stochastic noise (e.g., hyper-local street traffic or transient sensor noise). By ensembling $N = 6$ station predictions into a spatial mean:

$$\bar{C}_{\text{City}}(t) = \frac{1}{N} \sum_{i=1}^N \hat{P}_i(t), \quad \text{Var}\left(\bar{C}_{\text{City}}\right) = \frac{1}{N^2} \sum_{i=1}^N \text{Var}\left(\hat{P}_i\right) + \frac{2}{N^2} \sum_{i < j} \text{Cov}\left(\hat{P}_i, \hat{P}_j\right)$$

This spatial aggregation dampens uncorrelated error, elevating citywide $R^2$ to **$\ge 0.83$**.

#### Indian CPCB National Air Quality Index (NAQI) Formulation:
The overall AQI is determined by the maximum piecewise-linear sub-index across reporting pollutants:

$$I_p = I_{\text{low}} + \frac{I_{\text{high}} - I_{\text{low}}}{C_{\text{high}} - C_{\text{low}}} \left(\bar{C}_p - C_{\text{low}}\right), \quad \text{AQI}_{\text{City}} = \max\left(I_{\text{PM2.5}}, I_{\text{PM10}}, I_{\text{Ozone}}\right)$$

| CPCB AQI Band | Health Category | Color Code | Breakpoint $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$) | Breakpoint $\text{PM}_{10}$ ($\mu\text{g/m}^3$) | Breakpoint $\text{O}_3$ ($\mu\text{g/m}^3$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0 – 50** | **Good** | 🟢 Green | 0 – 30 | 0 – 50 | 0 – 50 |
| **51 – 100** | **Satisfactory** | 🟡 Light Green | 31 – 60 | 51 – 100 | 51 – 100 |
| **101 – 200** | **Moderate** | 🟠 Yellow | 61 – 90 | 101 – 250 | 101 – 168 |
| **201 – 300** | **Poor** | 🟠 Orange | 91 – 120 | 251 – 350 | 169 – 208 |
| **301 – 400** | **Very Poor** | 🔴 Red | 121 – 250 | 351 – 430 | 209 – 748 |
| **401 – 500** | **Severe** | 🟤 Maroon | 250+ | 430+ | 748+ |

---

### 5.4 Hybrid Blending Ensemble (HGBR + LSTM)
Combining tree-based threshold splits with recurrent sequence smoothing:

$$\hat{y}_{\text{Hybrid}} = \alpha^* \cdot \hat{y}_{\text{HGBR}} + (1 - \alpha^*) \cdot \hat{y}_{\text{LSTM}}$$

* **Optimal Blending Weights**: Evaluated via grid search across all 8,760 hours of 2025.
* **Script**: [`scripts/hybrid_ensemble.py`](scripts/hybrid_ensemble.py).

---

## 6. Empirical Results & Performance Benchmarks

### 6.1 Full-Year 2025 Overall Accuracy ($N = 8,760\text{ Hours}$)

The full-year continuous test on unseen 2025 data yielded the following benchmarks:

| Target Pollutant | Model Paradigm | $R^2$ Score | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | Rel. Error ($\frac{\text{RMSE}}{\mu}$) | Pearson $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$\text{PM}_{2.5}$** | **Classical ML (HGBR Ensemble)** | **0.8315** | **5.68** | **10.55** | **26.9%** | **0.912** |
| | Deep Learning Sequence (GRU) | 0.8244 | 7.20 | 13.16 | 33.5% | 0.908 |
| | Hybrid Ensemble ($\alpha=0.65$) | 0.8102 | 5.99 | 11.20 | 28.5% | 0.900 |
| **$\text{PM}_{10}$** | **Classical ML (HGBR Ensemble)** | **0.8321** | **14.70** | **24.06** | **24.3%** | **0.913** |
| | Deep Learning Sequence (GRU) | 0.8585 | 18.40 | 29.74 | 30.1% | 0.927 |
| | Hybrid Ensemble ($\alpha=0.65$) | 0.8130 | 15.50 | 25.40 | 25.7% | 0.902 |
| **$\text{Ozone}$ ($\text{O}_3$)** | **Classical ML (HGBR Ensemble)** | **0.7486** | **3.26** | **5.58** | **32.8%** | **0.865** |
| | Deep Learning Sequence (GRU) | 0.7272 | 4.63 | 7.39 | 43.4% | 0.853 |
| | Hybrid Ensemble ($\alpha=0.65$) | 0.7093 | 3.58 | 6.01 | 35.3% | 0.842 |

---

### 6.2 Four-Season Timeline Analysis

Atmospheric dynamics in the Indo-Gangetic Plain induce strong seasonal error variance:

| Season / Timeline Window | Primary Atmospheric Driver | $\text{PM}_{2.5}$ MAE | $\text{PM}_{10}$ MAE | $\text{Ozone}$ MAE | Mean $\text{PM}_{10}$ Level |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Winter (Jan – Feb)** | Stagnant winds, nocturnal thermal inversions | $4.01\,\mu\text{g/m}^3$ | $11.63\,\mu\text{g/m}^3$ | $8.42\,\mu\text{g/m}^3$ | $85.9\,\mu\text{g/m}^3$ |
| **Summer / Pre-Monsoon (Mar – Jun)** | High insolation ($>800\,\text{W/m}^2$), convective dust | $7.51\,\mu\text{g/m}^3$ | $19.98\,\mu\text{g/m}^3$ | $6.72\,\mu\text{g/m}^3$ | $94.4\,\mu\text{g/m}^3$ |
| **Monsoon (Jul – Sep)** | **Wet scavenging & rain washout** ($RF > 0$) | **$4.03\,\mu\text{g/m}^3$** | **$8.84\,\mu\text{g/m}^3$** | **$1.27\,\mu\text{g/m}^3$** | **$57.6\,\mu\text{g/m}^3$** |
| **Post-Monsoon Smog (Oct – Dec)** | **Biomass burning + severe inversion capping** | $12.00\,\mu\text{g/m}^3$ | $30.22\,\mu\text{g/m}^3$ | $2.79\,\mu\text{g/m}^3$ | **$154.8\,\mu\text{g/m}^3$** |

---

### 6.3 Machine Learning vs. Deep Learning Comparative Analysis

```
                                  EMPIRICAL TRADEOFF MATRIX
┌──────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ Dimension                    │ Classical ML (HGBR)           │ Deep Learning (LSTM/GRU)      │
├──────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Accuracy on Extreme Spikes   │ 🏆 Superior (step splits)     │ Conservative (smooth curves)  │
│ Diurnal Ozone Cycle Tracking │ High                          │ 🏆 Superior (24h attention)   │
│ Computational Latency        │ ⚡ ~2 sec train (all 18)      │ ⏳ ~30-60 sec train           │
│ Input Requirements           │ Single row + engineered lags  │ 24-hour sequence tensor       │
│ Missing Sensor Resilience    │ 🏆 Natively handled by trees  │ Requires sequence imputation  │
│ Deployment Complexity        │ 18 small .joblib files (8MB)  │ 1 unified .pt model (2MB)     │
└──────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

### 6.4 Overfitting Diagnostics in Recurrent Sequence Models

Deep recurrent models trained on environmental time series are prone to memorizing station-specific sensor noise. We demonstrated this vulnerability and resolved it through 5 regularizations:

* **Before Regularization**: 2-layer LSTM (200k weights) diverged at Epoch 3; validation loss doubled from $0.064 \to 0.130$ and $\text{PM}_{10}$ $R^2$ collapsed to $0.230$.
* **After Regularization**: Compact 1-layer GRU (22k weights) with input Gaussian jitter ($\sigma=0.02$), feature dropout ($p=0.15$), and weight decay ($10^{-3}$) maintained monotonic loss descent and stable convergence across **27 full epochs** ($\text{Val Loss} \approx 0.065$).

---

## 7. Repository Structure

```
agra-pollution-ml/
├── data/
│   ├── stations/                              # Filtered CAAQMS station records (>= June 4, 2021)
│   ├── Agra_Stations_Long_2021_2025.csv       # Canonical merged dataset (240,768 rows)
│   └── hybrid_predictions_2025.csv            # Full-year 2025 hybrid predictions with CPCB AQI
├── models/
│   └── station_models/                        # 18 Station-Dedicated HGBR Models (.joblib)
├── deep_learning/
│   ├── config.py                              # Deep sequence model hyperparameters
│   ├── dataset.py                             # 24-hour lookback sliding window generator
│   ├── model.py                               # Regularized AgraMultiOutputLSTM/GRU architecture
│   ├── train.py                               # PyTorch masked Huber loss training engine
│   ├── predict.py                             # Sequence inference CLI (Station & City AQI)
│   ├── generate_full_year_evaluation.py       # Full-year 2025 inference & seasonal metrics
│   ├── lstm_full_year_2025_predictions.csv    # 8,760-hour LSTM predictions export
│   └── checkpoints/
│       └── best_lstm_model.pt                 # Optimized PyTorch model weights
├── charts/
│   ├── real_vs_predicted_city.png             # Whole-city HGBR parity & residual plots
│   ├── real_vs_predicted_stations.png         # Station-level timeseries breakdown
│   ├── lstm_full_year_2025_real_vs_predicted.png # Full-year 2025 continuous DL tracking
│   ├── lstm_training_overfitting_curve.png    # Overfitting Before vs. After comparison
│   └── hybrid_ensemble_comparison.png         # Real vs. HGBR vs. LSTM vs. Hybrid plot
├── scripts/
│   ├── config.py                              # Central configuration & CPCB NAQI breakpoints
│   ├── feature_engineering.py                 # Domain transformations & feature extractors
│   ├── merge_stations.py                      # Multi-station data merger utility
│   ├── train_station_models.py                # 18-Model HGBR training pipeline
│   ├── predict.py                             # Unified prediction CLI (Station & Whole-City)
│   ├── plot_real_vs_predicted.py              # Visual diagnostics generator
│   ├── hybrid_ensemble.py                     # Hybrid blending stacking orchestrator
│   └── pipeline.py                            # Master end-to-end pipeline CLI
├── station_wise_engineered_results.csv        # Detailed HGBR station metrics table
├── requirements.txt                           # Python package dependencies
└── README.md                                  # Research documentation & technical report
```

---

## 8. Installation & Environment Setup

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)

```bash
# Clone the repository
git clone https://github.com/your-username/agra-pollution-ml.git
cd agra-pollution-ml

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 9. Execution & Quickstart Guide

### 1. Execute Complete Classical ML Pipeline
Runs data verification, feature engineering, 18-model training, diagnostic plotting, and city forecasting:
```bash
python3 scripts/pipeline.py --all
```

### 2. Whole-City Real-Time Forecasting & CPCB AQI
```bash
python3 scripts/predict.py --city --hours 12
```

### 3. Station-Specific Forecasting
```bash
python3 scripts/predict.py --station Sanjay_Palace --hours 6
```

### 4. Train Deep Learning Sequence Model (LSTM/GRU)
```bash
python3 deep_learning/train.py
```

### 5. Generate Full-Year 2025 Deep Learning Evaluation & Plots
```bash
python3 deep_learning/generate_full_year_evaluation.py
```

### 6. Run Hybrid Blending Ensemble (HGBR + LSTM)
```bash
python3 scripts/hybrid_ensemble.py
```

---

## 10. Citation & Acknowledgements

If you utilize this codebase, dataset, or methodology in your academic research or environmental policy analysis, please cite:

```bibtex
@article{agra_pollution_forecasting_2026,
  title={High-Resolution Atmospheric Air Quality Forecasting in the Taj Trapezium Zone: A Comparative Study of Physics-Informed Boosted Trees, Deep Sequence Networks, and Spatial Ensembles},
  author={Khushwant et al.},
  journal={Atmospheric Environment & Environmental Data Science},
  year={2026},
  publisher={GitHub Repository}
}
```

* **Data Sources**: Central Pollution Control Board (CPCB) and Uttar Pradesh Pollution Control Board (UPPCB) Continuous Ambient Air Quality Monitoring Stations (CAAQMS).
* **Regulatory Reference**: Indian National Air Quality Index (NAQI) Standard (CPCB, Ministry of Environment, Forest and Climate Change).
