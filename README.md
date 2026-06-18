# Household Power Consumption — Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

Short-term forecasting of household active power consumption using deep recurrent and convolutional-recurrent neural networks (LSTM, CNN-LSTM, CNN-BiLSTM) on the UCI *Individual Household Electric Power Consumption* dataset.

---

## Project Overview

This project builds and compares seven neural-network models for forecasting `Global_active_power` one step ahead, from a multivariate time series of minute-level electricity measurements collected over ~4 years in a single household near Paris.

The work covers the full pipeline: in-depth exploratory analysis, principled missing-value treatment, feature engineering, time-aware data splitting, model comparison, and error analysis. A central theme is that **model complexity must be justified by results** — the final recommended model is chosen on the combination of accuracy *and* stability, not on a single metric.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

| Property | Value |
|---|---|
| Measurements | 2,075,259 |
| Frequency | 1 minute |
| Period | December 2006 – November 2010 (47 months) |
| Location | Sceaux, France (7 km from Paris) |
| Features | 7 electrical quantities + 3 sub-meters |

**Variables**

- `Global_active_power` — household active power, kW *(forecast target)*
- `Global_reactive_power` — reactive power, kW
- `Voltage` — voltage, V
- `Global_intensity` — current intensity, A
- `Sub_metering_1` — kitchen (dishwasher, oven, microwave), Wh
- `Sub_metering_2` — laundry room (washer, dryer, fridge, light), Wh
- `Sub_metering_3` — electric water heater and air conditioner, Wh

> The dataset file `household_power_consumption.txt` is **not** included in this repository. Download it from the link above and place it in the project root before running the notebook.

---

## Methodology

### 1. Exploratory Data Analysis
- Correlation analysis (Pearson and Spearman) — near-perfect collinearity between active power and current; discrete operating modes in sub-meters.
- Missing-value characterisation — 1.3% missing per feature, occurring as **whole-minute gaps** (all 7 features missing together), clustered into 71 distinct intervals, dominated by a few multi-day outages.
- Seasonality analysis across hour-of-day, day-of-week, month, and season.
- Energy-balance decomposition: a derived `Other_Wh` residual circuit accounts for ~51% of total energy.

### 2. Missing-Value Strategies
Compared on a fixed baseline:
- **Mean imputation** — fill with the training-set mean.
- **Strategy A** — trim to the longest continuous segment.
- **Strategy B (seasonal)** — short gaps via time interpolation; long gaps from the same date/time in adjacent years.

Seasonal imputation is conceptually more principled but yields no measurable gain over simple mean imputation on the test set.

### 3. Feature Engineering
- **`Other_Wh`** — residual (unmetered) load derived from the energy balance.
- **Cyclic calendar features** — `hour`, `day-of-week` encoded via sin/cos to preserve cyclicality.
- **K-Means sub-metering discretisation** — non-zero values clustered into discrete appliance operating modes.
- **Resampling** — power features aggregated with `mean`, energy features with `sum`.

Ablation findings: calendar features gave the **largest** improvement; K-Means discretisation contributed positively; `StandardScaler` outperformed MinMax/Robust scaling.

### 4. Time-Aware Splitting
Data split chronologically into **train (2 years) / validation (1 year) / test (~1 year)**. Scalers and K-Means are fitted on the training set only, then applied to validation/test — **no data leakage**.

---

## Models

Seven LSTM-based variants were built and compared:

| Model | Description |
|---|---|
| **Baseline LSTM** | One-lag supervised framing, MinMaxScaler, hourly data |
| **Standard LSTM** | Enriched features, StandardScaler, 48-hour look-back |
| **Mean LSTM** | Standard pipeline + mean imputation, MinMaxScaler |
| **Seasonal LSTM** | Interpolation + seasonal year-offset imputation |
| **CNN + LSTM** | Conv1D feature extractor → LSTM, hourly data |
| **LSTM (daily)** | Daily resampling, 7-day look-back, compact architecture |
| **CNN + BiLSTM (daily)** | Conv1D → Bidirectional LSTM on daily data |

All trained models are saved in the [`models/`](models/) folder, so they can be loaded directly for inference without re-training.

---

## Results

### Hourly Models

| Model | Test RMSE | Train RMSE | Train/Test Gap |
|---|---|---|---|
| Mean LSTM | **0.477** | 0.522 | −8.7% |
| Seasonal LSTM | 0.487 | 0.527 | −7.7% |
| **Standard LSTM** ⭐ | 0.492 | 0.502 | **−2.1%** |
| CNN + LSTM | 0.502 | 0.528 | −5.0% |
| Baseline LSTM | 0.598 | 0.661 | −9.5% |

<!-- Add your saved figures here once committed to the repo, e.g. in an assets/ folder:
![RMSE comparison](assets/rmse_comparison.png)
![Actual vs Prediction](assets/actual_vs_prediction.png)
![Scatter](assets/scatter.png)
-->

**Recommended model: Standard LSTM.** Mean LSTM has the lowest absolute test RMSE (0.477), but its margin over Standard LSTM (0.492) is within the range of random-initialisation noise — the top three models are effectively statistically equivalent. The decisive factor is **stability**: Standard LSTM shows the smallest train/test gap (−2.1% vs −8.7%), meaning its error is nearly identical on seen and unseen data and therefore the most predictable. Standard LSTM trades a negligible 0.015 RMSE for substantially more reliable generalisation.

> **Note on the negative train/test gap:** all models show test RMSE below train RMSE. This is a property of the temporal split — the 2007–2008 training period is more volatile than the calmer 2010 test period — and **not** evidence of data leakage. Standard LSTM is simply the least affected by it.

### Daily Models

Daily-resampled models reached lower RMSE (≈ 0.24 kW) than hourly models (≈ 0.48 kW), as expected for a smoother aggregated signal. These are **not** directly comparable to hourly models, as they solve a different task (next-day vs next-hour forecast).

### Key Takeaways
- **Simplicity wins:** a rich feature set with simple mean imputation matches more complex seasonal imputation and CNN hybrids.
- **Calendar features** were the single most impactful engineering step.
- All models capture typical load well but exhibit **peak-smoothing bias** on extreme values — a known limitation of MSE-trained regressors.

---

## Tech Stack

- **Python** — pandas, NumPy, SciPy
- **TensorFlow / Keras** — LSTM, CNN-LSTM, CNN-BiLSTM architectures
- **scikit-learn** — scaling, K-Means clustering, metrics
- **Matplotlib / Seaborn** — visualisation

---

## Project Structure

```
TS_Power_consumption/
├── Project_6_Power_Consumption_EN.ipynb   # Main notebook (EDA + modelling)
├── README.md
└── Models/                                # Pre-trained Keras models
    ├── basismodel.keras                   # Baseline LSTM
    ├── standardmodel.keras                # Standard LSTM (recommended)
    ├── MeanLSTM_model.keras               # Mean-imputation LSTM
    ├── SeasonalLSTM_model.keras           # Seasonal-imputation LSTM
    ├── CNN_LSTM_model.keras               # CNN + LSTM (hourly)
    ├── daily_LSTM_model.keras             # LSTM (daily)
    └── CNN_BiLSTM_model.keras             # CNN + BiLSTM (daily)
```

> **Note:** the notebook currently saves and loads models from the project root
> (e.g. `model.save("standardmodel.keras")`). If you keep the trained models in
> `Models/`, update the corresponding paths in the notebook to `models/...`
> (or move the `.keras` files to the root before re-running).

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Eruhonya/TS_Power_consumption.git
cd TS_Power_consumption

# 2. Install dependencies
pip install pandas numpy scipy scikit-learn tensorflow matplotlib seaborn

# 3. Download the dataset
# Place household_power_consumption.txt in the project root
# (from the UCI link above)

# 4. Launch the notebook
jupyter notebook Power_Consumption_prediction.ipynb
```

The notebook trains all models from scratch. To skip training and use the
included pre-trained models instead, load them directly:

```python
from tensorflow.keras.models import load_model
model = load_model("Models/standardmodel.keras")   # recommended model
```

