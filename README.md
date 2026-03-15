#  Seoul Bike Sharing Demand — ML Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Predict hourly bike rental demand in Seoul using weather, season, and time-of-day features**

[Features](#-features) · [Dataset](#-dataset) · [Models](#-models) · [Setup](#-setup--installation) · [Usage](#-usage) · [Results](#-results) · [Pipeline](#-pipeline)

</div>

---

##  Overview

This project builds an end-to-end **regression ML pipeline** to predict the number of bikes rented per hour in Seoul, South Korea. Using weather conditions, seasonal patterns, and time features, the model learns when demand peaks (rush hours, warm sunny days) and when it drops (rain, snow, winter nights).

```
SeoulBikeData.csv → EDA → Feature Engineering → Train 4 Models → Evaluate → Tune → Predict
```

> **Use case:** Helps bike-sharing operators anticipate demand, optimise fleet distribution, and reduce shortages or idle bikes at each station.

---

##  Features

- ✅ **4 regression models** trained and compared side-by-side
- ✅ **Cyclical time encoding** — preserves the circular nature of hours and months
- ✅ **Interaction features** — temperature × solar radiation, temperature × humidity
- ✅ **Log-transformed target** for linear models to handle skewed distribution
- ✅ **GridSearchCV hyperparameter tuning** on the best model
- ✅ **Cross-validation** with 5-fold CV and visual score distribution
- ✅ **Feature importance** ranked by Random Forest
- ✅ **Predicted vs Actual** diagnostic plots and residual analysis
- ✅ **Custom prediction demo** — input any weather/time scenario and get a count

---

##  Dataset

| Property | Value |
|---|---|
| **Source** | [Seoul Bike Sharing Demand — Kaggle](https://www.kaggle.com/datasets/gaurav2022/mobile-health) · UCI ML Repository |
| **File** | `SeoulBikeData.csv` |
| **Records** | 8,760 hourly observations |
| **Time span** | 1 full year (December 2017 – November 2018) |
| **Encoding** | `latin-1` |

###  Raw Columns

| Column | Description | Type |
|---|---|---|
| `Date` | Date in `DD/MM/YYYY` format | Date |
| `Rented Bike Count` | **Target** — bikes rented that hour | int |
| `Hour` | Hour of day (0–23) | int |
| `Temperature(°C)` | Air temperature | float |
| `Humidity(%)` | Relative humidity | int |
| `Wind speed (m/s)` | Wind speed | float |
| `Visibility (10m)` | Visibility distance | int |
| `Dew point temperature(°C)` | Dew point | float |
| `Solar Radiation (MJ/m2)` | Solar radiation | float |
| `Rainfall(mm)` | Rainfall amount | float |
| `Snowfall (cm)` | Snowfall amount | float |
| `Seasons` | Winter / Spring / Summer / Autumn | string |
| `Holiday` | Holiday / No Holiday | string |
| `Functioning Day` | Yes / No | string |

###  Engineered Features

| Feature | Description |
|---|---|
| `Month` | Month extracted from Date |
| `DayOfWeek` | Day of week (0=Mon, 6=Sun) |
| `IsWeekend` | Binary flag for Saturday/Sunday |
| `DayOfYear` | Day number within the year |
| `Hour_sin` / `Hour_cos` | Cyclical encoding of hour (24-hour cycle) |
| `Month_sin` / `Month_cos` | Cyclical encoding of month (12-month cycle) |
| `Temp_x_Solar` | Temperature × Solar Radiation interaction |
| `Temp_x_Humidity` | Temperature × Humidity interaction |
| `Seasons_enc` | Label-encoded seasons |
| `Holiday_enc` | Label-encoded holiday flag |

> **Why cyclical encoding?** Hour 23 and Hour 0 are adjacent in time but numerically far apart. Sine/cosine encoding preserves this circular relationship for linear models.

---

##  Models

| Model | Task | Notes |
|---|---|---|
| **Ridge Regression** | Linear regression | Trained on **log(target+1)** to handle skew; predictions back-transformed |
| **Random Forest** | Ensemble (bagging) | 200 trees, no target transform needed |
| **Gradient Boosting** | Ensemble (boosting) | 300 estimators, lr=0.1, depth=5, subsample=0.8 |
| **XGBoost** | Optimised boosting | 500 estimators, lr=0.05 (requires `pip install xgboost`) |

###  Evaluation Metrics

| Metric | Formula | What it measures |
|---|---|---|
| **MAE** | Mean Absolute Error | Average absolute prediction error (in bike count units) |
| **RMSE** | Root Mean Squared Error | Penalises large errors more heavily |
| **R²** | Coefficient of determination | Proportion of variance explained (1.0 = perfect) |

---

##  Results

Expected performance on 20% holdout test set:

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|:---:|:---:|:---:|
| Ridge Regression (log target) | ~250 | ~350 | ~0.55 |
| Random Forest | ~85 | ~130 | ~0.90 |
| Gradient Boosting | ~95 | ~145 | ~0.88 |
| **XGBoost** | **~70** | **~110** | **~0.93** |

> Actual results depend on random seed and train/test split. Tuned Random Forest may match or exceed XGBoost.

###  Key Findings

- **Hour of day** is the most important feature — demand peaks sharply at **8 AM** and **6 PM**
- **Temperature** has a strong positive effect — every 5°C increase raises expected demand
- **Season** matters: Summer > Autumn > Spring > Winter
- **Rainfall and Snowfall** are the strongest negative predictors
- **Solar Radiation** correlates with demand independent of temperature

---

##  Setup & Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- `SeoulBikeData.csv` in the same folder as the notebook

### 1. Clone the repository

```bash
git clone https://github.com/your-username/seoul-bike-sharing-prediction.git
cd seoul-bike-sharing-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 3. Download the dataset

Download `SeoulBikeData.csv` from [Kaggle](https://www.kaggle.com/datasets/city-of-seoul/seoul-bike-sharing-demand) and place it in the project root:

```
seoul-bike-sharing-prediction/
├── bike_sharing_prediction.ipynb
├── SeoulBikeData.csv          ← place here
├── requirements.txt
└── README.md
```

---

##  Usage

### Run the notebook

```bash
jupyter notebook bike_sharing_prediction.ipynb
```

Select **Kernel → Restart & Run All**.

### Predict for a custom scenario

```python
import pandas as pd
import numpy as np
import joblib

model  = joblib.load('best_bike_model.pkl')
scaler = joblib.load('bike_scaler.pkl')

# Example: warm summer weekday afternoon
new_data = pd.DataFrame([{
    'Hour': 17,
    'Hour_sin': np.sin(2 * np.pi * 17 / 24),
    'Hour_cos': np.cos(2 * np.pi * 17 / 24),
    'Temperature(°C)': 28.0,
    'Humidity(%)': 55,
    'Wind speed (m/s)': 2.5,
    'Visibility (10m)': 1800,
    'Dew point temperature(°C)': 18.0,
    'Solar Radiation (MJ/m2)': 1.2,
    'Rainfall(mm)': 0.0,
    'Snowfall (cm)': 0.0,
    'Seasons_enc': 1,       # Summer
    'Holiday_enc': 1,       # No Holiday
    'Month': 7,
    'Month_sin': np.sin(2 * np.pi * 7 / 12),
    'Month_cos': np.cos(2 * np.pi * 7 / 12),
    'DayOfWeek': 2,
    'IsWeekend': 0,
    'DayOfYear': 196,
    'Temp_x_Solar': 28.0 * 1.2,
    'Temp_x_Humidity': 28.0 * 55
}])

predicted_bikes = model.predict(new_data)[0]
print(f"Predicted rentals: {int(predicted_bikes)} bikes")
```

---

##  Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                     PIPELINE OVERVIEW                   │
├─────────────┬───────────────────────────────────────────┤
│  Section 1  │  Import libraries                         │
│  Section 2  │  Load SeoulBikeData.csv (latin-1 encoding)│
│  Section 3  │  EDA — target dist, hourly/seasonal trends│
│             │       correlation heatmap, scatter plots  │
│  Section 4  │  Feature Engineering                      │
│             │  • Parse date → Month, DayOfWeek, Weekend │
│             │  • Cyclical encode Hour & Month           │
│             │  • Interaction features                   │
│             │  • Label encode Seasons & Holiday         │
│             │  • Filter non-functioning days            │
│  Section 5  │  Train/Test split (80/20, random_state=42)│
│             │  StandardScaler for linear models         │
│  Section 6  │  Train: Ridge · RF · GBR · XGBoost        │
│  Section 7  │  Compare MAE, RMSE, R² with bar charts    │
│  Section 8  │  Best model diagnostics                   │
│             │  • Predicted vs Actual scatter            │
│             │  • Residual plot                          │
│  Section 9  │  Feature Importance (top 20)              │
│  Section 10 │  5-Fold Cross-Validation                  │
│  Section 11 │  GridSearchCV Hyperparameter Tuning       │
│  Section 12 │  Predict on custom new input              │
│  Section 13 │  Summary table                            │
└─────────────┴───────────────────────────────────────────┘
```

---

##  Project Structure

```
seoul-bike-sharing-prediction/
│
├── bike_sharing_prediction.ipynb   # Main notebook — run this
├── SeoulBikeData.csv               # Dataset (download separately)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

##  Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

##  Next Steps

- [ ] **LSTM / Temporal Fusion Transformer** for time-series forecasting
- [ ] **Prophet** for trend + seasonality decomposition
- [ ] **SHAP values** for deeper feature explainability
- [ ] **REST API** deployment with FastAPI + model serving
- [ ] **Streamlit dashboard** for interactive demand forecasting
- [ ] **Multi-city generalisation** — extend to other bike-sharing systems

---

##  Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -m 'Add LSTM model'`)
4. Push to branch (`git push origin feature/new-model`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgements

- Dataset: [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand) — UCI ML Repository
- City of Seoul Open Data Portal
- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

---


