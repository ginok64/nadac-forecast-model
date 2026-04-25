# NADAC Forecast Model

This repository provides a trained LightGBM model and a prediction function for forecasting future NADAC prices.

The current model predicts **monthly price changes (delta)** rather than absolute prices. For each forecast month, the model predicts the expected change from the previous month, then adds that change to the latest known price.

## Repository Structure

```text
chicago-capstone-model/
│
├── models/
│   ├── lightgbm_model.pkl
│   ├── feature_metadata.json
│   └── predict_lightgbm.py
│
├── notebooks/
│   ├── LightGBM_predict.ipynb
│   └── LightGBM_train.ipynb
│
├── example.py
└── README.md
```

## Files

### `models/lightgbm_model.pkl`

The trained LightGBM model.

### `models/feature_metadata.json`

Stores the feature column order and lag settings used by the model.

### `models/predict_lightgbm.py`

Contains the `forecast_ndc()` function used to generate forecasts.

### `example.py`

Example script showing how to call the prediction function.

## Usage

Import the prediction function:

```python
from models.predict_lightgbm import forecast_ndc
import pandas as pd
```

Prepare historical monthly NADAC price data:

```python
history_df = pd.DataFrame({
    "month": [
        "2025-01-31", "2025-02-28", "2025-03-31",
        "2025-04-30", "2025-05-31", "2025-06-30",
        "2025-07-31", "2025-08-31", "2025-09-30",
        "2025-10-31", "2025-11-30", "2025-12-31"
    ],
    "NADAC Per Unit": [
        520, 520, 520, 520, 522, 522,
        522, 522, 522, 522, 522, 522
    ]
})
```

Generate a 12-month forecast:

```python
forecast = forecast_ndc(
    ndc="00002146080",
    history_df=history_df,
    steps=12
)

print(forecast)
```

## Input Format

`history_df` must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `month` | string or datetime | Month-end date for the historical price |
| `NADAC Per Unit` | float | Historical NADAC price |

The model requires at least **12 months of historical monthly prices**.

## Output Format

The function returns a list of dictionaries:

```json
[
    {
        "month": 1,
        "predicted_price": 523.14
    },
    {
        "month": 2,
        "predicted_price": 523.88
    }
]
```