import os
import json
import joblib
import pandas as pd
import numpy as np

# ===== CONFIG =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "lightgbm_model.pkl")
FEATURE_META_PATH = os.path.join(BASE_DIR, "feature_metadata.json")


# ===== LOAD MODEL =====
model = joblib.load(MODEL_PATH)

with open(FEATURE_META_PATH, "r") as f:
    meta = json.load(f)

feat_cols = meta["features"]
LAGS = meta["lags"]


# ===== HELPERS =====
def fill_missing(history_df):
    history_df = history_df.copy()
    history_df["month"] = pd.to_datetime(history_df["month"])

    history_df = history_df.set_index("month")

    full_idx = pd.date_range(
        start=history_df.index.min(),
        end=history_df.index.max(),
        freq="ME"
    )

    history_df = history_df.reindex(full_idx)
    history_df["NADAC Per Unit"] = history_df["NADAC Per Unit"].ffill()

    return history_df.reset_index().rename(columns={"index": "month"})


# ===== CORE FUNCTION =====
def forecast_ndc(ndc: str, history_df: pd.DataFrame, steps: int = 12):
    """
    history_df must contain:
    - month
    - NADAC Per Unit

    return format:
    [
        {"month": 1, "predicted_price": ...},
        {"month": 2, "predicted_price": ...}
    ]
    """

    history_df = history_df.copy()
    history_df["month"] = pd.to_datetime(history_df["month"])
    history_df["NADAC Per Unit"] = pd.to_numeric(
        history_df["NADAC Per Unit"],
        errors="coerce"
    )

    history_df = history_df.dropna(subset=["month", "NADAC Per Unit"])
    history_df = history_df.sort_values("month").reset_index(drop=True)

    history_df = fill_missing(history_df)

    if len(history_df) < 12:
        raise ValueError("Need at least 12 months of history")

    known = {
        row["month"]: float(row["NADAC Per Unit"])
        for _, row in history_df.iterrows()
    }

    last_month = max(known.keys())

    future_months = pd.date_range(
        start=last_month + pd.offsets.MonthEnd(1),
        periods=steps,
        freq="ME"
    )

    results = []

    for index, target_month in enumerate(future_months, start=1):
        feature_dict = {}

        for k in LAGS:
            lag_month = target_month - pd.offsets.MonthEnd(k)

            if lag_month not in known:
                raise ValueError(
                    f"Missing lag month {lag_month.strftime('%Y-%m')} "
                    f"for forecast month {target_month.strftime('%Y-%m')}"
                )

            feature_dict[f"lag_{k}"] = known[lag_month]

        vals_3 = [
            known[target_month - pd.offsets.MonthEnd(k)]
            for k in range(1, 4)
        ]

        vals_6 = [
            known[target_month - pd.offsets.MonthEnd(k)]
            for k in range(1, 7)
        ]

        feature_dict["roll_mean_3"] = float(np.mean(vals_3))
        feature_dict["roll_mean_6"] = float(np.mean(vals_6))
        feature_dict["roll_std_6"] = float(np.std(vals_6, ddof=1))

        feature_dict["month_num"] = int(target_month.month)

        X_pred = pd.DataFrame([feature_dict], columns=feat_cols)
        delta = float(model.predict(X_pred)[0])

        prev_month = target_month - pd.offsets.MonthEnd(1)
        last_price = known[prev_month]

        pred_price = last_price + delta

        results.append({
            "month": index,
            "predicted_price": pred_price
        })

        known[target_month] = pred_price

    return results