from models.predict_lightgbm import forecast_ndc
import pandas as pd

history_df = pd.DataFrame({
    "month": [
        "2025-01-31","2025-02-28","2025-03-31",
        "2025-04-30","2025-05-31","2025-06-30",
        "2025-07-31","2025-08-31","2025-09-30",
        "2025-10-31","2025-11-30","2025-12-31"
    ],
    "NADAC Per Unit": [
        520,520,520,520,522,522,522,522,522,522,522,522
    ]
})

forecast = forecast_ndc("00002146080", history_df, steps=12)

print(forecast)