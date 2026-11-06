"""Hàm dùng chung cho Lab 4. Người viết: Thành."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "lab4" / "figures"
for d in (RAW, PROCESSED, FIG):
    d.mkdir(parents=True, exist_ok=True)

COLS = ["load", "temperature", "humidity", "wind_speed"]
TARGET = "load"
L = 24                      # cửa sổ 24 giờ
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15   # test = 0.15 còn lại
SEED = 42

def load_data():
    df = pd.read_csv(PROCESSED / "load_weather_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
    df.index.freq = "h"
    return df[COLS]

def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE(%)": round(mape, 2)}
