"""Chia tập, chuẩn hóa, tạo cửa sổ cho Lab 4. Người viết: Bao Dinh. (Khung gợi ý, Bao Dinh kiểm tra và hoàn thiện.)"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import COLS, L, TRAIN_RATIO, VAL_RATIO

def split_index(n):
    """Vị trí cắt train / val / test theo thời gian."""
    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    return i1, i2

def fit_scale(df):
    """Fit StandardScaler CHỈ trên train, transform toàn bộ. Trả về df_scaled, scaler, (i1, i2)."""
    i1, i2 = split_index(len(df))
    scaler = StandardScaler().fit(df[COLS].iloc[:i1])
    scaled = pd.DataFrame(scaler.transform(df[COLS]), index=df.index, columns=COLS)
    return scaled, scaler, (i1, i2)

def inverse_load(z, scaler):
    return np.asarray(z) * scaler.scale_[0] + scaler.mean_[0]

def make_windows(scaled, L=L):
    """X[k] = 24 giờ (t-23..t) x 4 biến, y[k] = load(t+1). Trả về X (n, L, 4), y (n,), mốc thời gian của y."""
    arr = scaled[COLS].to_numpy()
    X = np.stack([arr[i - L:i] for i in range(L, len(arr))])
    y = arr[L:, 0]
    return X, y, scaled.index[L:]

def add_rolling(scaled):
    """Đặc trưng thêm cho XGBoost, tính đến thời điểm t (dự báo t+1 nên không rò rỉ)."""
    load = scaled["load"]
    return pd.DataFrame({
        "load_roll_mean_6": load.rolling(6).mean(),
        "load_roll_std_24": load.rolling(24).std(),
        "load_trend_24": load - load.shift(24),
    }, index=scaled.index)
