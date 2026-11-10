
"""
Mục 4, 5: Xử lý đặc trưng, chuẩn hóa và tạo cửa sổ dữ liệu
cho các mô hình dự báo chuỗi thời gian.

Người thực hiện: Bao Dinh

Các chức năng:
1. Chia dữ liệu Train / Validation / Test theo thứ tự thời gian.
2. Chuẩn hóa bằng StandardScaler, chỉ fit trên tập Train.
3. Khôi phục giá trị load về đơn vị MW.
4. Tạo dữ liệu cửa sổ 3D cho LSTM và các mô hình chuỗi thời gian.
5. Tạo đặc trưng rolling cho mô hình XGBoost.

Lưu ý:
- Không shuffle dữ liệu.
- Không sử dụng thông tin tương lai để tạo đặc trưng.
- Giữ tương thích với utils.py của project.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from utils import COLS, L, TRAIN_RATIO, VAL_RATIO


# ============================================================
# 1. KIỂM TRA DỮ LIỆU
# ============================================================

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Kiểm tra DataFrame có đủ cột và dữ liệu hợp lệ."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dữ liệu đầu vào phải là pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame không được rỗng.")

    if not df.columns.is_unique:
        raise ValueError("DataFrame có tên cột bị trùng.")

    missing_cols = [col for col in COLS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Thiếu các cột dữ liệu: {missing_cols}")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Dữ liệu phải được sắp xếp theo thứ tự thời gian.")

    values = df[COLS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Dữ liệu chứa NaN hoặc vô cực. "
            "Hãy xử lý dữ liệu thiếu trước khi tiếp tục."
        )


# ============================================================
# 2. CHIA TRAIN / VALIDATION / TEST
# ============================================================

def split_index(n_samples: int):
    """
    Chia chỉ số dữ liệu theo thứ tự thời gian.

    Tỷ lệ mặc định:
        Train      : 70%
        Validation : 15%
        Test       : 15%

    Returns:
        i1: vị trí kết thúc tập Train.
        i2: vị trí kết thúc tập Validation.

    Ví dụ:
        scaled, scaler, (i1, i2) = fit_scale(df)
    """

    if not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples phải là số nguyên.")

    if n_samples <= 0:
        raise ValueError("n_samples phải lớn hơn 0.")

    if not (0 < TRAIN_RATIO < 1):
        raise ValueError("TRAIN_RATIO phải nằm trong khoảng (0, 1).")

    if not (0 < VAL_RATIO < 1):
        raise ValueError("VAL_RATIO phải nằm trong khoảng (0, 1).")

    if TRAIN_RATIO + VAL_RATIO >= 1:
        raise ValueError(
            "Tổng TRAIN_RATIO và VAL_RATIO phải nhỏ hơn 1."
        )

    i1 = int(n_samples * TRAIN_RATIO)
    i2 = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

    if i1 == 0 or i2 <= i1 or i2 >= n_samples:
        raise ValueError("Số lượng mẫu không đủ để chia Train/Val/Test.")

    return i1, i2


# ============================================================
# 3. CHUẨN HÓA DỮ LIỆU
# ============================================================

def fit_scale(df: pd.DataFrame):
    """
    Chuẩn hóa các biến bằng StandardScaler.

    QUAN TRỌNG:
        - Chỉ fit scaler trên tập Train.
        - Transform toàn bộ dữ liệu bằng scaler đã fit.
        - Không fit riêng trên Validation hoặc Test.

    Returns:
        df_scaled: DataFrame sau chuẩn hóa.
        scaler: StandardScaler đã được fit.
        (i1, i2): Hai mốc chia dữ liệu.
    """

    _validate_dataframe(df)

    i1, i2 = split_index(len(df))

    scaler = StandardScaler()

    # Chỉ học mean và standard deviation từ Train.
    scaler.fit(df[COLS].iloc[:i1])

    # Dùng cùng scaler để biến đổi toàn bộ dữ liệu.
    df_scaled = df.copy()

    df_scaled.loc[:, COLS] = scaler.transform(df[COLS])

    return df_scaled, scaler, (i1, i2)


# ============================================================
# 4. KHÔI PHỤC GIÁ TRỊ LOAD
# ============================================================

def inverse_load(z, scaler: StandardScaler):
    """
    Đưa giá trị load sau chuẩn hóa về đơn vị MW ban đầu.

    Công thức:
        load_original = load_scaled * scale + mean

    Args:
        z: Giá trị load đã chuẩn hóa.
        scaler: StandardScaler đã fit trên tập Train.

    Returns:
        NumPy array chứa giá trị load theo đơn vị ban đầu.
    """

    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError("Scaler chưa được fit.")

    if "load" not in COLS:
        raise ValueError("Không tìm thấy cột load trong COLS.")

    z = np.asarray(z, dtype=float)

    load_idx = COLS.index("load")

    return z * scaler.scale_[load_idx] + scaler.mean_[load_idx]


# ============================================================
# 5. TẠO CỬA SỔ DỮ LIỆU 3D
# ============================================================

def make_windows(scaled: pd.DataFrame, L: int = L):
    """
    Tạo cửa sổ dữ liệu phục vụ dự báo chuỗi thời gian.

    Với L = 24:
        X[k] = dữ liệu của 24 giờ trước.
        y[k] = load tại giờ tiếp theo.

    Shape:
        X: (N, 24, 4)
        y: (N,)
        target_time: DatetimeIndex chứa thời gian của y.

    Returns:
        X, y, target_time

    Ví dụ:
        X, y, target_time = make_windows(scaled)
    """

    _validate_dataframe(scaled)

    if not isinstance(L, (int, np.integer)) or L <= 0:
        raise ValueError("L phải là số nguyên dương.")

    if len(scaled) <= L:
        raise ValueError(
            f"Cần nhiều hơn {L} mẫu để tạo cửa sổ dữ liệu."
        )

    data = scaled[COLS].to_numpy(dtype=np.float64)

    target_idx = COLS.index("load")

    # sliding_window_view tạo cửa sổ trượt.
    # Shape ban đầu: (N - L + 1, số_features, L)
    windows = np.lib.stride_tricks.sliding_window_view(
        data,
        window_shape=L,
        axis=0
    )

    # Đưa về shape (N - L + 1, L, số_features).
    windows = np.moveaxis(windows, -1, 1)

    # Mỗi cửa sổ dự báo giá trị ở thời điểm kế tiếp.
    X = windows[:-1].copy()

    y = data[L:, target_idx].copy()

    target_time = scaled.index[L:]

    return X, y, target_time


# ============================================================
# 6. TẠO ĐẶC TRƯNG ROLLING
# ============================================================

def add_rolling(scaled: pd.DataFrame):
    """
    Tạo 3 đặc trưng rolling cho mô hình XGBoost.

    Các đặc trưng:
        1. load_roll_mean_6:
           Trung bình load của 6 giờ gần nhất.

        2. load_roll_std_24:
           Độ lệch chuẩn load của 24 giờ gần nhất.

        3. load_trend_24:
           Chênh lệch load giữa hiện tại và 24 giờ trước.

    Tất cả đặc trưng tại thời điểm t chỉ sử dụng dữ liệu
    đến thời điểm t, không sử dụng thông tin tương lai.

    Returns:
        DataFrame gồm 3 cột rolling.
    """

    _validate_dataframe(scaled)

    load = scaled["load"]

    rolling_features = pd.DataFrame(
        {
            "load_roll_mean_6": load.rolling(
                window=6,
                min_periods=6
            ).mean(),

            "load_roll_std_24": load.rolling(
                window=24,
                min_periods=24
            ).std(),

            "load_trend_24": load - load.shift(24),
        },
        index=scaled.index
    )

    return rolling_features
