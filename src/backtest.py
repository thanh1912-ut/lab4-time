"""
Walk-forward validation dùng chung và các chỉ số đánh giá.
Người thực hiện: Bao Dinh
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from features import inverse_load


# ============================================================
# 1. TÍNH CÁC CHỈ SỐ ĐÁNH GIÁ
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Tính RMSE, MAE, MAPE.

    Đầu vào: Giá trị thực tế và dự báo (MW).
    """

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Hai mảng phải có cùng số lượng phần tử và không rỗng.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Dữ liệu đánh giá chứa NaN hoặc vô cực.")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE không xác định tại những điểm có giá trị thực tế bằng 0.
    mask = np.abs(y_true) > 1e-8

    if mask.any():
        mape = np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
    else:
        mape = np.nan

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape)
    }


# ============================================================
# 2. WALK-FORWARD VALIDATION
# ============================================================

def walk_forward(
    fit_fn,
    predict_fn,
    scaled,
    start,
    end,
    refit_every=None,
    scaler=None,
    train_end=None
):
    """
    Dự báo tuần tự từng giờ bằng Walk-forward validation.

    fit_fn(history_df):
        Huấn luyện mô hình từ dữ liệu quá khứ.

    predict_fn(model, history_df):
        Dự báo load tại giờ tiếp theo.
        Chỉ sử dụng history_df, không sử dụng dữ liệu tương lai.

    scaled:
        DataFrame dữ liệu đã chuẩn hóa.

    start:
        Vị trí bắt đầu dự báo.

    end:
        Vị trí kết thúc (không bao gồm end).

    refit_every:
        Số giờ giữa hai lần huấn luyện lại.
        None = chỉ huấn luyện một lần.

    scaler:
        StandardScaler dùng để đưa kết quả về MW.
        Nếu None, kết quả vẫn ở đơn vị chuẩn hóa.

    train_end:
        Lần huấn luyện đầu chỉ dùng scaled.iloc[:train_end].
        None = dùng mọi dữ liệu trước start (train + validation).
        Đặt bằng vị trí cuối train để mô hình học giống XGBoost/LSTM
        (validation chỉ để chọn cấu hình), dùng cho so sánh công bằng.

    Returns:
        DataFrame gồm:
            actual
            prediction

        Index là thời gian dự báo.
    """

    # Kiểm tra dữ liệu
    if not isinstance(scaled, pd.DataFrame):
        raise TypeError("scaled phải là pandas DataFrame.")

    if scaled.empty:
        raise ValueError("Dữ liệu không được rỗng.")

    if "load" not in scaled.columns:
        raise ValueError("Không tìm thấy cột load.")

    if not scaled.index.is_monotonic_increasing:
        raise ValueError("Dữ liệu phải theo thứ tự thời gian.")

    if not (isinstance(start, (int, np.integer)) and
            isinstance(end, (int, np.integer))):
        raise TypeError("start và end phải là số nguyên.")

    if start < 1 or end <= start or end > len(scaled):
        raise ValueError("Khoảng start/end không hợp lệ.")

    if refit_every is not None:
        if (not isinstance(refit_every, (int, np.integer))
                or isinstance(refit_every, (bool, np.bool_))
                or refit_every <= 0):
            raise ValueError("refit_every phải là số nguyên dương hoặc None.")

    if train_end is not None and not (
            isinstance(train_end, (int, np.integer)) and 1 <= train_end <= start):
        raise ValueError("train_end phải là số nguyên trong [1, start].")

    # Huấn luyện ban đầu chỉ bằng dữ liệu trước thời điểm start (hoặc trước train_end).
    model = fit_fn(scaled.iloc[:start if train_end is None else train_end].copy())

    predictions = []

    # Dự báo tuần tự từng giờ.
    for k, t in enumerate(range(start, end)):

        # Huấn luyện lại định kỳ.
        if refit_every is not None and k > 0 and k % refit_every == 0:
            model = fit_fn(scaled.iloc[:t].copy())

        # Chỉ truyền dữ liệu từ đầu đến t-1.
        history = scaled.iloc[:t].copy()

        prediction = predict_fn(model, history)

        prediction = np.asarray(prediction, dtype=float).reshape(-1)

        if prediction.size != 1:
            raise ValueError(
                "predict_fn phải trả về đúng một giá trị load mỗi lần."
            )

        predictions.append(float(prediction[0]))

    # Giá trị thực tế tại các thời điểm được dự báo.
    actual = scaled["load"].iloc[start:end].to_numpy(dtype=float)

    predictions = np.asarray(predictions, dtype=float)

    # Khôi phục về MW nếu có scaler.
    if scaler is not None:
        actual = inverse_load(actual, scaler)
        predictions = inverse_load(predictions, scaler)

    results = pd.DataFrame(
        {
            "actual": actual,
            "prediction": predictions
        },
        index=scaled.index[start:end]
    )

    results.index.name = "Datetime"

    return results


# ============================================================
# 3. LƯU KẾT QUẢ BACKTEST
# ============================================================

def save_backtest_results(results, output_path):
    """
    Lưu kết quả dự báo thành file CSV.
    """

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results phải là pandas DataFrame.")

    if not {"actual", "prediction"}.issubset(results.columns):
        raise ValueError("Thiếu cột actual hoặc prediction.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_path, index=True, encoding="utf-8-sig")

    print(f"Đã lưu kết quả: {output_path}")