
"""
Mục 10: Đánh giá và so sánh các mô hình dự báo.
Người thực hiện: Bao Dinh

So sánh:
- Naive
- Seasonal Naive (24h)
- VAR
- XGBoost
- LSTM

Chỉ sử dụng dữ liệu dự báo thực tế, không tạo dữ liệu giả.

So sánh công bằng: bảng chính dùng cùng một luật cho cả 3 mô hình
- học trên train, validation chỉ để chọn cấu hình / dừng sớm;
- không huấn luyện lại trên test; mỗi giờ test dự báo 1 bước từ lịch sử thật;
- cùng 3.946 giờ test, cùng scaler fit trên train, cùng công thức sai số.
Bản VAR/XGBoost có refit mỗi 168 giờ được báo riêng (compare_metrics_refit.csv).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data, PROCESSED, FIG


# ============================================================
# 1. CẤU HÌNH
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Bảng chính: cả 3 mô hình học một lần trên train, không refit trên test.
MODEL_FILES = {
    "VAR": "pred_var_fixed.csv",
    "XGBoost": "pred_xgb_fixed.csv",
    "LSTM": "pred_lstm.csv",
}

# Bảng phụ: bản walk-forward có refit mỗi 168 giờ của mục 6, 7.
REFIT_FILES = {
    "VAR (refit mỗi 168 giờ)": "pred_var.csv",
    "XGBoost (refit mỗi 168 giờ)": "pred_xgb.csv",
}

# Ghi chú cách huấn luyện cho từng dòng của bảng chính.
TRAINING_NOTES = {
    "Naive": "Không học; dự báo = load giờ trước",
    "Seasonal Naive (24h)": "Không học; dự báo = load cùng giờ hôm trước",
    "VAR": "VAR(26), p chọn theo BIC trên train; học 1 lần trên train; không refit",
    "XGBoost": "max_depth chọn trong {3,5,7,9} trên validation; học trên train, early stopping theo validation; không refit",
    "LSTM": "patience chọn trong {5,10} trên validation (TB 3 seed); xáo trộn cửa sổ train; seed 42; không refit",
}

MODEL_COLORS = {
    "VAR": "#0066cc",
    "XGBoost": "#009933",
    "LSTM": "#cc0000",
}


# ============================================================
# 2. ĐỌC VÀ CHUẨN HÓA FILE DỰ BÁO
# ============================================================

def load_prediction(model_name, file_name):
    """
    Đọc file dự báo và chuẩn hóa về:
    Datetime, y_true (nếu có), y_pred.
    """

    file_path = PROCESSED / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file {file_path}. "
            f"Vui lòng kiểm tra dữ liệu của mô hình {model_name}."
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"File {file_name} không có dữ liệu.")

    # Chuẩn hóa tên cột
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Tìm cột thời gian
    datetime_candidates = [
        "datetime", "date", "timestamp", "time"
    ]

    datetime_col = next(
        (c for c in datetime_candidates if c in df.columns),
        None
    )

    if datetime_col is None:
        raise ValueError(
            f"{file_name} thiếu cột thời gian. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    # Tìm cột dự báo
    prediction_candidates = [
        "y_pred", "pred", "prediction",
        "predicted", "forecast", "yhat", "load_pred"
    ]

    pred_col = next(
        (c for c in prediction_candidates if c in df.columns),
        None
    )

    # Tìm cột giá trị thực tế
    actual_candidates = [
        "y_true", "actual", "true",
        "actual_load", "load_actual"
    ]

    actual_col = next(
        (c for c in actual_candidates if c in df.columns),
        None
    )

    # Nếu không nhận diện được tên cột dự báo,
    # chỉ chấp nhận khi còn đúng một cột số có thể là dự báo.
    if pred_col is None:
        excluded = {datetime_col, actual_col}

        numeric_candidates = [
            c for c in df.columns
            if c not in excluded
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        if len(numeric_candidates) != 1:
            raise ValueError(
                f"Không xác định được cột dự báo trong {file_name}. "
                f"Các cột hiện có: {list(df.columns)}"
            )

        pred_col = numeric_candidates[0]

    # Chuyển đổi dữ liệu
    df[datetime_col] = pd.to_datetime(
        df[datetime_col], errors="coerce"
    )

    df[pred_col] = pd.to_numeric(
        df[pred_col], errors="coerce"
    )

    if actual_col is not None:
        df[actual_col] = pd.to_numeric(
            df[actual_col], errors="coerce"
        )

    # Bỏ các dòng thiếu thời gian hoặc dự báo
    df = df.dropna(subset=[datetime_col, pred_col])

    # Kiểm tra thời gian trùng lặp
    if df[datetime_col].duplicated().any():
        raise ValueError(
            f"{file_name} có thời gian bị trùng lặp."
        )

    df = df.set_index(datetime_col).sort_index()

    result = pd.DataFrame(index=df.index)

    result["y_pred"] = df[pred_col]

    if actual_col is not None:
        result["y_true"] = df[actual_col]

    print(
        f"Đã đọc {model_name}: "
        f"{len(result)} dòng từ {file_name}"
    )

    return result


# ============================================================
# 3. TÍNH CÁC CHỈ SỐ ĐÁNH GIÁ
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Tính RMSE, MAE, MAPE.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("Hai mảng dữ liệu không cùng độ dài.")

    if len(y_true) == 0:
        raise ValueError("Không có dữ liệu để đánh giá.")

    error = y_true - y_pred

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    # Tránh chia cho 0
    denominator = np.maximum(np.abs(y_true), 1e-8)

    mape = np.mean(
        np.abs(error) / denominator
    ) * 100

    return {
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2),
        "MAPE (%)": round(float(mape), 2),
    }


# ============================================================
# 4. CHƯƠNG TRÌNH SO SÁNH
# ============================================================

def compare_all_models():

    print("=" * 65)
    print("LAB 4 - MỤC 10: MODEL EVALUATION AND COMPARISON")
    print("Người thực hiện: Bao Dinh")
    print("=" * 65)

    # --------------------------------------------------------
    # 4.1. Đọc dữ liệu gốc
    # --------------------------------------------------------

    df = load_data().sort_index()

    if not df.index.is_unique:
        raise ValueError("Dữ liệu gốc có thời gian trùng lặp.")

    if "load" not in df.columns:
        raise ValueError("Dữ liệu gốc thiếu cột load.")

    n_samples = len(df)

    n_train = int(n_samples * TRAIN_RATIO)
    n_val = int(n_samples * VAL_RATIO)

    test_start = n_train + n_val

    df_test = df.iloc[test_start:].copy()

    if df_test.empty:
        raise ValueError("Tập Test không có dữ liệu.")

    print(f"\nTổng số mẫu: {n_samples}")
    print(f"Train: {n_train}")
    print(f"Validation: {n_val}")
    print(f"Test: {len(df_test)}")

    # --------------------------------------------------------
    # 4.2. Tạo DataFrame dự báo chung
    # --------------------------------------------------------

    comparison = pd.DataFrame(index=df_test.index)

    comparison.index.name = "Datetime"

    comparison["Actual"] = df_test["load"]

    # Baseline 1: Naive
    comparison["Naive"] = (
        df["load"].shift(1).reindex(df_test.index)
    )

    # Baseline 2: Seasonal Naive 24h
    comparison["Seasonal Naive (24h)"] = (
        df["load"].shift(24).reindex(df_test.index)
    )

    # --------------------------------------------------------
    # 4.3. Đọc ba mô hình thực tế
    # --------------------------------------------------------

    for model_name, file_name in MODEL_FILES.items():

        df_pred = load_prediction(model_name, file_name)

        # Đối chiếu thời gian và lấy dự báo
        comparison[model_name] = (
            df_pred["y_pred"].reindex(comparison.index)
        )

        # Nếu file có giá trị thực tế, kiểm tra với dữ liệu gốc
        if "y_true" in df_pred.columns:

            actual_from_model = (
                df_pred["y_true"].reindex(comparison.index)
            )

            valid = (
                comparison["Actual"].notna()
                & actual_from_model.notna()
            )

            if valid.any():

                if not np.allclose(
                    comparison.loc[valid, "Actual"],
                    actual_from_model.loc[valid],
                    rtol=1e-4,
                    atol=1e-3
                ):
                    raise ValueError(
                        f"Dữ liệu thực tế trong {file_name} "
                        "không khớp với tập Test gốc. "
                        "Không thể so sánh công bằng."
                    )

    # --------------------------------------------------------
    # 4.4. Đồng bộ dữ liệu
    # --------------------------------------------------------

    # Chỉ đánh giá trên những thời điểm mà tất cả mô hình
    # đều có dự báo và giá trị thực tế.
    comparison = comparison.replace(
        [np.inf, -np.inf], np.nan
    )

    comparison = comparison.dropna()

    if comparison.empty:
        raise ValueError(
            "Không có thời điểm chung giữa ba mô hình. "
            "Hãy kiểm tra lại Datetime trong các file CSV."
        )

    if len(comparison) < 2:
        raise ValueError(
            "Số lượng thời điểm chung quá ít để đánh giá."
        )

    print(
        f"\nSố thời điểm chung dùng để so sánh: "
        f"{len(comparison)}"
    )

    print(
        f"Thời gian bắt đầu: {comparison.index.min()}"
    )

    print(
        f"Thời gian kết thúc: {comparison.index.max()}"
    )

    # --------------------------------------------------------
    # 4.5. Tính metrics
    # --------------------------------------------------------

    summary = []

    model_names = [
        "Naive",
        "Seasonal Naive (24h)",
        "VAR",
        "XGBoost",
        "LSTM",
    ]

    y_true = comparison["Actual"].values

    for model_name in model_names:

        y_pred = comparison[model_name].values

        result = calculate_metrics(y_true, y_pred)

        result["Model"] = model_name

        summary.append(result)

    df_metrics = (
        pd.DataFrame(summary)
        .set_index("Model")
    )

    df_metrics = df_metrics[
        ["RMSE", "MAE", "MAPE (%)"]
    ]

    print("\n" + "=" * 65)
    print("BẢNG SO SÁNH CHỈ SỐ LỖI")
    print("=" * 65)

    print(df_metrics.to_string())

    # --------------------------------------------------------
    # 4.6. Xác định mô hình tốt nhất
    # --------------------------------------------------------

    best_model = df_metrics["RMSE"].idxmin()

    print("\nMÔ HÌNH CÓ RMSE THẤP NHẤT:")
    print(best_model)

    # --------------------------------------------------------
    # 4.7. Xuất CSV
    # --------------------------------------------------------

    PROCESSED.mkdir(parents=True, exist_ok=True)

    metrics_path = PROCESSED / "compare_metrics.csv"

    df_metrics.to_csv(
        metrics_path,
        encoding="utf-8-sig"
    )

    comparison_path = PROCESSED / "compare_predictions.csv"

    comparison.to_csv(
        comparison_path,
        encoding="utf-8-sig"
    )

    print(f"\nĐã lưu bảng metrics: {metrics_path}")
    print(f"Đã lưu dữ liệu so sánh: {comparison_path}")

    # Ghi chú cách huấn luyện, để bảng chính tự nói rõ luật so sánh.
    pd.DataFrame(
        {"Model": list(TRAINING_NOTES), "Cách huấn luyện": list(TRAINING_NOTES.values())}
    ).to_csv(PROCESSED / "compare_protocol.csv", index=False, encoding="utf-8-sig")

    # Bảng phụ: VAR/XGBoost có refit mỗi 168 giờ, trên cùng các mốc thời gian.
    refit_rows = []
    for model_name, file_name in REFIT_FILES.items():
        df_pred = load_prediction(model_name, file_name)
        y_pred = df_pred["y_pred"].reindex(comparison.index)
        if y_pred.isna().any():
            raise ValueError(f"{file_name} thiếu dự báo ở một số giờ test.")
        refit_rows.append({"Model": model_name,
                           **calculate_metrics(comparison["Actual"].values, y_pred.values)})
    df_refit = pd.DataFrame(refit_rows).set_index("Model")
    df_refit.to_csv(PROCESSED / "compare_metrics_refit.csv", encoding="utf-8-sig")
    print("\nBảng phụ (refit mỗi 168 giờ, không dùng cho xếp hạng chính):")
    print(df_refit.to_string())

    # --------------------------------------------------------
    # 4.8. Biểu đồ Actual vs Prediction
    # --------------------------------------------------------

    FIG.mkdir(parents=True, exist_ok=True)

    hours_view = min(48, len(comparison))

    plot_data = comparison.iloc[:hours_view]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(13, 9),
        sharex=True
    )

    for i, model_name in enumerate(
        ["VAR", "XGBoost", "LSTM"]
    ):

        color = MODEL_COLORS[model_name]

        axes[i].plot(
            plot_data.index,
            plot_data["Actual"],
            color="black",
            linewidth=1.8,
            label="Actual"
        )

        axes[i].plot(
            plot_data.index,
            plot_data[model_name],
            color=color,
            linewidth=1.6,
            label=f"{model_name} Prediction"
        )

        axes[i].set_title(
            f"Actual vs {model_name}",
            fontsize=11,
            fontweight="bold"
        )

        axes[i].set_ylabel("Load (MW)")

        axes[i].legend(loc="upper right")

        axes[i].grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

    axes[-1].set_xlabel("Datetime")

    fig.suptitle(
        "SO SÁNH DỰ BÁO GIỮA BA MÔ HÌNH (48 GIỜ)",
        fontsize=13,
        fontweight="bold"
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    plt.savefig(
        FIG / "s10_1tuan.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 4.9. Biểu đồ RMSE và MAE
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 5))

    df_metrics[["RMSE", "MAE"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "SO SÁNH RMSE VÀ MAE GIỮA CÁC MÔ HÌNH",
        fontweight="bold"
    )

    ax.set_ylabel("Giá trị lỗi (MW)")

    ax.set_xlabel("Mô hình")

    ax.tick_params(axis="x", rotation=0)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        FIG / "s10_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 4.10. Biểu đồ lỗi theo giờ
    # --------------------------------------------------------

    error_df = pd.DataFrame(index=comparison.index)

    error_df["hour"] = comparison.index.hour

    for model_name in model_names:

        error_df[model_name] = np.abs(
            comparison["Actual"] - comparison[model_name]
        )

    mae_by_hour = error_df.groupby("hour").mean()

    fig, ax = plt.subplots(figsize=(12, 5))

    for model_name in model_names:

        ax.plot(
            mae_by_hour.index,
            mae_by_hour[model_name],
            marker="o",
            linewidth=1.5,
            label=model_name
        )

    ax.set_title(
        "MAE TRUNG BÌNH THEO TỪNG GIỜ TRONG NGÀY",
        fontweight="bold"
    )

    ax.set_xlabel("Giờ trong ngày")

    ax.set_ylabel("MAE (MW)")

    ax.set_xticks(range(24))

    ax.legend()

    ax.grid(
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        FIG / "s10_loi_theo_gio.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 4.11. Hoàn thành
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("HOÀN THÀNH MỤC 10")
    print("=" * 65)

    print(f"Mô hình có RMSE thấp nhất: {best_model}")

    print("\nCác biểu đồ đã lưu:")
    print("- s10_1tuan.png")
    print("- s10_metrics.png")
    print("- s10_loi_theo_gio.png")

    print("\nKhông sử dụng dữ liệu dự báo mô phỏng.")


if __name__ == "__main__":
    compare_all_models()
