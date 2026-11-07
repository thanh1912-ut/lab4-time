
"""
Mục 12: Đánh giá Traffic Forecasting (METR-LA) trên Test.
Người làm: Phạm Lê Văn Bảo.
Chạy: python src/nc_evaluate.py

Đọc checkpoint GRU đã lưu, KHÔNG huấn luyện lại.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import RAW
from nc_traffic import (
    BATCH_SIZE,
    DEVICE,
    HORIZONS,
    INPUT_STEPS,
    MODEL_PATH,
    TrafficDataset,
    TrafficGRU,
    fill_missing_causally,
    historical_average,
    masked_metrics,
    predict,
)


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "lab4" / "figures"
RESULT_DIR = ROOT / "data" / "processed"


def score(y_true, y_pred, model_name):
    """Tính MAE, RMSE, MAPE cho từng horizon."""
    rows = []

    for j, horizon in enumerate(HORIZONS):
        if not np.all(np.isfinite(y_pred[:, j, :])):
            raise ValueError(
                f"Du bao {model_name} chua NaN/Inf tai horizon {horizon}."
            )

        mae, rmse, mape = masked_metrics(
            y_true[:, j, :],
            y_pred[:, j, :]
        )

        rows.append({
            "Model": model_name,
            "Horizon_min": horizon * 5,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_percent": mape,
        })

    return rows


def plot_horizon(metrics):
    """Biểu đồ so sánh ba mô hình theo ba khoảng dự báo."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    names = [
        "Last Value",
        "Historical Average",
        "GRU"
    ]

    x = np.arange(len(HORIZONS))
    width = 0.25

    columns = [
        ("MAE", "MAE"),
        ("RMSE", "RMSE"),
        ("MAPE_percent", "MAPE (%)"),
    ]

    for ax, (metric, label) in zip(axes, columns):
        for k, name in enumerate(names):
            subset = metrics[
                metrics["Model"] == name
            ].set_index("Horizon_min")

            values = subset.loc[
                [15, 30, 60], metric
            ].to_numpy()

            ax.bar(
                x + (k - 1) * width,
                values,
                width,
                label=name
            )

        ax.set_xticks(
            x,
            ["15 min", "30 min", "60 min"]
        )
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=8)

    fig.suptitle(
        "METR-LA - Test metrics by forecast horizon"
    )
    fig.tight_layout()

    path = FIG_DIR / "nc_horizon.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def plot_example(y_true, y_pred, target_times, sensor_names):
    """Vẽ thực tế và dự báo 60 phút của một cảm biến."""
    j = HORIZONS.index(12)

    valid = (
        np.isfinite(y_true[:, j, :])
        & (y_true[:, j, :] != 0)
    )

    sensor = int(
        np.argmax(valid.sum(axis=0))
    )

    mask = valid[:, sensor]
    window = min(288, len(mask))

    if window == 0 or not mask.any():
        raise ValueError(
            "Test khong co nhan hop le de ve vi du."
        )

    counts = np.convolve(
        mask.astype(np.int32),
        np.ones(window, dtype=np.int32),
        mode="valid"
    )

    start = int(np.argmax(counts))
    stop = start + window

    actual = y_true[
        start:stop, j, sensor
    ].copy()

    forecast = y_pred[
        start:stop, j, sensor
    ].copy()

    good = mask[start:stop]

    actual[~good] = np.nan
    forecast[~good] = np.nan

    times = pd.to_datetime(
        target_times[start:stop, j]
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(
        times,
        actual,
        label="Actual",
        linewidth=1.5
    )

    ax.plot(
        times,
        forecast,
        label="GRU prediction (60 min)",
        linewidth=1.3
    )

    ax.set_title(
        f"METR-LA - Sensor {sensor_names[sensor]} "
        "- 60-min forecast"
    )

    ax.set_xlabel("Target timestamp")
    ax.set_ylabel("Speed (mph)")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.autofmt_xdate()
    fig.tight_layout()

    path = FIG_DIR / "nc_vidu.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path, sensor_names[sensor]


def main():
    # 1. Kiểm tra checkpoint
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Khong tim thay checkpoint: {MODEL_PATH}"
        )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=== TEST METR-LA (KHONG TRAIN LAI) ===")
    print("Device:", DEVICE)

    # 2. Đọc dữ liệu
    df = pd.read_hdf(
        RAW / "metr-la.h5"
    )

    train_end = int(len(df) * 0.7)
    test_start = int(len(df) * 0.8)

    train = df.iloc[:train_end]
    test = df.iloc[test_start:]

    print("METR-LA:", df.shape)
    print("Train:", train.shape)
    print("Test:", test.shape)

    # 3. Thống kê chỉ từ Train
    train_array = train.to_numpy(
        dtype=np.float32
    )

    good = (
        np.isfinite(train_array)
        & (train_array != 0)
    )

    global_mean = float(
        train_array[good].mean(
            dtype=np.float64
        )
    )

    sensor_sums = np.where(
        good,
        train_array,
        0
    ).sum(axis=0, dtype=np.float64)

    sensor_counts = good.sum(axis=0)

    sensor_means = np.divide(
        sensor_sums,
        sensor_counts,
        out=np.full(
            train.shape[1],
            global_mean
        ),
        where=sensor_counts > 0
    )

    # 4. Khôi phục GRU đã train
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if int(checkpoint["num_sensors"]) != df.shape[1]:
        raise ValueError(
            "So cam bien trong checkpoint khong khop dataset."
        )

    mean = float(checkpoint["mean"])
    std = float(checkpoint["std"])

    if std <= 0:
        raise ValueError(
            "Checkpoint std khong hop le."
        )

    model = TrafficGRU(
        df.shape[1]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

    print(
        "Best Val Loss:",
        checkpoint["val_loss"]
    )

    # 5. Tiền xử lý giống lúc Train
    filled = fill_missing_causally(
        df,
        sensor_means
    )

    inputs = (filled - mean) / std

    targets = (
        df.replace(0, np.nan) - mean
    ) / std

    test_dataset = TrafficDataset(
        inputs.iloc[test_start:],
        targets.iloc[test_start:]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    # 6. Dự báo GRU
    y_true, gru_pred = predict(
        model,
        test_loader,
        mean,
        std
    )

    # 7. Baseline Last Value
    # Dùng dữ liệu đã điền thiếu theo chiều thời gian,
    # nhất quán với đầu vào của GRU.
    t = np.arange(
        INPUT_STEPS - 1,
        len(test) - max(HORIZONS)
    )

    target_positions = (
        t[:, None]
        + np.array(HORIZONS)[None, :]
    )

    filled_test = filled.iloc[test_start:].to_numpy(
        dtype=np.float32
    )

    last_pred = np.repeat(
        filled_test[t, None, :],
        len(HORIZONS),
        axis=1
    )

    # 8. Baseline Historical Average
    times = test.index.to_numpy()[
        target_positions
    ]

    hist_pred = historical_average(
        train,
        pd.DatetimeIndex(
            times.ravel()
        )
    )

    hist_pred = hist_pred.reshape(
        len(t),
        len(HORIZONS),
        df.shape[1]
    )

    # 9. Kiểm tra kích thước
    if not (
        y_true.shape
        == gru_pred.shape
        == last_pred.shape
        == hist_pred.shape
    ):
        raise ValueError(
            "Kich thuoc du bao khong khop nhau."
        )

    # 10. Đánh giá ba mô hình
    rows = []

    models = [
        ("Last Value", last_pred),
        ("Historical Average", hist_pred),
        ("GRU", gru_pred)
    ]

    for name, prediction in models:
        rows.extend(
            score(
                y_true,
                prediction,
                name
            )
        )

    metrics = pd.DataFrame(rows)

    print("\n=== KET QUA TEST ===")

    print(
        metrics.to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}"
        )
    )

    # 11. Xuất CSV
    csv_path = (
        RESULT_DIR / "nc_test_metrics.csv"
    )

    metrics.to_csv(
        csv_path,
        index=False
    )

    # 12. Xuất hai biểu đồ
    chart_path = plot_horizon(
        metrics
    )

    example_path, sensor = plot_example(
        y_true,
        gru_pred,
        times,
        df.columns
    )

    print("\n=== FILE DA XUAT ===")
    print(csv_path)
    print(chart_path)
    print(example_path)
    print("Cam bien vi du:", sensor)


if __name__ == "__main__":
    main()
