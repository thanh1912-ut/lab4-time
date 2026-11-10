
"""
Mục 12 nâng cao: Traffic forecasting (METR-LA / PEMS-BAY).
Người làm: Phạm Lê Văn Bảo.
Chi tiết: docs/lab4/phan-cong.md, mục 5.3.

Mô hình:
- Last Value Baseline
- Historical Average Baseline
- GRU dự báo 15, 30, 60 phút

Chia dữ liệu theo thời gian: 70% Train, 10% Validation, 20% Test.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils import RAW, SEED


INPUT_STEPS = 12
HORIZONS = (3, 6, 12)

HIDDEN_SIZE = 128
NUM_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "nc_gru_best.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_metrics(y_true, y_pred):
    """Tính MAE, RMSE, MAPE; bỏ qua nhãn 0."""
    mask = np.isfinite(y_true) & (y_true != 0)

    actual = y_true[mask].astype(np.float64)
    predicted = y_pred[mask].astype(np.float64)

    if actual.size == 0:
        return np.nan, np.nan, np.nan

    error = actual - predicted

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    mape = np.mean(np.abs(error / actual)) * 100

    return mae, rmse, mape


def evaluate_predictions(y_true, y_pred, name):
    print(f"\n=== {name} - VALIDATION ===")

    for i, horizon in enumerate(HORIZONS):
        mae, rmse, mape = masked_metrics(
            y_true[:, i, :],
            y_pred[:, i, :]
        )

        print(
            f"{horizon * 5:2d} phut | "
            f"MAE: {mae:.4f} | "
            f"RMSE: {rmse:.4f} | "
            f"MAPE: {mape:.2f}%"
        )


def historical_average(train, target_times):
    """Trung bình lịch sử theo thứ, giờ, phút của từng cảm biến."""
    values = train.to_numpy(dtype=np.float32).copy()
    values[~np.isfinite(values) | (values == 0)] = np.nan

    slots = (
        train.index.dayofweek.to_numpy() * 288
        + train.index.hour.to_numpy() * 12
        + train.index.minute.to_numpy() // 5
    )

    global_sum = np.nansum(values, axis=0)
    global_count = np.sum(np.isfinite(values), axis=0)

    global_mean = np.divide(
        global_sum,
        global_count,
        out=np.zeros(train.shape[1], dtype=np.float32),
        where=global_count > 0
    )

    slot_means = np.empty(
        (7 * 288, train.shape[1]),
        dtype=np.float32
    )

    for slot in range(7 * 288):
        selected = values[slots == slot]

        if len(selected) == 0:
            slot_means[slot] = global_mean
            continue

        counts = np.sum(np.isfinite(selected), axis=0)
        totals = np.nansum(selected, axis=0)

        slot_means[slot] = np.divide(
            totals,
            counts,
            out=global_mean.copy(),
            where=counts > 0
        )

    target_slots = (
        target_times.dayofweek.to_numpy() * 288
        + target_times.hour.to_numpy() * 12
        + target_times.minute.to_numpy() // 5
    )

    return slot_means[target_slots]


def fill_missing_causally(df, train_sensor_means):
    """
    Chỉ điền đầu vào:
    - 0 được xem là thiếu
    - Forward fill dùng dữ liệu quá khứ
    - Các vị trí đầu chuỗi thiếu được điền bằng mean Train

    Không thay đổi nhãn thực tế.
    """
    clean = df.replace(0, np.nan).ffill()

    return clean.fillna(
        pd.Series(train_sensor_means, index=df.columns)
    )


class TrafficDataset(Dataset):
    """
    Tạo cửa sổ khi DataLoader yêu cầu, giảm sử dụng RAM.

    X: (12, 207)
    y: (3, 207)
    """
    def __init__(self, inputs, targets):
        self.inputs = inputs.to_numpy(dtype=np.float32)
        self.targets = targets.to_numpy(dtype=np.float32)

        self.length = (
            len(self.inputs)
            - (INPUT_STEPS - 1)
            - max(HORIZONS)
        )

        if self.length <= 0:
            raise ValueError("Khong du du lieu de tao cua so.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t = idx + INPUT_STEPS - 1

        x = self.inputs[t - INPUT_STEPS + 1:t + 1]
        y = self.targets[
            [t + h for h in HORIZONS]
        ]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y)
        )


class TrafficGRU(nn.Module):
    def __init__(self, num_sensors):
        super().__init__()

        self.gru = nn.GRU(
            input_size=num_sensors,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )

        self.output = nn.Linear(
            HIDDEN_SIZE,
            len(HORIZONS) * num_sensors
        )

        self.num_sensors = num_sensors

    def forward(self, x):
        sequence, _ = self.gru(x)

        last_hidden = sequence[:, -1, :]
        prediction = self.output(last_hidden)

        return prediction.reshape(
            -1,
            len(HORIZONS),
            self.num_sensors
        )


def masked_mae_loss(prediction, target):
    """Loss chỉ tính trên mục tiêu hợp lệ."""
    mask = torch.isfinite(target) & (target != 0)

    # Target được chuẩn hóa; giá trị thiếu giữ là NaN.
    mask = torch.isfinite(target)

    safe_target = torch.where(
        mask,
        target,
        torch.zeros_like(target)
    )

    errors = torch.abs(prediction - safe_target)
    return (errors * mask).sum() / mask.sum().clamp(min=1)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_error = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            prediction = model(x)

            mask = torch.isfinite(y)
            safe_y = torch.where(
                mask, y, torch.zeros_like(y)
            )

            abs_errors = torch.abs(prediction - safe_y)
            loss = (
                (abs_errors * mask).sum()
                / mask.sum().clamp(min=1)
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0
                )
                optimizer.step()

        total_error += (
            (abs_errors.detach() * mask).sum().item()
        )
        total_count += mask.sum().item()

    return total_error / max(total_count, 1)


@torch.no_grad()
def predict(model, loader, mean, std):
    model.eval()

    predictions = []
    targets = []

    for x, y in loader:
        x = x.to(DEVICE)

        pred = model(x).cpu().numpy()
        true = y.numpy()

        predictions.append(pred * std + mean)
        targets.append(true * std + mean)

    return (
        np.concatenate(targets),
        np.concatenate(predictions)
    )


def main():
    set_seed(SEED)

    print("=== THIET BI HUAN LUYEN ===")
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 1. Đọc dữ liệu
    df = pd.read_hdf(RAW / "metr-la.h5")

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print("\n=== DATASET ===")
    print("METR-LA:", df.shape)
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    # 2. Tính thống kê chỉ từ Train
    train_array = train.to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(train_array) & (train_array != 0)

    valid_values = train_array[valid_mask]

    mean = float(valid_values.mean(dtype=np.float64))
    std = float(valid_values.std(dtype=np.float64))

    sensor_sums = np.where(
        valid_mask, train_array, 0
    ).sum(axis=0, dtype=np.float64)

    sensor_counts = valid_mask.sum(axis=0)

    sensor_means = np.divide(
        sensor_sums,
        sensor_counts,
        out=np.full(train.shape[1], mean),
        where=sensor_counts > 0
    )

    print(f"\nTrain mean: {mean:.4f}")
    print(f"Train std: {std:.4f}")

    # 3. Xử lý dữ liệu đầu vào, không sửa nhãn
    # Dùng toàn bộ lịch sử quá khứ cho forward fill.
    filled = fill_missing_causally(df, sensor_means)

    # Chỉ chuẩn hóa đầu vào bằng mean/std của Train
    normalized_input = (filled - mean) / std

    # Nhãn 0 được chuyển thành NaN để masked loss bỏ qua.
    normalized_target = df.replace(0, np.nan)
    normalized_target = (normalized_target - mean) / std

    # 4. Tạo Dataset cho từng tập
    train_ds = TrafficDataset(
        normalized_input.iloc[:train_end],
        normalized_target.iloc[:train_end]
    )

    val_ds = TrafficDataset(
        normalized_input.iloc[train_end:val_end],
        normalized_target.iloc[train_end:val_end]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    print("\n=== SO MAU ===")
    print("Train:", len(train_ds))
    print("Validation:", len(val_ds))
    print("Test: giu lai de danh gia cuoi cung")

    # 5. Đánh giá hai baseline trên Validation
    val_raw = val.to_numpy(dtype=np.float32)

    target_indices = np.arange(
        INPUT_STEPS - 1,
        len(val) - max(HORIZONS)
    )

    y_val = np.stack(
        [val_raw[target_indices + h] for h in HORIZONS],
        axis=1
    )

    # Last Value sử dụng dữ liệu đã xử lý thiếu giống GRU và Test.
    filled_val = filled.iloc[train_end:val_end].to_numpy(
        dtype=np.float32
    )

    last_predictions = np.repeat(
        filled_val[target_indices, None, :],
        len(HORIZONS),
        axis=1
    )

    evaluate_predictions(
        y_val, last_predictions, "LAST VALUE"
    )

    target_times = pd.DatetimeIndex(
        val.index[
            (target_indices[:, None] + np.array(HORIZONS)).ravel()
        ]
    )

    historical_predictions = historical_average(
        train, target_times
    ).reshape(
        len(target_indices),
        len(HORIZONS),
        df.shape[1]
    )

    evaluate_predictions(
        y_val,
        historical_predictions,
        "HISTORICAL AVERAGE"
    )

    # 6. Khởi tạo mô hình GRU
    model = TrafficGRU(df.shape[1]).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\n=== BAT DAU TRAIN GRU ===")
    print("Epoch toi da:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Hidden size:", HIDDEN_SIZE)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    # 7. Huấn luyện
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer
        )

        val_loss = run_epoch(
            model, val_loader
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}",
            flush=True
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "num_sensors": df.shape[1],
                },
                MODEL_PATH
            )

            print("  -> Da luu model tot nhat")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping: Validation khong cai thien.")
            break

    # 8. Đọc lại checkpoint tốt nhất
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state"])

    print("\n=== KET QUA TRAIN ===")
    print("Best epoch:", best_epoch)
    print(f"Best Val Loss: {best_val:.4f}")
    print("Model:", MODEL_PATH)

    # 9. Đánh giá GRU trên Validation
    y_true, y_pred = predict(
        model, val_loader, mean, std
    )

    # Nhãn thiếu được giữ là NaN sau khi giải chuẩn hóa.
    evaluate_predictions(
        y_true, y_pred, "GRU"
    )


if __name__ == "__main__":
    main()
