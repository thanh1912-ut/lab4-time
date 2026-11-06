
"""
Mục 4, 5: Chuẩn hóa, chia tập, tạo cửa sổ lag và rolling features.

Người thực hiện: Bao Dinh

Nội dung:
1. Tải và kiểm tra dữ liệu.
2. Chia Train / Validation / Test theo thứ tự thời gian.
3. Chuẩn hóa dữ liệu, chỉ fit trên Train.
4. In thống kê trước và sau chuẩn hóa.
5. Tạo cửa sổ dữ liệu 3D (N, 24, 4).
6. Tạo dữ liệu phẳng gồm 96 biến lag và 3 biến rolling.
7. Vẽ hình minh họa và lưu kết quả.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data, COLS, L, FIG, PROCESSED
from features import fit_scale, make_windows, add_rolling


# ============================================================
# 1. CHIA DỮ LIỆU VÀ IN THÔNG TIN
# ============================================================

def print_split_info(df, i1, i2):
    """In số lượng mẫu và khoảng thời gian của từng tập."""

    train = df.iloc[:i1]
    val = df.iloc[i1:i2]
    test = df.iloc[i2:]

    print("\n--- THÔNG TIN CHIA TẬP DỮ LIỆU ---")

    print(
        f"Train: {len(train):,} mẫu "
        f"({len(train) / len(df) * 100:.2f}%)"
    )

    print(
        f"Validation: {len(val):,} mẫu "
        f"({len(val) / len(df) * 100:.2f}%)"
    )

    print(
        f"Test: {len(test):,} mẫu "
        f"({len(test) / len(df) * 100:.2f}%)"
    )

    print(f"\nThời gian Train: {train.index[0]} -> {train.index[-1]}")
    print(f"Thời gian Val  : {val.index[0]} -> {val.index[-1]}")
    print(f"Thời gian Test : {test.index[0]} -> {test.index[-1]}")

    return train, val, test


# ============================================================
# 2. VẼ BIỂU ĐỒ CHIA TẬP
# ============================================================

def plot_data_split(train, val, test):
    """Vẽ và lưu biểu đồ phân chia Train / Validation / Test."""

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        train.index,
        train["load"],
        label="Train",
        linewidth=0.8
    )

    ax.plot(
        val.index,
        val["load"],
        label="Validation",
        linewidth=0.8
    )

    ax.plot(
        test.index,
        test["load"],
        label="Test",
        linewidth=0.8
    )

    ax.set_title(
        "Temporal Train / Validation / Test Split",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Datetime")
    ax.set_ylabel("Load (MW)")

    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()

    output_path = Path(FIG) / "s4_split.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nĐã lưu biểu đồ chia tập: {output_path}")


# ============================================================
# 3. IN THỐNG KÊ TRƯỚC VÀ SAU CHUẨN HÓA
# ============================================================

def print_scaling_info(df, df_scaled, i1):
    """So sánh Mean và Standard Deviation của tập Train."""

    train_before = df.iloc[:i1][COLS]
    train_after = df_scaled.iloc[:i1][COLS]

    print("\n--- THỐNG KÊ TRƯỚC KHI SCALE (TRAIN) ---")
    print(train_before.agg(["mean", "std"]).round(4))

    print("\n--- THỐNG KÊ SAU KHI SCALE (TRAIN) ---")
    print(train_after.agg(["mean", "std"]).round(4))


# ============================================================
# 4. VẼ MINH HỌA CỬA SỔ 24 GIỜ
# ============================================================

def plot_window_sample(X):
    """Vẽ một cửa sổ dữ liệu gồm 24 giờ và 4 biến."""

    if len(X) == 0:
        raise ValueError("Không có cửa sổ dữ liệu để vẽ.")

    sample = X[0]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(11, 9),
        sharex=True
    )

    for i, col in enumerate(COLS):

        axes[i].plot(
            range(1, L + 1),
            sample[:, i],
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=col
        )

        axes[i].set_ylabel(col)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.35)

    axes[0].set_title(
        f"One Input Window: {L} Hours × {len(COLS)} Features",
        fontsize=13,
        fontweight="bold"
    )

    axes[-1].set_xlabel("Time Step (Hour)")

    fig.tight_layout()

    output_path = Path(FIG) / "s5_window.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu biểu đồ cửa sổ: {output_path}")


# ============================================================
# 5. TẠO DỮ LIỆU PHẲNG CHO XGBOOST
# ============================================================

def create_flat_features(df_scaled):
    """
    Tạo dữ liệu phẳng:
        - 96 biến lag: 24 giờ × 4 biến.
        - 3 biến rolling.
        - 1 cột target load.

    Các biến đầu vào tại thời điểm t chỉ sử dụng dữ liệu
    từ thời điểm t-24 đến t-1.

    Target là load tại thời điểm t.
    """

    flat = pd.DataFrame(index=df_scaled.index)

    # 1. Tạo 96 biến lag.
    for col in COLS:
        for lag in range(1, L + 1):
            flat[f"{col}_lag_{lag}"] = df_scaled[col].shift(lag)

    # 2. Tạo 3 biến rolling từ dữ liệu quá khứ.
    load_history = df_scaled["load"].shift(1)

    flat["load_roll_mean_24"] = (
        load_history.rolling(window=24).mean()
    )

    flat["load_roll_std_24"] = (
        load_history.rolling(window=24).std()
    )

    flat["load_roll_max_24"] = (
        load_history.rolling(window=24).max()
    )

    # 3. Target: load tại thời điểm dự báo.
    flat["load"] = df_scaled["load"]

    # 4. Loại bỏ các dòng chưa đủ 24 giờ lịch sử.
    flat = flat.dropna()

    return flat


# ============================================================
# 6. HÀM CHẠY CHÍNH
# ============================================================

def run_preprocess_demo():

    print("=" * 65)
    print("LAB 4 - MỤC 4 & 5: PREPROCESSING AND FEATURE ENGINEERING")
    print("Người thực hiện: Bao Dinh")
    print("=" * 65)

    # --------------------------------------------------------
    # Bước 1: Tải dữ liệu
    # --------------------------------------------------------

    df = load_data()

    print("\n[1] DATA INFORMATION")
    print(f"Tổng số mẫu: {len(df):,}")
    print(f"Các biến: {COLS}")
    print(f"Khoảng thời gian: {df.index[0]} -> {df.index[-1]}")

    # --------------------------------------------------------
    # Bước 2: Chia Train / Validation / Test
    # --------------------------------------------------------

    i1 = int(len(df) * 0.70)
    i2 = int(len(df) * 0.85)

    train, val, test = print_split_info(df, i1, i2)

    # Vẽ biểu đồ phân chia.
    plot_data_split(train, val, test)

    # --------------------------------------------------------
    # Bước 3: Chuẩn hóa dữ liệu
    # --------------------------------------------------------

    print("\n[2] STANDARDIZATION")

    df_scaled, scaler, (i1, i2) = fit_scale(df)

    print("Đã fit StandardScaler trên tập Train.")

    print_scaling_info(df, df_scaled, i1)

    # --------------------------------------------------------
    # Bước 4: Tạo cửa sổ dữ liệu 3D
    # --------------------------------------------------------

    print("\n[3] 3D SLIDING WINDOW")

    X_win, y_win, target_time = make_windows(df_scaled)

    print(f"X shape: {X_win.shape}")
    print(f"y shape: {y_win.shape}")
    print(f"Target time shape: {target_time.shape}")

    print(
        f"Số đặc trưng: {X_win.shape[2]} | "
        f"Số bước thời gian: {X_win.shape[1]}"
    )

    print(
        f"Thời gian dự báo đầu tiên: {target_time[0]}"
    )

    print(
        f"Thời gian dự báo cuối cùng: {target_time[-1]}"
    )

    # Vẽ hình minh họa.
    plot_window_sample(X_win)

    # --------------------------------------------------------
    # Bước 5: Tạo dữ liệu phẳng và rolling cho XGBoost
    # --------------------------------------------------------

    print("\n[4] FLAT FEATURES AND ROLLING")

    flat_df = create_flat_features(df_scaled)

    print(f"Số dòng dữ liệu flat: {len(flat_df):,}")
    print(f"Số cột dữ liệu flat: {len(flat_df.columns)}")

    lag_columns = [
        col for col in flat_df.columns
        if "_lag_" in col
    ]

    rolling_columns = [
        "load_roll_mean_24",
        "load_roll_std_24",
        "load_roll_max_24"
    ]

    print(f"Số biến lag: {len(lag_columns)}")
    print(f"Số biến rolling: {len(rolling_columns)}")

    print("\nCác biến rolling:")
    print(rolling_columns)

    # --------------------------------------------------------
    # Bước 6: Lưu dữ liệu đã xử lý
    # --------------------------------------------------------

    output_path = Path(PROCESSED) / "features_flat.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flat_df.to_csv(output_path, index_label="Datetime")

    print(f"\nĐã lưu dữ liệu flat: {output_path}")

    # --------------------------------------------------------
    # Hoàn thành
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("HOÀN THÀNH MỤC 4 & 5")
    print("=" * 65)

    return {
        "df_scaled": df_scaled,
        "scaler": scaler,
        "split_indices": (i1, i2),
        "X_window": X_win,
        "y_window": y_win,
        "target_time": target_time,
        "flat_df": flat_df
    }


if __name__ == "__main__":
    run_preprocess_demo()
