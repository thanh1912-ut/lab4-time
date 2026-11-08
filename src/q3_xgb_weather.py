"""Câu hỏi 3: XGBoost học được gì từ thời tiết (và Câu 6: ML vs DL).
Người làm: Hoàng Thắng.
Chi tiết: docs/lab4/phan-cong.md, mục 5.5.
"""
import sys
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import partial_dependence

from utils import load_data, COLS, L, SEED, PROCESSED, FIG, metrics
from features import fit_scale, make_windows, add_rolling, inverse_load, split_index
from m2_xgboost import build_features

DPI = 150


def analyze_feature_importance(model, feature_names):
    """Phân tích độ quan trọng đặc trưng (Gain) của 99 cột:
    1. Tổng hợp Gain theo nhóm biến (load, temperature, humidity, wind_speed, rolling).
    2. Tổng hợp Gain theo từng bậc trễ (Lag 1 đến Lag 24).
    Lưu hình: q3_importance_nhom.png, q3_importance_lag.png.
    """
    booster = model.get_booster()
    gains = booster.get_score(importance_type="gain")

    # Đảm bảo tất cả 99 đặc trưng đều có giá trị (nếu không xuất hiện thì gain = 0)
    full_gains = {f: gains.get(f, 0.0) for f in feature_names}
    total_gain = sum(full_gains.values())

    # 1. Nhóm theo biến
    group_map = {
        "load": "Phụ tải (load lags)",
        "temperature": "Nhiệt độ (temperature lags)",
        "humidity": "Độ ẩm (humidity lags)",
        "wind_speed": "Tốc độ gió (wind_speed lags)",
        "rolling": "Đặc trưng cuộn (rolling & trend)",
    }
    group_gains = {k: 0.0 for k in group_map}

    for name, val in full_gains.items():
        if "roll" in name or "trend" in name:
            group_gains["rolling"] += val
        elif "load" in name:
            group_gains["load"] += val
        elif "temp" in name:
            group_gains["temperature"] += val
        elif "hum" in name:
            group_gains["humidity"] += val
        elif "wind" in name:
            group_gains["wind_speed"] += val

    df_group = pd.DataFrame([
        {
            "Nhóm": group_map[k],
            "Tổng Gain": group_gains[k],
            "Tỷ lệ (%)": (group_gains[k] / total_gain) * 100,
        }
        for k in group_gains
    ]).sort_values(by="Tổng Gain", ascending=False)

    print("\n" + "=" * 60)
    print("BẢNG 1: ĐỘ QUAN TRỌNG ĐẶC TRƯNG THEO NHÓM BIẾN (GAIN)")
    print("=" * 60)
    for _, row in df_group.iterrows():
        print(f"  {row['Nhóm']:<35}: {row['Tổng Gain']:>8.2f} ({row['Tỷ lệ (%)']:>5.2f}%)")
    print("=" * 60)

    # Vẽ biểu đồ nhóm biến
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.barh(df_group["Nhóm"], df_group["Tỷ lệ (%)"], color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(df_group["Tỷ lệ (%)"]) * 1.15)
    ax.invert_yaxis()
    ax.set_title("Độ quan trọng theo nhóm biến trong mô hình XGBoost (Gain %)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Tỷ lệ đóng góp Gain (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    plt.tight_layout()
    plt.savefig(FIG / "q3_importance_nhom.png")
    plt.close()
    print("Đã lưu biểu đồ: q3_importance_nhom.png")

    # 2. Nhóm theo bậc trễ (Lag 1 đến Lag 24)
    lag_gains = {lag: 0.0 for lag in range(1, 25)}
    lag_load = {lag: 0.0 for lag in range(1, 25)}
    lag_weather = {lag: 0.0 for lag in range(1, 25)}

    for name, val in full_gains.items():
        match = re.search(r"_lag(\d+)$", name)
        if match:
            lag = int(match.group(1))
            lag_gains[lag] += val
            if "load" in name:
                lag_load[lag] += val
            else:
                lag_weather[lag] += val

    # Vẽ biểu đồ theo độ trễ
    lags = list(range(1, 25))
    fig, ax = plt.subplots(figsize=(11, 5), dpi=DPI)
    ax.bar(lags, [lag_load[l] for l in lags], label="Phụ tải (load)", color="#1f77b4", alpha=0.9)
    ax.bar(lags, [lag_weather[l] for l in lags], bottom=[lag_load[l] for l in lags], label="Thời tiết (T, H, W)", color="#ff7f0e", alpha=0.9)

    ax.set_title("Độ quan trọng đặc trưng theo độ trễ (Lag 1 đến 24)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Bậc trễ (Giờ)", fontsize=11)
    ax.set_ylabel("Tổng Gain", fontsize=11)
    ax.set_xticks(lags)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(frameon=True, facecolor="white", shadow=True)

    # Chú thích lag 1 và lag 24
    ax.annotate("Lag 24 (Cùng giờ hôm qua)\nGain lớn nhất: chu kỳ ngày",
                xy=(24, lag_gains[24]), xytext=(17, lag_gains[24] * 0.8),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
                fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

    ax.annotate("Lag 1 (Giờ gần nhất)",
                xy=(1, lag_gains[1]), xytext=(3, lag_gains[24] * 0.3),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
                fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIG / "q3_importance_lag.png")
    plt.close()
    print("Đã lưu biểu đồ: q3_importance_lag.png")

    return df_group, lag_gains


def run_ablation_study(df_flat, y, scaler, i1, i2):
    """Ablation Study:
    So sánh mô hình đầy đủ (Full: Lags + Weather + Rolling) với:
    - Mô hình chỉ có phụ tải (Load Only: 24 lags load)
    - Mô hình phụ tải + rolling (Load + Rolling: không có biến thời tiết)
    """
    train_idx = slice(0, i1 - L)
    val_idx = slice(i1 - L, i2 - L)
    test_idx = slice(i2 - L, len(y))

    y_val_orig = inverse_load(y[val_idx], scaler)
    y_test_orig = inverse_load(y[test_idx], scaler)

    # Chuẩn bị danh sách cột
    load_cols = [f"load_lag{lag}" for lag in range(1, 25)]
    roll_cols = ["load_roll_mean_6", "load_roll_std_24", "load_trend_24"]
    load_roll_cols = load_cols + roll_cols

    configs = {
        "Full (96 lags + 3 rolling)": df_flat.columns.tolist(),
        "Load + Rolling (27 cột, bỏ thời tiết)": load_roll_cols,
        "Load Only (24 lags, bỏ thời tiết & rolling)": load_cols,
    }

    results = []
    print("\n" + "=" * 80)
    print("TIẾN HÀNH ABLATION STUDY (ĐÁNH GIÁ ĐÓNG GÓP CỦA THỜI TIẾT)")
    print("=" * 80)

    for name, cols in configs.items():
        X_sub = df_flat[cols]
        m = xgb.XGBRegressor(
            max_depth=7,
            n_estimators=1000,
            learning_rate=0.05,
            early_stopping_rounds=50,
            random_state=SEED,
            n_jobs=-1,
        )
        m.fit(X_sub.iloc[train_idx], y[train_idx], eval_set=[(X_sub.iloc[val_idx], y[val_idx])], verbose=False)

        # Validation metrics
        pred_val = inverse_load(m.predict(X_sub.iloc[val_idx]), scaler)
        score_val = metrics(y_val_orig, pred_val)

        # Test metrics
        pred_test = inverse_load(m.predict(X_sub.iloc[test_idx]), scaler)
        score_test = metrics(y_test_orig, pred_test)

        results.append({
            "Cấu hình": name,
            "Số đặc trưng": len(cols),
            "Best Trees": m.best_iteration,
            "Val RMSE": score_val["RMSE"],
            "Val MAE": score_val["MAE"],
            "Val MAPE": score_val["MAPE(%)"],
            "Test RMSE": score_test["RMSE"],
            "Test MAE": score_test["MAE"],
            "Test MAPE": score_test["MAPE(%)"],
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("=" * 80)

    return df_res


def plot_pdp_temperature(model, df_train, scaler, save_path):
    """Vẽ Partial Dependence Plot (PDP) của nhiệt độ trễ 1 giờ (temperature_lag1).
    Chuyển trục x sang °C thực và trục y sang độ lệch công suất tiêu thụ (MW).
    """
    pdp = partial_dependence(model, df_train, features=["temperature_lag1"], grid_resolution=50)
    temp_scaled = pdp["grid_values"][0]
    avg_scaled = pdp["average"][0]

    # Đổi sang °C và delta MW gốc
    temp_celsius = temp_scaled * scaler.scale_[1] + scaler.mean_[1]
    load_delta_mw = (avg_scaled - avg_scaled.mean()) * scaler.scale_[0]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=DPI)
    ax.plot(temp_celsius, load_delta_mw, color="#e6550d", linewidth=2.2, marker="o", markersize=4)

    ax.set_title("Ảnh hưởng cận biên của Nhiệt độ lên Phụ tải (PDP: temperature_lag1)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Nhiệt độ quan sát (°C)", fontsize=11)
    ax.set_ylabel("Độ lệch phụ tải biên (MW)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Ghi chú vùng tăng tải
    ax.axvspan(20, max(temp_celsius), color="red", alpha=0.08, label="Vùng dùng điều hòa nhiệt độ (>20°C)")
    ax.legend(frameon=True, facecolor="white", shadow=True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu biểu đồ PDP: {save_path.name}")


def print_q6_comparison():
    """In bảng và nhận xét so sánh giữa Machine Learning (XGBoost) và Deep Learning (LSTM) - Câu 6."""
    comp_text = """
========================================================================================
CÂU HỎI 6: KHÁC BIỆT GIỮA MACHINE LEARNING (XGBOOST) VÀ DEEP LEARNING (LSTM)
========================================================================================
| Tiêu chí                      | XGBoost (Machine Learning)        | LSTM (Deep Learning)              |
|-------------------------------|-----------------------------------|-----------------------------------|
| Dạng dữ liệu đầu vào          | Vector phẳng 2D (99 đặc trưng)    | Tensor 3D (24 bước x 4 biến)       |
| Kỹ nghệ đặc trưng             | Cần tự tạo (lag, rolling, trend)  | Tự động trích xuất qua cổng nhớ   |
| Chuẩn hóa dữ liệu             | Không bắt buộc (dựa trên phân vị)  | Rất quan trọng (cần StandardScaler)|
| Số lượng tham số              | Dựa trên số cây & lá (nhẹ)        | 30.113 tham số (LSTM 64->32->Dense)|
| Thời gian huấn luyện          | Rất nhanh (~2s đơn, ~75s walk-fwd)| Chậm hơn đáng kể (~3-5 phút)      |
| Sai số kiểm thử (Test RMSE)   | 6.07 MW (MAPE: 1.86%)              | ~6.5 - 7.5 MW (tương đương/nhỉnh hơn)|
| Khả năng giải thích (XAI)     | Rõ ràng (Gain, Split, PDP)        | Khó giải thích (hộp đen nơ-ron)   |
| Độ ổn định với seed ngẫu nhiên| Rất cao, lặp lại nhất quán        | Nhạy cảm với khởi tạo trọng số    |
| Lượng dữ liệu cần thiết       | Hoạt động tốt từ dữ liệu vừa/nhỏ  | Đòi hỏi lượng dữ liệu lớn để hội tụ|
========================================================================================
Nhận xét kết luận:
1. Trong bài toán dự báo 1 bước với dữ liệu chuỗi có tính chu kỳ ngày/tuần mạnh, XGBoost kết hợp
   với kỹ nghệ đặc trưng miền (lag 24, rolling statistics) đem lại độ chính xác vượt trội hơn hoặc
   tương đương LSTM nhưng thời gian huấn luyện nhanh hơn gấp nhiều lần.
2. XGBoost vượt trội hoàn toàn về tính minh bạch (interpretability): thông qua Feature Importance
   và PDP, ta dễ dàng chứng minh được vai trò chi phối của chu kỳ 24h và tác động phi tuyến của nhiệt độ.
3. Ngược lại, LSTM có lợi thế không đòi hỏi kỹ nghệ đặc trưng thủ công và xử lý tự nhiên chuỗi đa biến 3D,
   nhưng tốn kém chi phí tính toán và yêu cầu tinh chỉnh siêu tham số phức tạp hơn.
========================================================================================
"""
    print(comp_text)


def run_q3():
    print("===== BẮT ĐẦU CÂU HỎI 3 & CÂU HỎI 6 (HOÀNG THẮNG) =====")
    df = load_data()
    scaled, scaler, (i1, i2) = fit_scale(df)
    df_flat, y, y_time, feature_names = build_features(scaled, L=L)

    train_idx = slice(0, i1 - L)
    val_idx = slice(i1 - L, i2 - L)

    # 1. Huấn luyện mô hình tối ưu đã xác định ở Mục 7 (max_depth=7, early stopping)
    print("Huấn luyện mô hình XGBoost tốt nhất để trích xuất đặc trưng...")
    model = xgb.XGBRegressor(
        max_depth=7,
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(df_flat.iloc[train_idx], y[train_idx], eval_set=[(df_flat.iloc[val_idx], y[val_idx])], verbose=False)

    # 2. Phân tích Feature Importance
    df_group, lag_gains = analyze_feature_importance(model, feature_names)

    # 3. Ablation Study
    df_ablation = run_ablation_study(df_flat, y, scaler, i1, i2)

    # 4. Vẽ PDP cho nhiệt độ
    plot_pdp_temperature(model, df_flat.iloc[train_idx], scaler, FIG / "q3_pdp_temp.png")

    # 5. Câu hỏi 6: So sánh ML vs DL
    print_q6_comparison()

    print("===== HOÀN THÀNH CÂU HỎI 3 & 6 =====\n")
    return df_group, df_ablation


if __name__ == "__main__":
    run_q3()
