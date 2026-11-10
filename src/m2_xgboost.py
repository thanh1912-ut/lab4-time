"""Mục 7: Mô hình XGBoost.
Người làm: Hoàng Thắng.
Chi tiết: docs/lab4/phan-cong.md, mục 5.5.
"""
import sys
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from utils import load_data, COLS, L, SEED, PROCESSED, FIG, metrics
from features import fit_scale, make_windows, add_rolling, inverse_load, split_index

DPI = 150


def build_features(scaled, L=L):
    """Xây dựng 99 đặc trưng phẳng cho XGBoost:
    - 96 đặc trưng trễ (24 độ trễ x 4 biến: load, temperature, humidity, wind_speed)
    - 3 đặc trưng thống kê cuộn (rolling) tính đến thời điểm t
    y[k] là load(t+1) tương ứng.
    """
    X_win, y, y_time = make_windows(scaled, L=L)

    feature_names = []
    X_lags = []
    for c, col in enumerate(COLS):
        for lag in range(1, L + 1):
            feature_names.append(f"{col}_lag{lag}")
            X_lags.append(X_win[:, L - lag, c])

    roll = add_rolling(scaled)
    roll_cols = list(roll.columns)
    feature_names.extend(roll_cols)

    # Lấy các giá trị rolling tại thời điểm t (ngay trước t+1)
    roll_arr = roll.iloc[L - 1 : -1].bfill().to_numpy()
    X_flat = np.column_stack([np.column_stack(X_lags), roll_arr])

    df_flat = pd.DataFrame(X_flat, columns=feature_names, index=y_time)
    return df_flat, y, y_time, feature_names


def tune_xgboost(df_flat, y, scaler, i1, i2):
    """Thử nghiệm các giá trị max_depth trong {3, 5, 7, 9} với Early Stopping trên validation."""
    train_idx = slice(0, i1 - L)
    val_idx = slice(i1 - L, i2 - L)

    X_train, y_train = df_flat.iloc[train_idx], y[train_idx]
    X_val, y_val = df_flat.iloc[val_idx], y[val_idx]
    y_val_orig = inverse_load(y_val, scaler)

    depth_candidates = [3, 5, 7, 9]
    tuning_results = []
    best_depth = None
    best_rmse = float("inf")
    best_model = None
    best_evals_result = None

    print("-" * 65)
    print(f"{'max_depth':<10}{'Best Iter':<12}{'Val RMSE':<12}{'Val MAE':<12}{'Val MAPE (%)':<12}")
    print("-" * 65)

    for depth in depth_candidates:
        evals_result = {}
        model = xgb.XGBRegressor(
            max_depth=depth,
            n_estimators=2000,
            learning_rate=0.05,
            early_stopping_rounds=50,
            random_state=SEED,
            n_jobs=-1,
            eval_metric="rmse",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        evals_result = model.evals_result()

        preds_val_scaled = model.predict(X_val)
        preds_val_orig = inverse_load(preds_val_scaled, scaler)
        score = metrics(y_val_orig, preds_val_orig)

        tuning_results.append({
            "max_depth": depth,
            "best_iteration": model.best_iteration,
            "RMSE": score["RMSE"],
            "MAE": score["MAE"],
            "MAPE(%)": score["MAPE(%)"],
        })

        print(f"{depth:<10}{model.best_iteration:<12}{score['RMSE']:<12.2f}{score['MAE']:<12.2f}{score['MAPE(%)']:<12.2f}")

        if score["RMSE"] < best_rmse:
            best_rmse = score["RMSE"]
            best_depth = depth
            best_model = model
            best_evals_result = evals_result

    print("-" * 65)
    print(f"=> Mô hình tối ưu: max_depth={best_depth}, best_iteration={best_model.best_iteration}, Val RMSE={best_rmse:.2f} MW")

    return best_model, best_depth, best_evals_result, pd.DataFrame(tuning_results)


def plot_learning_curve(evals_result, best_iteration, save_path):
    """Vẽ đường cong học (Learning Curve) RMSE của tập Train và Validation theo số cây."""
    train_rmse = evals_result["validation_0"]["rmse"]
    val_rmse = evals_result["validation_1"]["rmse"]
    rounds = range(len(train_rmse))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    ax.plot(rounds, train_rmse, label="Train RMSE (chuẩn hóa)", color="#1f77b4", linewidth=1.8)
    ax.plot(rounds, val_rmse, label="Val RMSE (chuẩn hóa)", color="#ff7f0e", linewidth=1.8)
    ax.axvline(best_iteration, color="red", linestyle="--", linewidth=1.5, label=f"Best Iteration ({best_iteration})")

    ax.set_title("Đường cong huấn luyện XGBoost (Learning Curve)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Số cây tăng cường (Boosting Rounds)", fontsize=11)
    ax.set_ylabel("Hàm mất mát RMSE (thang chuẩn hóa)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu biểu đồ học: {save_path.name}")


def walk_forward_test(df_flat, y, scaler, i2, best_depth, best_trees, refit_every=168):
    """Walk-forward validation trên toàn bộ tập test:
    - Dự báo từng giờ t+1 bằng cách lấy đúng các giá trị quan sát thật tới giờ t.
    - Tái huấn luyện (refit) mô hình sau mỗi 168 giờ (1 tuần) trên toàn bộ lịch sử đã qua.
    """
    test_start = i2 - L
    n_test = len(y) - test_start
    test_timestamps = df_flat.index[test_start:]

    print(f"\nBắt đầu Walk-Forward Validation trên tập Test ({n_test} giờ, refit mỗi {refit_every} giờ)...")
    t0 = time.time()
    preds_scaled = []
    current_model = None

    for step in range(n_test):
        idx = test_start + step
        if step % refit_every == 0:
            X_hist = df_flat.iloc[:idx]
            y_hist = y[:idx]
            current_model = xgb.XGBRegressor(
                max_depth=best_depth,
                n_estimators=best_trees,
                learning_rate=0.05,
                random_state=SEED,
                n_jobs=-1,
            )
            current_model.fit(X_hist, y_hist, verbose=False)
            if step % (refit_every * 4) == 0 or step == 0:
                elapsed = time.time() - t0
                pct = (step / n_test) * 100
                print(f"  [Walk-Forward] Giờ {step:>4}/{n_test} ({pct:>5.1f}%) - Đã chạy {elapsed:>4.1f}s")

        p = current_model.predict(df_flat.iloc[idx : idx + 1])[0]
        preds_scaled.append(p)

    total_time = time.time() - t0
    print(f"Hoàn thành Walk-Forward trong {total_time:.1f}s!")

    # Chuyển đổi về thang đo MW gốc
    preds_orig = inverse_load(preds_scaled, scaler)
    y_test_orig = inverse_load(y[test_start:], scaler)

    pred_df = pd.DataFrame({
        "Datetime": test_timestamps,
        "y_true": np.round(y_test_orig, 3),
        "y_pred": np.round(preds_orig, 3),
    })
    pred_path = PROCESSED / "pred_xgb.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Đã lưu kết quả dự đoán: {pred_path}")

    test_score = metrics(y_test_orig, preds_orig)
    print("\n" + "=" * 45)
    print(f"KẾT QUẢ XGBOOST TRÊN TẬP TEST (Thang MW gốc):")
    print(f"  - RMSE:     {test_score['RMSE']:.2f} MW")
    print(f"  - MAE:      {test_score['MAE']:.2f} MW")
    print(f"  - MAPE:     {test_score['MAPE(%)']:.2f} %")
    print(f"  - Thời gian: {total_time:.1f} s")
    print("=" * 45)

    return pred_df, test_score, total_time


def fixed_test(best_model, df_flat, y, scaler, i2):
    """Bản dùng cho so sánh công bằng (mục 10): giữ nguyên mô hình học trên train
    (early stopping theo validation), không refit trên test. Mỗi giờ test vẫn dùng
    các lag thật tới giờ t, giống walk-forward một bước."""
    test_start = i2 - L
    preds_orig = inverse_load(best_model.predict(df_flat.iloc[test_start:]), scaler)
    y_test_orig = inverse_load(y[test_start:], scaler)
    pred_df = pd.DataFrame({
        "Datetime": df_flat.index[test_start:],
        "y_true": np.round(y_test_orig, 3),
        "y_pred": np.round(preds_orig, 3),
    })
    pred_df.to_csv(PROCESSED / "pred_xgb_fixed.csv", index=False)
    score = metrics(y_test_orig, preds_orig)
    print(f"Bản công bằng (học trên train, không refit): {score}")
    return pred_df, score


def plot_sample_week(pred_df, save_path):
    """Vẽ so sánh thực tế vs dự đoán trong 1 tuần (168 giờ) tiêu biểu của tập test."""
    # Chọn tuần từ 04/08/2014 đến 10/08/2014 (cao điểm hè)
    pred_df_idx = pred_df.copy()
    pred_df_idx["Datetime"] = pd.to_datetime(pred_df_idx["Datetime"])
    pred_df_idx.set_index("Datetime", inplace=True)

    week_start = pd.Timestamp("2014-08-04 00:00:00")
    week_end = week_start + pd.Timedelta(hours=167)
    sample_week = pred_df_idx.loc[week_start:week_end]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI)
    ax.plot(sample_week.index, sample_week["y_true"], label="Thực tế (Ground Truth)", color="#2ca02c", linewidth=2.0)
    ax.plot(sample_week.index, sample_week["y_pred"], label="XGBoost Dự báo", color="#d62728", linestyle="--", linewidth=1.8)

    ax.set_title("Dự báo Phụ tải Điện năng 1 Tuần (XGBoost Walk-Forward)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Thời gian (Giờ)", fontsize=11)
    ax.set_ylabel("Công suất tiêu thụ (MW)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu biểu đồ 1 tuần: {save_path.name}")


def run_m2():
    print("===== BẮT ĐẦU MỤC 7: MÔ HÌNH XGBOOST (HOÀNG THẮNG) =====")
    df = load_data()
    scaled, scaler, (i1, i2) = fit_scale(df)
    df_flat, y, y_time, feature_names = build_features(scaled, L=L)

    print(f"Bộ dữ liệu sau trích xuất đặc trưng: {df_flat.shape[0]} mẫu, {df_flat.shape[1]} cột")
    print(f"Tập Train: {i1 - L} mẫu | Validation: {i2 - i1} mẫu | Test: {len(y) - (i2 - L)} mẫu")

    # 1. Tuning max_depth
    best_model, best_depth, evals_result, df_tune = tune_xgboost(df_flat, y, scaler, i1, i2)

    # 2. Vẽ learning curve
    plot_learning_curve(evals_result, best_model.best_iteration, FIG / "m2_learning_curve.png")

    # 3. Walk-forward trên test
    pred_df, test_score, total_time = walk_forward_test(
        df_flat, y, scaler, i2,
        best_depth=best_depth,
        best_trees=best_model.best_iteration,
        refit_every=168,
    )

    # 4. Bản so sánh công bằng: không refit (cùng luật với VAR/LSTM ở mục 10)
    fixed_test(best_model, df_flat, y, scaler, i2)

    # 5. Vẽ 1 tuần tiêu biểu
    plot_sample_week(pred_df, FIG / "m2_xgb_1tuan.png")

    print("===== HOÀN THÀNH MỤC 7 =====\n")
    return best_model, df_tune, test_score, total_time


if __name__ == "__main__":
    run_m2()
