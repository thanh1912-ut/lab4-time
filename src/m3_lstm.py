"""Mục 8: LSTM dự báo điện năng một giờ tới. Người làm: Hoàng Nam.

Chạy từ gốc repo: python src/m3_lstm.py
Mô hình chính dùng 24 giờ x 4 biến; xuất pred_lstm.csv.
Patience chọn trên validation (trung bình 3 seed), mô hình chính dùng seed 42.
"""
from __future__ import annotations

import argparse
import json
from lstm_common import PATIENCE_CANDIDATES, RESULTS_DIR, TUNE_SEEDS, prepare_data, tune_main
from utils import FIG, PROCESSED, ROOT, metrics
import matplotlib.pyplot as plt
import pandas as pd


def table(frame):
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def export_main(data, result, tuned):
    times = data["times_test"]
    truth = data["df"].loc[times, "load"].to_numpy()
    frame = pd.DataFrame({"Datetime": times, "y_true": truth, "y_pred": result["predictions"]["test"]})
    frame.to_csv(PROCESSED / "pred_lstm.csv", index=False)
    rows = []
    for name, lag in [("Naive", 1), ("Seasonal naive (24h)", 24)]:
        pred = data["df"]["load"].shift(lag).loc[times].to_numpy()
        rows.append({"Model": name, **metrics(truth, pred)})
    rows.append({"Model": "LSTM (L=24)", **result["metadata"]["split_metrics"]["test"]})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "m3_baselines.csv", index=False)
    pd.DataFrame([{"Split": s, **v} for s, v in result["metadata"]["split_metrics"].items()]).to_csv(
        RESULTS_DIR / "m3_split_metrics.csv", index=False)
    week = frame.iloc[:168]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(week["Datetime"], week["y_true"], label="Thực tế", linewidth=1.6)
    ax.plot(week["Datetime"], week["y_pred"], label="LSTM: dự báo 1 giờ", linewidth=1.2)
    ax.set(title="LSTM — tuần đầu tập test (168 giờ)", xlabel="Thời gian", ylabel="Load (MW)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "m3_lstm_1tuan.png", dpi=150)
    plt.close(fig)
    write_reports(data, result, comparison, tuned)
    print(f"Đã lưu pred_lstm.csv ({len(frame)} giờ).", flush=True)


def write_reports(data, result, comparison, tuned):
    meta, cfg = result["metadata"], result["metadata"]["config"]
    sel, tune_table = tuned["selection"], tuned["table"].copy()
    tune_table["train_seconds"] = tune_table["train_seconds"].round(1)
    chosen = tune_table[tune_table.patience == sel["patience"]]
    splits = pd.DataFrame([{"Tập": s, **v} for s, v in meta["splits"].items()])
    scores = pd.DataFrame([{"Tập": s, **v} for s, v in meta["split_metrics"].items()])
    main_rmse = meta["split_metrics"]["test"]["RMSE"]
    seasonal_rmse = float(comparison.iloc[1]["RMSE"])
    delta_pct = (main_rmse / seasonal_rmse - 1) * 100
    baseline_direction = "cao hơn" if delta_pct >= 0 else "thấp hơn"
    if delta_pct >= 0:
        baseline_note = ("Kết quả này cho thấy cần giữ baseline đơn giản trong bảng đánh giá; "
                         "kiến trúc phức tạp hơn không tự bảo đảm sai số thấp hơn.")
    else:
        baseline_note = ("LSTM tận dụng được cả lịch sử 24 giờ của 4 biến nên vượt baseline chỉ dùng "
                         "load cùng giờ hôm trước.")
    parts = ROOT / "docs" / "lab4" / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    report = f"""# Mục 8. Mô hình LSTM — Hoàng Nam

## Bài toán và dữ liệu

Dùng 24 giờ quan sát gần nhất của 4 biến `load`, `temperature`, `humidity`, `wind_speed` để dự báo load giờ tiếp theo. Dữ liệu chung có {len(data['df']):,} giờ. StandardScaler chỉ fit trên train. Mỗi cửa sổ được xếp vào tập theo thời điểm của y, nên đầu validation/test được dùng lịch sử trước ranh giới.

{table(splits)}

## Kiến trúc và huấn luyện

`Input(24,4) → LSTM(64, return_sequences=True) → LSTM(32) → Dense(1)`.

- Số tham số: **{meta['parameters']:,}**.
- Loss MSE trên dữ liệu chuẩn hóa; Adam learning rate 0,001; seed {cfg['seed']}.
- Tối đa {cfg['epochs']} epoch, batch {cfg['batch_size']}. **Xáo trộn thứ tự các cửa sổ train mỗi epoch**; validation và test giữ thứ tự thời gian.
  Mỗi cửa sổ đã chứa đủ 24 giờ quá khứ của chính nó nên xáo trộn không làm lộ tương lai; nó chỉ tránh việc mỗi epoch học từ 2012 đến hết 2014 theo đúng thứ tự,
  vì khi đó các batch cuối mỗi epoch luôn là mùa đông đầu 2014 và có thể kéo trọng số lệch về giai đoạn này.
- EarlyStopping theo `val_loss`, patience {cfg['patience']} (chọn trên validation, xem bên dưới), khôi phục trọng số tốt nhất.
- Thực chạy **{meta['epochs_run']} epoch**, chọn epoch **{meta['selected_epoch']}**; thời gian `fit`: **{meta['train_seconds']:.3f} giây**, thiết bị {', '.join(meta['devices'])}. Thời gian không gồm đọc dữ liệu, dự báo, xuất hình.
- Python {cfg['python']}, TensorFlow {cfg['tensorflow']}, numpy {cfg['numpy']}, pandas {cfg['pandas']}.

## Chọn patience trên validation

Thử patience {', '.join(map(str, PATIENCE_CANDIDATES))}, mỗi giá trị {len(TUNE_SEEDS)} seed ({', '.join(map(str, TUNE_SEEDS))}).
Chọn patience có RMSE validation trung bình thấp nhất: {', '.join(f"patience {k}: {v:.2f} MW" for k, v in sel['mean_validation_rmse'].items())}
→ chọn **patience {sel['patience']}**; mô hình chính là seed {sel['main_seed']}. Cột test chỉ để báo cáo, không dùng để chọn.

{table(tune_table)}

Với patience {sel['patience']}, test RMSE qua {len(chosen)} seed: **{sel['test_rmse_mean']:.2f} ± {sel['test_rmse_std']:.2f} MW**
(trung bình ± độ lệch chuẩn), cho biết mức dao động do khởi tạo ngẫu nhiên.

## Backtesting và kết quả

Walk-forward một bước với trọng số cố định: tại mỗi giờ test, đầu vào là 24 giờ **giá trị thật quá khứ**. Không refit trên test. Dự báo các cửa sổ theo batch cho cùng phép đánh giá vì trọng số cố định. Đây không phải dự báo đệ quy nhiều ngày từ một mốc duy nhất.

Đã đảo chuẩn hóa về MW trước khi tính chỉ số bằng `utils.metrics()`:

{table(scores)}

So với hai baseline trên đúng cùng các giờ test:

{table(comparison)}

RMSE LSTM **{baseline_direction} seasonal naive 24 giờ {abs(delta_pct):.2f}%**
({main_rmse:.2f} so với {seasonal_rmse:.2f} MW). {baseline_note}
Không thay đổi checkpoint hoặc cấu hình dựa trên kết quả test này.

![LSTM một tuần](../figures/m3_lstm_1tuan.png)

Đầu ra: `data/processed/pred_lstm.csv`, {len(data['times_test'])} dòng, ba cột `Datetime,y_true,y_pred`. Số liệu sinh từ mô hình huấn luyện trên dữ liệu thật.

## Giới hạn

Chỉ 3 seed nên độ lệch chuẩn còn thô. Chọn patience và checkpoint theo validation; test chỉ dùng báo cáo. Dữ liệu chung đã được nhóm nội suy/điền thiếu từ trước, có phương pháp dùng quan sát tương lai; thí nghiệm này không chứng minh toàn bộ quy trình dữ liệu thô vận hành trực tuyến. Kết quả áp dụng cho tập test và cách tạo dữ liệu đã nêu.
"""
    process = f"""# Quy trình mục 8 — Hoàng Nam

## Chạy lại

Từ gốc repo: `python src/m3_lstm.py`. Mặc định epochs=50, batch=64; chạy {len(PATIENCE_CANDIDATES) * len(TUNE_SEEDS)} lần train (patience × seed). Huấn luyện lại dù đã có cache: `python src/m3_lstm.py --force`. Tham số `--epochs` dùng thử cấu hình khác; báo cáo tự ghi số epoch thực chạy.

## Các bước và kết quả trung gian

1. `utils.load_data()` đọc CSV chung; kiểm tra giờ liên tục, duy nhất và giá trị hữu hạn.
2. `features.fit_scale()` fit scaler trên {data['split_indices'][0]} giờ train gốc. `features.make_windows()` tạo cửa sổ 24 giờ. Chia theo thời điểm y: train {len(data['y_train'])}, validation {len(data['y_val'])}, test {len(data['y_test'])}; không chia lại tỉ lệ trên len(X).
3. Deterministic ops; tf.data xáo trộn các cửa sổ train mỗi epoch theo seed, giới hạn thread CPU. Kiến trúc đúng đề, Adam 0,001, MSE. EarlyStopping chỉ theo validation.
4. `lstm_common.tune_main()`: patience {PATIENCE_CANDIDATES} × seed {TUNE_SEEDS}; chọn patience {sel['patience']} theo RMSE validation trung bình; lưu `lstm_tuning.csv`, `lstm_selection.json`.
5. Mô hình chính `{sel['main_name']}`: chọn epoch {meta['selected_epoch']} theo `val_loss`, sau {meta['epochs_run']} epoch thực chạy.
6. Dự báo theo batch; `features.inverse_load()` đưa về MW. `y_true` lấy nguyên từ CSV nguồn; tính RMSE/MAE/MAPE bằng `utils.metrics()`.
7. Lưu mô hình, dự báo ba tập, lịch sử, metadata trong `data/processed/lstm/`. Cache chỉ tái dùng khi dữ liệu, mã helper, cấu hình và runtime khớp.
8. Xuất `pred_lstm.csv`, hình 168 giờ đầu test, bảng baseline, báo cáo. Test bắt đầu {data['times_test'][0]}, kết thúc {data['times_test'][-1]}.

## File để kiểm tra

- Code: `src/m3_lstm.py`, `src/lstm_common.py`; chuẩn hóa và tạo cửa sổ qua `features.py`.
- Mô hình: `data/processed/lstm/models/{sel['main_name']}.keras`.
- Lịch sử: `data/processed/lstm/{sel['main_name']}_history.csv`.
- Cấu hình, thời gian, hash, số tham số và sai số: `data/processed/lstm/{sel['main_name']}_metadata.json`.
- Bảng chọn patience: `data/processed/lstm/lstm_tuning.csv`, `lstm_selection.json`.
- Hash SHA-256 dữ liệu: `{data['data_sha256']}`.
- Tài liệu EarlyStopping: https://keras.io/api/callbacks/early_stopping/ .

Đường dẫn ảnh tương đối từ `docs/lab4/parts/` là `../figures/`.
"""
    (parts / "m3_report.md").write_text(report, encoding="utf-8")
    (parts / "m3_process.md").write_text(process, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    # main.py truyền nguyên argv khi gọi từng phần bằng runpy.
    parser.add_argument("--rebuild-data", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    data = prepare_data()
    tuned = tune_main(data, epochs=args.epochs, batch_size=args.batch_size, force=args.force)
    result = tuned["main"]
    export_main(data, result, tuned)
    print(json.dumps(result["metadata"]["split_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
