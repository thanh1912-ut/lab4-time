"""Câu 4: chu kỳ; Câu 5: overfitting của LSTM. Người làm: Hoàng Nam.

Chạy: python src/q4_q5_lstm.py. Số liệu, hình và Markdown lấy từ kết quả thật.
Mô hình chính luôn dùng cửa sổ 24 giờ, 4 biến; biến thể lịch là thí nghiệm thêm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lstm_common import RESULTS_DIR, prepare_data, save_json, train_experiment, tune_main

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import inverse_load
from utils import COLS, FIG, SEED, metrics

PARTS = Path(__file__).resolve().parents[1] / "docs" / "lab4" / "parts"
DAY_NAMES = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]


def markdown_table(frame):
    """Bảng Markdown không cần thư viện tabulate."""
    def value(x):
        return f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x).replace("|", "\\|")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(value(x) for x in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def truth(data, split="test"):
    # Giữ nguyên độ chính xác CSV nguồn, giống bảng kết quả mô hình chính.
    return data["df"].loc[data[f"times_{split}"], "load"].to_numpy(dtype=float)


def grouped_errors(times, y_true, predictions, group):
    """Sai số theo giờ hoặc thứ, giữ cùng các quan sát cho mọi cấu hình."""
    index = pd.DatetimeIndex(times)
    key = index.hour if group == "hour" else index.dayofweek
    rows = []
    for name, y_pred in predictions.items():
        for number in sorted(np.unique(key)):
            mask = key == number
            rows.append({group: int(number), "Model": name, "n": int(mask.sum()),
                         **metrics(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])})
    return pd.DataFrame(rows)


def plot_groups(frame, group, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for name, part in frame.groupby("Model", sort=False):
        for ax, metric in zip(axes, ["RMSE", "MAE"]):
            ax.plot(part[group], part[metric], marker="o", linewidth=1.8, label=name)
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        ax.set_ylabel(f"{metric} (MW)")
        ax.set_xlabel("Giờ trong ngày" if group == "hour" else "Thứ trong tuần")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        if group == "weekday":
            ax.set_xticks(range(7), DAY_NAMES, rotation=25)
        else:
            ax.set_xticks(range(0, 24, 3))
    fig.suptitle("Câu 4: sai số dự báo một giờ trên cùng tập test")
    save_figure(fig, filename)


def recursive_diagnostic(base, data, hours=48):
    """Đệ quy load với thời tiết tương lai thực tế, một giả định chẩn đoán."""
    df = data["df"]
    start = df.index.get_loc(data["times_test"][0])
    end = min(start + hours, len(df))
    history = np.asarray(data["scaler"].transform(df[COLS]), dtype=np.float32)
    predicted = []
    window = int(data.get("window", 24))
    for pos in range(start, end):
        x = history[pos - window:pos][None, ...]
        z = float(np.asarray(base["model"](x, training=False)).reshape(-1)[0])
        predicted.append(z)
        history[pos, 0] = z
    out = pd.DataFrame({
        "Datetime": df.index[start:end], "y_true": df["load"].iloc[start:end].to_numpy(),
        "y_pred_recursive": inverse_load(predicted, data["scaler"]),
        "y_pred_one_step": np.asarray(base["predictions"]["test"])[:end - start],
    })
    out.to_csv(RESULTS_DIR / "q4_recursive_48.csv", index=False)
    score = metrics(out.y_true, out.y_pred_recursive)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(out.Datetime, out.y_true, color="#172554", label="Thực tế", linewidth=2)
    ax.plot(out.Datetime, out.y_pred_one_step, label="Một bước, cập nhật load thật", alpha=.8)
    ax.plot(out.Datetime, out.y_pred_recursive, label="Đệ quy, dùng thời tiết thực tế", linestyle="--")
    ax.set(title="Câu 4: chẩn đoán dự báo đệ quy 48 giờ", xlabel="Thời gian", ylabel="Load (MW)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    save_figure(fig, "q4_48h.png")
    return {"hours": len(out), "start": str(out.Datetime.iloc[0]),
            "end": str(out.Datetime.iloc[-1]), **score}


def analyse_q4(base, calendar, data, recursive=True):
    times = pd.DatetimeIndex(data["times_test"])
    true = truth(data)
    predictions = {
        "LSTM 24h": np.asarray(base["predictions"]["test"]).reshape(-1),
        "LSTM 24h + lịch": np.asarray(calendar["predictions"]["test"]).reshape(-1),
        "Seasonal naive 24h": data["df"]["load"].shift(24).reindex(times).to_numpy(),
    }
    if not all(len(v) == len(times) and np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("Dự báo câu 4 phải đủ cùng mốc test và không có giá trị thiếu.")
    summary = pd.DataFrame([{"Model": name, **metrics(true, v)} for name, v in predictions.items()])
    summary.to_csv(RESULTS_DIR / "q4_comparison.csv", index=False)
    by_day = grouped_errors(times, true, predictions, "weekday")
    by_hour = grouped_errors(times, true, predictions, "hour")
    by_day.to_csv(RESULTS_DIR / "q4_errors_weekday.csv", index=False)
    by_hour.to_csv(RESULTS_DIR / "q4_errors_hour.csv", index=False)
    plot_groups(by_day, "weekday", "q4_loi_theo_thu.png")
    plot_groups(by_hour, "hour", "q4_loi_theo_gio.png")
    base_rmse, cal_rmse = float(summary.iloc[0].RMSE), float(summary.iloc[1].RMSE)
    change = (base_rmse - cal_rmse) / base_rmse * 100 if base_rmse else 0.0
    direction = "giảm" if change >= 0 else "tăng"
    calendar_vs_baseline = []
    for metric in ["RMSE", "MAE", "MAPE(%)"]:
        cal_value = float(summary.iloc[1][metric])
        baseline_value = float(summary.iloc[2][metric])
        relation = "thấp hơn" if cal_value < baseline_value else "cao hơn hoặc bằng"
        calendar_vs_baseline.append(
            f"{metric} {relation} ({cal_value:.2f} so với {baseline_value:.2f})"
        )
    week = by_day[by_day.Model == "LSTM 24h"]
    worst, best = week.loc[week.RMSE.idxmax()], week.loc[week.RMSE.idxmin()]
    recursion = recursive_diagnostic(base, data) if recursive else None
    extra = ""
    if recursion:
        extra = f"""
### Thí nghiệm thêm: đệ quy 48 giờ

Từ {recursion['start']} đến {recursion['end']}, load dự báo được đưa trở lại cửa sổ,
không cập nhật load thực tế sau thời điểm bắt đầu. RMSE = **{recursion['RMSE']:.2f} MW**,
MAE = **{recursion['MAE']:.2f} MW**, MAPE = **{recursion['MAPE(%)']:.2f}%**.
Thời tiết tương lai dùng giá trị quan sát thực tế: đây là chẩn đoán có điều kiện với thời tiết đã biết,
không phải đánh giá vận hành khi chưa có dự báo thời tiết. Mô hình chính vẫn dự báo một bước.

![Dự báo đệ quy 48 giờ](../figures/q4_48h.png)
"""
    report = f"""# Câu 4: LSTM học được chu kỳ nào? — Hoàng Nam

## Thiết kế thí nghiệm

Mô hình chính dùng 24 giờ gần nhất của load, nhiệt độ, độ ẩm và tốc độ gió; kiến trúc
LSTM(64) → LSTM(32) → Dense(1). Đánh giá dự báo một bước trên **{len(times):,} giờ test**
từ {times[0]} đến {times[-1]}, dùng các quan sát thật trong cửa sổ quá khứ.
Biến thể giữ cửa sổ 24 giờ, thêm bốn đặc trưng sin/cos cho giờ trong ngày và thứ trong tuần
tại các mốc đầu vào. Số chiều đầu vào tăng từ 4 lên 8, nên số tham số cũng thay đổi.
Mô hình chính có **{base['metadata']['parameters']:,} tham số**, thời gian huấn luyện
**{base['metadata']['train_seconds']:.2f} giây**. Biến thể lịch có
**{calendar['metadata']['parameters']:,} tham số**, thời gian huấn luyện
**{calendar['metadata']['train_seconds']:.2f} giây**. Thời gian là số đo của từng lần chạy,
không bao gồm khâu chuẩn bị dữ liệu hay dự báo và có thể thay đổi theo tải máy.
Mô hình chính và biến thể dùng cùng cách train: xáo trộn cửa sổ train mỗi epoch, seed {SEED},
EarlyStopping patience {base['metadata']['config']['patience']} (patience chọn ở mục 8 bằng validation).
Tập test chỉ dùng để báo cáo, không dùng để chọn lại mô hình chính.

## Kết quả trên cùng tập test

{markdown_table(summary)}

Đơn vị RMSE và MAE là MW; MAPE là %. Với seed {SEED}, thêm đặc trưng lịch làm RMSE
{direction} **{abs(change):.2f}%** so với mô hình chính. Mỗi cấu hình chạy một seed;
chưa đủ để kết luận ưu thế ổn định. Biến thể lịch là thí nghiệm bổ sung;
`pred_lstm.csv` vẫn chứa dự báo mô hình chính 24 giờ, 4 biến.

Đối chiếu biến thể lịch với seasonal naive 24 giờ: {'; '.join(calendar_vs_baseline)}.
Vì các chỉ số đo các mặt khác nhau của sai số, cần đọc cả RMSE, MAE và MAPE
trước khi kết luận một cấu hình tốt hơn tổng thể.

## Sai số theo chu kỳ

Trong mô hình chính, {DAY_NAMES[int(worst.weekday)]} có RMSE cao nhất (**{worst.RMSE:.2f} MW**)
và {DAY_NAMES[int(best.weekday)]} thấp nhất (**{best.RMSE:.2f} MW**). Sai số theo thứ còn chịu ảnh hưởng
của mùa, mức tải và sự kiện trong tập test; khác biệt này không tự chứng minh mạng đã
hoặc chưa học được chu kỳ tuần.

![Sai số theo thứ](../figures/q4_loi_theo_thu.png)

![Sai số theo giờ](../figures/q4_loi_theo_gio.png)

## Trả lời câu 4

Cửa sổ 24 giờ cung cấp thông tin hình dạng và mức tải trong một ngày trước giờ dự báo.
Mạng có thể khai thác quan hệ giữa các giờ và các biến trong cửa sổ đó; sai số test đo mức hữu ích
của thông tin này. Cửa sổ không chứa trực tiếp quan sát cùng giờ của tuần trước (lag 168),
nên thí nghiệm lịch bổ sung tín hiệu vị trí trong ngày và tuần để kiểm tra phần thông tin còn thiếu.
RMSE {direction} {abs(change):.2f}% trong lần chạy này, nhưng số đầu vào và số tham số là yếu tố cùng thay đổi.
Kết quả hỗ trợ nhận xét về lợi ích đặc trưng lịch trong cấu hình đã thử,
không phải bằng chứng tách biệt rằng LSTM đã học trọn chu kỳ tuần.
{extra}
"""
    process = f"""# Quy trình Câu 4: Chu kỳ LSTM — Hoàng Nam

1. Dùng lại mô hình chính, lịch sử, dự báo thật và metadata trong `data/processed/lstm/`.
2. Gọi `prepare_data(calendar=True)` thêm sin/cos giờ-trong-ngày và thứ-trong-tuần tại các mốc quá khứ.
3. Giữ cùng mốc mục tiêu và cùng train/validation/test; scaler bốn biến gốc chỉ fit trên train.
4. Huấn luyện `lstm_calendar` cùng số epoch tối đa, batch, seed, loss, optimizer và EarlyStopping của mô hình chính.
5. Đổi dự báo về MW, đối chiếu đúng {len(times):,} thời điểm test; tính RMSE/MAE/MAPE, so thêm seasonal naive dùng load cùng giờ hôm trước.
6. Nhóm sai số theo giờ/thứ; từng nhóm lưu số quan sát `n` để diễn giải phạm vi.
7. Lưu `q4_comparison.csv`, `q4_errors_hour.csv`, `q4_errors_weekday.csv`, hình `q4_*.png`.
8. {'Chạy thêm 48 giờ đệ quy đầu tập test; đưa load dự báo vào cửa sổ, giữ thời tiết tương lai thực tế và ghi rõ giả định.' if recursion else 'Bỏ thí nghiệm đệ quy theo tùy chọn --skip-recursive.'}

## Cách chạy

```bash
python src/m3_lstm.py
python src/q4_q5_lstm.py
```

`--force` huấn luyện lại; `--epochs` và `--batch-size` điều chỉnh cấu hình được lưu trong metadata.
Mặc định 50 epoch tối đa, batch 64, seed {SEED}. Cấu hình chốt trước khi xem test.
Đặc trưng lịch tính từ timestamp đã biết, không lấy load tương lai.
Một seed chưa cho biết độ ổn định; chưa kiểm định thống kê hay tách ảnh hưởng số tham số.
"""
    (PARTS / "q4_report.md").write_text(report.rstrip() + "\n", encoding="utf-8")
    (PARTS / "q4_process.md").write_text(process.rstrip() + "\n", encoding="utf-8")
    save_json(RESULTS_DIR / "q4_analysis.json", {"calendar_rmse_change_pct": change, "recursive": recursion})
    return summary


def analyse_q5(experiments):
    labels = {"lstm_main": "LSTM chính + EarlyStopping",
              "lstm_no_early_stopping": "LSTM đủ epoch, không EarlyStopping",
              "lstm_dropout": "LSTM + Dropout(0.2) + EarlyStopping"}
    rows, gap_rows = [], []
    for name, experiment in experiments.items():
        meta = experiment["metadata"]
        scores = meta["split_metrics"]
        for split in ["train", "validation", "test"]:
            rows.append({"Model": labels[name], "Split": split, **scores[split]})
        gap_rows.append({"Model": labels[name], "Epochs": meta["epochs_run"],
                         "Best val epoch": meta["best_epoch"],
                         "Train RMSE": scores["train"]["RMSE"],
                         "Validation RMSE": scores["validation"]["RMSE"],
                         "Test RMSE": scores["test"]["RMSE"], "Training seconds": meta["train_seconds"]})
    scores, summary = pd.DataFrame(rows), pd.DataFrame(gap_rows)
    scores.to_csv(RESULTS_DIR / "q5_split_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "q5_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, experiment) in zip(axes, experiments.items()):
        history = experiment["history"]
        epoch = np.arange(1, len(history["loss"]) + 1)
        ax.plot(epoch, history["loss"], label="Train", color="#2563eb")
        ax.plot(epoch, history["val_loss"], label="Validation", color="#ea580c")
        ax.axvline(experiment["metadata"]["best_epoch"], linestyle="--", color="#64748b", label="Val tốt nhất")
        ax.set(xlabel="Epoch", ylabel="MSE trên thang chuẩn hóa", title=labels[name].replace(" + ", "\n"))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Câu 5: loss train/validation — chọn trọng số bằng validation")
    save_figure(fig, "q5_loss.png")
    base = experiments["lstm_main"]
    history = base["history"]
    best_idx = int(base["metadata"]["best_epoch"]) - 1
    val_best, val_last = float(history["val_loss"][best_idx]), float(history["val_loss"][-1])
    main_scores = base["metadata"]["split_metrics"]
    train_rmse, val_rmse = float(main_scores["train"]["RMSE"]), float(main_scores["validation"]["RMSE"])
    difference = val_rmse - train_rmse
    noes_history = experiments["lstm_no_early_stopping"]["history"]
    noes_best_idx = int(np.argmin(noes_history["val_loss"]))
    noes_best = float(noes_history["val_loss"][noes_best_idx])
    noes_last = float(noes_history["val_loss"][-1])
    noes_train_at_best = float(noes_history["loss"][noes_best_idx])
    noes_train_last = float(noes_history["loss"][-1])
    noes_last_epoch = len(noes_history["loss"])
    epochs_after_best = noes_last_epoch - noes_best_idx - 1
    train_delta = noes_train_last - noes_train_at_best
    val_delta = noes_last - noes_best
    train_drop_pct = -train_delta / max(abs(noes_train_at_best), 1e-12) * 100
    val_rise_pct = val_delta / max(abs(noes_best), 1e-12) * 100
    tail_size = min(5, max(epochs_after_best, 1))
    late_train_mean = float(np.mean(noes_history["loss"][-tail_size:]))
    late_val_mean = float(np.mean(noes_history["val_loss"][-tail_size:]))
    # Chẩn đoán cần train còn cải thiện trong khi validation xấu đi, không chỉ
    # so val cuối với min(val), một bất đẳng thức luôn đúng theo định nghĩa.
    sustained_divergence = (
        epochs_after_best >= 5 and train_drop_pct > 1 and val_rise_pct > 1
        and late_train_mean < noes_train_at_best * .99
        and late_val_mean > noes_best * 1.01
    )
    if sustained_divergence:
        assessment = (
            "**Có dấu hiệu overfitting khi kéo dài huấn luyện ở biến thể không EarlyStopping.** "
            "Sau epoch có validation loss tốt nhất, train loss tiếp tục giảm nhưng validation loss tăng; "
            "trung bình các epoch cuối cũng cho cùng hướng phân kỳ. Mô hình chính dùng checkpoint "
            "tốt nhất theo validation để hạn chế tác động này; đây không phải khẳng định mô hình chính "
            "đã loại bỏ hoàn toàn overfitting."
        )
    elif train_delta < 0 and val_delta > 0:
        assessment = (
            "**Có xu hướng train giảm trong khi validation tăng, nhưng bằng chứng overfitting còn yếu.** "
            "Mức thay đổi nhỏ, khoảng theo dõi sau epoch tốt nhất ngắn hoặc trung bình các epoch cuối "
            "chưa cho cùng hướng phân kỳ rõ. Cần thêm lần chạy và kiểm tra đường loss trước khi kết luận mạnh."
        )
    else:
        assessment = (
            "**Chưa thấy bằng chứng rõ của overfitting từ diễn biến train/validation sau epoch tốt nhất.** "
            "Không quan sát đồng thời train loss tiếp tục giảm và validation loss tăng ở hai mốc so sánh. "
            "Việc validation cuối lớn hơn giá trị nhỏ nhất của chính nó không đủ để kết luận overfitting."
        )
    late_improvement = ""
    if noes_best_idx + 1 > int(base["metadata"]["epochs_run"]):
        late_improvement = (
            f"Ở lần chạy đủ epoch, validation tốt nhất xuất hiện tại epoch {noes_best_idx + 1}, "
            f"sau thời điểm mô hình chính đã dừng (epoch {base['metadata']['epochs_run']}). "
            f"Validation có thể cải thiện trở lại sau một đoạn chững; patience={base['metadata']['config']['patience']} của mô hình chính "
            "đã dừng trước lần cải thiện muộn này. EarlyStopping không bảo đảm tìm được checkpoint "
            "tốt nhất của toàn bộ lịch huấn luyện dài hơn. Đây là quan sát trên validation; "
            "mô hình chính vẫn giữ đúng cấu hình đã định trước."
        )
    trajectory = pd.DataFrame([
        {"Mốc không EarlyStopping": "Validation tốt nhất", "Epoch": noes_best_idx + 1,
         "Train MSE": noes_train_at_best, "Validation MSE": noes_best},
        {"Mốc không EarlyStopping": "Epoch cuối", "Epoch": noes_last_epoch,
         "Train MSE": noes_train_last, "Validation MSE": noes_last},
    ])
    trajectory.to_csv(RESULTS_DIR / "q5_training_trajectory.csv", index=False)
    save_json(RESULTS_DIR / "q5_overfitting_diagnostic.json", {
        "best_validation_epoch": noes_best_idx + 1, "last_epoch": noes_last_epoch,
        "epochs_after_best": epochs_after_best, "train_mse_at_best_validation": noes_train_at_best,
        "last_train_mse": noes_train_last, "best_validation_mse": noes_best,
        "last_validation_mse": noes_last, "train_mse_delta": train_delta,
        "validation_mse_delta": val_delta, "train_mse_drop_pct": train_drop_pct,
        "validation_mse_rise_pct": val_rise_pct, "tail_epochs": tail_size,
        "tail_mean_train_mse": late_train_mean, "tail_mean_validation_mse": late_val_mean,
        "sustained_divergence": bool(sustained_divergence),
        "diagnostic_rule": "at least 5 epochs after best validation; train falls >1% and validation rises >1%; final up-to-5-epoch means agree",
    })
    gap_text = "cao hơn" if difference >= 0 else "thấp hơn"
    if epochs_after_best == 0:
        # Validation tốt nhất ở epoch cuối: không có đoạn nào sau mốc tốt nhất để so.
        trajectory_text = (
            f"Ở biến thể đủ epoch, validation loss thấp nhất nằm ở **epoch cuối ({noes_last_epoch})** "
            f"(train MSE **{noes_train_last:.6f}**, validation MSE **{noes_last:.6f}**, thang chuẩn hóa). "
            "Validation vẫn còn giảm tới khi dừng, tức mô hình **chưa hội tụ** trong số epoch đã đặt: "
            "đây là dấu hiệu học chưa đủ (underfitting), không phải overfitting."
        )
    else:
        trajectory_text = f"""Ở biến thể đủ epoch, từ epoch **{noes_best_idx + 1}** (validation tốt nhất) tới epoch
**{noes_last_epoch}**, train MSE đổi từ **{noes_train_at_best:.6f}** thành **{noes_train_last:.6f}**
(chênh lệch cuối − đầu **{train_delta:+.6f}**, mức giảm **{train_drop_pct:+.2f}%**);
validation MSE từ **{noes_best:.6f}** thành **{noes_last:.6f}**
(chênh lệch **{val_delta:+.6f}**, mức tăng **{val_rise_pct:+.2f}%**).
Trung bình {tail_size} epoch cuối: train MSE **{late_train_mean:.6f}**,
validation MSE **{late_val_mean:.6f}**. Các đại lượng này đều ở thang chuẩn hóa.
Chẩn đoán phân kỳ rõ dùng quy tắc mô tả: có ít nhất 5 epoch sau mốc tốt nhất,
train giảm trên 1%, validation tăng trên 1%, và trung bình tối đa 5 epoch cuối cùng cùng chiều.
Đây là quy tắc đọc kết quả của thí nghiệm, không phải kiểm định thống kê."""
    report = f"""# Câu 5: LSTM có bị overfitting không? — Hoàng Nam

## Thí nghiệm kiểm tra

So sánh ba cấu hình định trước, cùng seed {SEED}, cùng cửa sổ 24 giờ và cùng dữ liệu:
mô hình chính có EarlyStopping, mô hình đủ epoch không EarlyStopping,
và mô hình thêm Dropout(0.2) sau mỗi lớp LSTM với EarlyStopping. Cấu hình chính giữ nguyên;
không chọn biến thể theo sai số test.

## Kết quả

{markdown_table(summary)}

RMSE bằng MW sau khi đổi về thang gốc. Với EarlyStopping, trọng số lưu là epoch có
validation loss thấp nhất. Với biến thể không EarlyStopping, đánh giá trọng số **epoch cuối**,
dù cột `Best val epoch` có thể nằm sớm hơn. Thời gian đo chỉ gồm huấn luyện.

![Loss train và validation](../figures/q5_loss.png)

{markdown_table(scores)}

## Trả lời câu 5

{assessment}

{late_improvement}

Mô hình chính chạy {base['metadata']['epochs_run']} epoch và dùng trọng số ở epoch
{base['metadata']['best_epoch']}. Validation MSE tốt nhất {val_best:.6f},
so với {val_last:.6f} ở epoch cuối đã chạy. RMSE validation {gap_text} RMSE train
**{abs(difference):.2f} MW** ({val_rmse:.2f} so với {train_rmse:.2f} MW).
Chênh lệch cần đọc cùng đường loss: các tập là các giai đoạn thời gian khác nhau,
nên thay đổi mùa và phân phối cũng có thể tạo chênh lệch, không chỉ overfitting.

{trajectory_text}

EarlyStopping hạn chế tiếp tục học khi validation không cải thiện; Dropout là thử nghiệm
regularization bổ sung. Biến thể chạy một seed (mô hình chính có thêm 2 seed ở mục 8),
chưa đủ để khẳng định mức cải thiện ổn định.
"""
    process = f"""# Quy trình Câu 5: Overfitting — Hoàng Nam

1. Giữ nguyên cách chia thời gian và scaler fit trên train; dùng cùng mẫu cho ba cấu hình.
2. Mạng chính LSTM(64) → LSTM(32) → Dense(1), Adam(0.001), MSE, batch/epoch theo CLI, xáo trộn cửa sổ train mỗi epoch (seed {SEED}).
3. EarlyStopping theo val_loss, patience={experiments['lstm_main']['metadata']['config']['patience']} (chọn ở mục 8 bằng validation), khôi phục trọng số tốt nhất. Ghi số epoch đã chạy và epoch chọn bằng validation.
4. Chạy `lstm_no_early_stopping` đủ số epoch đặt trước, đánh giá trọng số epoch cuối để quan sát tác động huấn luyện kéo dài.
5. Chạy `lstm_dropout`: Dropout(0.2) sau mỗi lớp LSTM, giữ các cấu hình còn lại và EarlyStopping.
6. Với trọng số chọn của mỗi cấu hình, dự báo train/validation/test, inverse về MW rồi tính cùng `utils.metrics()`.
7. So sánh train/validation MSE ở epoch validation tốt nhất và epoch cuối của biến thể không EarlyStopping; kiểm tra trung bình tối đa 5 epoch cuối. Chỉ xem phân kỳ rõ khi có ít nhất 5 epoch theo dõi, train giảm trên 1%, validation tăng trên 1% và các trung bình cuối cùng chiều; quy tắc này chỉ mô tả, không là kiểm định thống kê.
8. Lưu `q5_comparison.csv`, `q5_split_metrics.csv`, `q5_training_trajectory.csv`, `q5_overfitting_diagnostic.json`, lịch sử loss, metadata, mô hình Keras và hình `q5_loss.png`.

## Cách chạy

```bash
python src/q4_q5_lstm.py
```

Mặc định 50 epoch tối đa, batch 64, seed {SEED}; biến thể không EarlyStopping chạy đủ 50 epoch.
Nếu dùng --epochs khác, bảng/metadata phản ánh đúng số epoch thực tế, không gọi đó là kết quả đủ 50 epoch.
--force bỏ cache và huấn luyện lại. Tập test không dùng để chọn epoch, dropout hoặc mô hình chính.

## Lưu ý diễn giải

Loss train trong lịch sử là trung bình batch trong lúc cập nhật trọng số; RMSE train được tính lại
với trọng số cuối/chọn, nên hai đại lượng không phải cùng một phép đo. Dropout chỉ hoạt động khi train.
Chênh lệch train/validation còn chịu ảnh hưởng chuyển mùa. Chạy một seed và chưa kiểm định thống kê.
"""
    (PARTS / "q5_report.md").write_text(report, encoding="utf-8")
    (PARTS / "q5_process.md").write_text(process, encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Huấn luyện lại dù có cache khớp cấu hình")
    parser.add_argument("--rebuild-data", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-recursive", action="store_true", help="Bỏ chẩn đoán đệ quy 48 giờ")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=2)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs và batch-size phải là số nguyên dương")
    for path in [RESULTS_DIR, PARTS, FIG]:
        path.mkdir(parents=True, exist_ok=True)
    data, calendar_data = prepare_data(), prepare_data(calendar=True)
    if not pd.DatetimeIndex(data["times_test"]).equals(pd.DatetimeIndex(calendar_data["times_test"])):
        raise ValueError("Mốc test của mô hình chính và biến thể lịch phải giống hệt nhau.")
    # Mô hình chính lấy từ bước chọn patience của mục 8 (dùng cache nếu đã chạy m3_lstm.py).
    tuned = tune_main(data, epochs=args.epochs, batch_size=args.batch_size, force=args.force, verbose=args.verbose)
    base = tuned["main"]
    kwargs = dict(epochs=args.epochs, batch_size=args.batch_size, patience=tuned["selection"]["patience"],
                  verbose=args.verbose, force=args.force)
    calendar = train_experiment("lstm_calendar", calendar_data, **kwargs)
    q4 = analyse_q4(base, calendar, data, recursive=not args.skip_recursive)
    dropout = train_experiment("lstm_dropout", data, dropout=.2, **kwargs)
    no_es = train_experiment("lstm_no_early_stopping", data, early_stopping=False, **kwargs)
    q5 = analyse_q5({"lstm_main": base, "lstm_no_early_stopping": no_es, "lstm_dropout": dropout})
    print("\nCÂU 4 — cùng tập test\n", q4.to_string(index=False))
    print("\nCÂU 5 — trọng số chọn bằng validation hoặc epoch cuối\n", q5.to_string(index=False))
    print("\nĐã lưu kết quả thật, hình và bốn bản nháp Markdown cho câu 4/5.")


if __name__ == "__main__":
    main()
