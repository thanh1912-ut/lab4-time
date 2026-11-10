# Quy trình mục 9 và 10 — Bao Dinh

## Mục 9: `src/backtest.py`

1. `walk_forward(fit_fn, predict_fn, scaled, start, end, refit_every=None, scaler=None, train_end=None)` kiểm tra đầu vào: DataFrame không rỗng, có cột `load`, thời gian tăng dần, `start`/`end` hợp lệ, `refit_every` là số nguyên dương hoặc None, `train_end` trong `[1, start]`.
2. Lần học đầu: `fit_fn(scaled.iloc[:start])`, hoặc `scaled.iloc[:train_end]` khi chỉ muốn học trên train.
3. Mỗi giờ `t`: nếu tới kỳ học lại (`k % refit_every == 0`) thì `fit_fn(scaled.iloc[:t])`; dự báo bằng `predict_fn(model, scaled.iloc[:t])`, không truyền dữ liệu từ giờ `t` trở đi.
4. Trả DataFrame `actual`, `prediction` theo `Datetime`. Có `scaler` thì đổi về MW bằng `features.inverse_load`.
5. Ban đầu hàm trả mảng numpy. Sau khi đổi sang DataFrame thì `m1_var.py` bị lỗi; đã sửa `m1_var.py` để lấy cột `prediction`.
6. Thêm `train_end` để VAR có bản học một lần trên train, cùng luật với XGBoost và LSTM.

## Mục 10: `src/s10_compare.py`

1. `load_data()`; chia 70/15/15 theo thời gian; test 3.946 giờ từ 20/07/2014 14:00.
2. Hai baseline tính trực tiếp trên dữ liệu gốc: naive `load.shift(1)`, seasonal naive `load.shift(24)`.
3. Đọc dự báo theo luật chung (học trên train, không học lại): `pred_var_fixed.csv`, `pred_xgb_fixed.csv`, `pred_lstm.csv`. Mỗi file có `Datetime, y_true, y_pred`.
4. Kiểm tra `y_true` trong từng file khớp load gốc (sai lệch ≤ 1e-3 MW), ghép theo `Datetime` và chỉ giữ các giờ mọi mô hình đều có dự báo. Kết quả: đủ 3.946 giờ.
5. Tính RMSE, MAE, MAPE; lưu `compare_metrics.csv`, `compare_predictions.csv`, `compare_protocol.csv` (cách huấn luyện từng mô hình).
6. Bảng phụ: tính lại cho `pred_var.csv`, `pred_xgb.csv` (bản refit mỗi 168 giờ) trên cùng các giờ; lưu `compare_metrics_refit.csv`.
7. Vẽ `s10_1tuan.png` (48 giờ đầu test), `s10_metrics.png`, `s10_loi_theo_gio.png`.

## Vì sao đổi luật so sánh

Bản đầu so VAR và XGBoost có học lại mỗi tuần (thấy cả validation và phần test đã qua) với LSTM chỉ học trên train và không học lại. LSTM khi đó còn train không xáo trộn mẫu nên RMSE 13,96 MW. Khi đưa cả ba về cùng luật:
- Học lại hằng tuần chỉ làm VAR/XGBoost tốt hơn 0,02–0,10 MW.
- LSTM sau khi sửa cách train đạt 6,14 MW.

Vì vậy kết luận cũ "XGBoost/VAR tốt hơn LSTM rõ rệt" là do khác cách huấn luyện, không phải do mô hình.

## Chạy lại

```bash
python src/m1_var.py        # pred_var.csv, pred_var_fixed.csv
python src/m2_xgboost.py    # pred_xgb.csv, pred_xgb_fixed.csv
python src/m3_lstm.py       # pred_lstm.csv
python src/s10_compare.py
```
