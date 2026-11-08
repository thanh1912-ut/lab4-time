# Quy trình mục 8 — Hoàng Nam

## Chạy lại

Từ gốc repo: `python src/m3_lstm.py`. Mặc định epochs=50, batch=64. Huấn luyện lại dù đã có cache: `python src/m3_lstm.py --force`. Tham số `--epochs` dùng thử cấu hình khác; báo cáo tự ghi số epoch thực chạy.

## Các bước và kết quả trung gian

1. `utils.load_data()` đọc CSV chung; kiểm tra giờ liên tục, duy nhất và giá trị hữu hạn.
2. `features.fit_scale()` fit scaler trên 18412 giờ train gốc. `features.make_windows()` tạo cửa sổ 24 giờ. Chia theo thời điểm y: train 18388, validation 3946, test 3946; không chia lại tỉ lệ trên len(X).
3. Seed 42, deterministic ops; tf.data giữ thứ tự mẫu, giới hạn thread CPU. Kiến trúc đúng đề, Adam 0,001, MSE. EarlyStopping chỉ theo validation.
4. Lưu loss mỗi epoch. Chọn epoch 16 theo `val_loss`, sau 21 epoch thực chạy.
5. Dự báo theo batch; `features.inverse_load()` đưa về MW. `y_true` lấy nguyên từ CSV nguồn; tính RMSE/MAE/MAPE bằng `utils.metrics()`.
6. Lưu mô hình, dự báo ba tập, lịch sử, metadata trong `data/processed/lstm/`. Cache chỉ tái dùng khi dữ liệu, mã helper, cấu hình và runtime khớp.
7. Xuất `pred_lstm.csv`, hình 168 giờ đầu test, bảng baseline, báo cáo. Test bắt đầu 2014-07-20 14:00:00, kết thúc 2014-12-31 23:00:00.

## File để kiểm tra

- Code: `src/m3_lstm.py`, `src/lstm_common.py`; chuẩn hóa và tạo cửa sổ qua `features.py`.
- Mô hình: `data/processed/lstm/models/lstm_main.keras`.
- Lịch sử: `data/processed/lstm/lstm_main_history.csv`.
- Cấu hình, thời gian, hash, số tham số và sai số: `data/processed/lstm/lstm_main_metadata.json`.
- Hash SHA-256 dữ liệu: `d1782f8e7dc1a7aae06c17699edf6ccaec8bf2f8b958aac9a4da8d108d552ce3`.
- Tài liệu EarlyStopping: https://keras.io/api/callbacks/early_stopping/ .

Đường dẫn ảnh tương đối từ `docs/lab4/parts/` là `../figures/`.
