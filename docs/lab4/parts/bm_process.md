# Quy trình benchmark bổ sung — Thành

1. **Dự báo đầu vào**, cùng luật (học trên train, không học lại trên test):
   - `pred_var_fixed.csv` (mục 6).
   - `pred_xgb_fixed.csv` (mục 7).
   - `lstm_p10_s*_test.csv` (mục 8).
   - Kiểm tra mỗi file có đúng 3.946 mốc test và `y_true` khớp dữ liệu gốc (≤ 1e-3 MW).
   - Kiểm tra `pred_lstm.csv` trùng lần chạy seed 42.
2. **Thêm seed LSTM** 45, 46: dùng `lstm_common.train_experiment` với patience đã chọn, có cache. Seed thêm chỉ để đo độ ổn định, không dùng để chọn cấu hình.
3. **Baseline** tính trực tiếp trên load gốc: `shift(1)`, `shift(24)`, `shift(168)`. **Kết hợp** là trung bình đều ba dự báo.
4. **Chỉ số:** RMSE, MAE, MAPE, MASE (thang = MAE của seasonal naive 24 giờ trên train, 7,22 MW), skill so với seasonal naive 24 giờ.
5. **Bootstrap:** `block_indices()` chọn ngẫu nhiên điểm đầu của các khối 168 giờ (seed 42), ghép rồi cắt về 3.946 giờ, 2000 lần. Dùng cùng chỉ số khối cho mọi mô hình. Khoảng tin cậy lấy phân vị 2,5% và 97,5%.
6. **Diebold–Mariano** (`dm_test()`):
   - $d_t = L(e_{A,t}) - L(e_{B,t})$.
   - Thống kê $\bar d / \sqrt{\hat\sigma^2_{NW}/n}$, trong đó Newey–West dùng kernel Bartlett, trễ 24.
   - Nhân hệ số HLN, tính p hai phía theo $t_{n-1}$.
   - Hiệu chỉnh Holm trên 10 cặp, làm riêng cho từng loại loss.
   - Lặp lại với trễ 168.
7. **Nhóm:**
   - Khung giờ: 0–6, 7–11, 12–17, 18–23.
   - Loại ngày: ngày thường, cuối tuần, lễ / kỳ nghỉ cuối năm. Lễ gồm 15/08, 08/12, 25/12, cộng thêm 24–31/12. Các ngày 5/10, 1/11, 1/12 không còn là ngày lễ chính thức ở Bồ Đào Nha trong giai đoạn 2013–2015.
   - Tháng.
8. **Chi phí:**
   - Học lại VAR trên train và đo thời gian.
   - Học lại XGBoost (depth 7, early stopping theo validation), đo thời gian và kiểm tra ra đúng `pred_xgb_fixed.csv`. Đếm số cây và số lá.
   - LSTM lấy thời gian học từ metadata (trung vị các seed), đo thời gian dự báo của mô hình chính.
9. **Lưu kết quả:**
   - `data/processed/benchmark/`: `bm_metrics.csv`, `bm_metrics_refit.csv`, `bm_dm.csv`, `bm_lstm_seeds.csv`, `bm_segments.csv`, `bm_month.csv`, `bm_cost.csv`, `bm_environment.json`.
   - Hình `bm_ci.png`, `bm_thang.png`.
   - `parts/bm_report.md`, `parts/bm_process.md`.

Hình dùng một màu cố định cho mỗi mô hình (VAR xanh dương, XGBoost cam, LSTM xanh ngọc, kết hợp vàng); bảng màu đã được kiểm tra cho người mù màu. Mọi giá trị trên hình đều có trong bảng.
