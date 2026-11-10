# Câu hỏi 3: Quy trình phân tích ảnh hưởng của thời tiết trong XGBoost (Hoàng Thắng)

**File code:** `src/q3_xgb_weather.py` · **Chạy:** `python src/q3_xgb_weather.py` (khoảng 15 đến 20 giây)
**Đầu vào:** `data/processed/load_weather_hourly.csv`, mô hình XGBoost đã tinh chỉnh ở Mục 7
**Đầu ra:** `docs/lab4/figures/q3_importance_nhom.png`, `q3_importance_lag.png`, `q3_pdp_temp.png`

## 1. Phương pháp & Công cụ

- **Độ lợi đặc trưng (Feature Importance - Gain):** Sử dụng hàm `get_score(importance_type="gain")` từ Booster của XGBoost. Chỉ số Gain phản ánh độ giảm hàm mất mát trung bình của mỗi phép cắt nhánh sử dụng đặc trưng đó, là thước đo tin cậy nhất trong các bài toán hồi quy (vượt trội hơn `weight`/`frequency` vốn chỉ đếm số lần xuất hiện).
- **Thử nghiệm loại trừ đặc trưng (Ablation Study):** Huấn luyện lại các mô hình trên các tập con đặc trưng (Full, Load + Rolling, Load Only) với cùng siêu tham số (`max_depth=7`, `learning_rate=0.05`, `early_stopping_rounds=50`), đảm bảo mọi khác biệt về sai số chỉ bắt nguồn từ sự hiện diện của các nhóm đặc trưng đang thử nghiệm.
- **Biểu đồ phụ thuộc một phần (Partial Dependence Plot - PDP):** Sử dụng `sklearn.inspection.partial_dependence` để tính tích phân biên (marginal effect) của một đặc trưng lên kỳ vọng đầu ra:
  $$\hat{f}_{S}(x_S) = \mathbb{E}_{X_C}[\hat{f}(x_S, X_C)] \approx \frac{1}{n} \sum_{i=1}^n \hat{f}(x_S, x_{C}^{(i)})$$
  trong đó $S$ là biến nhiệt độ khảo sát, $C$ là tập hợp tất cả các đặc trưng còn lại.

---

## 2. Các bước triển khai chi tiết

| # | Bước thực hiện | Hàm / Kỹ thuật | Chi tiết & Mục đích | Kết quả trung gian |
|---|---|---|---|---|
| 1 | Huấn luyện mô hình tối ưu | `xgb.XGBRegressor(max_depth=7, n_estimators=1000)` | Fit trên tập Train, kiểm tra trên Validation để kích hoạt Early Stopping | Mô hình tối ưu tại `best_iteration=153` |
| 2 | Trích xuất và gom nhóm Gain | `get_score(importance_type="gain")`, gom theo tiền tố biến | Gom 99 cột vào 5 nhóm: Phụ tải, Nhiệt độ, Độ ẩm, Tốc độ gió, Thống kê cuộn | Phụ tải: 97,19%; Rolling: 1,02%; Nhiệt độ: 0,94%; Độ ẩm: 0,47%; Gió: 0,39% |
| 3 | Gom nhóm Gain theo bậc trễ | Biểu thức chính quy `re.search(r'_lag(\d+)$', name)` | Tổng hợp đóng góp từ Lag 1 đến Lag 24, phân tách giữa phụ tải và các biến thời tiết | Lag 24 đạt Gain lớn nhất (137,72), kế tiếp là Lag 1 (2,43) |
| 4 | Thực hiện Ablation Study | Huấn luyện 3 mô hình tương ứng 3 tập con cột | Đo đạc RMSE, MAE, MAPE trên cả Validation và Test | Mô hình Full giảm 19,1% RMSE Val so với mô hình Load Only |
| 5 | Tính toán và chuẩn hóa PDP | `sklearn.inspection.partial_dependence(..., features=['temperature_lag1'])` | Lưới khảo sát 50 điểm; nhân ngược độ lệch chuẩn và cộng kỳ vọng để hiển thị °C và MW gốc | Đường cong thể hiện rõ ngưỡng nhiệt độ bắt đầu bật điều hòa ($>20^\circ\text{C}$) |
| 6 | Xuất hình ảnh và bảng biểu | `matplotlib.pyplot` | Lưu 3 hình ảnh với DPI=150, ghi chú trực quan | `q3_importance_nhom.png`, `q3_importance_lag.png`, `q3_pdp_temp.png` |

---

## 3. Lỗi gặp phải và cách xử lý

| Vấn đề phát sinh | Nguyên nhân | Cách khắc phục |
|---|---|---|
| **Lỗi KeyError trong scikit-learn mới khi đọc kết quả PDP** | Trong các phiên bản scikit-learn mới (1.9+), đối tượng Bunch trả về từ `partial_dependence` sử dụng khóa `grid_values` thay cho `values` | Sử dụng `pdp["grid_values"][0]` để lấy tọa độ các mốc lưới đặc trưng |
| **Trục tọa độ PDP ở thang đo chuẩn hóa khó diễn giải** | Giá trị đầu vào là Z-score (quanh 0) và đầu ra là Z-score chuẩn hóa, người đọc không thấy được nhiệt độ thực tế | Nhân ngược với `scaler.scale_[1]` và cộng `scaler.mean_[1]` để ra $^\circ\text{C}$; nhân độ lệch đầu ra với `scaler.scale_[0]` để ra MW công suất thực tế |
| **Xung đột tên trễ khi gom nhóm** | Điều kiện `if "_lag1" in name` sẽ bắt dính cả `_lag10` đến `_lag19` | Dùng Regex kiểm tra kết thúc chuỗi `r'_lag(\d+)$'` |

---

## 4. Môi trường thực thi

- Python 3.14.6
- Thư viện: `xgboost` 3.4.1, `scikit-learn` 1.9.1, `matplotlib` 3.11.2, `pandas` 3.0.3, `numpy` 2.5.1.
