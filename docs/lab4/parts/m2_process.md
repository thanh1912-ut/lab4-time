# Mục 7: Quy trình xây dựng mô hình XGBoost (Hoàng Thắng)

**File code:** `src/m2_xgboost.py` · **Chạy:** `python src/m2_xgboost.py` (từ thư mục gốc, khoảng 75 đến 80 giây gồm cả tuning và walk-forward)
**Đầu vào:** `data/processed/load_weather_hourly.csv`
**Đầu ra:** `data/processed/pred_xgb.csv`, `docs/lab4/figures/m2_learning_curve.png`, `m2_xgb_1tuan.png`

## 1. Thư viện sử dụng

- `xgboost` (bản 3.4.1): Xây dựng và tối ưu mô hình Gradient Boosted Trees đa luồng (`n_jobs=-1`).
- `scikit-learn`: Hàm chuẩn hóa `StandardScaler`, hàm đo lường và đánh giá.
- `pandas`, `numpy`: Biến đổi cấu trúc mảng 3D cửa sổ trượt sang bảng phẳng 2D.
- `matplotlib`: Trực quan hóa đường cong hàm mất mát và chuỗi thời gian so sánh.

---

## 2. Các bước triển khai chi tiết

| # | Bước thực hiện | Hàm / Tham số | Mục đích & Lý do kỹ thuật | Kết quả đạt được |
|---|---|---|---|---|
| 1 | Đọc và chuẩn hóa dữ liệu | `load_data()`, `fit_scale()` | Đọc 26.304 giờ dữ liệu 4 biến; chỉ fit `StandardScaler` trên tập Train để tránh rò rỉ dữ liệu tương lai | `scaled` shape (26304, 4), scaler lưu mean và scale của tập train |
| 2 | Tạo vector đặc trưng phẳng | `make_windows(scaled, L=24)`, `add_rolling(scaled)` | Trải phẳng 24 bước thời gian của 4 biến thành 96 cột trễ; bổ sung 3 cột thống kê cuộn (mean 6h, std 24h, trend 24h) | DataFrame 26.280 mẫu x 99 đặc trưng, không có giá trị NaN |
| 3 | Phân chia tập dữ liệu | `split_index()` | Chia theo mốc thời gian của nhãn $y$: Train (18.388), Validation (3.946), Test (3.946) | Giữ nguyên thứ tự thời gian, mẫu đầu của tập sau dùng lịch sử của tập trước |
| 4 | Tinh chỉnh độ sâu cây (`max_depth`) | `XGBRegressor(max_depth=d, n_estimators=2000, learning_rate=0.05, early_stopping_rounds=50)` | Khảo sát $d \in \{3, 5, 7, 9\}$ với cơ chế dừng sớm trên tập Validation | Tìm ra `max_depth = 7` đạt Val RMSE nhỏ nhất (4,52 MW) tại vòng lặp 153 |
| 5 | Xuất biểu đồ học (Learning Curve) | `model.evals_result()`, `plot_learning_curve()` | Đánh giá xu hướng giảm của Train RMSE và Val RMSE, xác nhận mô hình không bị overfitting | Biểu đồ `m2_learning_curve.png` |
| 6 | Kiểm định Walk-Forward trên tập Test | Vòng lặp 3.946 giờ, `refit_every = 168` | Dự báo từng giờ kế tiếp; sau mỗi 1 tuần (168 giờ) cập nhật mô hình trên toàn bộ dữ liệu quá khứ | Tập dự đoán trên test với Test RMSE = 6,07 MW, MAE = 3,71 MW, MAPE = 1,86% |
| 7 | Lưu kết quả dự đoán | `inverse_load()`, `pred_df.to_csv()` | Quy đổi từ thang chuẩn hóa về đơn vị MW gốc, lưu file theo đúng quy ước chung | File `data/processed/pred_xgb.csv` gồm 3 cột `Datetime, y_true, y_pred` |
| 8 | Trực quan hóa 1 tuần kiểm thử | `plot_sample_week()` | Trích xuất 168 giờ tuần cao điểm hè (04/08/2014 – 10/08/2014) để so sánh trực quan | Biểu đồ `m2_xgb_1tuan.png` |

---

## 3. Lỗi gặp phải và cách xử lý

| Lỗi / Vấn đề phát sinh | Nguyên nhân | Giải pháp xử lý |
|---|---|---|
| **Rò rỉ dữ liệu khi gắn đặc trưng rolling** | Nếu lấy `add_rolling(scaled)` tại vị trí $i$ (ứng với thời điểm $t+1$ của nhãn $y$) thì các thống kê cuộn đã chứa giá trị phụ tải của chính thời điểm cần dự báo | Căn chỉnh lấy giá trị đặc trưng cuộn tại vị trí $i-1$ (tương ứng thời điểm $t$), đảm bảo tại thời điểm ra quyết định chỉ sử dụng thông tin quá khứ |
| **Giá trị NaN ở dòng đầu tiên của `load_trend_24`** | `load - load.shift(24)` tại $t=23$ chưa đủ 24 bước lùi về trước nên sinh giá trị `NaN` | Áp dụng `.bfill()` trên tập đặc trưng cuộn để điền giá trị mẫu đầu tiên một cách hợp lý và an toàn |
| **Lỗi `UnicodeEncodeError` trên Windows console** | Windows PowerShell mặc định sử dụng bảng mã `cp1252` nên không hiển thị được tiếng Việt có dấu khi gọi `print` | Bổ sung đoạn mã `sys.stdout.reconfigure(encoding="utf-8")` ở đầu tệp tin thực thi |
| **Tên đặc trưng trong Booster bị xung đột chuỗi khi lọc Regex** | Tìm kiếm chuỗi đơn giản `_lag2` dễ bị khớp nhầm với `_lag20`, `_lag21`, ..., `_lag24` | Sử dụng biểu thức chính quy chính xác `re.search(r'_lag(\d+)$', name)` |

---

## 4. Môi trường thực thi

- Hệ điều hành: Windows 11 (win32)
- Python 3.14.6
- Các thư viện: `xgboost` 3.4.1, `scikit-learn` 1.9.1, `pandas` 3.0.3, `numpy` 2.5.1, `matplotlib` 3.11.2, `seaborn` 0.13.2.
