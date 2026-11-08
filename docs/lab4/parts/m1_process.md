# Quy trình Mục 6 - VAR (Lê Quỳnh)

## Các bước, hàm và tham số
| Bước | Hàm / tham số | Lý do |
|---|---|---|
| Đọc và chuẩn hóa | `utils.load_data()`, `features.fit_scale()` (scaler chỉ fit trên train 70%) | Dùng chung với XGBoost, LSTM để so sánh được; không rò rỉ tương lai |
| Kiểm tra tính dừng | `adfuller(autolag="AIC")` từng biến trên train | VAR giả định chuỗi dừng; p > 0,05 thì không bác bỏ nghiệm đơn vị |
| Sai phân | `DataFrame.diff()` bậc 1 cho cả 4 biến (chỉ khi có biến không dừng) | Cùng một cách cho cả 4 biến để đảo ngược dễ: `load_t + Δload_dự_báo` |
| Chọn bậc trễ | `VAR(train).select_order(maxlags=48)`, so AIC và BIC | 48 giờ = 2 chu kỳ ngày; BIC phạt tham số nặng hơn nên p nhỏ hơn, chọn theo `CRITERION` trong `m1_var.py` |
| Huấn luyện | `VAR(data).fit(p)` | Số tham số hệ số = 4 x 4 x p |
| Granger | `res.test_causality("load", [biến], kind="f")` | Kiểm tra thời tiết có giúp dự báo load không |
| Walk-forward | `backtest.walk_forward`, `refit_every=168`, `predict_fn` dùng `res.forecast(last_p, steps=1)` | Mỗi giờ chỉ dùng dữ liệu trước giờ đó, giá trị thật được thêm vào lịch sử sau mỗi bước |
| Thang đo gốc | `features.inverse_load` rồi `utils.metrics` | Quy ước chung của nhóm |

Lần huấn luyện đầu dùng train + validation (đến hết validation), sau đó huấn luyện lại mỗi 168 giờ trên toàn bộ dữ liệu trước thời điểm đó.

## Kết quả trung gian
Lấy từ `m1_report.md` (bảng ADF, bảng AIC/BIC, p chọn, kiểm định Granger) và các file `data/processed/var_aic_bic.csv`, `var_order.json`.

## Lỗi gặp phải và cách sửa
(Điền khi chạy thật; ví dụ: lỗi thiếu `statsmodels`, thời gian walk-forward quá lâu nên đặt `refit_every=168`, cảnh báo index thời gian của statsmodels.)
