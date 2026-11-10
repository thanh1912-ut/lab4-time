# Đối chiếu yêu cầu Practice 4 với bài nhóm (10/11/2026)

Repo: https://github.com/thanh1912-ut/lab4-time. Số liệu trong file này lấy theo bản đã có [PR #5](https://github.com/thanh1912-ut/lab4-time/pull/5) (so sánh công bằng) và [PR #6](https://github.com/thanh1912-ut/lab4-time/pull/6) (benchmark). Hai PR này **chưa merge**, nên `main` hiện vẫn còn số LSTM cũ (13,96) và số giả ở Câu 6.

Ký hiệu trạng thái:
- ✅ đúng và đủ
- ⚠️ đã làm nhưng cần chỉnh hoặc giải thích thêm
- ❌ còn thiếu

## Kết luận nhanh

**Phần code và kết quả đã đủ** cho cả 12 mục và 6 câu hỏi của cô, chạy lại ra đúng số.

**Chưa nộp được ngay** vì còn 3 việc:
1. Merge PR #5 và PR #6.
2. Ghép báo cáo cuối `docs/lab4/report.md` và `process.md`. Hiện từ mục 3 trở đi, hai file này mới chỉ là khung trỏ tới `parts/`.
3. Viết phần chữ cho mục 4–5 (`parts/s4_s5_report.md`, `s4_s5_process.md`). Code và hình của hai mục này đã có.

## 1. Mục tiêu

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| Mô hình hóa chuỗi thời gian đa biến | 4 biến `load, temperature, humidity, wind_speed` | ✅ |
| Ba nhóm mô hình VAR, XGBoost, LSTM | `src/m1_var.py`, `src/m2_xgboost.py`, `src/m3_lstm.py` | ✅ |
| Đánh giá bằng RMSE, MAE, MAPE | `utils.metrics()` dùng chung, bảng ở `s10_compare.py` | ✅ |

## 2. Dataset và bài toán

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| Electricity Load Diagrams (UCI) với 4 biến | Điện lấy từ UCI LD2011_2014, 318 khách hàng, đổi sang MW. **UCI không có thời tiết**, nên ghép thêm nhiệt độ, độ ẩm, gió của trạm Lisboa (NOAA ISD). Kết quả: `data/processed/load_weather_hourly.csv`, 26.304 giờ, 2012–2014 | ⚠️ Cần ghi rõ trong báo cáo lý do ghép thêm nguồn thời tiết (đã có ở mục 2 của `report.md`) |
| Vector quan sát $X_t$ = [load, temp, humidity, wind] | `utils.COLS` | ✅ |
| Dùng 24 giờ gần nhất dự báo load giờ tiếp theo | `features.make_windows(L=24)`: 24 giờ đến $t$ để dự báo load $t+1$ | ✅ |

## 3. Data Understanding (Phạm Lê Văn Bảo)

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| Missing values | Bảng 3.1: cả 4 biến 0 giá trị thiếu sau mục 2. Mục 2 đã sửa 6 giờ đổi giờ mùa hè và điền các giờ thời tiết trống | ✅ |
| Outliers | IQR và Z-score theo từng biến (load chỉ 9 điểm theo IQR), xét thêm phần dư load; hình `s3_outlier.png` | ✅ |
| Seasonality ngày và tuần | Load trung bình theo giờ và theo thứ (`s3_chu_ky.png`); ACF lag 24 = 0,987, lag 168 = 0,964 (`s3_acf.png`) | ✅ |
| Biểu đồ load theo thời gian | `s3_load.png` | ✅ |
| Biểu đồ load vs temperature | `s3_load_vs_temp.png`, có tách theo mùa | ✅ |
| Ma trận tương quan | `s3_corr.png`, Pearson | ✅ |

## 4. Preprocessing (Bao Dinh)

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| StandardScaler cho mọi biến | `features.fit_scale()` chỉ fit trên train, dùng chung cho cả 3 mô hình | ✅ code |
| Chia theo thời gian, không shuffle, tỉ lệ 70/15/15 | Train 18.412 giờ (đến 06/02/2014 03:00), validation 3.946 giờ, test 3.946 giờ (20/07–31/12/2014); hình `s4_split.png` | ✅ code |
| Phần viết | `parts/s4_s5_report.md`, `s4_s5_process.md` chưa có | ❌ |

> **Lưu ý về chữ "không shuffle":** cô yêu cầu không xáo trộn khi **chia tập**, và nhóm làm đúng điều đó. Riêng LSTM (PR #5) có xáo trộn **thứ tự các cửa sổ bên trong tập train** mỗi epoch. Mỗi cửa sổ đã chứa sẵn 24 giờ quá khứ của nó, còn validation và test vẫn giữ nguyên thứ tự thời gian, nên không có rò rỉ dữ liệu. Nếu không xáo trộn, LSTM chỉ đạt RMSE 13,96, kém cả cách "lấy cùng giờ hôm qua" (13,42). Báo cáo nên giải thích rõ điểm này để cô không hiểu nhầm.

## 5. Feature Engineering (Bao Dinh)

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| Cửa sổ trễ L = 24 cho 4 biến | `make_windows`: X có dạng (n, 24, 4); hình `s5_window.png` | ✅ |
| Rolling mean 6 giờ, rolling std 24 giờ, trend $load_t - load_{t-24}$, chỉ dùng cho XGBoost | `features.add_rolling()`; XGBoost có 99 đặc trưng = 96 lag + 3 rolling | ✅ |

## 6. VAR (Lê Quỳnh)

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| Kiểm tra dừng bằng ADF; không dừng thì sai phân | ADF trên train: cả 4 biến dừng (p của load là 0,017) nên không sai phân; code có sẵn nhánh sai phân | ✅ |
| Chọn bậc trễ theo AIC hoặc BIC | `select_order(48)`: AIC chọn 48, BIC chọn 26, dùng **p = 26 theo BIC** | ✅ |
| Huấn luyện VAR, dự báo cả vector, chỉ đánh giá load | `VAR(...).fit(26)`, ổn định; `forecast` trả về 4 biến, chỉ lấy cột load. Có thêm kiểm định Granger | ✅ |
| Kết quả test | RMSE 6,07, MAE 4,11, MAPE 2,06% (học 1 lần); 6,05 / 4,06 / 2,03 (học lại mỗi tuần) | ✅ |

## 7. XGBoost (Hoàng Thắng)

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| Đầu vào là vector phẳng từ cửa sổ lag; đầu ra là load $t+1$ | 99 cột (`build_features`) | ✅ |
| Điều chỉnh max_depth | Thử {3, 5, 7, 9} trên validation, chọn 7 (val RMSE 4,52) | ✅ |
| Điều chỉnh n_estimators bằng early stopping trên validation | Tối đa 2000 cây, early stopping 50 vòng, dừng ở 153; có hình learning curve | ✅ |
| Kết quả test | RMSE 6,17, MAE 3,81, MAPE 1,89% (học 1 lần); 6,07 / 3,71 / 1,86 (học lại mỗi tuần) | ✅ |

## 8. LSTM (Hoàng Nam)

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| Tensor 3D (batch, 24, 4) | `prepare_data()`: X_train có dạng (18.388, 24, 4) | ✅ |
| LSTM(64) → LSTM(32) → Dense(1) | Đúng kiến trúc, 30.113 tham số | ✅ |
| Loss MSE, optimizer Adam | MSE, Adam với learning rate 0,001; EarlyStopping theo validation | ✅ |
| Kết quả test | RMSE 6,14, MAE 4,39, MAPE 2,25% (seed 42); qua 5 seed: 6,05 ± 0,16 | ✅ (sau PR #5) |

## 9. Backtesting walk-forward

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| Áp dụng cho cả 3 mô hình: train, dự báo t+1, thêm vào chuỗi, dự báo t+2… | `backtest.walk_forward()`: mỗi giờ test dự báo 1 bước, sau đó **đưa giá trị thật** của giờ đó vào lịch sử rồi dự báo giờ kế tiếp. Có tùy chọn học lại mỗi 168 giờ (VAR, XGBoost) | ✅ |
| (Thêm) dự báo nhiều bước, đưa **giá trị dự báo** vào chuỗi | VAR: h = 1/24/72 giờ (Câu 2); LSTM: đệ quy 48 giờ (Câu 4) | ✅ |

> Câu "thêm vào chuỗi" có thể hiểu là thêm giá trị thật hoặc thêm giá trị dự báo. Bảng chính dùng giá trị thật, đúng như vận hành thực tế; còn cách thêm giá trị dự báo được thử riêng ở Câu 2 và Câu 4. Báo cáo nên ghi rõ cả hai.

## 10. Đánh giá mô hình

Test 3.946 giờ. Cả 3 mô hình theo cùng luật: học trên train, chọn cấu hình bằng validation, không học lại trên test.

| Mô hình | RMSE (MW) | MAE (MW) | MAPE (%) |
|---|---|---|---|
| Naive (giờ trước) | 25,50 | 16,34 | 8,78 |
| Seasonal naive 24h | 13,42 | 7,68 | 3,82 |
| VAR(26) | 6,07 | 4,11 | 2,06 |
| XGBoost | 6,17 | 3,81 | 1,89 |
| LSTM | 6,14 | 4,39 | 2,25 |

Trạng thái: ✅ (`s10_compare.py`, `parts/s9_s10_report.md`).

Phần thêm ngoài yêu cầu (PR #6):
- Benchmark có khoảng tin cậy 95% bằng bootstrap.
- Kiểm định Diebold–Mariano. Với sai số bình phương, ba mô hình không khác biệt có ý nghĩa.
- Kết hợp trung bình 3 mô hình cho RMSE 5,36.

## 11. Câu hỏi phân tích

| Câu hỏi | Trả lời ngắn (file) | Trạng thái |
|---|---|---|
| 1. Biến thời tiết nào ảnh hưởng mạnh nhất? | **Nhiệt độ**: Pearson 0,66, mạnh nhất vào mùa hè (0,72); độ ẩm và gió yếu hơn nhiều. Khớp với Gain của XGBoost (`q1_report.md`) | ✅ |
| 2. Khi nào VAR thất bại? | p quá nhỏ (p = 2 cho RMSE 16,42), giờ tải đổi nhanh 9–10h, kỳ nghỉ cuối năm, dự báo xa (RMSE h = 72 là 18,4) (`q2_report.md`) | ⚠️ Chạy lại `q2_var_fail.py` sẽ **ghi đè** phần trả lời viết tay. Ngày 1/11/2014 không phải ngày lễ chính thức (nên viết "6/10 giờ trong kỳ nghỉ cuối năm") |
| 3. XGBoost học được gì từ thời tiết? | Nhiệt độ quan trọng nhất trong nhóm thời tiết; PDP cho thấy load tăng khi trên khoảng 20 °C (`q3_report.md`) | ⚠️ Viết quá tay: trên test, bỏ thời tiết vẫn ra RMSE 6,16 so với 6,17; mức giảm 19% chủ yếu nhờ đặc trưng rolling; PDP có dạng chữ J chứ không phải chữ U |
| 4. LSTM học được chu kỳ nào? | Cửa sổ 24 giờ bắt được chu kỳ ngày; thêm sin/cos giờ và thứ làm RMSE giảm từ 6,14 xuống 5,77, tức còn thiếu thông tin chu kỳ tuần (`q4_report.md`) | ✅ (sau PR #5) |
| 5. LSTM có overfitting không? | Có, khi chạy đủ 50 epoch: sau epoch 16 train loss giảm 34% nhưng val loss tăng 12%; EarlyStopping chặn được (`q5_report.md`) | ✅ (sau PR #5) |
| 6. Khác biệt ML và DL? | Cùng luật huấn luyện thì độ chính xác ngang nhau; XGBoost học nhanh hơn khoảng 50 lần, ổn định và dễ giải thích hơn; LSTM nhạy với seed và cách train (`q6_report.md`) | ✅ (sau PR #5; bản trên `main` còn số giả) |

## 12. Bài tập nâng cao: Traffic (Phạm Lê Văn Bảo)

| Yêu cầu của cô | Bài nhóm | Trạng thái |
|---|---|---|
| METR-LA, 207 trạm, 5 phút/lần | `nc_traffic.py` đọc `metr-la.h5` | ✅ |
| Dùng 12 bước quá khứ dự báo X(t+3), X(t+6), X(t+12), tức 15/30/60 phút | GRU so với Last Value và Historical Average. MAE của GRU: 3,40 / 3,60 / 4,00; Last Value: 3,50 / 4,21 / 5,38 | ✅ |
| Đối chiếu bài báo DCRNN | So với MAE công bố 2,77 / 3,15 / 3,60, có nêu lý do chênh lệch (`nc_report.md`) | ✅ |
| Chạy lại được | `torch` chưa có trong `requirements.txt`; `metr-la.h5` không nằm trong repo và chưa ghi link tải cụ thể | ⚠️ |

## Cấu trúc nộp bài cô yêu cầu

| Thư mục / file | Trạng thái |
|---|---|
| `data/raw/`, `data/processed/` | ✅ (file gốc 710 MB không đưa lên Git, có `s2_data.py` để tải lại) |
| `src/` có `main.py` chạy toàn bộ | ✅ |
| `README.md` | ✅ |
| `docs/lab4/report.md` | ❌ Mới có mục 1–2; mục 3–13 chỉ là dòng trỏ tới `parts/` |
| `docs/lab4/process.md` | ❌ Mới có mục 0–2; các mục còn lại chỉ là dòng trỏ tới `parts/` |

## Việc còn lại trước khi nộp

1. **Thành:**
   - Merge PR #5 rồi PR #6.
   - Ghép `parts/*.md` vào `report.md` và `process.md`: đánh số hình và bảng, đổi đường dẫn ảnh, chốt một bộ số liệu, xóa `parts/`.
2. **Bao Dinh:** viết `s4_s5_report.md` và `s4_s5_process.md`.
3. **Lê Quỳnh:**
   - Không để `q2_var_fail.py` ghi đè phần trả lời viết tay.
   - Sửa ý ngày lễ 1/11.
4. **Hoàng Thắng:** sửa kết luận Câu 3 cho khớp số ablation trên test.
5. **Phạm Lê Văn Bảo:** thêm `torch` vào requirements, ghi link và sha256 của `metr-la.h5`.
