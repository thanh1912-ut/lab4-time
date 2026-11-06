# Quy trình Lab 4: Dự báo chuỗi thời gian đa biến

> Bản đang ghép. Mục 0 và 2 đã xong; các mục còn lại ghép từ `parts/*_process.md`.

## 0. Luồng dữ liệu

```
UCI LD2011_2014.txt (15 phút, 370 khách hàng)      NOAA ISD trạm LPPT (30 phút, UTC)
        │                                                   │
        └──────────── src/s2_data.py (Thành) ───────────────┘
                     gộp theo giờ, sửa giờ đổi giờ, điền giờ thiếu
                                   │
                                   ▼
             data/processed/load_weather_hourly.csv  (26.304 giờ x 4 biến)
                  │                                  │
                  │                                  └─► s3_eda.py, q1_weather.py (Phạm Lê Văn Bảo)
                  ▼
     src/features.py, src/backtest.py (Bao Dinh): scaler fit trên train, chia 70/15/15, cửa sổ 24 giờ
                  │
     ┌────────────┼──────────────────┐
     ▼            ▼                  ▼
 m1_var.py     m2_xgboost.py      m3_lstm.py
 (Lê Quỳnh)    (Hoàng Thắng)      (Hoàng Nam)
 pred_var.csv  pred_xgb.csv       pred_lstm.csv
     └────────────┬──────────────────┘
                  ▼
        s10_compare.py (Bao Dinh) → compare_metrics.csv, hình so sánh
                  ▼
        docs/lab4/report.md, process.md (Thành)
```

Chạy toàn bộ: `python src/main.py` (thêm `--rebuild-data` để tải lại dữ liệu gốc). Bài nâng cao chạy riêng: `python src/nc_traffic.py`.

Cấu hình chung trong `src/utils.py`: `COLS = [load, temperature, humidity, wind_speed]`, `L = 24`, chia 70/15/15 theo thời gian, `SEED = 42`, sai số RMSE, MAE, MAPE trên thang đo gốc qua hàm `metrics()`.

## 2. Mục 2: Chuẩn bị dữ liệu (Thành)

**File code:** `src/s2_data.py` · **Chạy:** `python src/s2_data.py` (từ thư mục gốc, khoảng 15 giây sau khi đã tải xong)
**Đầu vào:** `data/raw/LD2011_2014.txt`, `data/raw/isd/085360-99999-{2012,2013,2014}.gz` (script tự tải nếu chưa có)
**Đầu ra:** `data/processed/load_weather_hourly.csv`, `docs/lab4/figures/s2_4bien.png`, `s2_dst.png`

### Thư viện

`pandas`, `numpy`, `matplotlib`; thư viện chuẩn `urllib` (tải file), `zipfile`, `gzip`, `hashlib` (kiểm tra mã băm).

### Lấy dữ liệu

| Dữ liệu | Cách lấy | Kiểm tra |
|---|---|---|
| Điện năng | Thử tải zip từ UCI; nếu không được thì tải bản sao `LD2011_2014.txt` trên GitHub | SHA-256 `d51565f2...0175124b` (710.998.915 byte), trùng ở 4 kho GitHub độc lập lưu file của UCI |
| Thời tiết | Tải 3 file năm 2012, 2013, 2014 của trạm 085360-99999 từ kho NOAA ISD trên AWS (`noaa-isd-pds`) | Dòng đầu: `METAR LPPT 010000Z 35003KT 9999 FEW012 12/10`, đọc ra gió 1,5 m/s, nhiệt độ 12,0 °C, điểm sương 10,0 °C, khớp bản tin |

Ban đầu nhóm định dùng Open-Meteo (dữ liệu mô phỏng ERA5), nhưng chuyển sang ISD vì đây là số đo thật tại trạm và tải trực tiếp được.

### Các bước: điện năng

| # | Việc làm | Hàm / tham số | Lý do | Kết quả trung gian |
|---|---|---|---|---|
| 1 | Đọc file | `read_csv(sep=";", decimal=",", index_col=0, parse_dates=True)` | File dùng dấu `;` và dấu phẩy thập phân | 140.256 mốc 15 phút x 370 khách hàng, 01/01/2011 00:15 đến 01/01/2015 00:00 |
| 2 | Lọc 2012 đến 2014 | `.loc["2012-01-01 00:15":"2015-01-01 00:00"]` | Năm 2011 nhiều khách hàng chưa đăng ký (toàn 0) | 105.216 mốc |
| 3 | Chọn khách hàng | Mốc khác 0 đầu tiên ≤ 02/01/2012 và mốc khác 0 cuối cùng ≥ 30/12/2014 | Khách hàng mới vào giữa chừng làm tổng nhảy bậc, mô hình sẽ hiểu nhầm là xu hướng | Giữ 318, bỏ 52 |
| 4 | Tổng và gộp theo giờ | `sum(axis=1)`, `resample("h", closed="right", label="left").mean() / 1000` | Mỗi mốc là cuối khoảng 15 phút nên giờ H gồm (H:00, H+1:00]; trung bình 4 mốc là công suất trung bình trong giờ; đổi kW sang MW | 26.304 giờ |
| 5 | Sửa giờ đổi giờ | Tính Chủ nhật cuối tháng 3 và tháng 10; đặt giờ 01:00 thành NaN; `interpolate(method="time")` | Theo mô tả của UCI: tháng 3 giờ này bằng 0, tháng 10 giờ này cộng dồn 2 giờ | 6 giờ được sửa (bảng dưới) |

| Giờ | Trước khi sửa (MW) | Giờ trước / giờ sau (MW) | Sau khi sửa (MW) |
|---|---|---|---|
| 25/03/2012 01:00 | 32,9 | 104,1 / 114,7 | nội suy |
| 28/10/2012 01:00 | 206,7 | 166,6 / 114,9 | nội suy |
| 31/03/2013 01:00 | 32,5 | 99,2 / 110,1 | nội suy |
| 27/10/2013 01:00 | 193,9 | 159,7 / 107,5 | nội suy |
| 30/03/2014 01:00 | 28,1 | 93,5 / 101,1 | 97,3 |
| 26/10/2014 01:00 | 202,2 | 161,8 / 111,7 | 136,7 |

### Các bước: thời tiết

| # | Việc làm | Hàm / tham số | Lý do | Kết quả trung gian |
|---|---|---|---|---|
| 1 | Đọc định dạng ISD | Cắt theo vị trí ký tự: ngày giờ `[15:27]`, tốc độ gió `[65:69]` (m/s x 10), nhiệt độ `[87:92]` (°C x 10), điểm sương `[93:98]` | ISD là file văn bản độ rộng cố định | 52.720 bản tin (52.206 METAR định kỳ, 514 bản tin đặc biệt) |
| 2 | Bỏ giá trị lỗi | Giá trị 9999 là thiếu; mã chất lượng 2, 3, 6, 7 là nghi ngờ/sai | | Bỏ 1 nhiệt độ, 25 điểm sương |
| 3 | Tính độ ẩm | RH = 100 · exp(17,625·Td/(243,04+Td)) / exp(17,625·T/(243,04+T)) | ISD không có độ ẩm tương đối, chỉ có điểm sương | |
| 4 | Đổi giờ | `tz_localize("UTC").tz_convert("Europe/Lisbon")` | Dữ liệu điện ghi theo giờ địa phương | |
| 5 | Gộp theo giờ | `groupby(floor("h")).mean()`; `reindex` đủ 26.304 giờ | Giờ H lấy trung bình bản tin lúc H:00 và H:30 | 104 giờ không có bản tin |
| 6 | Điền giờ thiếu | Đoạn ≤ 3 giờ: `interpolate`; đoạn dài hơn: trung bình giá trị cùng giờ của t−24 và t+24, rồi `interpolate` phần còn lại | Nội suy thẳng qua 24 giờ sẽ xóa mất chu kỳ ngày của nhiệt độ | 0 giá trị thiếu |

Các đoạn thiếu thời tiết dài: 22/03/2012 09:00 (15 giờ), 14/11/2012 00:00 (24 giờ), 27/06/2013 01:00 (23 giờ), 08/11/2013 01:00 (23 giờ), 03/12/2013 19:00 (2 giờ), 12/07/2014 14:00 (10 giờ); thêm 7 giờ thiếu lẻ.

### Ghép và lưu

`pd.concat([load, weather], axis=1)`, kiểm tra đủ 26.304 giờ và không còn NaN (`assert`), làm tròn 3 chữ số, lưu `load_weather_hourly.csv`. Tương quan nhanh với load: temperature 0,656; humidity −0,528; wind_speed 0,280.

### Lỗi gặp và cách sửa

| Lỗi / vấn đề | Cách sửa |
|---|---|
| Không vào được UCI và Open-Meteo từ máy chạy | Dùng bản sao file UCI trên GitHub (kiểm tra mã băm) và dữ liệu NOAA ISD trên AWS; script vẫn thử UCI trước |
| Nếu giữ cả 370 khách hàng thì tổng tăng bậc mỗi khi có khách hàng mới | Chỉ giữ 318 khách hàng có dữ liệu suốt 3 năm |
| Dùng `resample("h")` mặc định thì giờ H gồm H:00 đến H:45, lệch 15 phút so với ý nghĩa của mốc thời gian | `closed="right", label="left"` |
| Đổi UTC sang giờ địa phương làm giờ mùa đông bị lặp và giờ mùa hè bị mất | Gộp theo giờ bằng trung bình (giờ lặp được gộp lại), giờ bị mất được điền như giờ thiếu |

### Môi trường đã chạy

Python 3.13, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2.

## 3. Mục 3: Data Understanding (Phạm Lê Văn Bảo)

_`parts/s3_process.md`, `parts/q1_process.md`_

## 4. Mục 4, 5: Preprocessing và Feature Engineering (Bao Dinh)

_`parts/s4_s5_process.md`_

## 5. Mục 6: VAR (Lê Quỳnh)

_`parts/m1_process.md`, `parts/q2_process.md`_

## 6. Mục 7: XGBoost (Hoàng Thắng)

_`parts/m2_process.md`, `parts/q3_process.md`, `parts/q6_process.md`_

## 7. Mục 8: LSTM (Hoàng Nam)

_`parts/m3_process.md`, `parts/q4_process.md`, `parts/q5_process.md`_

## 8. Mục 9, 10: Backtesting và so sánh (Bao Dinh)

_`parts/s9_s10_process.md`_

## 9. Mục 12: Nâng cao (Phạm Lê Văn Bảo)

_`parts/nc_process.md`_

## 10. Phân công

| Thành viên | Phần việc |
|---|---|
| Thành | Dựng dự án, README, mục 2 (dữ liệu), ghép báo cáo, `main.py` |
| Bao Dinh | Mục 4, 5, 9, 10 |
| Phạm Lê Văn Bảo | Mục 3, Câu 1, mục 12 nâng cao |
| Lê Quỳnh | Mục 6 (VAR), Câu 2 |
| Hoàng Thắng | Mục 7 (XGBoost), Câu 3, Câu 6 |
| Hoàng Nam | Mục 8 (LSTM), Câu 4, Câu 5 |
