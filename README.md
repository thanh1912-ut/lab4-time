# Lab 4: Dự báo chuỗi thời gian đa biến (Điện năng + Thời tiết)

Bài thực hành 4, môn Phân tích chuỗi thời gian.

## 1. Giới thiệu

Tại mỗi giờ t có vector X_t = [load_t, temperature_t, humidity_t, wind_speed_t]. Dùng 24 giờ gần nhất để dự báo điện năng giờ tiếp theo load_{t+1}.

Mục tiêu:
- Hiểu cách mô hình hóa chuỗi thời gian đa biến
- Xây dựng và so sánh 3 nhóm mô hình: VAR (thống kê), XGBoost (Machine Learning), LSTM (Deep Learning)
- Đánh giá bằng walk-forward validation với RMSE, MAE, MAPE
- Trả lời 6 câu hỏi phân tích; bài nâng cao dự báo giao thông (METR-LA / PEMS-BAY)

## 2. Data Introduction

Bộ dữ liệu điện của UCI **không có biến thời tiết**, nên nhóm ghép thêm dữ liệu thời tiết cùng khu vực (Bồ Đào Nha) và cùng thời gian.

| Nguồn | Nội dung | Link |
|---|---|---|
| UCI ElectricityLoadDiagrams20112014 | Điện tiêu thụ 15 phút/lần của 370 khách hàng ở Bồ Đào Nha, 2011 đến 2014 | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 |
| Open-Meteo Historical Weather | Nhiệt độ, độ ẩm, tốc độ gió theo giờ tại Lisbon | https://open-meteo.com/en/docs/historical-weather-api |

File dùng chung sau khi ghép: `data/processed/load_weather_hourly.csv` (cột `Datetime, load, temperature, humidity, wind_speed`, theo giờ, 2012 đến 2014). _(Thành điền số dòng, thống kê sau khi chạy `s2_data.py`.)_

## 3. Môi trường

- Python 3.11 hoặc 3.12 (TensorFlow chưa chắc hỗ trợ bản mới hơn)
- Thư viện: xem `requirements.txt`

## 4. Cài đặt và chạy

```bash
git clone https://github.com/thanh1912-ut/lab4-time.git
cd lab4-time
pip install -r requirements.txt
python src/s2_data.py       # tải dữ liệu gốc vào data/raw/ (không có trên Git vì quá lớn) và tạo file đã ghép
python src/main.py          # chạy toàn bộ
python src/nc_traffic.py    # bài nâng cao, chạy riêng
```

Kết quả: hình trong `docs/lab4/figures/`, dự đoán và bảng so sánh trong `data/processed/`, báo cáo ở `docs/lab4/report.md` và `docs/lab4/process.md`.

## 5. Cấu trúc thư mục

```
lab4-time/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          dữ liệu gốc, không sửa, không đưa lên Git
│   └── processed/    dữ liệu sau xử lý, file dự đoán
├── src/              mã nguồn, mỗi file một phần việc
└── docs/lab4/
    ├── report.md     nội dung và kết quả
    ├── process.md    quy trình, phương pháp, kết quả trung gian
    ├── phan-cong.md  phân công chi tiết
    ├── figures/
    └── parts/        bản nháp từng người
```

## 6. Phân công

| Thành viên | Phần việc | File |
|---|---|---|
| Thành | Dựng dự án, README, mục 2 (dữ liệu), ghép báo cáo | `utils.py`, `s2_data.py`, `main.py` |
| Bao Dinh | Mục 4, 5 (tiền xử lý, đặc trưng), mục 9 (walk-forward), mục 10 (so sánh) | `features.py`, `backtest.py`, `s4_s5_preprocess.py`, `s10_compare.py` |
| Phạm Lê Văn Bảo | Mục 3 (Data Understanding), Câu 1, mục 12 nâng cao | `s3_eda.py`, `q1_weather.py`, `nc_traffic.py` |
| Lê Quỳnh | Mục 6 (VAR), Câu 2 | `m1_var.py`, `q2_var_fail.py` |
| Hoàng Thắng | Mục 7 (XGBoost), Câu 3, Câu 6 | `m2_xgboost.py`, `q3_xgb_weather.py` |
| Hoàng Nam | Mục 8 (LSTM), Câu 4, Câu 5 | `m3_lstm.py`, `q4_q5_lstm.py` |

Chi tiết từng bước, phần tự tìm hiểu và checklist nộp bài: [docs/lab4/phan-cong.md](docs/lab4/phan-cong.md).
