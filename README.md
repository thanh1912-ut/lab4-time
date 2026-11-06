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

Bộ dữ liệu điện của UCI **không có biến thời tiết**, nên nhóm ghép thêm số đo thời tiết thật của trạm sân bay Lisbon, cùng khu vực (Bồ Đào Nha) và cùng thời gian.

| Nguồn | Nội dung | Link |
|---|---|---|
| UCI ElectricityLoadDiagrams20112014 | Điện tiêu thụ 15 phút/lần của 370 khách hàng ở Bồ Đào Nha, 2011 đến 2014 | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 |
| NOAA Integrated Surface Database (ISD) | Bản tin quan trắc 30 phút/lần của trạm Lisboa/Portela (LPPT, mã 085360-99999): nhiệt độ, điểm sương, tốc độ gió | https://registry.opendata.aws/noaa-isd/ |

File dùng chung sau khi ghép: `data/processed/load_weather_hourly.csv`, **26.304 giờ** từ 01/01/2012 00:00 đến 31/12/2014 23:00, không thiếu.

| Biến | Đơn vị | Ý nghĩa | Trung bình | Nhỏ nhất | Lớn nhất |
|---|---|---|---|---|---|
| `load` | MW | Công suất tiêu thụ trung bình trong giờ của 318 khách hàng có dữ liệu suốt 3 năm | 203,3 | 75,7 | 445,8 |
| `temperature` | °C | Nhiệt độ không khí | 16,4 | 0,0 | 40,0 |
| `humidity` | % | Độ ẩm tương đối (tính từ nhiệt độ và điểm sương) | 72,8 | 10,1 | 100,0 |
| `wind_speed` | m/s | Tốc độ gió | 4,1 | 0,0 | 15,2 |

Đã xử lý: 52 khách hàng đăng ký muộn hoặc ngừng sớm (bỏ), 6 giờ đổi giờ bị sai trong dữ liệu điện (nội suy), 104 giờ thiếu bản tin thời tiết (điền). Chi tiết: `docs/lab4/parts/s2_report.md`.

## 3. Môi trường

- Python 3.11 hoặc 3.12 (TensorFlow chưa chắc hỗ trợ bản mới hơn). Mục 2 đã chạy thử với Python 3.13, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2.
- Thư viện: xem `requirements.txt`

## 4. Cài đặt và chạy

```bash
git clone https://github.com/thanh1912-ut/lab4-time.git
cd lab4-time
pip install -r requirements.txt
python src/main.py          # chạy toàn bộ (dùng data/processed/load_weather_hourly.csv đã có sẵn)
python src/main.py --rebuild-data   # tải lại dữ liệu gốc (khoảng 710 MB, không có trên Git) và tạo lại file đã ghép
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
