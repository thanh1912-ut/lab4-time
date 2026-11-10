# Phân công Bài thực hành 4 (Lab 4): Dự báo chuỗi thời gian đa biến (Điện năng + Thời tiết)

**6 người làm:** Thành, Bao Dinh, Hoàng Thắng, Phạm Lê Văn Bảo, Hoàng Nam, Lê Quỳnh
(Trần Minh Sơn không nhận phần trong Lab này. Thành nhận vai dựng dự án, chuẩn bị dữ liệu và ghép báo cáo, giống Lab 3.)

**Repo:** https://github.com/thanh1912-ut/lab4-time

**Đề bài tóm tắt:** tại mỗi giờ t có vector X_t = [load_t, temp_t, humidity_t, wind_t]. Dùng 24 giờ gần nhất X_{t−23:t} để dự báo load_{t+1}. Xây dựng và so sánh 3 mô hình **VAR** (thống kê), **XGBoost** (Machine Learning), **LSTM** (Deep Learning), đánh giá bằng walk-forward với RMSE, MAE, MAPE, trả lời 6 câu hỏi phân tích, và làm bài nâng cao dự báo giao thông (METR-LA / PEMS-BAY).

---

## 0. Quy trình làm bài (đúng thứ tự mục trong đề)

| Mục đề | Nội dung | Người làm |
|---|---|---|
| 2. Dataset | Tải dữ liệu điện UCI, **ghép thêm dữ liệu thời tiết** (xem lưu ý mục 4), gộp về theo giờ, lưu 1 file dùng chung | **Thành** |
| 3. Data Understanding | Missing values, outliers, chu kỳ ngày/tuần; biểu đồ load theo thời gian, load vs temperature, ma trận tương quan | **Phạm Lê Văn Bảo** |
| 4. Preprocessing | StandardScaler; chia train 70% / validation 15% / test 15% theo thời gian, không shuffle | **Bao Dinh** |
| 5. Feature Engineering | Cửa sổ trễ L = 24 cho cả 4 biến; rolling mean 6h, rolling std 24h, load trend (chỉ cho XGBoost) | **Bao Dinh** |
| 6. Mô hình 1: VAR | ADF test, differencing, chọn bậc trễ theo AIC/BIC, huấn luyện, chỉ đánh giá cột load | **Lê Quỳnh** |
| 7. Mô hình 2: XGBoost | Input là vector phẳng từ cửa sổ lag, tune `max_depth`, `n_estimators`, early stopping trên validation | **Hoàng Thắng** |
| 8. Mô hình 3: LSTM | Tensor (mẫu, 24, 4); LSTM(64) → LSTM(32) → Dense(1); loss MSE, Adam | **Hoàng Nam** |
| 9. Backtesting | Hàm walk-forward dùng chung (Bao Dinh viết); mỗi người chạy cho mô hình của mình | **Bao Dinh** + 3 người làm mô hình |
| 10. Đánh giá | Bảng RMSE, MAE, MAPE của 3 mô hình trên cùng tập test, biểu đồ so sánh | **Bao Dinh** |
| 11. Câu hỏi 1 | Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng? | **Phạm Lê Văn Bảo** |
| 11. Câu hỏi 2 | Khi nào VAR thất bại? | **Lê Quỳnh** |
| 11. Câu hỏi 3 | XGBoost học được thông tin gì từ thời tiết? | **Hoàng Thắng** |
| 11. Câu hỏi 4 | LSTM học được chu kỳ nào (ngày, tuần)? | **Hoàng Nam** |
| 11. Câu hỏi 5 | LSTM có bị overfitting không? | **Hoàng Nam** |
| 11. Câu hỏi 6 | Khác biệt giữa Machine Learning và Deep Learning trong bài này? | **Hoàng Thắng** (lấy số LSTM của Hoàng Nam) |
| 12. Nâng cao | Traffic forecasting trên METR-LA (hoặc PEMS-BAY), dự báo 15, 30, 60 phút | **Phạm Lê Văn Bảo** |
| Báo cáo | `README.md`, `report.md`, `process.md`, `main.py` | **Thành** |

Mỗi câu trả lời phải có **số liệu hoặc hình của nhóm** làm bằng chứng, không chỉ lý thuyết.

---

## 1. Bảng phân công

| # | Thành viên | Phần việc | File code của bạn |
|---|---|---|---|
| 1 | **Thành** | Dựng dự án, README; **mục 2** (tải, gộp theo giờ, ghép thời tiết); cuối kỳ ghép `report.md`, `process.md`, viết `main.py` | `utils.py`, `s2_data.py`, `main.py` |
| 2 | **Bao Dinh** | **Mục 4, 5** (chuẩn hóa, chia tập, cửa sổ lag, rolling) + **mục 9** (hàm walk-forward dùng chung) + **mục 10** (bảng so sánh 3 mô hình) | `features.py`, `backtest.py`, `s4_s5_preprocess.py`, `s10_compare.py` |
| 3 | **Phạm Lê Văn Bảo** | **Mục 3** (Data Understanding) + **Câu 1** + **mục 12 nâng cao** (traffic) | `s3_eda.py`, `q1_weather.py`, `nc_traffic.py` |
| 4 | **Lê Quỳnh** | **Mục 6: VAR** + **Câu 2** | `m1_var.py`, `q2_var_fail.py` |
| 5 | **Hoàng Thắng** | **Mục 7: XGBoost** + **Câu 3** + **Câu 6** | `m2_xgboost.py`, `q3_xgb_weather.py` |
| 6 | **Hoàng Nam** | **Mục 8: LSTM** + **Câu 4** + **Câu 5** | `m3_lstm.py`, `q4_q5_lstm.py` |

**Vì sao chia như vậy:**
- Giữ vai giống Lab 3 để mọi người làm tiếp cái mình đã quen: Thành làm dữ liệu và báo cáo; Bao Dinh làm phần "nền" mà mọi mô hình đều import; Hoàng Thắng tiếp tục XGBoost; Lê Quỳnh tiếp tục mô hình thống kê (Lab 3 làm ARIMA, Lab 4 làm VAR); Hoàng Nam tiếp tục phần dự báo nhiều bước / mạng nơ-ron.
- Phạm Lê Văn Bảo làm EDA xong sớm (chỉ cần file dữ liệu của Thành), sau đó rảnh nhất nên nhận bài nâng cao.
- Mỗi người làm mô hình thì trả lời luôn câu hỏi về mô hình đó, vì số liệu nằm trong tay mình.

---

## 2. Lịch làm việc

Gọi **H** là hạn nộp của cô.

| Mốc | Ai | Việc phải xong |
|---|---|---|
| Ngày 1 | **Thành** | Mời 5 bạn vào repo; chạy thử tải dữ liệu |
| Ngày 2 | **Thành** | `data/processed/load_weather_hourly.csv` (**cả nhóm cần file này**) |
| Ngày 2 đến 3 | **Bao Dinh** | `src/features.py` và `src/backtest.py` chạy được (**Lê Quỳnh, Hoàng Thắng, Hoàng Nam đều import**) |
| Ngày 3 | **Phạm Lê Văn Bảo** | Xong EDA (mục 3), gửi nhóm nhận xét chu kỳ và tương quan |
| Ngày 4 đến 5 | Lê Quỳnh, Hoàng Thắng, Hoàng Nam | Mỗi mô hình có file dự đoán `pred_<tên>.csv` trên tập test |
| **H trừ 3 ngày** | 5 bạn (trừ Thành) | Nộp đủ code, hình, kết quả và 2 file nháp `.md` |
| H trừ 2 ngày | **Bao Dinh** | Bảng so sánh mục 10 từ 3 file `pred_*.csv` |
| H trừ 1 ngày | **Thành** | Ghép xong `report.md`, `process.md`, gửi nhóm đọc lại |
| H | **Thành** + 1 bạn | Clone repo về máy sạch, chạy `python src/main.py`, nộp bài |

**Trong lúc chờ:** đọc phần "Phải tự tìm hiểu" của mình và viết khung code với dữ liệu giả (ví dụ `np.random.randn(1000, 4)`), khi có file thật thì thay vào.

---

## 3. Cấu trúc dự án (theo yêu cầu của cô)

Khung thư mục, `requirements.txt`, `src/utils.py` và file code rỗng có tên từng người **đã được tạo sẵn trong repo**.

```
lab4-time/
├── README.md                       ← Thành
├── requirements.txt                ← đã có sẵn
├── data/
│   ├── raw/                        ← dữ liệu gốc, KHÔNG sửa, KHÔNG đưa lên Git (quá 100 MB)
│   │   ├── LD2011_2014.txt                 (UCI, Thành tải bằng s2_data.py)
│   │   ├── isd/085360-99999-201x.gz        (NOAA ISD, trạm sân bay Lisbon, Thành tải bằng s2_data.py)
│   │   └── metr-la.h5                      (Phạm Lê Văn Bảo tải, bài nâng cao)
│   └── processed/                  ← được đưa lên Git
│       ├── load_weather_hourly.csv         (Thành) file chính cả nhóm dùng
│       ├── pred_var.csv                    (Lê Quỳnh)
│       ├── pred_xgb.csv                    (Hoàng Thắng)
│       ├── pred_lstm.csv                   (Hoàng Nam)
│       └── compare_metrics.csv             (Bao Dinh)
├── src/
│   ├── utils.py                (Thành) đường dẫn, đọc dữ liệu, metrics dùng chung
│   ├── s2_data.py              (Thành) tải + gộp theo giờ + ghép thời tiết
│   ├── s3_eda.py               (Phạm Lê Văn Bảo)
│   ├── q1_weather.py           (Phạm Lê Văn Bảo)
│   ├── features.py             (Bao Dinh) chia tập, scaler, cửa sổ lag, rolling
│   ├── backtest.py             (Bao Dinh) walk-forward dùng chung
│   ├── s4_s5_preprocess.py     (Bao Dinh) chạy minh họa mục 4, 5
│   ├── m1_var.py               (Lê Quỳnh)
│   ├── q2_var_fail.py          (Lê Quỳnh)
│   ├── m2_xgboost.py           (Hoàng Thắng)
│   ├── q3_xgb_weather.py       (Hoàng Thắng)
│   ├── m3_lstm.py              (Hoàng Nam)
│   ├── q4_q5_lstm.py           (Hoàng Nam)
│   ├── s10_compare.py          (Bao Dinh)
│   ├── nc_traffic.py           (Phạm Lê Văn Bảo) bài nâng cao
│   └── main.py                 (Thành) chạy lần lượt tất cả
└── docs/
    └── lab4/
        ├── phan-cong.md            ← file này
        ├── report.md               ← Thành ghép
        ├── process.md              ← Thành ghép
        ├── figures/
        └── parts/                  ← bản nháp từng bạn, ghép xong thì xóa
```

### Quy ước bắt buộc cho cả nhóm

1. **Code là file `.py` trong `src/`**, chạy được từ thư mục gốc, ví dụ `python src/m2_xgboost.py`.
2. **Đọc dữ liệu qua `utils.load_data()`; chia tập, chuẩn hóa, tạo cửa sổ qua `features.py`; backtest qua `backtest.py`.** Không ai tự chia tập hay tự fit scaler riêng, để 3 mô hình so sánh được với nhau.
3. **Cấu hình chung (không ai tự đổi), nằm trong `utils.py`:**
   - Biến: `COLS = ["load", "temperature", "humidity", "wind_speed"]`, mục tiêu là `load`.
   - Cửa sổ `L = 24`, dự báo 1 bước (giờ tiếp theo).
   - Chia theo thời gian: train 70%, validation 15%, test 15%.
   - `SEED = 42` cho numpy, XGBoost, TensorFlow.
4. **StandardScaler chỉ được `fit` trên tập train**, rồi dùng `transform` cho validation và test. Fit trên toàn bộ dữ liệu là rò rỉ thông tin của tương lai.
5. **Sai số tính trên thang đo gốc** (đổi `load` ngược lại bằng `features.inverse_load()`), dùng hàm `metrics()` trong `utils.py` (RMSE, MAE, MAPE).
6. Mỗi mô hình lưu dự đoán walk-forward trên **toàn bộ tập test** vào `data/processed/pred_<tên>.csv` với 3 cột `Datetime, y_true, y_pred` (thang đo gốc). Bao Dinh đọc 3 file này để so sánh.
7. **Hình** lưu vào `docs/lab4/figures/`, `.png`, `dpi=150`, có tiêu đề, tên trục, chú thích, tên bắt đầu bằng mã phần (`s3_`, `m1_`, `q4_`, ...).
8. **Mỗi bạn nộp 2 file Markdown cho mỗi phần** vào `docs/lab4/parts/`: `<mã>_report.md` (kết quả, nhận xét, có con số cụ thể) và `<mã>_process.md` (các bước, hàm, tham số và lý do, kết quả trung gian, lỗi và cách sửa).
9. Đầu mỗi file `.py` ghi comment: tên phần, tên người làm, file làm gì.

---

## 4. Dữ liệu

### Lưu ý quan trọng: dataset UCI không có cột thời tiết

Đề ghi "Electricity Load Diagrams Dataset (UCI) gồm load, temperature, humidity, wind_speed", nhưng bộ **ElectricityLoadDiagrams20112014** của UCI **chỉ có điện năng** của 370 khách hàng ở Bồ Đào Nha. Nhóm phải **tự ghép thời tiết** cùng khu vực, cùng thời gian, và ghi rõ điều này trong báo cáo (mục Dataset).

### Điện năng (UCI)

| Thuộc tính | Giá trị |
|---|---|
| Tên | ElectricityLoadDiagrams20112014 |
| Link | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 |
| File | `LD2011_2014.txt` (giải nén từ zip), phân cách bằng `;`, **dấu thập phân là dấu phẩy** |
| Nội dung | Điện tiêu thụ (kW trong mỗi 15 phút) của 370 khách hàng `MT_001` đến `MT_370`, Bồ Đào Nha |
| Thời gian | 2011 đến 2014, mỗi 15 phút |
| Điểm đặc biệt (theo mô tả UCI) | Khách hàng đăng ký sau 2011 có giá trị 0 ở giai đoạn trước; ngày đổi giờ tháng 3 có giá trị 0 lúc 1:00 đến 2:00; ngày đổi giờ tháng 10 thì 1:00 đến 2:00 cộng dồn 2 giờ |

**Cách tạo biến `load` (đã làm):** chỉ dùng **2012-01-01 đến 2014-12-31**; giữ **318 khách hàng** có dữ liệu suốt 3 năm (bỏ 52 khách hàng đăng ký muộn hoặc ngừng sớm, để tổng không bị nhảy bậc); cộng các khách hàng tại mỗi mốc 15 phút, lấy trung bình 4 mốc trong giờ, đổi sang **MW**. 6 giờ đổi giờ bị sai được đặt NaN rồi nội suy.

### Thời tiết (NOAA ISD: số đo thật của trạm sân bay Lisbon)

Kế hoạch ban đầu dùng Open-Meteo, nhưng khi làm thì nhóm chuyển sang **số đo thật** của trạm khí tượng sân bay Lisbon trong kho NOAA Integrated Surface Database (ISD): đây là dữ liệu quan trắc, không phải mô phỏng, và tải trực tiếp được.

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | NOAA ISD trên AWS Open Data: `https://noaa-isd-pds.s3.amazonaws.com/data/{năm}/085360-99999-{năm}.gz` |
| Trạm | 085360-99999, Lisboa/Portela (mã ICAO LPPT), vĩ độ 38,767, kinh độ −9,133 |
| Tần suất gốc | Bản tin METAR mỗi 30 phút (giờ UTC) |
| Biến | `temperature` (°C), `humidity` (%, tính từ nhiệt độ và điểm sương bằng công thức Magnus), `wind_speed` (**m/s**) |
| Múi giờ | Đổi UTC sang giờ Lisbon (`Europe/Lisbon`) để khớp giờ địa phương của dữ liệu điện, rồi lấy trung bình theo giờ |

### File chính cả nhóm dùng

`data/processed/load_weather_hourly.csv` với 5 cột: `Datetime, load, temperature, humidity, wind_speed`, theo giờ, đã sắp xếp, không trùng, không thiếu.

File có **26.304 giờ** (01/01/2012 00:00 đến 31/12/2014 23:00). Chia 70/15/15: **train 18.412 giờ** (01/01/2012 đến 06/02/2014 03:00), **validation 3.946 giờ** (06/02/2014 04:00 đến 20/07/2014 13:00), **test 3.946 giờ** (20/07/2014 14:00 đến 31/12/2014 23:00).

### Nội dung `src/utils.py` (đã có trong repo)

```python
"""Hàm dùng chung cho Lab 4. Người viết: Thành."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "lab4" / "figures"
for d in (RAW, PROCESSED, FIG):
    d.mkdir(parents=True, exist_ok=True)

COLS = ["load", "temperature", "humidity", "wind_speed"]
TARGET = "load"
L = 24                      # cửa sổ 24 giờ
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15   # test = 0.15 còn lại
SEED = 42

def load_data():
    df = pd.read_csv(PROCESSED / "load_weather_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
    df.index.freq = "h"
    return df[COLS]

def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE(%)": round(mape, 2)}
```

---

## 5. Chi tiết việc của từng người

### 5.1. Thành: Dựng dự án, mục 2 (dữ liệu), ghép báo cáo

**Phần A. Dựng dự án** ✅ (đã xong, còn việc mời 5 bạn vào repo: Settings → Collaborators)

**Phần B. Mục 2: Chuẩn bị dữ liệu** ✅ trong `src/s2_data.py` (chi tiết ở `parts/s2_report.md`, `parts/s2_process.md`)
1. Tải `LD2011_2014.txt` (thử UCI trước, không được thì dùng bản sao trên GitHub, kiểm tra mã băm SHA-256).
2. Gộp điện theo giờ như mục 4, sửa 6 giờ đổi giờ.
3. Tải 3 file ISD 2012 đến 2014, đọc định dạng cố định độ rộng, bỏ giá trị có mã chất lượng xấu, tính độ ẩm, đổi giờ UTC sang giờ Lisbon, trung bình theo giờ.
4. Điền giờ thiếu thời tiết: đoạn ngắn nội suy tuyến tính; đoạn dài (10 đến 24 giờ) lấy trung bình cùng giờ hôm trước và hôm sau.
5. Lưu `data/processed/load_weather_hourly.csv`; hình `s2_4bien.png`, `s2_dst.png`.

**Phần C. README.md** (4 phần như các Lab trước): giới thiệu, Data Introduction (2 nguồn dữ liệu và lý do phải ghép thời tiết), môi trường, cài đặt và chạy (gồm bước tải dữ liệu vì `data/raw` không có trên Git), bảng phân công.

**Phần D. Ghép báo cáo cuối kỳ**
1. `src/main.py` chạy theo thứ tự: `s2_data` → `s3_eda` → `q1_weather` → `s4_s5_preprocess` → `m1_var` → `q2_var_fail` → `m2_xgboost` → `q3_xgb_weather` → `m3_lstm` → `q4_q5_lstm` → `s10_compare` (`nc_traffic` chạy riêng vì nặng).
2. Ghép `docs/lab4/report.md` theo đúng thứ tự mục của đề:

| Mục trong report.md | Lấy từ |
|---|---|
| 1. Mục tiêu | Thành |
| 2. Dataset và bài toán | Thành |
| 3. Data Understanding | Phạm Lê Văn Bảo |
| 4. Preprocessing | Bao Dinh |
| 5. Feature Engineering | Bao Dinh |
| 6. VAR | Lê Quỳnh |
| 7. XGBoost | Hoàng Thắng |
| 8. LSTM | Hoàng Nam |
| 9. Backtesting | Bao Dinh (cách làm) + 3 người làm mô hình (kết quả) |
| 10. Đánh giá và so sánh | Bao Dinh |
| 11. Câu hỏi 1 đến 6 | Bảo (1), Quỳnh (2), Thắng (3, 6), Nam (4, 5) |
| 12. Nâng cao: traffic | Phạm Lê Văn Bảo |
| 13. Kết luận | Thành |

3. Ghép `process.md`: sơ đồ luồng dữ liệu, rồi process của từng phần, cuối cùng bảng phân công.
4. Kiểm tra số liệu không mâu thuẫn (RMSE trong phần của từng mô hình phải khớp bảng của Bao Dinh), đánh số hình/bảng, sửa đường dẫn ảnh thành `figures/...`, xóa `parts/`, chạy lại trên máy sạch rồi nộp.

**Phải tự tìm hiểu**
- Đọc file CSV có `sep=";"`, `decimal=","`; `resample`; gọi API bằng `requests` và đọc JSON thành DataFrame.
- Đổi giờ mùa hè ở châu Âu (cuối tháng 3, cuối tháng 10) ảnh hưởng dữ liệu theo giờ thế nào.
- `merge`/`join` theo index thời gian.

**Kết quả phải nộp**
- [x] `src/s2_data.py` chạy được, `data/processed/load_weather_hourly.csv`
- [x] Hình `s2_4bien.png`, `s2_dst.png`
- [x] `parts/s2_report.md`: 2 nguồn dữ liệu, số giờ, khoảng thời gian, các giờ đã sửa, thống kê mô tả 4 biến
- [x] `parts/s2_process.md`
- [x] README, khung `report.md`, `process.md` (mục 1, 2 đã viết), `main.py`
- [ ] Cuối kỳ: ghép `report.md`, `process.md` từ phần của 5 bạn

---

### 5.2. Bao Dinh: Mục 4, 5 (tiền xử lý, đặc trưng), mục 9 (walk-forward), mục 10 (so sánh)

> 3 bạn làm mô hình đều import `features.py` và `backtest.py`. Làm 2 file này **sớm nhất** (ngày 2 đến 3), phần minh họa và so sánh làm sau.

**Phần A. `src/features.py`** (khung gợi ý, bạn hoàn thiện và tự kiểm tra kích thước)

```python
"""Chia tập, chuẩn hóa, tạo cửa sổ cho Lab 4. Người viết: Bao Dinh."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import COLS, L, TRAIN_RATIO, VAL_RATIO

def split_index(n):
    """Vị trí cắt train / val / test theo thời gian."""
    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    return i1, i2

def fit_scale(df):
    """Fit StandardScaler CHỈ trên train, transform toàn bộ. Trả về df_scaled, scaler, (i1, i2)."""
    i1, i2 = split_index(len(df))
    scaler = StandardScaler().fit(df[COLS].iloc[:i1])
    scaled = pd.DataFrame(scaler.transform(df[COLS]), index=df.index, columns=COLS)
    return scaled, scaler, (i1, i2)

def inverse_load(z, scaler):
    return np.asarray(z) * scaler.scale_[0] + scaler.mean_[0]

def make_windows(scaled, L=L):
    """X[k] = 24 giờ (t-23..t) x 4 biến, y[k] = load(t+1). Trả về X (n, L, 4), y (n,), mốc thời gian của y."""
    arr = scaled[COLS].to_numpy()
    X = np.stack([arr[i - L:i] for i in range(L, len(arr))])
    y = arr[L:, 0]
    return X, y, scaled.index[L:]

def add_rolling(scaled):
    """Đặc trưng thêm cho XGBoost, tính đến thời điểm t (dự báo t+1 nên không rò rỉ)."""
    load = scaled["load"]
    return pd.DataFrame({
        "load_roll_mean_6": load.rolling(6).mean(),
        "load_roll_std_24": load.rolling(24).std(),
        "load_trend_24": load - load.shift(24),
    }, index=scaled.index)
```

Lưu ý quan trọng: **mỗi mẫu được xếp vào train/val/test theo thời điểm của y (giờ cần dự báo)**. Mẫu đầu tiên của tập test vẫn được dùng 24 giờ cuối của validation làm đầu vào, như vậy không rò rỉ vì đó là quá khứ.

**Phần B. `src/backtest.py`** (walk-forward dùng chung cho 3 mô hình)

```python
"""Walk-forward validation dùng chung. Người viết: Bao Dinh."""
import numpy as np

def walk_forward(fit_fn, predict_fn, scaled, start, end, refit_every=None):
    """Dự báo lần lượt từng giờ t trong [start, end) chỉ dùng dữ liệu trước t.
    fit_fn(history_df) -> model ; predict_fn(model, history_df) -> load(t) đã chuẩn hóa.
    Sau mỗi bước, giá trị THẬT của giờ t được thêm vào lịch sử (đúng như đề:
    train -> predict t+1 -> thêm vào chuỗi -> predict t+2 ...).
    refit_every: số giờ giữa 2 lần huấn luyện lại (None = không huấn luyện lại)."""
    model = fit_fn(scaled.iloc[:start])
    preds = []
    for k, t in enumerate(range(start, end)):
        if refit_every and k > 0 and k % refit_every == 0:
            model = fit_fn(scaled.iloc[:t])
        preds.append(predict_fn(model, scaled.iloc[:t]))
    return np.array(preds)
```

**Phần C. Mục 4, 5 minh họa** trong `src/s4_s5_preprocess.py`
1. `fit_scale(load_data())`; in mean và std của từng biến trên train trước/sau chuẩn hóa (sau phải xấp xỉ 0 và 1); in mean/std trên test (không đúng 0 và 1, giải thích vì sao vẫn đúng).
2. In số mẫu và mốc thời gian đầu/cuối của train/val/test; vẽ 3 đoạn 3 màu (`s4_split.png`).
3. `make_windows`: in `X.shape` (mong đợi `(n, 24, 4)`), vẽ 1 cửa sổ mẫu 4 biến (`s5_window.png`).
4. Bản phẳng cho XGBoost: `X.reshape(n, 24*4)` + `add_rolling`; in tên 99 cột (96 lag + 3 rolling).

**Phần D. Mục 10** trong `src/s10_compare.py` (làm sau khi 3 bạn nộp `pred_*.csv`)
1. Đọc 3 file, kiểm tra cùng mốc thời gian; thêm baseline **naive** (`load_t` dự báo cho `t+1`) và **seasonal naive** (`load_{t−23}`, cùng giờ hôm qua).
2. Bảng RMSE, MAE, MAPE của 5 dòng; lưu `data/processed/compare_metrics.csv`.
3. Vẽ: 1 tuần trong test, thực tế và 3 mô hình (`s10_1tuan.png`); biểu đồ cột 3 chỉ số (`s10_metrics.png`); sai số theo giờ trong ngày của 3 mô hình (`s10_loi_theo_gio.png`).

**Phải tự tìm hiểu**
- StandardScaler; vì sao chỉ fit trên train (rò rỉ dữ liệu).
- Cửa sổ trượt cho chuỗi đa biến; dạng dữ liệu 3D cho LSTM và dạng phẳng cho XGBoost.
- Walk-forward validation (còn gọi rolling-origin evaluation) khác chia train/test 1 lần thế nào.

**Kết quả phải nộp**
- [ ] `src/features.py`, `backtest.py`, `s4_s5_preprocess.py`, `s10_compare.py` chạy được
- [ ] `data/processed/compare_metrics.csv`
- [ ] Hình `s4_split.png`, `s5_window.png`, `s10_1tuan.png`, `s10_metrics.png`, `s10_loi_theo_gio.png`
- [ ] `parts/s4_s5_report.md`, `parts/s9_s10_report.md` (cách walk-forward, bảng so sánh, 5 đến 7 dòng nhận xét mô hình nào tốt nhất và vì sao)
- [ ] 2 file `_process.md` tương ứng

---

### 5.3. Phạm Lê Văn Bảo: Mục 3 (Data Understanding), Câu 1, Mục 12 (nâng cao traffic)

**Phần A. Mục 3** trong `src/s3_eda.py` (chỉ cần file của Thành)
1. **Missing values:** `isna().sum()` từng biến; nếu Thành đã nội suy thì ghi số giờ đã nội suy.
2. **Outliers:** IQR và Z-score trên từng biến; với `load`, áp lên phần dư sau khi trừ trung bình theo giờ-trong-tuần để không nhầm đỉnh mùa vụ là outlier; đánh dấu chấm đỏ (`s3_outlier.png`).
3. **Seasonality:** trung bình load theo giờ trong ngày và theo thứ trong tuần (`s3_chu_ky.png`); ACF của load 200 lags, chỉ ra đỉnh ở lag 24 và 168 (`s3_acf.png`).
4. **Trực quan hóa theo đề:** load theo thời gian (`s3_load.png`); load vs temperature dạng scatter, tô màu theo mùa (`s3_load_vs_temp.png`); heatmap tương quan 4 biến (`s3_corr.png`).

**Phần B. Câu 1** trong `src/q1_weather.py`: biến thời tiết nào ảnh hưởng mạnh nhất?
1. Tương quan Pearson và Spearman của load với từng biến thời tiết.
2. Vì quan hệ load và nhiệt độ thường hình chữ U (lạnh thì sưởi, nóng thì điều hòa), tính thêm tương quan theo mùa, hoặc tương quan với `|temperature − 18|`.
3. Tương quan có trễ: load(t) với temp(t−k), k = 0..24 (`q1_lag_corr.png`).
4. Loại bỏ ảnh hưởng chu kỳ ngày: tương quan giữa phần dư của load và nhiệt độ sau khi trừ trung bình theo giờ trong ngày.
5. Kết luận biến nào mạnh nhất, đối chiếu với feature importance của Hoàng Thắng (Câu 3).

**Phần C. Mục 12 nâng cao: Traffic forecasting** trong `src/nc_traffic.py`
1. Dữ liệu **METR-LA**: 207 cảm biến tốc độ xe ở Los Angeles, mỗi 5 phút, khoảng 4 tháng năm 2012. Tải `metr-la.h5` theo hướng dẫn trong repo GitHub của paper DCRNN (liyaguang/DCRNN), đọc bằng `pd.read_hdf` (cần thư viện `tables`). Nếu không tải được thì dùng PEMS-BAY cùng nguồn.
2. Giá trị 0 là cảm biến lỗi: coi là thiếu, không tính vào sai số (masked MAE/RMSE/MAPE).
3. Chia theo thời gian 70/10/20 (như paper), chuẩn hóa fit trên train.
4. Bài toán: 12 bước quá khứ (60 phút) của 207 trạm → dự báo bước 3, 6, 12 (15, 30, 60 phút).
5. Mô hình: baseline giá trị cuối cùng, baseline trung bình lịch sử theo giờ-trong-tuần, và 1 mô hình học sâu (LSTM hoặc GRU với đầu ra 207 × 3). Máy yếu thì dùng 1 tháng dữ liệu hoặc 50 trạm, ghi rõ.
6. Bảng MAE, RMSE, MAPE theo 3 tầm xa (`nc_horizon.png`), 1 trạm ví dụ thực tế vs dự báo (`nc_vidu.png`); so với bảng kết quả trong paper DCRNN và giải thích vì sao DCRNN tốt hơn (dùng đồ thị kết nối giữa các trạm).

**Phải tự tìm hiểu**
- Tương quan Pearson, Spearman; ACF; IQR, Z-score.
- Quan hệ nhiệt độ và điện năng (heating/cooling degree).
- Đọc lướt paper DCRNN: ý tưởng đồ thị cảm biến và diffusion convolution; metric có mask.

**Kết quả phải nộp**
- [ ] `src/s3_eda.py`, `q1_weather.py`, `nc_traffic.py` chạy được
- [ ] Hình `s3_*.png` (7 hình), `q1_lag_corr.png`, `nc_horizon.png`, `nc_vidu.png`
- [ ] `parts/s3_report.md`: bảng missing, outlier, chu kỳ có số cụ thể, 3 hình theo đề kèm nhận xét
- [ ] `parts/q1_report.md`: bảng tương quan, trả lời câu 1 trong 5 đến 7 dòng
- [ ] `parts/nc_report.md`: mô tả dữ liệu, bảng sai số 3 tầm xa, so với paper
- [ ] 3 file `_process.md` tương ứng

---

### 5.4. Lê Quỳnh: Mục 6 (VAR) và Câu 2

> Dùng `features.py` và `backtest.py` của Bao Dinh. Trong lúc chờ: làm trước ADF test và chọn bậc trễ trên file của Thành.

**Phần A. Mục 6** trong `src/m1_var.py`
1. **Kiểm tra tính dừng:** `adfuller` (statsmodels) cho từng biến trên tập train; bảng thống kê ADF, p-value. Biến nào p > 0,05 thì lấy sai phân bậc 1; kiểm tra lại ADF sau sai phân. Nên dùng **cùng 1 cách sai phân cho cả 4 biến** để dễ đảo ngược.
2. **Chọn bậc trễ:** `VAR(train).select_order(maxlags=48)`; bảng AIC, BIC theo p; ghi p theo AIC và theo BIC, chọn 1 và giải thích (BIC thường chọn p nhỏ hơn).
3. **Huấn luyện** `VAR(train).fit(p)`; in tóm tắt hệ số phương trình của `load`.
4. **Walk-forward trên test** bằng `walk_forward`: `predict_fn` gọi `results.forecast(y=history[-p:], steps=1)`, lấy cột `load`; nếu đã sai phân thì cộng lại `load_t` để ra mức. Huấn luyện lại mỗi 168 giờ (`refit_every=168`) nếu đủ thời gian.
5. Đổi về thang đo gốc, tính sai số, lưu `data/processed/pred_var.csv`; vẽ 1 tuần thực tế vs dự báo (`m1_var_1tuan.png`).
6. (Nên làm) Kiểm định Granger: thời tiết có "gây ra" load theo nghĩa Granger không (`results.test_causality("load", ["temperature"])`).

**Phần B. Câu 2** trong `src/q2_var_fail.py`: khi nào VAR thất bại?
1. Sai số VAR theo giờ trong ngày và theo thứ trong tuần (`q2_loi_theo_gio.png`): VAR tuyến tính với p nhỏ khó bắt chu kỳ tuần (168 giờ).
2. Thử p = 2, 24, 48 (và 168 nếu máy chịu được): sai số và số tham số (4 × 4 × p) tăng thế nào.
3. Những giờ/ngày VAR sai nhiều nhất (top 10): có phải ngày lễ, ngày đổi giờ, đợt nóng/lạnh bất thường không.
4. Dự báo nhiều bước (24 giờ liền không cập nhật giá trị thật) để thấy VAR nhanh chóng hội tụ về trung bình.
5. Kết luận: VAR thất bại khi quan hệ phi tuyến (load vs nhiệt độ hình chữ U), chu kỳ dài cần p rất lớn, chuỗi không dừng/có gãy cấu trúc, có sự kiện đặc biệt, số biến lớn (số tham số tăng theo bình phương số biến).

**Phải tự tìm hiểu**
- Mô hình VAR(p): X_t = Σ A_i X_{t−i} + ε_t; số tham số; khác ARIMA một biến thế nào.
- ADF test, sai phân và cách đảo ngược sai phân khi dự báo.
- AIC, BIC; kiểm định Granger.
- `statsmodels.tsa.api.VAR`, `select_order`, `forecast`.

**Kết quả phải nộp**
- [ ] `src/m1_var.py`, `q2_var_fail.py` chạy được
- [ ] `data/processed/pred_var.csv`
- [ ] Hình `m1_var_1tuan.png`, `q2_loi_theo_gio.png` (+ hình theo p)
- [ ] `parts/m1_report.md`: bảng ADF, bảng AIC/BIC, p đã chọn, RMSE/MAE/MAPE trên test
- [ ] `parts/q2_report.md`: số liệu các thí nghiệm, trả lời câu 2 trong 6 đến 8 dòng
- [ ] 2 file `_process.md` tương ứng

---

### 5.5. Hoàng Thắng: Mục 7 (XGBoost), Câu 3, Câu 6

> Dùng `features.py` và `backtest.py` của Bao Dinh. Câu 6 cần số của Hoàng Nam (LSTM).

**Phần A. Mục 7** trong `src/m2_xgboost.py`
1. Đầu vào: `make_windows` → `X.reshape(n, 96)`, đặt tên cột `load_lag1..24, temperature_lag1..24, ...`; ghép 3 cột `add_rolling` tại thời điểm t. Đầu ra `load(t+1)`. Chia mẫu theo `split_index`.
2. **Tune** `max_depth ∈ {3, 5, 7, 9}` và `n_estimators` (đặt lớn, ví dụ 2000, để early stopping tự dừng) với `learning_rate=0.05`, `early_stopping_rounds=50`, `eval_set=[(X_val, y_val)]`. Bảng RMSE validation theo `max_depth`, ghi `best_iteration`.
3. Mô hình tốt nhất: walk-forward trên test (huấn luyện lại mỗi 168 giờ nếu đủ thời gian, gộp train + val khi huấn luyện lại; ghi rõ cách làm).
4. Đổi về thang đo gốc, tính sai số, lưu `pred_xgb.csv`; vẽ 1 tuần (`m2_xgb_1tuan.png`) và đường cong RMSE train/val theo số cây (`m2_learning_curve.png`).

**Phần B. Câu 3** trong `src/q3_xgb_weather.py`: XGBoost học được gì từ thời tiết?
1. Feature importance (gain) của 99 cột; **cộng gain theo nhóm biến** (load, temperature, humidity, wind, rolling) và theo độ trễ (`q3_importance_nhom.png`, `q3_importance_lag.png`).
2. **Ablation:** huấn luyện lại chỉ với lag của load (bỏ 3 biến thời tiết), so sai số với mô hình đầy đủ. Chênh lệch chính là phần đóng góp của thời tiết.
3. (Nên làm) Partial dependence của `temperature_lag1` (`sklearn.inspection.PartialDependenceDisplay`) để thấy hình dạng quan hệ nhiệt độ và load (`q3_pdp_temp.png`).

**Phần C. Câu 6:** khác biệt giữa ML và DL trong bài này
So XGBoost và LSTM theo: sai số trên test, thời gian huấn luyện, số tham số/độ phức tạp, cần chuẩn hóa hay không, cần tự tạo đặc trưng (rolling, lag phẳng) hay mô hình tự học từ chuỗi 3D, khả năng giải thích (feature importance vs hộp đen), độ ổn định khi chạy lại với seed khác, lượng dữ liệu cần. Viết thành bảng + 5 đến 7 dòng kết luận.

**Phải tự tìm hiểu**
- Dữ liệu chuỗi đa biến dạng phẳng cho mô hình cây; early stopping với `eval_set`.
- Feature importance `gain`; ablation study; partial dependence plot.

**Kết quả phải nộp**
- [x] `src/m2_xgboost.py`, `q3_xgb_weather.py` chạy được
- [x] `data/processed/pred_xgb.csv`
- [x] Hình `m2_xgb_1tuan.png`, `m2_learning_curve.png`, `q3_importance_nhom.png`, `q3_importance_lag.png` (+ `q3_pdp_temp.png`)
- [x] `parts/m2_report.md`: bảng tune, tham số chọn, sai số test
- [x] `parts/q3_report.md`: importance theo nhóm, kết quả ablation, trả lời câu 3
- [x] `parts/q6_report.md`: bảng so sánh ML vs DL, trả lời câu 6
- [x] 3 file `_process.md` tương ứng

---

### 5.6. Hoàng Nam: Mục 8 (LSTM), Câu 4, Câu 5

> Dùng `features.py` và `backtest.py` của Bao Dinh. Gửi số sai số và thời gian huấn luyện cho Hoàng Thắng để làm Câu 6.

**Phần A. Mục 8** trong `src/m3_lstm.py`
1. Dữ liệu: `make_windows` → `X` dạng `(n, 24, 4)`, `y` dạng `(n,)`; chia theo `split_index`.
2. Kiến trúc đúng đề (Keras):
   ```python
   model = keras.Sequential([
       keras.Input(shape=(24, 4)),
       keras.layers.LSTM(64, return_sequences=True),
       keras.layers.LSTM(32),
       keras.layers.Dense(1),
   ])
   model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))
   ```
3. Huấn luyện: `epochs=50`, `batch_size=64`, `validation_data=(X_val, y_val)`, `EarlyStopping(patience=5, restore_best_weights=True)`; đặt `keras.utils.set_random_seed(42)`. Lưu lịch sử loss.
4. Walk-forward trên test: mỗi giờ lấy 24 giờ thật gần nhất làm đầu vào (không huấn luyện lại, ghi rõ). Dự báo cả tập test 1 lần bằng `model.predict(X_test)` cho kết quả giống hệt vì đầu vào đều là giá trị thật, có thể dùng để chạy nhanh; ghi trong báo cáo.
5. Đổi về thang đo gốc, tính sai số, lưu `pred_lstm.csv`; vẽ 1 tuần (`m3_lstm_1tuan.png`). Ghi số tham số (`model.count_params()`) và thời gian huấn luyện.

**Phần B. Câu 4** trong `src/q4_q5_lstm.py`: LSTM học được chu kỳ nào?
1. Cửa sổ chỉ 24 giờ nên về lý thuyết LSTM **thấy được chu kỳ ngày nhưng không thấy trực tiếp chu kỳ tuần**. Kiểm chứng: sai số theo giờ trong ngày và theo thứ trong tuần (`q4_loi_theo_thu.png`); nếu sai số thứ Hai và Chủ nhật cao hơn (chuyển giữa cuối tuần và ngày thường) là dấu hiệu không học được chu kỳ tuần.
2. Thí nghiệm: huấn luyện lại với cửa sổ L = 168 (hoặc thêm 2 cột giờ-trong-ngày, thứ-trong-tuần dạng sin/cos) và so sai số. Ghi rõ đây là thí nghiệm thêm, mô hình chính vẫn L = 24.
3. (Nên làm) Dự báo 48 giờ liên tiếp bằng cách đưa dự báo vào lại đầu vào (thời tiết lấy giá trị thật) để xem đường dự báo có giữ được hình dạng ngày không (`q4_48h.png`).

**Phần C. Câu 5:** LSTM có overfitting không?
1. Đường cong loss train và validation theo epoch (`q5_loss.png`); epoch tốt nhất và khoảng cách 2 đường.
2. So RMSE trên train, validation, test (thang đo gốc).
3. Thử bỏ EarlyStopping (chạy đủ 50 epoch) và thêm `Dropout(0.2)` để so sánh; kết luận có overfitting không và cách chống.

**Phải tự tìm hiểu**
- LSTM: cổng quên, cổng vào, cổng ra; `return_sequences`; dữ liệu 3D (mẫu, bước thời gian, biến).
- Keras (TensorFlow): `Sequential`, `fit`, `EarlyStopping`, `History`. **Lưu ý:** TensorFlow thường chỉ hỗ trợ đến Python 3.12, nên dùng Python 3.11 hoặc 3.12 (hoặc Google Colab). Muốn dùng PyTorch thì giữ đúng kiến trúc và báo nhóm.
- Overfitting, early stopping, dropout.

**Kết quả phải nộp**
- [ ] `src/m3_lstm.py`, `q4_q5_lstm.py` chạy được
- [ ] `data/processed/pred_lstm.csv`
- [ ] Hình `m3_lstm_1tuan.png`, `q4_loi_theo_thu.png`, `q5_loss.png` (+ `q4_48h.png`)
- [ ] `parts/m3_report.md`: kiến trúc, số tham số, epoch dừng, thời gian, sai số test
- [ ] `parts/q4_report.md`, `parts/q5_report.md`: số liệu thí nghiệm, trả lời mỗi câu 5 đến 7 dòng
- [ ] 3 file `_process.md` tương ứng

---

## 6. Ai phụ thuộc vào ai

```
Thành (load_weather_hourly.csv, utils.py)
   │
   ├──► Phạm Lê Văn Bảo (EDA, Câu 1) ──► gửi nhận xét chu kỳ/tương quan cho cả nhóm
   │                      └──► bài nâng cao traffic (độc lập, dữ liệu riêng)
   ▼
Bao Dinh (features.py, backtest.py)
   │
   ├──► Lê Quỳnh (VAR, Câu 2)      ──► pred_var.csv  ─┐
   ├──► Hoàng Thắng (XGBoost, Câu 3) ─► pred_xgb.csv  ─┼──► Bao Dinh (mục 10 so sánh)
   └──► Hoàng Nam (LSTM, Câu 4, 5)  ──► pred_lstm.csv ─┘          │
                │                                                 │
                └── số liệu LSTM ──► Hoàng Thắng (Câu 6)          ▼
                                                    Thành ghép report.md, process.md
```
