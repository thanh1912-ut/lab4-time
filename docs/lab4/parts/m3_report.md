# Mục 8. Mô hình LSTM — Hoàng Nam

## Bài toán và dữ liệu

Dùng 24 giờ quan sát gần nhất của 4 biến `load`, `temperature`, `humidity`, `wind_speed` để dự báo load giờ tiếp theo. Dữ liệu chung có 26,304 giờ. StandardScaler chỉ fit trên train. Mỗi cửa sổ được xếp vào tập theo thời điểm của y, nên đầu validation/test được dùng lịch sử trước ranh giới.

| Tập | samples | start | end |
| --- | --- | --- | --- |
| train | 18388 | 2012-01-02 00:00:00 | 2014-02-06 03:00:00 |
| validation | 3946 | 2014-02-06 04:00:00 | 2014-07-20 13:00:00 |
| test | 3946 | 2014-07-20 14:00:00 | 2014-12-31 23:00:00 |

## Kiến trúc và huấn luyện

`Input(24,4) → LSTM(64, return_sequences=True) → LSTM(32) → Dense(1)`.

- Số tham số: **30,113**.
- Loss MSE trên dữ liệu chuẩn hóa; Adam learning rate 0,001; seed 42.
- Tối đa 50 epoch, batch 64. **Xáo trộn thứ tự các cửa sổ train mỗi epoch**; validation và test giữ thứ tự thời gian.
  Mỗi cửa sổ đã chứa đủ 24 giờ quá khứ của chính nó nên xáo trộn không làm lộ tương lai; nó chỉ tránh việc mỗi epoch học từ 2012 đến hết 2014 theo đúng thứ tự,
  vì khi đó các batch cuối mỗi epoch luôn là mùa đông đầu 2014 và có thể kéo trọng số lệch về giai đoạn này.
- EarlyStopping theo `val_loss`, patience 10 (chọn trên validation, xem bên dưới), khôi phục trọng số tốt nhất.
- Thực chạy **26 epoch**, chọn epoch **16**; thời gian `fit`: **139.509 giây**, thiết bị /physical_device:CPU:0. Thời gian không gồm đọc dữ liệu, dự báo, xuất hình.
- Python 3.13.18, TensorFlow 2.20.0, numpy 2.5.3, pandas 3.0.5.

## Chọn patience trên validation

Thử patience 5, 10, mỗi giá trị 3 seed (42, 43, 44).
Chọn patience có RMSE validation trung bình thấp nhất: patience 5: 6.17 MW, patience 10: 5.87 MW
→ chọn **patience 10**; mô hình chính là seed 42. Cột test chỉ để báo cáo, không dùng để chọn.

| patience | seed | epochs_run | selected_epoch | train_seconds | train_RMSE | val_RMSE | test_RMSE | test_MAE | test_MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 42 | 21 | 16 | 128.8 | 6.02 | 5.9 | 6.14 | 4.39 | 2.25 |
| 5 | 43 | 25 | 20 | 193.1 | 5.97 | 6.23 | 6.58 | 4.58 | 2.31 |
| 5 | 44 | 19 | 14 | 127.0 | 5.94 | 6.38 | 6.46 | 4.7 | 2.48 |
| 10 | 42 | 26 | 16 | 139.5 | 6.02 | 5.9 | 6.14 | 4.39 | 2.25 |
| 10 | 43 | 50 | 43 | 333.0 | 5.01 | 5.76 | 5.92 | 4.27 | 2.22 |
| 10 | 44 | 36 | 26 | 228.8 | 5.56 | 5.94 | 6.0 | 4.25 | 2.14 |

Với patience 10, test RMSE qua 3 seed: **6.02 ± 0.11 MW**
(trung bình ± độ lệch chuẩn), cho biết mức dao động do khởi tạo ngẫu nhiên.

## Backtesting và kết quả

Walk-forward một bước với trọng số cố định: tại mỗi giờ test, đầu vào là 24 giờ **giá trị thật quá khứ**. Không refit trên test. Dự báo các cửa sổ theo batch cho cùng phép đánh giá vì trọng số cố định. Đây không phải dự báo đệ quy nhiều ngày từ một mốc duy nhất.

Đã đảo chuẩn hóa về MW trước khi tính chỉ số bằng `utils.metrics()`:

| Tập | RMSE | MAE | MAPE(%) |
| --- | --- | --- | --- |
| train | 6.02 | 4.25 | 2.16 |
| validation | 5.9 | 4.26 | 2.34 |
| test | 6.14 | 4.39 | 2.25 |

So với hai baseline trên đúng cùng các giờ test:

| Model | RMSE | MAE | MAPE(%) |
| --- | --- | --- | --- |
| Naive | 25.5 | 16.34 | 8.78 |
| Seasonal naive (24h) | 13.42 | 7.68 | 3.82 |
| LSTM (L=24) | 6.14 | 4.39 | 2.25 |

RMSE LSTM **thấp hơn seasonal naive 24 giờ 54.25%**
(6.14 so với 13.42 MW). LSTM tận dụng được cả lịch sử 24 giờ của 4 biến nên vượt baseline chỉ dùng load cùng giờ hôm trước.
Không thay đổi checkpoint hoặc cấu hình dựa trên kết quả test này.

![LSTM một tuần](../figures/m3_lstm_1tuan.png)

Đầu ra: `data/processed/pred_lstm.csv`, 3946 dòng, ba cột `Datetime,y_true,y_pred`. Số liệu sinh từ mô hình huấn luyện trên dữ liệu thật.

## Giới hạn

Chỉ 3 seed nên độ lệch chuẩn còn thô. Chọn patience và checkpoint theo validation; test chỉ dùng báo cáo. Dữ liệu chung đã được nhóm nội suy/điền thiếu từ trước, có phương pháp dùng quan sát tương lai; thí nghiệm này không chứng minh toàn bộ quy trình dữ liệu thô vận hành trực tuyến. Kết quả áp dụng cho tập test và cách tạo dữ liệu đã nêu.
