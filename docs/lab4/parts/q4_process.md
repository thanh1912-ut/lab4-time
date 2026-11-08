# Quy trình Câu 4: Chu kỳ LSTM — Hoàng Nam

1. Dùng lại mô hình chính, lịch sử, dự báo thật và metadata trong `data/processed/lstm/`.
2. Gọi `prepare_data(calendar=True)` thêm sin/cos giờ-trong-ngày và thứ-trong-tuần tại các mốc quá khứ.
3. Giữ cùng mốc mục tiêu và cùng train/validation/test; scaler bốn biến gốc chỉ fit trên train.
4. Huấn luyện `lstm_calendar` cùng số epoch tối đa, batch, seed, loss, optimizer và EarlyStopping của mô hình chính.
5. Đổi dự báo về MW, đối chiếu đúng 3,946 thời điểm test; tính RMSE/MAE/MAPE, so thêm seasonal naive dùng load cùng giờ hôm trước.
6. Nhóm sai số theo giờ/thứ; từng nhóm lưu số quan sát `n` để diễn giải phạm vi.
7. Lưu `q4_comparison.csv`, `q4_errors_hour.csv`, `q4_errors_weekday.csv`, hình `q4_*.png`.
8. Chạy thêm 48 giờ đệ quy đầu tập test; đưa load dự báo vào cửa sổ, giữ thời tiết tương lai thực tế và ghi rõ giả định.

## Cách chạy

```bash
python src/m3_lstm.py
python src/q4_q5_lstm.py
```

`--force` huấn luyện lại; `--epochs` và `--batch-size` điều chỉnh cấu hình được lưu trong metadata.
Mặc định 50 epoch tối đa, batch 64, seed 42. Cấu hình chốt trước khi xem test.
Đặc trưng lịch tính từ timestamp đã biết, không lấy load tương lai.
Một seed chưa cho biết độ ổn định; chưa kiểm định thống kê hay tách ảnh hưởng số tham số.
