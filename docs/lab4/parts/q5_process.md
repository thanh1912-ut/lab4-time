# Quy trình Câu 5: Overfitting — Hoàng Nam

1. Giữ nguyên cách chia thời gian và scaler fit trên train; dùng cùng mẫu cho ba cấu hình.
2. Mạng chính LSTM(64) → LSTM(32) → Dense(1), Adam(0.001), MSE, batch/epoch theo CLI, xáo trộn cửa sổ train mỗi epoch (seed 42).
3. EarlyStopping theo val_loss, patience=10 (chọn ở mục 8 bằng validation), khôi phục trọng số tốt nhất. Ghi số epoch đã chạy và epoch chọn bằng validation.
4. Chạy `lstm_no_early_stopping` đủ số epoch đặt trước, đánh giá trọng số epoch cuối để quan sát tác động huấn luyện kéo dài.
5. Chạy `lstm_dropout`: Dropout(0.2) sau mỗi lớp LSTM, giữ các cấu hình còn lại và EarlyStopping.
6. Với trọng số chọn của mỗi cấu hình, dự báo train/validation/test, inverse về MW rồi tính cùng `utils.metrics()`.
7. So sánh train/validation MSE ở epoch validation tốt nhất và epoch cuối của biến thể không EarlyStopping; kiểm tra trung bình tối đa 5 epoch cuối. Chỉ xem phân kỳ rõ khi có ít nhất 5 epoch theo dõi, train giảm trên 1%, validation tăng trên 1% và các trung bình cuối cùng chiều; quy tắc này chỉ mô tả, không là kiểm định thống kê.
8. Lưu `q5_comparison.csv`, `q5_split_metrics.csv`, `q5_training_trajectory.csv`, `q5_overfitting_diagnostic.json`, lịch sử loss, metadata, mô hình Keras và hình `q5_loss.png`.

## Cách chạy

```bash
python src/q4_q5_lstm.py
```

Mặc định 50 epoch tối đa, batch 64, seed 42; biến thể không EarlyStopping chạy đủ 50 epoch.
Nếu dùng --epochs khác, bảng/metadata phản ánh đúng số epoch thực tế, không gọi đó là kết quả đủ 50 epoch.
--force bỏ cache và huấn luyện lại. Tập test không dùng để chọn epoch, dropout hoặc mô hình chính.

## Lưu ý diễn giải

Loss train trong lịch sử là trung bình batch trong lúc cập nhật trọng số; RMSE train được tính lại
với trọng số cuối/chọn, nên hai đại lượng không phải cùng một phép đo. Dropout chỉ hoạt động khi train.
Chênh lệch train/validation còn chịu ảnh hưởng chuyển mùa. Chạy một seed và chưa kiểm định thống kê.
