# Câu hỏi 6: Quy trình so sánh Machine Learning và Deep Learning (Hoàng Thắng)

**File liên quan:** `src/q3_xgb_weather.py`, `src/m2_xgboost.py` (Mục 7 & Câu 3), đối chiếu thông số LSTM quy định tại Mục 5.6 (`phan-cong.md`)

## 1. Phương pháp & Căn cứ so sánh

Để đưa ra so sánh khách quan và khoa học giữa đại diện Machine Learning (XGBoost) và Deep Learning (LSTM), chúng tôi xây dựng khung đánh giá đa chiều dựa trên các bằng chứng thực nghiệm cụ thể thu thập được trong quá trình chạy mô hình:

1. **Về cấu trúc và số lượng tham số:**
   - **XGBoost:** Mô hình tối ưu gồm 153 cây, độ sâu tối đa 7. Số lượng nút lá tối đa cho mỗi cây là $2^7 = 128$ lá.
   - **LSTM:** Kiến trúc quy định bởi đề bài:
     - Lớp 1: $\text{LSTM}(64, \text{return\_sequences}=\text{True})$ với đầu vào $(24, 4)$:
       $$\text{Params}_1 = 4 \times [(units + input\_dim) \times units + units] = 4 \times [(64 + 4) \times 64 + 64] = 17.664$$
     - Lớp 2: $\text{LSTM}(32)$:
       $$\text{Params}_2 = 4 \times [(32 + 64) \times 32 + 32] = 12.416$$
     - Lớp 3: $\text{Dense}(1)$:
       $$\text{Params}_3 = 32 \times 1 + 1 = 33$$
     - Tổng cộng mạng LSTM có đúng **30.113 tham số** cần tối ưu hóa.

2. **Về thời gian tính toán và tài nguyên:**
   - Ghi nhận thời gian đo bằng hàm `time.time()` trong Python:
     - XGBoost huấn luyện đơn trên 18.388 mẫu mất 1,95 giây.
     - Toàn bộ quá trình Walk-forward (chạy 3.946 bước, huấn luyện lại 24 lần trên toàn bộ tập dữ liệu mở rộng dần) hoàn thành trong 75,2 giây trên CPU thông thường.
     - Ngược lại, việc huấn luyện một mạng LSTM với 50 epochs (dù có Early Stopping ở epoch ~20) mất tối thiểu 3 đến 5 phút trên CPU, và việc tái huấn luyện 24 lần theo kiểu Walk-forward sẽ tốn hàng chục phút nếu không có GPU chuyên dụng.

3. **Về tính minh bạch và diễn giải (Explainability):**
   - XGBoost cung cấp cơ chế tính toán Gain chính xác đến từng thuộc tính và cho phép vẽ đường cong ảnh hưởng biên (PDP).
   - LSTM về bản chất là ma trận trọng số phi tuyến lồng nhau, rất khó để tách bạch bao nhiêu phần trăm đóng góp thuộc về nhiệt độ, độ ẩm hay chu kỳ phụ tải.

---

## 2. Quy trình trích xuất và tổng hợp

1. Chạy hoàn chỉnh `src/m2_xgboost.py` để lấy số liệu thực tế về RMSE, MAE, MAPE, thời gian huấn luyện và số cây của XGBoost.
2. Chạy `src/q3_xgb_weather.py` để trích xuất Feature Importance và PDP.
3. Tổng hợp các đại lượng định lượng vào bảng ma trận 9 tiêu chí so sánh.
4. Rút ra 4 nhận xét kết luận súc tích về bài học kinh nghiệm lựa chọn mô hình trong các bài toán dự báo chuỗi thời gian công nghiệp.
