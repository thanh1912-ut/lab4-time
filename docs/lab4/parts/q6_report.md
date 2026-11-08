# Câu hỏi 6: Khác biệt giữa Machine Learning và Deep Learning trong bài này (Hoàng Thắng)

## 1. Bảng so sánh tổng hợp giữa XGBoost (ML) và LSTM (DL)

Dựa trên quá trình triển khai thực tế trên bộ dữ liệu điện năng và thời tiết Bồ Đào Nha (26.304 giờ), chúng tôi tổng hợp sự khác biệt cốt lõi giữa hai trường phái mô hình:

| Tiêu chí so sánh | XGBoost (Machine Learning) | LSTM (Deep Learning) | Phân tích & Đánh giá |
|---|---|---|---|
| **Cấu trúc dữ liệu đầu vào** | Vector phẳng 2D: ma trận $(N, 99)$ gồm 96 lag và 3 đặc trưng cuộn | Tensor 3D: hình dạng $(N, 24, 4)$ (mẫu, bước thời gian, số biến) | XGBoost xem mỗi bước trễ là một biến độc lập; LSTM duy trì cấu trúc không gian - thời gian tự nhiên qua các bước |
| **Kỹ nghệ đặc trưng (Feature Engineering)** | Bắt buộc phải tạo thủ công (trích xuất lag 1..24, rolling mean 6h, rolling std 24h, trend 24h) | Tự động trích xuất biểu diễn (Representation Learning) qua cơ chế cổng nhớ (Forget, Input, Output Gates) | XGBoost phụ thuộc nhiều vào kiến thức miền (domain knowledge); LSTM tự học liên kết theo chiều thời gian |
| **Yêu cầu chuẩn hóa dữ liệu** | Không phụ thuộc vào thang đo (Scale-invariant) vì thuật toán phân nhánh theo điểm chia (split) | Rất nhạy cảm, bắt buộc phải chuẩn hóa (StandardScaler) để tránh nổ/biến mất gradient trong các hàm phi tuyến ($\tanh$, $\text{sigmoid}$) | Với LSTM, chuẩn hóa không đúng (như fit trên toàn bộ tập) sẽ gây rò rỉ thông tin nghiêm trọng |
| **Số lượng tham số & Độ phức tạp** | Tham số phân bố theo các lá cây trong 153 cây tăng cường (mô hình gọn nhẹ, không lưu ma trận trọng số dày đặc) | **30.113 tham số** có thể huấn luyện:<br>- LSTM(64): 17.664 tham số<br>- LSTM(32): 12.416 tham số<br>- Dense(1): 33 tham số | LSTM có số tham số lớn hơn nhiều, đòi hỏi không gian bộ nhớ và tài nguyên tính toán liên tục |
| **Thời gian huấn luyện** | **Rất nhanh:** ~1,95 giây (huấn luyện đơn) và **75,2 giây** cho toàn bộ 24 lần walk-forward refit | **Chậm:** ~3 đến 5 phút trên CPU cho 50 epochs (kể cả khi dùng Early Stopping) | XGBoost tận dụng tối ưu đa luồng CPU (`n_jobs=-1`), nhanh hơn LSTM từ 5 đến 10 lần |
| **Sai số kiểm thử (Test Metrics)** | **RMSE: 6,07 MW**<br>**MAE: 3,71 MW**<br>**MAPE: 1,86 %** | **RMSE: ~6,5 – 7,5 MW**<br>**MAE: ~4,2 – 4,8 MW**<br>**MAPE: ~2,1 – 2,4 %** | Nhờ có đặc trưng chỉ định rõ `load_lag24` và `load_trend_24`, XGBoost hội tụ tới nghiệm chính xác hơn khi kích thước mẫu ở mức vừa phải |
| **Khả năng giải thích (Explainability / XAI)** | **Rất cao (White-box / Glass-box):** Dễ dàng tính Feature Gain, độ quan trọng theo nhóm, vẽ Partial Dependence Plot (PDP) | **Thấp (Black-box):** Các ma trận trọng số và trạng thái ẩn $h_t, c_t$ rất trừu tượng, khó phân tách rạch ròi đóng góp của thời tiết | XGBoost thân thiện với báo cáo nghiệp vụ và giám sát vận hành hệ thống điện thực tế |
| **Độ ổn định với seed ngẫu nhiên** | **Rất ổn định:** Kết quả chia nhánh của cây là tất định (deterministic) khi cố định `random_state` | Nhạy cảm hơn với khởi tạo trọng số ngẫu nhiên ban đầu (Glorot/Xavier initialization) và xáo trộn batch | LSTM có phương sai nghiệm giữa các lần chạy cao hơn |
| **Nhu cầu về khối lượng dữ liệu** | Hoạt động rất tốt trên tập dữ liệu quy mô vừa (18.000 mẫu train) | Cần tập dữ liệu lớn hơn nhiều để phát huy tối đa khả năng khái quát hóa mà không bị overfitting | Kích thước dữ liệu 3 năm theo giờ là "vừa vặn" cho XGBoost nhưng hơi ít đối với Deep Learning phức tạp |

---

## 2. Nhận xét kết luận (5 – 7 dòng)

1. Trong bài toán dự báo chuỗi thời gian phụ tải 1 bước với dữ liệu dạng bảng có tính chu kỳ ngày/tuần mạnh mẽ, **XGBoost kết hợp kỹ nghệ đặc trưng miền (đặc biệt là lag 24 và trend 24) mang lại độ chính xác vượt trội hơn LSTM**, với sai số MAPE chỉ 1,86% (so với mức > 2,1% của mạng nơ-ron).
2. **Ưu thế lớn nhất của XGBoost là tốc độ và khả năng giải thích minh bạch**: thời gian huấn luyện nhanh hơn gần 10 lần và có thể trực quan hóa chính xác quy luật phản ứng phi tuyến của phụ tải theo nhiệt độ thông qua Partial Dependence Plot.
3. Tuy nhiên, **LSTM sở hữu ưu thế tự động hóa biểu diễn dữ liệu chuỗi 3D** mà không cần chuyên gia con người phải tự tay thiết kế các biến thống kê cuộn (rolling statistics), tạo tiền đề lý tưởng cho các bài toán dự báo đa biến cực lớn hoặc chuỗi có quan hệ phụ thuộc dài hạn phức tạp.
4. Đối với các bài toán vận hành hệ thống năng lượng thực tế yêu cầu tái huấn luyện liên tục (walk-forward/online learning) và cần tính giải trình cao đối với cơ quan điều độ lưới điện, **mô hình học máy dạng cây (XGBoost) hiện vẫn là giải pháp thực dụng và tối ưu hơn so với mạng học sâu (LSTM)**.
