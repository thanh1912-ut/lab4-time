# Câu hỏi 6: Khác biệt giữa Machine Learning và Deep Learning trong bài này (Hoàng Thắng)

## 1. Bảng so sánh tổng hợp giữa XGBoost (ML) và LSTM (DL)

Dựa trên quá trình triển khai thực tế trên bộ dữ liệu điện năng và thời tiết Bồ Đào Nha (26.304 giờ), chúng tôi tổng hợp sự khác biệt cốt lõi giữa hai trường phái mô hình:

| Tiêu chí so sánh | XGBoost (Machine Learning) | LSTM (Deep Learning) | Phân tích & Đánh giá |
|---|---|---|---|
| **Cấu trúc dữ liệu đầu vào** | Vector phẳng 2D: ma trận $(N, 99)$ gồm 96 lag và 3 đặc trưng cuộn | Tensor 3D: hình dạng $(N, 24, 4)$ (mẫu, bước thời gian, số biến) | XGBoost xem mỗi bước trễ là một biến độc lập; LSTM duy trì cấu trúc không gian - thời gian tự nhiên qua các bước |
| **Kỹ nghệ đặc trưng (Feature Engineering)** | Bắt buộc phải tạo thủ công (trích xuất lag 1..24, rolling mean 6h, rolling std 24h, trend 24h) | Tự động trích xuất biểu diễn (Representation Learning) qua cơ chế cổng nhớ (Forget, Input, Output Gates) | XGBoost phụ thuộc nhiều vào kiến thức miền (domain knowledge); LSTM tự học liên kết theo chiều thời gian |
| **Yêu cầu chuẩn hóa dữ liệu** | Không phụ thuộc vào thang đo (Scale-invariant) vì thuật toán phân nhánh theo điểm chia (split) | Rất nhạy cảm, bắt buộc phải chuẩn hóa (StandardScaler) để tránh nổ/biến mất gradient trong các hàm phi tuyến ($\tanh$, $\text{sigmoid}$) | Với LSTM, chuẩn hóa không đúng (như fit trên toàn bộ tập) sẽ gây rò rỉ thông tin nghiêm trọng |
| **Số lượng tham số & Độ phức tạp** | Tham số phân bố theo các lá cây trong 153 cây tăng cường (mô hình gọn nhẹ, không lưu ma trận trọng số dày đặc) | **30.113 tham số** có thể huấn luyện:<br>- LSTM(64): 17.664 tham số<br>- LSTM(32): 12.416 tham số<br>- Dense(1): 33 tham số | LSTM có số tham số lớn hơn nhiều, đòi hỏi không gian bộ nhớ và tài nguyên tính toán liên tục |
| **Thời gian huấn luyện** | **Rất nhanh:** khoảng 2 giây cho một lần học 153 cây (số đo ở mục 7); walk-forward có refit 24 lần khoảng 72–75 giây | **Chậm hơn nhiều:** khoảng 2–6 phút mỗi lần học trên CPU 4 nhân (mô hình chính 26 epoch, tối đa 50); chọn patience với 3 seed tốn khoảng 19 phút | Một lần học XGBoost nhanh hơn LSTM vài chục lần (khoảng 2 giây so với khoảng 140 giây), nên refit hằng tuần rất rẻ với XGBoost nhưng đắt với LSTM |
| **Sai số kiểm thử (cùng luật: học một lần trên train, không refit)** | **RMSE: 6,17 MW**<br>**MAE: 3,81 MW**<br>**MAPE: 1,89 %**<br>(bản refit hằng tuần: 6,07 / 3,71 / 1,86) | **RMSE: 6,14 MW**<br>**MAE: 4,39 MW**<br>**MAPE: 2,25 %**<br>(seed 42; trung bình 3 seed RMSE 6,02 ± 0,11) | RMSE gần như bằng nhau (chênh 0,03 MW, nhỏ hơn dao động giữa các seed của LSTM). XGBoost tốt hơn rõ ở MAE và MAPE: phần lớn các giờ sai ít hơn, nhưng có một số giờ sai lớn nên RMSE không thấp hơn |
| **Khả năng giải thích (Explainability / XAI)** | **Rất cao (White-box / Glass-box):** Dễ dàng tính Feature Gain, độ quan trọng theo nhóm, vẽ Partial Dependence Plot (PDP) | **Thấp (Black-box):** Các ma trận trọng số và trạng thái ẩn $h_t, c_t$ rất trừu tượng, khó phân tách rạch ròi đóng góp của thời tiết | XGBoost thân thiện với báo cáo nghiệp vụ và giám sát vận hành hệ thống điện thực tế |
| **Độ ổn định với seed và cách huấn luyện** | **Rất ổn định:** không dùng subsample nên chạy lại cho đúng cùng kết quả | **Nhạy:** đổi seed làm test RMSE dao động 5,92–6,14 MW (patience 10); không xáo trộn mẫu train thì RMSE lên 13,96 MW | Muốn kết luận về LSTM phải chạy nhiều seed và ghi rõ cách huấn luyện |
| **Nhu cầu về khối lượng dữ liệu** | Hoạt động tốt với 18.388 mẫu train | Với cùng 18.388 mẫu, LSTM vẫn đạt sai số ngang XGBoost khi train đúng cách | Ở quy mô 3 năm dữ liệu theo giờ, lượng dữ liệu không phải điểm nghẽn chính; điểm khác là chi phí và độ nhạy khi huấn luyện |

---

## 2. Nhận xét kết luận (5 – 7 dòng)

1. Khi cả hai được huấn luyện theo cùng một luật (học trên train, chọn cấu hình bằng validation, không refit trên test), **XGBoost và LSTM có độ chính xác gần như ngang nhau**: RMSE 6,17 so với 6,14 MW. XGBoost tốt hơn ở MAE (3,81 so với 4,39 MW) và MAPE (1,89% so với 2,25%).
2. **Khác biệt lớn nhất là chi phí và độ nhạy khi huấn luyện.** XGBoost học trong khoảng 2 giây và cho cùng kết quả mỗi lần chạy. LSTM mất vài phút mỗi lần, và kết quả phụ thuộc mạnh vào cách train: không xáo trộn mẫu thì RMSE là 13,96 MW, xáo trộn thì 6,14 MW, đổi seed thì dao động khoảng ±0,11 MW.
3. **XGBoost dễ giải thích hơn**: Gain theo nhóm biến và PDP ở Câu 3 cho thấy trực tiếp vai trò của `load_lag24` và ngưỡng nhiệt độ; LSTM không có công cụ tương đương đơn giản.
4. **LSTM không cần tự tạo đặc trưng rolling**, nhưng đổi lại phải chọn patience, xáo trộn mẫu và chạy nhiều seed mới có kết quả tin được.
5. Với bài toán dự báo 1 giờ tới trên dữ liệu này, **XGBoost là lựa chọn thực dụng hơn vì cùng độ chính xác nhưng rẻ hơn, ổn định hơn và dễ giải thích hơn**, chứ không phải vì LSTM kém chính xác.
