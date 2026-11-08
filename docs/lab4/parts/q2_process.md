# Quy trình Câu 2 - Khi nào VAR thất bại? (Lê Quỳnh)

## Các bước, hàm và tham số
| Thí nghiệm | Cách làm | Lý do |
|---|---|---|
| Sai số theo giờ và theo thứ | Đọc `pred_var.csv`, `groupby(hour)` và `groupby(dayofweek)` lấy MAE, MAPE; ACF phần dư tại lag 24 và 168 | VAR tuyến tính với p nhỏ khó bắt chu kỳ tuần (168 giờ) |
| Đổi bậc trễ p | p = 2, 24, 48 (thêm 168 bằng `--p168`), chạy lại walk-forward bằng `m1_var.run_walk_forward` | Xem sai số giảm và số tham số 4 x 4 x p tăng thế nào |
| Top 10 giờ sai nhiều nhất | `nlargest(10, abs_err)`, đối chiếu ngày lễ Bồ Đào Nha, ngày đổi giờ 26/10/2014, nhiệt độ ngoài P5 và P95 | Kiểm tra có phải sự kiện đặc biệt không |
| Dự báo nhiều bước | VAR(p) huấn luyện đến hết validation, dự báo 72 giờ liền từ 100 mốc ngẫu nhiên (`SEED = 42`) không cập nhật giá trị thật; thống kê RMSE theo h và biên độ dao động ngày 3 so với thực tế | Cho thấy dự báo hội tụ về trung bình |

Mọi sai số tính trên thang đo gốc (MW).

## Kết quả trung gian
Các bảng lưu ở `data/processed/q2_theo_p.csv`, `q2_top10.csv`; số liệu tổng hợp trong `q2_report.md`.

## Lỗi gặp phải và cách sửa

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| Nhận xét tự sinh trong `q2_report.md` không khớp số liệu: ghi top 10 chỉ có 2/10 giờ bất thường, ghi "dự báo dần phẳng về trung bình", ghi MAE cao nhất vào thứ 5 như thể khác biệt lớn | Script chỉ đếm ngày lễ chính thức (không tính cả kỳ nghỉ cuối năm 24 đến 31/12) và dùng câu mẫu theo giả thuyết lý thuyết, chưa đối chiếu với kết quả thật | Đọc lại bảng số liệu và 3 hình, viết lại phần trả lời bằng tay: 7/10 giờ rơi vào kỳ nghỉ lễ; biên độ dao động ngày 3 vẫn đạt 98% thực tế nên dự báo không phẳng về trung bình; MAE giữa các thứ chỉ chênh khoảng 6% |
| Giả thuyết "VAR hội tụ nhanh về trung bình" trong đề không thấy trong kết quả | VAR(26) có các hệ số lớn ở lag 24 và 25 nên giữ được chu kỳ ngày khi dự báo nhiều bước | Ghi đúng kết quả quan sát được; sai số tăng (5,3 MW lúc h = 1 lên 18,4 MW lúc h = 72) vì lệch pha và mức nền tích lũy, không phải vì dự báo phẳng |
