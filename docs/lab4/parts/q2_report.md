# Câu 2 - Khi nào VAR thất bại? (Lê Quỳnh)

Mô hình chính: VAR(26), không sai phân, walk-forward trên test (20/07 - 31/12/2014).

## 1. Sai số theo giờ và theo thứ
| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAE | 4.0700 | 2.1400 | 2.3500 | 2.0800 | 2.0200 | 1.7300 | 1.7000 | 2.2200 | 4.9200 | 7.7500 | 8.1800 | 6.5700 | 4.7000 | 4.8600 | 4.4700 | 3.9300 | 4.0500 | 4.4000 | 4.0300 | 3.6200 | 3.7300 | 3.7400 | 5.0900 | 5.0700 |
| MAPE | 3.1400 | 1.9100 | 2.1400 | 1.9900 | 1.9200 | 1.6000 | 1.4000 | 1.6700 | 3.2100 | 3.4900 | 3.2400 | 2.5100 | 1.7200 | 1.7800 | 1.6300 | 1.4000 | 1.4800 | 1.5300 | 1.4300 | 1.4100 | 1.4500 | 1.4700 | 2.2600 | 2.9700 |

| index | Thứ 2 | Thứ 3 | Thứ 4 | Thứ 5 | Thứ 6 | Thứ 7 | CN |
|---|---|---|---|---|---|---|---|
| MAE | 4.0600 | 3.9200 | 4.0500 | 4.1700 | 4.1200 | 4.1300 | 3.9700 |
| MAPE | 2.0600 | 1.8700 | 1.9800 | 2.2300 | 1.9600 | 2.0800 | 2.0400 |

Hình: `figures/q2_loi_theo_gio.png`. ACF phần dư: lag 24 = -0.233, lag 168 = 0.197.

## 2. Thay đổi bậc trễ p
| p | số tham số (4x4xp) | RMSE | MAE | MAPE(%) | thời gian (s) |
|---|---|---|---|---|---|
| 2.00 | 32.00 | 16.42 | 10.65 | 5.49 | 1.00 |
| 24.00 | 384.00 | 9.22 | 6.10 | 3.17 | 6.00 |
| 26.00 | 416.00 | 6.05 | 4.06 | 2.03 | 0.00 |
| 48.00 | 768.00 | 5.92 | 4.00 | 2.02 | 9.00 |

Hình: `figures/q2_theo_p.png`.

## 3. Top 10 giờ sai nhiều nhất
| giờ | y_true | y_pred | err | thứ | temperature | ngày lễ | đổi giờ | nhiệt độ cực đoan |
|---|---|---|---|---|---|---|---|---|
| 2014-12-26 00:00 | 113.0500 | 73.6000 | 39.4500 | Thứ 6 | 9.5000 |  |  |  |
| 2014-09-11 10:00 | 325.9700 | 286.8700 | 39.1000 | Thứ 5 | 22.6700 |  |  |  |
| 2014-12-25 09:00 | 91.4900 | 129.7700 | -38.2900 | Thứ 5 | 8.0000 | Giáng sinh |  | lạnh (<P5) |
| 2014-12-26 09:00 | 183.8700 | 149.1400 | 34.7300 | Thứ 6 | 9.5000 |  |  |  |
| 2014-12-31 19:00 | 173.8000 | 207.7800 | -33.9900 | Thứ 4 | 9.5000 |  |  |  |
| 2014-08-03 09:00 | 256.5100 | 222.6100 | 33.9000 | CN | 20.0000 |  |  |  |
| 2014-12-27 00:00 | 128.3300 | 161.8400 | -33.5100 | Thứ 7 | 10.6700 |  |  |  |
| 2014-11-01 11:00 | 278.2800 | 245.5200 | 32.7600 | Thứ 7 | 20.0000 | Lễ Các Thánh |  |  |
| 2014-12-24 19:00 | 180.5500 | 212.4000 | -31.8500 | Thứ 4 | 11.5000 |  |  |  |
| 2014-09-11 11:00 | 313.7900 | 345.5200 | -31.7400 | Thứ 5 | 22.6700 |  |  |  |

7/10 giờ sai nhiều nhất rơi vào ngày nghỉ lễ: 6 giờ trong kỳ nghỉ cuối năm 24 đến 31/12 (Giáng sinh, Năm mới) và 1 giờ vào lễ Các Thánh 1/11. Không có giờ nào rơi vào ngày đổi giờ 26/10.

## 4. Dự báo nhiều bước (không cập nhật giá trị thật, 100 mốc ngẫu nhiên trong test)
RMSE: h = 1 là 5.3 MW, h = 24 là 9.3 MW, h = 72 là 18.4 MW.
Biên độ dao động trong ngày của dự báo so với thực tế: ngày 1 là 99%, ngày 3 là 98%.
Hình: `figures/q2_nhieu_buoc.png`.

## Trả lời Câu 2

1. Chu kỳ ngày: sai số tập trung vào lúc tải đổi nhanh, MAE cao nhất lúc 10h (8,18 MW) và 9h (7,75 MW), thấp nhất lúc 6h (1,70 MW). Chu kỳ tuần: MAE gần như đều giữa các thứ (3,92 đến 4,17 MW), nên VAR(26) không kém riêng ngày nào, nhưng ACF phần dư vẫn còn lag 24 = -0,23 và lag 168 = 0,20, tức vẫn còn một ít cấu trúc ngày và tuần chưa được mô hình hóa.
2. Bậc trễ: p = 2 cho RMSE 16,42 MW (32 tham số), p = 24 cho 9,22 MW (384 tham số), p = 26 cho 6,05 MW (416 tham số), p = 48 cho 5,92 MW (768 tham số). VAR cần p tối thiểu khoảng 26 để bắt chu kỳ ngày; tăng thêm lên 48 chỉ cải thiện RMSE khoảng 2% nhưng gần gấp đôi số tham số.
3. Giờ sai nhiều nhất: 7/10 giờ nằm trong kỳ nghỉ lễ (6 giờ trong 24 đến 31/12 quanh Giáng sinh và Năm mới, 1 giờ vào lễ Các Thánh 1/11); VAR chỉ dựa vào 26 giờ trước đó nên không biết nhu cầu điện thay đổi vào ngày nghỉ. Không giờ nào trong top 10 rơi vào ngày đổi giờ 26/10.
4. Dự báo nhiều bước (không cập nhật giá trị thật): RMSE tăng từ 5,3 MW (h = 1) lên 9,3 MW (h = 24) và 18,4 MW (h = 72). Biên độ dao động ngày 3 vẫn đạt 98% thực tế, nên VAR(26) giữ được hình dạng chu kỳ ngày, nhưng sai lệch pha và mức nền tích lũy dần nên sai số tăng.
5. Kết luận: VAR thất bại khi (a) p quá nhỏ để chứa chu kỳ ngày (p = 2 sai số gấp gần 3 lần p = 26), (b) có sự kiện đặc biệt như ngày lễ, (c) cần dự báo xa mà không có dữ liệu thật để sửa sai. Hai nguyên nhân còn lại trong lý thuyết là quan hệ phi tuyến (load theo nhiệt độ hình chữ U) và số tham số tăng theo bình phương số biến; nhóm không kiểm chứng trực tiếp hai điểm này.