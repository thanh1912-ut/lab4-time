# Quy trình mục 12: Dự báo giao thông METR-LA

**Người thực hiện:** Phạm Lê Văn Bảo

**File huấn luyện:** `src/nc_traffic.py`  
**File đánh giá:** `src/nc_evaluate.py`

## 1. Mục tiêu và đầu vào

Thực hiện bài toán dự báo tốc độ giao thông trên METR-LA bằng hai phương pháp cơ sở (Last Value, Historical Average) và mô hình GRU. Sử dụng 12 bước quan sát trước đó để dự báo tốc độ tại 207 cảm biến ở các mốc 15, 30 và 60 phút.

**Đầu vào:** `data/raw/metr-la.h5`

Dữ liệu được đọc bằng `pandas.read_hdf()`, sử dụng thư viện `tables` để hỗ trợ định dạng HDF5.

Kích thước dữ liệu sau khi đọc: `(34272, 207)`, tương ứng 34.272 thời điểm và 207 cảm biến, với khoảng cách giữa hai lần đo là 5 phút.

## 2. Tiền xử lý dữ liệu

### 2.1. Chia dữ liệu

Chia dữ liệu theo thứ tự thời gian, không shuffle:

| Tập | Số thời điểm |
|---|---:|
| Train | 23.990 |
| Validation | 3.427 |
| Test | 6.855 |

Tỷ lệ tương ứng là 70/10/20, theo cấu hình của bài nâng cao.

### 2.2. Xử lý dữ liệu thiếu

Giá trị tốc độ bằng 0 được xem là dữ liệu không hợp lệ của cảm biến.

Quy trình xử lý:

1. Thay các giá trị 0 bằng NaN trong dữ liệu dùng làm đầu vào.
2. Điền giá trị thiếu bằng quan sát hợp lệ gần nhất ở quá khứ (forward fill).
3. Với các vị trí chưa có quan sát hợp lệ trước đó, sử dụng trung bình của cảm biến tính trên tập Train.
4. Giữ nguyên giá trị thực tế ban đầu để tính sai số có mask, bỏ qua các vị trí không hợp lệ.

Forward fill chỉ sử dụng quan sát trước thời điểm hiện tại, không lấy giá trị tương lai để điền dữ liệu.

### 2.3. Chuẩn hóa

Trung bình và độ lệch chuẩn được tính từ các giá trị hợp lệ trong tập Train:

- Train mean: 58,5843
- Train std: 12,8229

Sử dụng hai thống kê này để chuẩn hóa đầu vào Train, Validation và Test. Không tính lại mean và std trên Validation hoặc Test nhằm tránh rò rỉ thông tin.

Khi đánh giá, kết quả dự báo được đưa về đơn vị tốc độ gốc (mph).

## 3. Tạo mẫu dự báo

Mỗi mẫu gồm:

- Đầu vào: 12 bước thời gian liên tiếp, mỗi bước có 207 giá trị tốc độ.
- Đầu ra: tốc độ của 207 cảm biến tại các bước tương lai 3, 6 và 12.

Như vậy, kích thước đầu vào của một mẫu là `(12, 207)` và đầu ra là `(3, 207)`.

Số mẫu được tạo:

| Tập | Số mẫu |
|---|---:|
| Train | 23.967 |
| Validation | 3.404 |

Các mẫu không đủ số bước quá khứ hoặc bước dự báo tương lai được loại bỏ. Việc tạo mẫu thực hiện riêng trong từng tập để không đưa nhãn tương lai từ tập khác vào quá trình huấn luyện.

## 4. Xây dựng các phương pháp dự báo

### 4.1. Last Value

Lấy giá trị tốc độ hợp lệ cuối cùng trong cửa sổ đầu vào làm dự báo cho cả ba mốc tương lai.

Phương pháp này đơn giản, không cần huấn luyện, được sử dụng làm mốc so sánh.

### 4.2. Historical Average

Tính tốc độ trung bình lịch sử cho từng cảm biến theo vị trí thời gian trong tuần. Các giá trị trung bình chỉ được xây dựng từ dữ liệu Train.

Khi dự báo, mô hình sử dụng trung bình tương ứng với thời điểm cần dự báo.

### 4.3. GRU

Sử dụng mạng GRU để học quan hệ theo thời gian giữa các quan sát tốc độ.

Cấu hình huấn luyện:

| Tham số | Giá trị |
|---|---|
| Input steps | 12 |
| Input features | 207 |
| Forecast steps | 3, 6, 12 |
| Hidden size | 128 |
| GRU layers | 1 |
| Batch size | 64 |
| Learning rate | 0,001 |
| Optimizer | Adam |
| Max epochs | 20 |
| Early Stopping patience | 5 |
| Random seed | 42 |

Đầu ra GRU được chuyển thành ma trận dự báo `(3, 207)`, tương ứng ba tầm dự báo cho tất cả cảm biến.

Thiết bị tính toán được lựa chọn tự động: sử dụng CUDA khi PyTorch nhận diện được GPU tương thích, nếu không thì sử dụng CPU.

## 5. Huấn luyện và lựa chọn mô hình

Chạy từ thư mục gốc dự án:

```bash
python src/nc_traffic.py
```

Mô hình được huấn luyện trên Train và theo dõi loss trên Validation sau mỗi epoch. Khi Validation Loss cải thiện, trọng số được lưu vào:

`models/nc_gru_best.pt`

Kết quả huấn luyện:

| Thông tin | Giá trị |
|---|---:|
| Epoch tốt nhất | 13 |
| Validation Loss tốt nhất | 0,256091 |
| Epoch dừng | 18 |
| Train Loss tại epoch 18 | 0,2280 |
| Validation Loss tại epoch 18 | 0,2609 |

Từ epoch 14 đến 18, Train Loss tiếp tục giảm nhưng Validation Loss không cải thiện. Early Stopping được kích hoạt sau 5 epoch, giữ lại checkpoint tốt nhất ở epoch 13 thay vì sử dụng trọng số của epoch cuối.

## 6. Đánh giá trên tập Test

Sau khi hoàn thành huấn luyện, chạy:

```bash
python src/nc_evaluate.py
```

File này nạp checkpoint đã lưu, thực hiện dự báo trên Test và không huấn luyện lại GRU.

Các chỉ số sử dụng:

- **MAE:** trung bình sai số tuyệt đối.
- **RMSE:** căn bậc hai của trung bình bình phương sai số.
- **MAPE:** trung bình sai số phần trăm tuyệt đối.

Cả ba chỉ số sử dụng mask để loại bỏ các vị trí có giá trị thực tế bằng 0 hoặc không hợp lệ.

Kết quả GRU trên Test:

| Tầm dự báo | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| 15 phút | 3,3970 | 6,1699 | 9,4553 |
| 30 phút | 3,6015 | 6,7160 | 10,2635 |
| 60 phút | 3,9969 | 7,5864 | 11,8432 |

Kết quả của Last Value và Historical Average được đánh giá trên cùng tập Test, sử dụng cùng cách tính chỉ số. Bảng đầy đủ được lưu tại `data/processed/nc_test_metrics.csv`.

## 7. Xuất kết quả và trực quan hóa

| Đầu ra | Nội dung |
|---|---|
| `models/nc_gru_best.pt` | Checkpoint GRU tốt nhất |
| `data/processed/nc_test_metrics.csv` | MAE, RMSE, MAPE của ba mô hình trên Test |
| `docs/lab4/figures/nc_horizon.png` | So sánh ba chỉ số theo tầm dự báo |
| `docs/lab4/figures/nc_vidu.png` | Tốc độ thực tế và dự báo GRU tại cảm biến 767541 |

Biểu đồ `nc_horizon.png` thể hiện kết quả so sánh ba mô hình ở các tầm dự báo 15, 30 và 60 phút.

Biểu đồ `nc_vidu.png` sử dụng cảm biến 767541 để minh họa dự báo trước 60 phút. Kết quả cho thấy GRU bám được xu hướng chung nhưng chưa dự báo chính xác toàn bộ những lần giảm tốc đột ngột.

## 8. Vấn đề gặp phải và cách xử lý

| Vấn đề | Cách xử lý |
|---|---|
| Giá trị 0 trong dữ liệu cảm biến không phản ánh tốc độ hợp lệ | Loại khỏi phép tính sai số bằng mask; điền giá trị thiếu trong đầu vào |
| Last Value ban đầu sử dụng dữ liệu thô trong khi GRU sử dụng dữ liệu đã điền thiếu | Đồng bộ Last Value để sử dụng chuỗi đầu vào đã xử lý, bảo đảm cách so sánh nhất quán |
| Train Loss tiếp tục giảm nhưng Validation Loss không cải thiện | Sử dụng Early Stopping và checkpoint tốt nhất |
| Huấn luyện GRU mất thời gian khi cần kiểm tra lại Test | Tách `nc_evaluate.py` để nạp checkpoint và đánh giá mà không train lại |
| Máy của các thành viên có cấu hình khác nhau | Tự động lựa chọn CUDA hoặc CPU, không cố định tên GPU trong code |

Việc đồng bộ cách xử lý dữ liệu cho Last Value được thực hiện để bảo đảm tính nhất quán của thực nghiệm, không nhằm tối ưu tham số GRU dựa trên Test. Sau khi hoàn tất, mô hình được đánh giá lại và ghi nhận kết quả cuối cùng.

## 9. Cách chạy lại và lưu ý chuyển giao

Từ thư mục gốc của dự án:

**Bước 1:** Chuẩn bị dữ liệu `data/raw/metr-la.h5` và cài đặt các thư viện cần thiết, bao gồm PyTorch, pandas, numpy, matplotlib và tables.

**Bước 2:** Huấn luyện nếu chưa có checkpoint:

```bash
python src/nc_traffic.py
```

**Bước 3:** Đánh giá và xuất kết quả:

```bash
python src/nc_evaluate.py
```

Nếu đã có checkpoint `models/nc_gru_best.pt`, có thể chạy bước 3 trực tiếp mà không cần train lại.

Bài nâng cao sử dụng bộ dữ liệu và cấu hình riêng nên chạy độc lập với `src/main.py` của phần dự báo điện năng chính.

## 10. Kết luận quy trình

Quy trình đã hoàn thành các bước đọc và xử lý METR-LA, xây dựng ba phương pháp dự báo, huấn luyện GRU, lựa chọn checkpoint bằng Validation và đánh giá trên Test.

Kết quả cho thấy GRU có MAE và RMSE thấp nhất trong ba mô hình tại cả ba tầm dự báo. Các bảng số liệu và hình minh họa được lưu riêng để phục vụ việc tổng hợp báo cáo chung của nhóm.

### Dữ liệu đầu vào và hướng dẫn tái chạy

- **Bộ dữ liệu:** METR-LA (dữ liệu tốc độ giao thông).
- **Định dạng:** `.h5`.
- **Tên file:** `metr-la.h5`.
- **Dung lượng:** khoảng 54,4 MB.
- **Nguồn tham khảo:** [DCRNN – GitHub](https://github.com/liyaguang/DCRNN).
- **Đường dẫn lưu dữ liệu:** `data/raw/metr-la.h5`.

**Lưu ý:** File dữ liệu gốc không được đưa lên GitHub nhằm hạn chế dung lượng repository. Khi chạy lại Câu 12, cần tải bộ dữ liệu METR-LA từ nguồn DCRNN và đặt đúng đường dẫn trên.

Mô hình GRU đã huấn luyện được lưu tại `models/nc_gru_best.pt`. Có thể sử dụng checkpoint này để đánh giá mà không cần huấn luyện lại, nhưng vẫn cần dữ liệu đầu vào.