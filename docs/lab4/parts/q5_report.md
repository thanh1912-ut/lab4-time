# Câu 5: LSTM có bị overfitting không? — Hoàng Nam

## Thí nghiệm kiểm tra

So sánh ba cấu hình định trước, cùng seed 42, cùng cửa sổ 24 giờ và cùng dữ liệu:
mô hình chính có EarlyStopping, mô hình đủ epoch không EarlyStopping,
và mô hình thêm Dropout(0.2) sau mỗi lớp LSTM với EarlyStopping. Cấu hình chính giữ nguyên;
không chọn biến thể theo sai số test.

## Kết quả

| Model | Epochs | Best val epoch | Train RMSE | Validation RMSE | Test RMSE | Training seconds |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM chính + EarlyStopping | 26 | 16 | 6.02 | 5.90 | 6.14 | 139.51 |
| LSTM đủ epoch, không EarlyStopping | 50 | 16 | 5.31 | 6.23 | 5.92 | 234.35 |
| LSTM + Dropout(0.2) + EarlyStopping | 40 | 30 | 6.07 | 6.15 | 6.03 | 197.72 |

RMSE bằng MW sau khi đổi về thang gốc. Với EarlyStopping, trọng số lưu là epoch có
validation loss thấp nhất. Với biến thể không EarlyStopping, đánh giá trọng số **epoch cuối**,
dù cột `Best val epoch` có thể nằm sớm hơn. Thời gian đo chỉ gồm huấn luyện.

![Loss train và validation](../figures/q5_loss.png)

| Model | Split | RMSE | MAE | MAPE(%) |
| --- | --- | --- | --- | --- |
| LSTM chính + EarlyStopping | train | 6.02 | 4.25 | 2.16 |
| LSTM chính + EarlyStopping | validation | 5.90 | 4.26 | 2.34 |
| LSTM chính + EarlyStopping | test | 6.14 | 4.39 | 2.25 |
| LSTM đủ epoch, không EarlyStopping | train | 5.31 | 3.76 | 1.89 |
| LSTM đủ epoch, không EarlyStopping | validation | 6.23 | 4.37 | 2.30 |
| LSTM đủ epoch, không EarlyStopping | test | 5.92 | 4.22 | 2.11 |
| LSTM + Dropout(0.2) + EarlyStopping | train | 6.07 | 4.44 | 2.40 |
| LSTM + Dropout(0.2) + EarlyStopping | validation | 6.15 | 4.55 | 2.54 |
| LSTM + Dropout(0.2) + EarlyStopping | test | 6.03 | 4.47 | 2.35 |

## Trả lời câu 5

**Có dấu hiệu overfitting khi kéo dài huấn luyện ở biến thể không EarlyStopping.** Sau epoch có validation loss tốt nhất, train loss tiếp tục giảm nhưng validation loss tăng; trung bình các epoch cuối cũng cho cùng hướng phân kỳ. Mô hình chính dùng checkpoint tốt nhất theo validation để hạn chế tác động này; đây không phải khẳng định mô hình chính đã loại bỏ hoàn toàn overfitting.



Mô hình chính chạy 26 epoch và dùng trọng số ở epoch
16. Validation MSE tốt nhất 0.005432,
so với 0.006205 ở epoch cuối đã chạy. RMSE validation thấp hơn RMSE train
**0.12 MW** (5.90 so với 6.02 MW).
Chênh lệch cần đọc cùng đường loss: các tập là các giai đoạn thời gian khác nhau,
nên thay đổi mùa và phân phối cũng có thể tạo chênh lệch, không chỉ overfitting.

Ở biến thể đủ epoch, từ epoch **16** (validation tốt nhất) tới epoch
**50**, train MSE đổi từ **0.006383** thành **0.004200**
(chênh lệch cuối − đầu **-0.002183**, mức giảm **+34.20%**);
validation MSE từ **0.005432** thành **0.006070**
(chênh lệch **+0.000638**, mức tăng **+11.74%**).
Trung bình 5 epoch cuối: train MSE **0.004308**,
validation MSE **0.006292**. Các đại lượng này đều ở thang chuẩn hóa.
Chẩn đoán phân kỳ rõ dùng quy tắc mô tả: có ít nhất 5 epoch sau mốc tốt nhất,
train giảm trên 1%, validation tăng trên 1%, và trung bình tối đa 5 epoch cuối cùng cùng chiều.
Đây là quy tắc đọc kết quả của thí nghiệm, không phải kiểm định thống kê.

EarlyStopping hạn chế tiếp tục học khi validation không cải thiện; Dropout là thử nghiệm
regularization bổ sung. Biến thể chạy một seed (mô hình chính có thêm 2 seed ở mục 8),
chưa đủ để khẳng định mức cải thiện ổn định.
