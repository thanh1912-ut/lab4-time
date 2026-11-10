# Kiểm tra Lab 4 — bài nộp của 5 bạn (10/11/2026)

Repo: https://github.com/thanh1912-ut/lab4-time

Cách kiểm tra:
- Clone repo và gộp thử 5 nhánh vào một nhánh tạm (chưa đẩy lên GitHub).
- Chạy lại từng script trên code đã gộp.
- So số liệu với file các bạn nộp.
- Đọc từng `parts/*.md` theo checklist trong `docs/lab4/phan-cong.md`.

## 1. Tóm tắt

- **Gộp code:** 5 nhánh gộp vào `main` không xung đột.
  - #1 Bao Dinh
  - #2 Phạm Lê Văn Bảo
  - #3 Lê Quỳnh
  - #4 Hoàng Thắng (từ fork)
  - Nhánh `hoang-nam-lab4` của Hoàng Nam, **chưa mở PR**
- **Số liệu chạy lại:** khớp với bài nộp, sai khác chỉ ở mức làm tròn máy (dưới 0,001 MW).
  - Đã chạy: EDA, Câu 1, tiền xử lý, VAR, Câu 2, XGBoost, Câu 3, LSTM, Câu 4–5, bảng so sánh.
  - Môi trường: Python 3.13, TensorFlow 2.20, 4 CPU.
- **Một lỗi làm hỏng cả pipeline:** `m1_var.py` của Quỳnh chạy riêng thì được, nhưng gộp với `backtest.py` mới của Bao Dinh thì crash. Sửa bằng 1 dòng (mục 3.1).
- **Còn 4 việc lớn phải xong trước khi ghép báo cáo:**
  1. Bao Dinh chưa nộp 4 file `parts`.
  2. Câu 6 của Thắng dùng số LSTM không có thật.
  3. LSTM của Nam kém cả baseline seasonal naive, do huấn luyện chưa tới (không phải overfitting). Tui thử chỉ bật xáo trộn mẫu train thì test RMSE giảm từ 13,96 xuống 6,14.
  4. `q2_var_fail.py` ghi đè phần trả lời viết tay của Quỳnh mỗi lần chạy.

## 2. Bảng kết quả chung (test 20/07–31/12/2014, 3.946 giờ, đã chạy lại)

| Mô hình | RMSE (MW) | MAE (MW) | MAPE (%) |
|---|---|---|---|
| Naive (giờ trước) | 25,50 | 16,34 | 8,78 |
| Seasonal naive 24h | 13,42 | 7,68 | 3,82 |
| VAR(26) | 6,05 | 4,06 | 2,03 |
| XGBoost (depth 7) | 6,07 | 3,71 | 1,86 |
| LSTM (24h × 4 biến) | 13,96 | 9,57 | 4,75 |

VAR và XGBoost gần như ngang nhau: VAR thắng nhẹ RMSE, XGBoost thắng MAE và MAPE. LSTM đứng sau cả seasonal naive.

## 2b. So sánh có công bằng không?

**Phần đánh giá thì công bằng:**
- Cả ba mô hình dùng cùng 3.946 giờ test.
- Cùng scaler, chỉ fit trên train.
- Mỗi giờ đều dự báo 1 bước từ lịch sử thật.
- Cùng hàm `utils.metrics()`.
- Không mô hình nào dùng test để chọn cấu hình.

**Phần huấn luyện thì lệch, bất lợi cho LSTM:**

| | VAR | XGBoost | LSTM (Nam) |
|---|---|---|---|
| Dữ liệu học | train + val, refit mỗi 168 giờ trên cả phần test đã qua | train (val để dừng sớm), sau đó refit mỗi 168 giờ trên train + val + test đã qua | chỉ train (val để dừng sớm), không refit |
| Chọn cấu hình | p theo BIC | thử 4 giá trị depth trên val | cố định, `shuffle=False` |
| Độ dài lịch sử | 26 giờ | 24 giờ + 3 rolling | 24 giờ |

**Đo lại khi cả ba chỉ học một lần trước test, không refit:**

| Mô hình | Test RMSE không refit | Test RMSE có refit (bảng chính) |
|---|---|---|
| VAR(26), học trên train + val | 6,04 | 6,05 |
| XGBoost depth 7, học trên train (số từ ablation của Thắng) | 6,17 | 6,07 |
| LSTM, chỉ bật xáo trộn mẫu | 6,14 | — |
| LSTM của Nam | 13,96 | — |

**Kết luận:**
- Refit gần như không làm thay đổi kết quả.
- Khoảng cách lớn của LSTM là do cách train (`shuffle=False`), không phải do bản chất mô hình.
- Trong cùng điều kiện, ba mô hình ngang nhau (khoảng 6,0–6,2 MW). Vì vậy câu "XGBoost/VAR tốt hơn LSTM rõ rệt" (Câu 6, mục 10) chưa đủ cơ sở.

**Nên làm:**
- Nam bật xáo trộn mẫu khi train.
- Bao Dinh thêm cột "cách huấn luyện" vào bảng so sánh, hoặc ghi chú dưới bảng.
- LSTM mới chạy 1 seed, nên ghi rõ giới hạn này.

## 3. Từng người

### 3.1. Lê Quỳnh (VAR, Câu 2)

Code chạy ra đúng số: VAR(26) theo BIC, RMSE 6,05.

**Phải sửa**

1. **`m1_var.py` crash khi gộp với code của Bao Dinh.** `backtest.walk_forward` giờ trả về DataFrame, không còn là mảng numpy. Sửa ở `src/m1_var.py` dòng 84–85:
   ```python
   pred_z = walk_forward(make_fit_fn(p, diff), make_predict_fn(diff),
                         scaled, i2, len(scaled), refit_every=refit_every)["prediction"].to_numpy()
   ```
   `q2_var_fail.py` dùng lại hàm này nên cũng hết lỗi. Đã thử: số liệu khớp đến 1e-12.

2. **`q2_var_fail.py` ghi đè `parts/q2_report.md`.** Mỗi lần chạy (cả khi chạy `main.py`), phần "Trả lời Câu 2" Quỳnh viết tay bị thay bằng đoạn mẫu tự sinh. Ví dụ: câu "7/10 giờ sai nhiều nhất rơi vào kỳ nghỉ" bị đổi thành "2/10 giờ…". Nên cho script ghi ra file khác (vd. `q2_auto.md`) hoặc chỉ ghi phần bảng.

**Nên sửa**

3. Câu 2 ghi "7/10 giờ rơi vào ngày nghỉ lễ", cần chỉnh lại:
   - Ngày 1/11/2014 không phải ngày lễ chính thức ở Bồ Đào Nha, vì lễ Các Thánh bị tạm bỏ trong giai đoạn 2013–2015, và hôm đó lại là thứ Bảy.
   - Các ngày 24, 26, 27 và 31/12 cũng không phải ngày lễ chính thức.
   - Nên viết "6/10 giờ nằm trong kỳ nghỉ cuối năm (24–31/12)", không gọi là "ngày lễ".
4. Bảng "Thay đổi bậc trễ p" in `p` và số tham số dạng `2.00`, `32.00`. Nên in số nguyên.
   - Cột thời gian khác nhau theo máy (p = 26 ghi 0 giây vì lấy từ lần chạy chính).

### 3.2. Hoàng Thắng (XGBoost, Câu 3, Câu 6)

Code chạy ra đúng số:
- Val RMSE theo depth 3/5/7/9 là 4,69/4,55/4,52/4,65.
- Chọn depth 7, 153 cây.
- Test RMSE 6,07; walk-forward khoảng 72 giây.

**Phải sửa**

1. **Câu 6 dùng số LSTM không có thật.** Bảng ghi LSTM "RMSE ~6,5–7,5 MW, MAE ~4,2–4,8, MAPE ~2,1–2,4 %", và câu kết luận ghi "> 2,1%". Số thật là:
   - Mô hình chính: RMSE **13,96**, MAE **9,57**, MAPE **4,75**.
   - Các biến thể: RMSE từ 11,36 đến 12,95.

   Cần sửa cả `docs/lab4/parts/q6_report.md` lẫn `print_q6_comparison()` trong `src/q3_xgb_weather.py`. Kết luận "XGBoost tốt hơn" vẫn đúng, thậm chí rõ hơn. Nhưng câu "tương đương/nhỉnh hơn" thì sai.

2. **Thời gian ở Câu 6 phải dùng số đo thật.**
   - LSTM chính chạy 21 epoch (không phải 50), mất 113–164 giây tùy máy.
   - XGBoost huấn luyện một lần khoảng 2 giây, walk-forward khoảng 72 giây.
   - Vậy câu "nhanh hơn 5–10 lần" không khớp với cặp số nào. Nên ghi rõ đang so cái gì với cái gì.

3. **Câu 3 kết luận quá tay về thời tiết.** Số ablation của chính Thắng cho thấy:
   - Bỏ toàn bộ thời tiết (Load + Rolling): test RMSE **6,16**, còn mô hình Full là **6,17**. Trên test, thời tiết **không** giúp.
   - Mức giảm 19,1% (5,59 → 4,52 trên validation) phần lớn là nhờ 3 đặc trưng rolling (5,59 → 4,71). Thời tiết chỉ thêm 4,71 → 4,52, khoảng 4%.
   - Tập validation này cũng là tập đã dùng để chọn depth 7 cho mô hình Full, nên kết quả hơi thiên về Full.
   - Câu "biến thời tiết là yếu tố duy nhất giúp dự báo tải tăng vọt" chưa có số nào chứng minh.
   - PDP cho thấy đường phẳng dưới 18 °C và tăng trên 20 °C, tức là dạng chữ J (ngưỡng), không phải chữ U như mục 1 viết.

**Nên sửa (nhỏ)**

4. `n_estimators=best_iteration` thiếu 1 cây. `best_iteration` đếm từ 0, nên phải là `best_iteration + 1`.
5. Trong `build_features`, lệnh `.bfill()` điền dòng rolling đầu tiên bằng giá trị sau nó. Chỉ ảnh hưởng 1 dòng đầu train, nhưng nên ghi chú.
6. Thắng tự viết vòng walk-forward riêng, không dùng `backtest.walk_forward`. Không sai, nhưng nên nói trong report.
7. Test RMSE trong ablation (6,17) khác mục 7 (6,07) vì ablation không refit. Nên ghi một câu để người đọc không tưởng hai chỗ mâu thuẫn.

### 3.3. Hoàng Nam (LSTM, Câu 4, Câu 5)

Code chạy lại ra đúng số:
- Dừng ở epoch 21, chọn epoch 16, test RMSE 13,96.
- Dự báo lệch tối đa 0,00015 MW so với file nộp.
- Biến thể lịch ra đúng 12,40.

Code cẩn thận: có kiểm tra cache, seed, deterministic, và chia mẫu theo mốc y.

**Phải sửa**

1. **Chưa mở PR** cho nhánh `hoang-nam-lab4`. Nam mở PR, hoặc Thành gộp thẳng nhánh này.

2. **LSTM kém cả seasonal naive (13,96 so với 13,42) vì huấn luyện chưa tới, không phải overfitting.** Bằng chứng:
   - Mô hình chạy đủ 50 epoch vẫn có validation tốt nhất ở **epoch 50**, nghĩa là còn đang giảm.
   - Train RMSE (12,02) còn cao hơn validation (9,76).
   - `shuffle=False` làm mỗi epoch đi theo đúng thứ tự thời gian, từ 2012 đến 2014.
   - Patience 5 dừng quá sớm.

   **Thử nghiệm của tui (không ghi vào repo):** giữ nguyên kiến trúc, seed và patience 5, chỉ xáo trộn các cửa sổ train mỗi epoch. Kết quả:

   | Cấu hình | Epoch chạy / chọn | Train RMSE | Val RMSE | Test RMSE | Test MAE | Test MAPE |
   |---|---|---|---|---|---|---|
   | Hiện tại (`shuffle=False`) | 21 / 16 | 12,02 | 9,76 | 13,96 | 9,57 | 4,75 % |
   | Chỉ thêm xáo trộn | 21 / 16 | 6,02 | 5,90 | **6,14** | 4,39 | 2,25 % |

   Test RMSE giảm từ 13,96 xuống 6,14, ngang VAR (6,05) và XGBoost (6,07). Thời gian huấn luyện gần như không đổi (khoảng 2 phút). Epoch tốt nhất vẫn được chọn theo validation, không dựa vào test.

   Xáo trộn các cửa sổ train không gây rò rỉ, vì mỗi cửa sổ đã là một mẫu độc lập; validation và test vẫn giữ thứ tự thời gian. Nhóm quyết định có đổi cấu hình chính hay không. Nếu giữ nguyên thì báo cáo phải nói rõ LSTM đang underfit.

3. **`q5_report.md` có đoạn tự sinh vô nghĩa:** "từ epoch **50** tới epoch **50**, train MSE đổi từ 0.006034 thành 0.006034 (… +0,00%)". Lỗi xảy ra vì epoch tốt nhất trùng epoch cuối. Khi gặp trường hợp này, script nên in "validation vẫn giảm tới epoch cuối, mô hình chưa hội tụ". Phần trả lời Câu 5 cũng nên nói thẳng là **underfitting**.

**Nên sửa (nhỏ)**

4. File `requirements-nam.txt` riêng ghi Python 3.12. Tui chạy bằng Python 3.13, numpy 2.5 và pandas 3.0.5 vẫn ra đúng số. Nên gộp vào `requirements.txt` rồi xóa file riêng.
5. Mỗi lần chạy, `m3/q4/q5_report.md` được sinh lại với thời gian của máy đang chạy (máy Nam 164 giây, máy tui 113 giây). Khi ghép báo cáo cuối, nên chốt một bộ số.
6. Số viết kiểu tiếng Anh ("3,946 giờ", "164.33 giây"), trong khi các phần khác viết kiểu Việt ("26.304"). Cần thống nhất khi ghép.

### 3.4. Bao Dinh (features, backtest, Mục 4–5, so sánh)

Code chạy được và bảng `compare_metrics.csv` ra đúng số. Có đủ 5 hình: `s4_split`, `s5_window`, `s10_1tuan`, `s10_metrics`, `s10_loi_theo_gio`.

**Phải sửa**

1. **Thiếu 4 file `parts`** mà checklist yêu cầu:
   - `s4_s5_report.md`
   - `s4_s5_process.md`
   - `s9_s10_report.md`, gồm cách walk-forward, bảng so sánh và 5–7 dòng nhận xét mô hình nào tốt nhất, vì sao
   - `s9_s10_process.md`
2. **`data/processed/features_flat.csv` nặng 52 MB nằm trong git.** File này được sinh lại mỗi lần chạy nên mỗi lần commit lại làm repo phình thêm 52 MB, mà GitHub cảnh báo từ 50 MB. Cách xử lý: thêm vào `.gitignore` rồi chạy `git rm --cached`.

**Lưu ý**

3. Bao Dinh đổi `walk_forward` sang trả DataFrame. Thay đổi này hợp lý, nhưng làm hỏng `m1_var.py` (xem 3.1). Các bạn dùng chung hàm này thì nên báo nhau trước khi đổi.

### 3.5. Phạm Lê Văn Bảo (EDA, Câu 1, nâng cao)

Có đủ 6 file `parts`. `s3_eda.py` và `q1_weather.py` chạy lại ra cùng kết quả.

**Nên sửa**

1. Phần nâng cao (`nc_traffic.py`, `nc_evaluate.py`) cần 2 thứ mà repo chưa có, nên tui chưa chạy lại được:
   - Thư viện `torch`, chưa có trong `requirements.txt`.
   - File `data/raw/metr-la.h5`. `nc_process.md` chỉ ghi "tải từ nguồn DCRNN".

   Nên thêm `torch` vào requirements (hoặc tạo `requirements-nc.txt`) và ghi đường link cụ thể cùng sha256 của file `.h5`.
2. Kết quả GRU (MAE 3,40 / 3,60 / 4,00 cho 15 / 30 / 60 phút) hợp lý so với baseline Last Value (3,50 / 4,21 / 5,38), nhưng tui chưa đối chiếu được vì thiếu dữ liệu.

## 4. Chạy toàn bộ

Sau khi sửa dòng `m1_var.py`, chạy `python src/main.py` sẽ chạy hết theo thứ tự trong `ORDER`.

- Phần LSTM chiếm nhiều thời gian nhất. Trên 4 CPU, mô hình chính mất khoảng 2 phút, còn `q4_q5_lstm.py` mất khoảng 10 phút. Ba biến thể Câu 4–5 cũng chạy lại ra đúng số: 12,40 / 11,36 / 12,95. Có cache nên lần sau nhanh hơn.
- Chạy `main.py` sẽ **ghi đè** 4 file sau. Chỉ `q2_report.md` có phần viết tay bị mất:
  - `m1_report.md`
  - `q2_report.md`
  - `m3_*.md`
  - `q4/q5_*.md`

## 5. Đã gộp (10/11/2026)

- Merge 4 PR, theo thứ tự #1, #2, #3, #4.
- Gộp nhánh `hoang-nam-lab4` vào `main` (commit 508ca27).
- Sửa `m1_var.py` và thêm `features_flat.csv` vào `.gitignore` (commit 9543bf4 và 55bd5a1).
- Clone lại `main` sạch rồi chạy VAR: RMSE 6,05.
- Việc còn lại là phần nội dung của từng bạn trong mục 3, và 4 file `parts` của Bao Dinh.

### Thứ tự gộp đã đề xuất trước đó

1. Gộp #1 (Bao Dinh), rồi #2 (Bảo), rồi #3 (Quỳnh).
2. Sửa 1 dòng `m1_var.py` ngay sau khi gộp #3.
3. Gộp #4 (Thắng).
4. Gộp nhánh của Nam.
5. Thêm `features_flat.csv` vào `.gitignore`.
6. Mỗi bạn sửa phần nội dung của mình theo mục 3. Riêng Bao Dinh nộp thêm 4 file `parts`.

## 6. So sánh công bằng: đã làm (PR #5, chờ Thành merge)

https://github.com/thanh1912-ut/lab4-time/pull/5

Luật chung cho cả ba mô hình:
- Học trên train.
- Validation chỉ để chọn cấu hình hoặc dừng sớm.
- Không học lại trên test.
- Mỗi giờ test dự báo 1 bước từ lịch sử thật.

| Mô hình | RMSE | MAE | MAPE (%) |
|---|---|---|---|
| Seasonal naive 24h | 13,42 | 7,68 | 3,82 |
| VAR(26) | 6,07 | 4,11 | 2,06 |
| XGBoost | 6,17 | 3,81 | 1,89 |
| LSTM (seed 42) | 6,14 | 4,39 | 2,25 |

Nhận xét:
- LSTM chạy 3 seed cho RMSE 6,02 ± 0,11 MW.
- Câu 4: thêm đặc trưng lịch cho RMSE 5,77 MW.
- Câu 5: chạy đủ 50 epoch thì validation xấu đi sau epoch 16, tức là có overfitting, và EarlyStopping chặn được.
- Các mục 3.1, 3.2 và 3.3 phía trên đã được xử lý trong PR này, gồm số liệu Câu 6, đoạn lỗi Câu 5, và LSTM train đúng cách.
- File `s9_s10` của Bao Dinh cũng đã có.

**Còn lại:**
- `s4_s5_report.md` và `s4_s5_process.md` (Bao Dinh).
- `q2_var_fail.py` ghi đè phần trả lời viết tay của Câu 2.
- Câu 3 kết luận quá tay về thời tiết.
- Câu 2 ghi sai ngày lễ.
- Phần nâng cao thiếu `torch` và link dữ liệu.
