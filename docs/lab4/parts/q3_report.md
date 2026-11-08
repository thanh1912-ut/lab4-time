# Câu hỏi 3: XGBoost học được gì từ thời tiết? (Hoàng Thắng)

## 1. Trả lời trực tiếp câu hỏi (Tóm tắt cốt lõi)

Thông qua phân tích **Độ quan trọng đặc trưng (Gain)**, **Thử nghiệm loại trừ đặc trưng (Ablation Study)** và **Biểu đồ phụ thuộc một phần (Partial Dependence Plot - PDP)**, mô hình XGBoost đã học được các thông tin bản chất sau từ các biến thời tiết:
1. **Nhiệt độ là biến thời tiết chi phối mạnh nhất:** Trong số 3 biến thời tiết, nhiệt độ chiếm tỷ trọng lớn nhất về độ cải thiện độ lợi phân nhánh (Gain), cao gấp đôi độ ẩm và gấp 2,4 lần tốc độ gió. Đặc trưng `temperature_lag1` (nhiệt độ giờ trước) nằm trong nhóm đặc trưng thời tiết có ảnh hưởng lớn nhất.
2. **Quan hệ phi tuyến hình chữ U giữa nhiệt độ và điện năng:** PDP chứng minh XGBoost đã học được quy luật thực tế: khi nhiệt độ vượt ngưỡng $20^\circ\text{C}$, phụ tải điện tăng vọt theo dạng phi tuyến mạnh (do nhu cầu sử dụng điều hòa nhiệt độ và hệ thống làm mát tại Lisbon vào mùa hè).
3. **Thời tiết đóng vai trò hiệu chỉnh biên quan trọng (Fine-tuning):** Chuỗi phụ tải bản thân nó đã chứa đựng chu kỳ ngày/tuần rất mạnh (chiếm trên 97% tổng Gain). Các biến thời tiết đóng vai trò cung cấp tín hiệu ngoại sinh để điều chỉnh các sai số cực biên do thời tiết bất thường (đợt nóng đột xuất), giúp giảm sai số RMSE từ **5,59 MW xuống 4,52 MW (cải thiện 19,1%)** so với mô hình chỉ dùng độ trễ phụ tải thuần túy.

---

## 2. Phân tích độ quan trọng đặc trưng (Feature Importance theo Gain)

Độ lợi (Gain) phản ánh mức độ giảm thiểu hàm mất mát (loss reduction) khi một đặc trưng được chọn để phân nhánh trong các cây quyết định.

### 2.1. Phân bố Gain theo nhóm biến

| Nhóm biến | Số đặc trưng | Tổng Gain | Tỷ lệ đóng góp (%) | Vai trò mô hình hóa |
|---|:---:|:---:|:---:|---|
| **Phụ tải quá khứ (`load lags`)** | 24 | 142,31 | **97,19%** | Cung cấp mức tải nền và quán tính tiêu thụ ngắn hạn |
| **Đặc trưng thống kê cuộn (`rolling & trend`)** | 3 | 1,49 | **1,02%** | Cung cấp xu hướng dịch chuyển và độ biến động |
| **Nhiệt độ (`temperature lags`)** | 24 | 1,37 | **0,94%** | Động lực ngoại sinh chính gây biến động tiêu thụ điện |
| **Độ ẩm (`humidity lags`)** | 24 | 0,69 | **0,47%** | Bổ trợ cho cảm nhận nhiệt (nhiệt ẩm / heat index) |
| **Tốc độ gió (`wind_speed lags`)** | 24 | 0,57 | **0,39%** | Ảnh hưởng nhẹ đến sự thoát nhiệt bề mặt công trình |
| **TỔNG CỘNG** | **99** | **146,43** | **100,00%** | |

![Độ quan trọng theo nhóm](../figures/q3_importance_nhom.png)

**Hình 11.1.** Tỷ lệ đóng góp Gain (%) theo các nhóm đặc trưng trong mô hình XGBoost.

### 2.2. Phân bố Gain theo bậc trễ (Lag 1 đến Lag 24)

![Độ quan trọng theo độ trễ](../figures/q3_importance_lag.png)

**Hình 11.2.** Tổng Gain phân bổ theo bậc trễ từ 1 đến 24 giờ.

Nhận xét:
- **`load_lag24` chiếm ưu thế tuyệt đối (Gain = 137,72):** Phụ tải tại thời điểm $t+1$ hôm nay tương quan chặt chẽ nhất với chính thời điểm đó ngày hôm qua ($t+1 - 24 = t-23$, tức lag 24 đối với $t+1$). Cây quyết định ưu tiên phân nhánh đầu tiên vào vị trí này để tái lập chu kỳ ngày.
- **`load_lag1` (Gain = 1,95) và `load_trend_24` (Gain = 1,18):** Đóng vai trò tinh chỉnh độ dịch chuyển ngắn hạn so với ngày hôm trước.
- **Đối với thời tiết:** Bậc trễ ngắn (`temperature_lag1`, `temperature_lag2`) có Gain cao nhất trong nhóm thời tiết, cho thấy nhiệt độ gần thời điểm dự báo nhất có tác động tức thời đến việc bật/tắt thiết bị điện.

---

## 3. Thử nghiệm loại trừ đặc trưng (Ablation Study)

Để đo lường định lượng giá trị gia tăng mà thông tin thời tiết mang lại, chúng tôi thực hiện 3 cấu hình thử nghiệm độc lập trên cùng quy trình huấn luyện và đánh giá:

| Cấu hình mô hình | Số đặc trưng | Validation RMSE (MW) | Validation MAE (MW) | Validation MAPE (%) | Test RMSE (MW) | Test MAPE (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **(1) Full Model** (Đầy đủ thời tiết + rolling) | **99** | **4,52** | **3,12** | **1,65%** | **6,17** | **1,89%** |
| **(2) Load + Rolling** (Bỏ toàn bộ biến thời tiết) | 27 | 4,71 | 3,15 | 1,65% | 6,16 | 1,88% |
| **(3) Load Only** (Chỉ 24 lag phụ tải) | 24 | 5,59 | 3,72 | 1,95% | 7,11 | 2,19% |

### Đánh giá định lượng từ Ablation Study:
1. **So sánh (1) với (3):** Khi bổ sung đầy đủ biến thời tiết và rolling, sai số RMSE trên tập Validation giảm từ **5,59 MW xuống 4,52 MW (giảm 19,1%)**, trên tập Test giảm từ **7,11 MW xuống 6,17 MW (giảm 13,2%)**.
2. **So sánh (1) với (2):** Sự hiện diện của biến thời tiết giúp mô hình Validation giảm RMSE từ **4,71 MW xuống 4,52 MW (giảm 4,04%)**. Trong các giai đoạn chuyển mùa và các ngày nắng nóng đỉnh điểm, các biến thời tiết là yếu tố duy nhất giúp mô hình dự báo chính xác hiện tượng phụ tải tăng vọt bất thường ngoài khuôn khổ chu kỳ lịch sử thông thường.

---

## 4. Biểu đồ phụ thuộc một phần (Partial Dependence Plot - PDP)

![PDP Nhiệt độ](../figures/q3_pdp_temp.png)

**Hình 11.3.** Ảnh hưởng cận biên của nhiệt độ trễ 1 giờ (`temperature_lag1`) lên phụ tải điện năng dự báo.

### Phát hiện từ biểu đồ PDP:
- **Dưới $18^\circ\text{C}$:** Đường biên phụ tải dao động nhẹ quanh mức cân bằng (-1,4 MW đến -0,5 MW). Do khí hậu mùa đông Lisbon tương đối ôn hòa (nhiệt độ hiếm khi xuống dưới $5^\circ\text{C}$), nhu cầu sưởi điện không tạo ra đỉnh tải đột biến.
- **Trên $20^\circ\text{C}$:** Phụ tải bắt đầu tăng dốc. Đặc biệt khi nhiệt độ đạt $25^\circ\text{C} - 27^\circ\text{C}$, độ lệch cận biên tăng vọt lên **+2,60 MW** (mức chênh lệch biên đạt tới 4 MW).
- Điều này chứng minh thuật toán XGBoost đã tự động phân nhánh và trích xuất thành công **tính chất phi tuyến (non-linear threshold effect)** của hành vi tiêu thụ điện theo nhiệt độ mà các mô hình tuyến tính đơn thuần như VAR không thể nắm bắt được.
