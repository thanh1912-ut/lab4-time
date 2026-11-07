
# Câu 1 – Quy trình phân tích mối quan hệ giữa thời tiết và phụ tải điện

## 1. Mục đích thực hiện

Câu 1 tập trung xác định biến thời tiết có mối tương quan mạnh nhất với phụ tải điện, đồng thời kiểm tra sự thay đổi của mối quan hệ này theo mùa, độ trễ thời gian và sau khi loại bỏ chu kỳ giờ trong ngày.

Quá trình thực hiện gồm năm nội dung chính:

1. Tính hệ số tương quan Pearson và Spearman giữa phụ tải và các biến thời tiết.
2. Phân tích tương quan theo mùa và kiểm tra độ lệch nhiệt độ so với 18°C.
3. Khảo sát tương quan giữa phụ tải hiện tại và nhiệt độ ở các độ trễ từ 0 đến 24 giờ.
4. Tính lại tương quan sau khi loại bỏ thành phần trung bình theo giờ trong ngày.
5. Trực quan hóa kết quả, tổng hợp nhận xét và chuẩn bị đối chiếu với feature importance của Câu 3.

Câu 1 là bài toán phân tích dữ liệu, không thực hiện huấn luyện hoặc đánh giá mô hình dự báo.

## 2. Dữ liệu và môi trường thực hiện

### 2.1. Dữ liệu đầu vào

Dữ liệu được sử dụng từ tệp đã xử lý ở Mục 2:

`data/processed/load_weather_hourly.csv`

Thông tin dữ liệu:

| Thuộc tính | Giá trị |
|---|---|
| Số quan sát | 26.304 |
| Tần suất | Theo giờ |
| Bắt đầu | 01/01/2012 00:00 |
| Kết thúc | 31/12/2014 23:00 |
| Biến mục tiêu | `load` |
| Biến thời tiết | `temperature`, `humidity`, `wind_speed` |

Các biến có đơn vị tương ứng: phụ tải (MW), nhiệt độ (°C), độ ẩm tương đối (%) và tốc độ gió (m/s).

Dữ liệu được đọc thông qua hàm `load_data()` trong `src/utils.py`. Hàm này đọc dữ liệu với cột `Datetime` làm chỉ số thời gian và trả về bốn biến cần thiết.

Câu 1 sử dụng dữ liệu sau tiền xử lý, không thực hiện lại các bước làm sạch hoặc bổ sung dữ liệu.

### 2.2. Công cụ sử dụng

| Công cụ | Vai trò |
|---|---|
| Python | Ngôn ngữ triển khai |
| pandas | Xử lý chuỗi thời gian và tính tương quan |
| NumPy | Tạo dãy giá trị độ trễ trên biểu đồ |
| Matplotlib | Trực quan hóa và lưu biểu đồ |
| VS Code | Viết mã và chạy chương trình |

Mã nguồn chính được triển khai trong `src/q1_weather.py`.

## 3. Quy trình thực hiện

### Bước 1. Đọc và kiểm tra dữ liệu

Sử dụng hàm `load_data()` từ `src/utils.py`, sau đó sắp xếp dữ liệu theo thời gian tăng dần.

```python
df = load_data().sort_index()
```

Kiểm tra số quan sát và khoảng thời gian:

```python
print(len(df))
print(df.index.min(), df.index.max())
```

Kết quả xác nhận dữ liệu gồm 26.304 quan sát, từ ngày 01/01/2012 đến hết ngày 31/12/2014.

Việc sắp xếp theo thời gian là cần thiết trước khi tính tương quan trễ, vì phép dịch chuỗi phụ thuộc vào thứ tự các quan sát.

### Bước 2. Tính tương quan Pearson và Spearman

Sử dụng hai phương pháp để đánh giá mối quan hệ giữa phụ tải và từng biến thời tiết.

**Pearson** đo mức độ liên hệ tuyến tính. Hệ số nằm trong khoảng từ -1 đến 1; dấu của hệ số thể hiện chiều tương quan, còn giá trị tuyệt đối thể hiện độ mạnh của mối liên hệ tuyến tính.

**Spearman** đo mức độ liên hệ đơn điệu thông qua thứ hạng dữ liệu. Phương pháp này giúp kiểm tra xem xu hướng tương quan có nhất quán khi không chỉ xét quan hệ tuyến tính hay không.

Các phép tính được thực hiện bằng pandas:

```python
pearson = df[WEATHER_COLS].corrwith(df["load"])

spearman = df[WEATHER_COLS].corrwith(
    df["load"],
    method="spearman"
)
```

Sau khi tính, lấy giá trị tuyệt đối của Pearson và sắp xếp giảm dần:

```python
result["|Pearson|"] = result["Pearson"].abs()

result = result.sort_values(
    "|Pearson|",
    ascending=False
)
```

Sử dụng giá trị tuyệt đối nhằm đánh giá độ mạnh tương quan mà không bỏ qua các biến có tương quan âm.

Kết quả xác định nhiệt độ có tương quan mạnh nhất với phụ tải trong ba biến khảo sát, với Pearson 0.6559 và Spearman 0.5922.

### Bước 3. Phân tích tương quan theo mùa

Dữ liệu được chia theo bốn mùa khí tượng dựa trên tháng của chỉ số thời gian:

| Mùa | Các tháng |
|---|---|
| Đông | 12, 1, 2 |
| Xuân | 3, 4, 5 |
| Hè | 6, 7, 8 |
| Thu | 9, 10, 11 |

Với mỗi mùa, lọc dữ liệu theo tháng và tính Pearson giữa phụ tải với từng biến thời tiết.

```python
subset = df[df.index.month.isin(months)]

correlation = subset["load"].corr(
    subset["temperature"]
)
```

Quy trình này được áp dụng tương tự cho độ ẩm và tốc độ gió.

Kết quả cho thấy tương quan nhiệt độ – phụ tải cao nhất vào mùa hè (0.7203) và thấp nhất vào mùa đông (0.4337).

Việc chia mùa giúp kiểm tra liệu tương quan trên toàn bộ dữ liệu có phản ánh giống nhau ở các giai đoạn trong năm hay không.

### Bước 4. Kiểm tra độ lệch nhiệt độ so với 18°C

Do mối quan hệ giữa nhiệt độ và nhu cầu sử dụng điện có thể không hoàn toàn tuyến tính, thực hiện thêm phép biến đổi:

`temperature_deviation = |temperature - 18|`

Trong Python:

```python
deviation = (df["temperature"] - 18).abs()
```

Sau đó tính Pearson và Spearman giữa `load` và `deviation`:

```python
pearson = df["load"].corr(deviation)

spearman = df["load"].corr(
    deviation,
    method="spearman"
)
```

Kết quả:

- Pearson: 0.1008.
- Spearman: -0.0057.

Mục đích của phép biến đổi là kiểm tra xem phụ tải có xu hướng tăng khi nhiệt độ cách xa mốc tham chiếu 18°C hay không.

Vì các hệ số gần 0, phép biến đổi này không cho thấy mối tương quan rõ rệt trên toàn bộ dữ liệu. Tuy nhiên, đây chỉ là một cách kiểm tra đơn giản, không phải phép kiểm định đầy đủ cho quan hệ hình chữ U.

### Bước 5. Tính tương quan theo độ trễ 0–24 giờ

Để xem xét mối liên hệ giữa phụ tải hiện tại và nhiệt độ trong quá khứ, khảo sát các độ trễ `k` từ 0 đến 24 giờ.

Tại mỗi độ trễ, tính:

`corr(load(t), temperature(t-k))`

Trong đó:

- `k = 0`: Nhiệt độ và phụ tải cùng thời điểm.
- `k = 1`: Nhiệt độ trước phụ tải một giờ.
- `k = 24`: Nhiệt độ trước phụ tải 24 giờ.

Đoạn mã thực hiện:

```python
for lag in range(25):
    past_temperature = df["temperature"].shift(lag)

    pearson = df["load"].corr(past_temperature)

    spearman = df["load"].corr(
        past_temperature,
        method="spearman"
    )
```

Phép `shift(lag)` dịch các giá trị nhiệt độ theo số hàng, tương ứng với số giờ vì dữ liệu có tần suất một giờ và được sắp xếp theo thời gian.

Khi `lag > 0`, những vị trí đầu chuỗi không có nhiệt độ quá khứ tương ứng sẽ xuất hiện giá trị thiếu. Hàm `corr()` của pandas tự loại các cặp quan sát thiếu khi tính hệ số.

Sau khi tính đủ 25 độ trễ, lưu kết quả vào bảng gồm ba cột: `Lag`, `Pearson` và `Spearman`.

Xác định độ trễ có tương quan Pearson mạnh nhất theo giá trị tuyệt đối:

```python
best_lag = int(
    lag_table["Pearson"].abs().idxmax()
)
```

Kết quả thu được:

| Độ trễ | Pearson | Spearman |
|---|---:|---:|
| 0 giờ | 0.6559 | 0.5922 |
| 12 giờ | -0.0447 | -0.0225 |
| 24 giờ | 0.6535 | 0.5899 |

Lag 0 có tương quan mạnh nhất trong khoảng khảo sát. Lag 24 có giá trị gần tương đương, phù hợp với đặc điểm chu kỳ ngày của dữ liệu.

### Bước 6. Loại bỏ thành phần chu kỳ giờ trong ngày

Tương quan giữa nhiệt độ và phụ tải có thể chịu ảnh hưởng từ việc cả hai chuỗi cùng biến đổi theo giờ trong ngày.

Để kiểm tra mối quan hệ sau khi giảm ảnh hưởng của chu kỳ này, thực hiện điều chỉnh riêng cho hai biến.

**6.1. Tính giá trị trung bình theo giờ**

Với mỗi giờ từ 0 đến 23, tính trung bình phụ tải và nhiệt độ của tất cả các quan sát thuộc giờ đó.

```python
load_hour_mean = df.groupby(
    df.index.hour
)["load"].transform("mean")

temp_hour_mean = df.groupby(
    df.index.hour
)["temperature"].transform("mean")
```

**6.2. Tạo chuỗi phần dư**

Lấy giá trị thực tế trừ đi giá trị trung bình tương ứng với giờ trong ngày:

```python
load_residual = df["load"] - load_hour_mean

temp_residual = df["temperature"] - temp_hour_mean
```

Ví dụ, phụ tải tại 18:00 được trừ đi phụ tải trung bình của tất cả các quan sát lúc 18:00 trong tập dữ liệu.

Như vậy, phần dư thể hiện mức chênh lệch so với giá trị trung bình của cùng giờ, thay vì giá trị phụ tải hoặc nhiệt độ ban đầu.

**6.3. Tính lại tương quan**

```python
adjusted_pearson = load_residual.corr(
    temp_residual
)

adjusted_spearman = load_residual.corr(
    temp_residual,
    method="spearman"
)
```

Kết quả:

| Hệ số | Ban đầu | Sau điều chỉnh |
|---|---:|---:|
| Pearson | 0.6559 | 0.7572 |
| Spearman | 0.5922 | 0.7885 |

Việc trừ trung bình theo giờ giúp loại bỏ thành phần trung bình lặp lại trong ngày, nhưng chưa kiểm soát các yếu tố khác như mùa hoặc ngày trong tuần.

### Bước 7. Trực quan hóa kết quả tương quan trễ

Sử dụng Matplotlib để biểu diễn hệ số tương quan theo độ trễ.

Biểu đồ gồm:

- Trục X: Độ trễ nhiệt độ từ 0 đến 24 giờ.
- Trục Y: Hệ số tương quan.
- Đường xanh dương: Pearson.
- Đường cam: Spearman.
- Hai vị trí được chú thích: lag 0 và lag 24.

Lag 0 được đánh dấu vì có giá trị tuyệt đối Pearson lớn nhất trong phạm vi khảo sát. Lag 24 được đánh dấu để thể hiện mức tương quan tăng trở lại sau một chu kỳ ngày.

Đường ngang tại hệ số 0 hỗ trợ phân biệt tương quan dương và âm.

Biểu đồ được lưu bằng:

```python
output = FIG / "q1_lag_corr.png"

fig.savefig(
    output,
    dpi=170,
    bbox_inches="tight"
)
```

Đường dẫn đầu ra:

`docs/lab4/figures/q1_lag_corr.png`

Biểu đồ thể hiện tương quan theo độ trễ, không phải kết quả dự báo của một mô hình học máy.

## 4. Tổ chức mã nguồn

| Tệp | Vai trò |
|---|---|
| `src/q1_weather.py` | Thực hiện các phép phân tích và xuất biểu đồ |
| `src/utils.py` | Cung cấp hàm đọc dữ liệu và đường dẫn dùng chung |
| `data/processed/load_weather_hourly.csv` | Dữ liệu đã xử lý |
| `docs/lab4/figures/q1_lag_corr.png` | Biểu đồ tương quan theo độ trễ |
| `docs/lab4/parts/q1_report.md` | Trình bày kết quả và nhận xét |
| `docs/lab4/parts/q1_process.md` | Ghi lại quy trình thực hiện |

Trong `q1_weather.py`, các bước phân tích được tổ chức thành các hàm riêng để dễ kiểm tra:

| Hàm | Chức năng |
|---|---|
| `correlation_table()` | Tính Pearson và Spearman cho ba biến thời tiết |
| `seasonal_correlations()` | Tính tương quan theo bốn mùa |
| `temperature_deviation()` | Kiểm tra độ lệch nhiệt độ so với 18°C |
| `lag_correlations()` | Tính tương quan theo độ trễ 0–24 giờ |
| `adjusted_correlations()` | Tính tương quan sau khi loại bỏ trung bình theo giờ |
| `plot_lag_correlations()` | Vẽ và lưu biểu đồ |
| `main()` | Điều phối các bước và in kết quả |

## 5. Cách chạy và kiểm tra kết quả

### 5.1. Chạy chương trình

Mở Terminal tại thư mục gốc của dự án:

```powershell
python src/q1_weather.py
```

Chương trình thực hiện lần lượt:

1. Đọc dữ liệu đã xử lý.
2. Tính tương quan Pearson và Spearman.
3. Tính tương quan theo mùa.
4. Kiểm tra độ lệch nhiệt độ so với 18°C.
5. Tính tương quan theo độ trễ.
6. Tính tương quan sau điều chỉnh chu kỳ giờ.
7. Lưu biểu đồ và in đường dẫn đầu ra.

### 5.2. Đối chiếu kết quả

Sau khi chạy, kiểm tra các giá trị chính trên Terminal:

| Nội dung | Kết quả kỳ vọng |
|---|---:|
| Số quan sát | 26.304 |
| Pearson nhiệt độ – phụ tải | 0.6559 |
| Spearman nhiệt độ – phụ tải | 0.5922 |
| Pearson mùa hè | 0.7203 |
| Pearson tại lag 0 | 0.6559 |
| Pearson tại lag 12 | -0.0447 |
| Pearson tại lag 24 | 0.6535 |
| Pearson sau điều chỉnh | 0.7572 |
| Spearman sau điều chỉnh | 0.7885 |

Các giá trị trên là kết quả thực tế của lần chạy trên tập dữ liệu hiện tại, được sử dụng để kiểm tra tính nhất quán khi chạy lại.

### 5.3. Kiểm tra biểu đồ

Mở tệp:

`docs/lab4/figures/q1_lag_corr.png`

Kiểm tra:

- Biểu đồ có đủ hai đường Pearson và Spearman.
- Trục độ trễ hiển thị từ 0 đến 24 giờ.
- Lag 0 có Pearson bằng 0.6559.
- Lag 24 có Pearson bằng 0.6535.
- Các chú thích không che khuất đường biểu diễn.
- Biểu đồ được hiển thị đúng trong `q1_report.md`.

## 6. Kết quả hoàn thành và công việc tiếp theo

Câu 1 đã hoàn thành các phép phân tích tương quan, kiểm tra theo mùa, độ trễ, điều chỉnh chu kỳ giờ và trực quan hóa kết quả.

Các sản phẩm gồm:

1. `src/q1_weather.py`
2. `docs/lab4/figures/q1_lag_corr.png`
3. `docs/lab4/parts/q1_report.md`
4. `docs/lab4/parts/q1_process.md`

**Phần cần bổ sung:** Khi Hoàng Thắng hoàn thành Câu 3, sử dụng kết quả feature importance để đối chiếu với thứ hạng tương quan của các biến thời tiết trong Câu 1. Không cần huấn luyện lại mô hình của Câu 3.
