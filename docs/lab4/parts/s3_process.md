
# NHẬT KÝ QUY TRÌNH – MỤC 3. DATA UNDERSTANDING

## 1. Thông tin thực hiện

**Nội dung:** Khám phá và phân tích dữ liệu (Exploratory Data Analysis – EDA).

**Chương trình:** `src/s3_eda.py`

**Dữ liệu đầu vào:** `data/processed/load_weather_hourly.csv`

**Dữ liệu đầu ra:** Sáu biểu đồ PNG và kết quả thống kê trên Terminal.

**Thư mục lưu biểu đồ:** `docs/lab4/figures/`

**Tài liệu kết quả:** `docs/lab4/parts/s3_report.md`

Mục 3 được thực hiện sau khi dữ liệu điện năng và thời tiết đã được đồng bộ, làm sạch ở Mục 2.

Phạm vi công việc chỉ bao gồm phân tích khám phá dữ liệu, không chỉnh sửa dữ liệu nguồn, không chuẩn hóa đặc trưng và không huấn luyện mô hình.

## 2. Công cụ và thư viện

Các thư viện được sử dụng:

| Thư viện | Mục đích |
|---|---|
| pandas | Đọc dữ liệu, nhóm dữ liệu, tính thống kê và xử lý thời gian |
| numpy | Tính toán số học, phân vị và hồi quy tuyến tính |
| matplotlib | Vẽ biểu đồ và lưu hình PNG |
| seaborn | Thiết lập giao diện và vẽ heatmap |
| statsmodels | Tính hàm tự tương quan ACF |

Chương trình sử dụng các cấu hình chung được khai báo trong `src/utils.py`, gồm đường dẫn dữ liệu, đường dẫn hình và danh sách các biến.

Việc sử dụng hàm `load_data()` giúp tránh khai báo lại đường dẫn và bảo đảm tính nhất quán giữa các phần của nhóm.

## 3. Quy trình tổng thể

Các bước thực hiện theo thứ tự:

1. Đọc bộ dữ liệu đã xử lý.
2. Kiểm tra số quan sát, khoảng thời gian và giá trị thiếu.
3. Tính IQR và Z-score cho từng biến gốc.
4. Tính phần dư phụ tải theo giờ-trong-tuần.
5. Áp dụng IQR và Z-score lên phần dư.
6. Vẽ biểu đồ phụ tải theo thời gian.
7. Vẽ biểu đồ phụ tải và nhiệt độ theo mùa.
8. Tính và trực quan hóa tương quan Pearson.
9. Vẽ biểu đồ ngoại lai.
10. Phân tích chu kỳ theo giờ và theo thứ.
11. Tính ACF trong 200 độ trễ.
12. Kiểm tra các số liệu và sáu biểu đồ đầu ra.

Toàn bộ quy trình được triển khai trong `src/s3_eda.py`.

## 4. Đọc và kiểm tra dữ liệu

### 4.1. Đọc dữ liệu

Sử dụng:

```python
from utils import load_data

df = load_data().sort_index()
```

Hàm `load_data()` đọc tệp CSV đã xử lý, chuyển `Datetime` thành chỉ số thời gian và chọn bốn biến phân tích:

```python
["load", "temperature", "humidity", "wind_speed"]
```

Dữ liệu được sắp xếp theo thời gian tăng dần để phục vụ phân tích chuỗi thời gian.

### 4.2. Kiểm tra cấu trúc

Chương trình kiểm tra:

- Dữ liệu có rỗng hay không.
- Chỉ số thời gian có bị trùng hay không.
- Số quan sát có đủ để tính ACF 200 độ trễ hay không.
- Có giá trị thiếu trong các biến phân tích hay không.

Kết quả:

```text
Observations: 26,304
Period: 2012-01-01 00:00:00 to 2014-12-31 23:00:00
```

Bộ dữ liệu có 26.304 quan sát theo giờ trong ba năm.

### 4.3. Kiểm tra giá trị thiếu

Sử dụng:

```python
missing = pd.DataFrame({
    "Missing": df.isna().sum(),
    "Missing (%)": df.isna().mean() * 100,
})
```

Kết quả:

| Biến | Missing | Missing (%) |
|---|---:|---:|
| load | 0 | 0,0 |
| temperature | 0 | 0,0 |
| humidity | 0 | 0,0 |
| wind_speed | 0 | 0,0 |

Không có giá trị thiếu trong bộ dữ liệu sau xử lý.

Theo quy trình của Mục 2, dữ liệu đã được xử lý các bất thường thời gian và bổ sung những giá trị thời tiết còn thiếu. Vì vậy, Mục 3 không thực hiện nội suy lại.

## 5. Phát hiện ngoại lai

### 5.1. Lựa chọn phương pháp

Sử dụng đồng thời IQR và Z-score nhằm đánh giá các quan sát có độ lệch lớn theo hai tiêu chí thống kê khác nhau.

IQR dựa trên phân vị nên ít nhạy với những giá trị cực trị hơn các thống kê dựa trên trung bình.

Z-score đo độ lệch của quan sát so với trung bình theo đơn vị độ lệch chuẩn.

Việc sử dụng cả hai phương pháp giúp kiểm tra mức độ trùng nhau của các quan sát được đánh dấu.

### 5.2. IQR

Công thức:

\[
IQR=Q_3-Q_1
\]

Ngưỡng dưới:

\[
LB=Q_1-1.5IQR
\]

Ngưỡng trên:

\[
UB=Q_3+1.5IQR
\]

Một quan sát được đánh dấu khi:

\[
x<LB \quad \text{hoặc} \quad x>UB
\]

Cách triển khai:

```python
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1

iqr_mask = (
    data.lt(q1 - 1.5 * iqr)
    | data.gt(q3 + 1.5 * iqr)
)
```

Sử dụng hệ số 1,5 theo quy tắc IQR thông dụng.

### 5.3. Z-score

Công thức:

\[
z_i=\frac{x_i-\mu}{\sigma}
\]

Ngưỡng đánh dấu:

\[
|z_i|>3
\]

Cách triển khai trong chương trình:

```python
std = data.std(ddof=0).replace(0, np.nan)
z_scores = (data - data.mean()) / std
z_mask = z_scores.abs().gt(3).fillna(False)
```

Tham số `ddof=0` được sử dụng để tính độ lệch chuẩn theo công thức chia cho số quan sát.

Việc thay độ lệch chuẩn bằng 0 thành NaN giúp tránh phép chia cho 0.

### 5.4. Kết quả trên dữ liệu gốc

| Biến | IQR | IQR (%) | Z-score | Z-score (%) |
|---|---:|---:|---:|---:|
| load | 9 | 0,034 | 3 | 0,011 |
| temperature | 219 | 0,833 | 125 | 0,475 |
| humidity | 127 | 0,483 | 54 | 0,205 |
| wind_speed | 321 | 1,220 | 173 | 0,658 |

Các kết quả này được in ra Terminal ở mục `OUTLIERS BY VARIABLE`.

Với các biến thời tiết, IQR và Z-score được áp dụng trực tiếp lên chuỗi gốc của từng biến.

Riêng `load`, chương trình tiếp tục thực hiện một bước điều chỉnh theo giờ-trong-tuần để phân tích ngoại lai phù hợp hơn với cấu trúc chu kỳ.

## 6. Tính phần dư phụ tải theo giờ-trong-tuần

### 6.1. Lý do thực hiện

Phụ tải điện có mức trung bình khác nhau giữa các giờ trong ngày.

Nếu áp dụng cùng một ngưỡng cho toàn bộ chuỗi, các giờ cao điểm có thể bị đánh giá không phù hợp so với những giờ thấp điểm.

Theo yêu cầu của Mục 3, chương trình tính phần dư bằng cách trừ phụ tải trung bình ứng với cùng một giờ trong tuần.

### 6.2. Xác định nhóm giờ-trong-tuần

Một tuần có:

\[
7\times24=168
\]

nhóm giờ.

Chỉ số nhóm được tính:

\[
g_t=24d_t+h_t
\]

Trong đó:

- \(d_t\): thứ trong tuần, từ 0 đến 6.
- \(h_t\): giờ trong ngày, từ 0 đến 23.

Ví dụ:

- Thứ Hai lúc 00:00 thuộc nhóm 0.
- Thứ Hai lúc 18:00 thuộc nhóm 18.
- Thứ Ba lúc 00:00 thuộc nhóm 24.
- Chủ Nhật lúc 23:00 thuộc nhóm 167.

### 6.3. Tính trung bình và phần dư

Phụ tải trung bình của nhóm \(g\):

\[
\overline{L}_g=
\frac{1}{N_g}\sum_{t:g_t=g}L_t
\]

Phần dư:

\[
r_t=L_t-\overline{L}_{g_t}
\]

Cách triển khai:

```python
hour_of_week = (
    df.index.day_of_week * 24
    + df.index.hour
)

expected = (
    df["load"]
    .groupby(hour_of_week)
    .transform("mean")
)

residual = df["load"] - expected
```

Hàm `transform("mean")` trả về giá trị trung bình tương ứng cho từng quan sát, bảo đảm có thể trừ trực tiếp với chuỗi phụ tải gốc.

### 6.4. Ý nghĩa phần dư

Nếu phần dư bằng 0, phụ tải tại thời điểm đó bằng mức trung bình của nhóm giờ-trong-tuần tương ứng.

Nếu phần dư dương, phụ tải cao hơn mức trung bình nhóm.

Nếu phần dư âm, phụ tải thấp hơn mức trung bình nhóm.

Cách điều chỉnh này loại bỏ thành phần trung bình lặp lại theo giờ và thứ, nhưng không loại bỏ hoàn toàn mọi yếu tố mùa vụ.

Đặc biệt, phần dư vẫn có thể chứa biến động theo mùa hằng năm.

### 6.5. Kết quả ngoại lai trên phần dư

| Phương pháp | Số điểm | Tỷ lệ |
|---|---:|---:|
| IQR | 2.099 | 7,980% |
| Z-score | 271 | 1,030% |

Chương trình tiếp tục kiểm tra giao nhau giữa hai phương pháp:

```python
iqr_only = iqr_mask & ~z_mask
z_only = z_mask & ~iqr_mask
both = iqr_mask & z_mask
```

Kết quả:

```text
IQR only: 1,828
Z-score only: 0
Both methods: 271
IQR total: 2,099
Z-score total: 271
```

Kiểm tra tính nhất quán:

\[
1828+271=2099
\]

Toàn bộ các điểm được Z-score đánh dấu cũng được IQR đánh dấu.

### 6.6. Giới hạn

Phần dư được tính bằng trung bình giờ-trong-tuần trên toàn bộ giai đoạn quan sát.

Đây là phép điều chỉnh phục vụ EDA, không phải bước tạo đặc trưng dự báo sử dụng trực tiếp cho tập test.

Nếu muốn sử dụng trung bình nhóm làm đặc trưng trong mô hình, cần ước lượng các thống kê này chỉ từ dữ liệu huấn luyện để tránh sử dụng thông tin tương lai.

Ngoài ra, các cụm phần dư dương lớn trong mùa hè có thể phản ánh biến động mùa vụ thực tế. Không được xem tất cả các điểm được đánh dấu là dữ liệu sai.

## 7. Biểu đồ ngoại lai – s3_outlier.png

### 7.1. Cấu trúc

Biểu đồ gồm hai phần:

- Phần trên: phụ tải điện gốc theo thời gian.
- Phần dưới: phần dư sau điều chỉnh giờ-trong-tuần và các điểm ngoại lai.

### 7.2. Cách hiển thị

Chuỗi phụ tải gốc được vẽ bằng biểu đồ đường.

Chuỗi phần dư được vẽ bằng đường màu xám.

Các điểm ngoại lai được phân màu:

- Cam: chỉ IQR.
- Đỏ: chỉ Z-score, nếu có.
- Tím: cả hai phương pháp.

Trong dữ liệu hiện tại không có điểm thuộc nhóm chỉ Z-score, nên biểu đồ thực tế chỉ xuất hiện các điểm ngoại lai màu cam và tím.

### 7.3. Kết quả

Các điểm ngoại lai dương tập trung thành cụm trong những giai đoạn phụ tải tăng mạnh.

Một số điểm âm lớn xuất hiện vào những thời điểm phụ tải giảm đột ngột.

Biểu đồ thể hiện rõ kết quả của hai phương pháp và không làm mất các điểm được cả hai phương pháp phát hiện.

### 7.4. Vấn đề đã xử lý

Trong quá trình trình bày, cần phân biệt tổng số điểm IQR với số điểm chỉ IQR.

Số 1.828 là nhóm chỉ IQR, không phải tổng số điểm IQR.

Vì vậy, chú thích biểu đồ hiển thị riêng:

- IQR total: 2.099.
- Z-score total: 271.
- IQR only: 1.828.
- Both methods: 271.

Cách trình bày này tránh việc cộng nhầm hoặc hiểu rằng một phần dữ liệu đã bị loại bỏ.

## 8. Phân tích chu kỳ – s3_chu_ky.png

### 8.1. Chu kỳ theo giờ

Sử dụng:

```python
hourly = (
    df["load"]
    .groupby(df.index.hour)
    .mean()
)
```

Kết quả gồm 24 giá trị trung bình, tương ứng 24 giờ trong ngày.

Phụ tải trung bình thấp nhất vào 04:00 và cao nhất vào 18:00.

Biểu đồ đường được lựa chọn vì các giờ có thứ tự liên tục, giúp thể hiện quá trình tăng giảm phụ tải trong ngày.

Hai điểm cực trị được đánh dấu để dễ quan sát.

### 8.2. Chu kỳ theo thứ

Sử dụng:

```python
weekly = (
    df["load"]
    .groupby(df.index.day_of_week)
    .mean()
    .reindex(range(7))
)
```

Kết quả:

| Thứ | Trung bình |
|---|---:|
| Mon | 201,5 |
| Tue | 203,3 |
| Wed | 202,9 |
| Thu | 203,8 |
| Fri | 206,2 |
| Sat | 205,7 |
| Sun | 199,9 |

Biểu đồ cột được sử dụng để so sánh bảy nhóm ngày.

Các giá trị trung bình được ghi trực tiếp phía trên cột.

### 8.3. Nhận xét kỹ thuật

Sự khác biệt giữa các giờ trong ngày lớn hơn sự khác biệt giữa trung bình các thứ.

Tuy nhiên, biểu đồ trung bình theo thứ chưa thể hiện toàn bộ cấu trúc lặp lại theo tuần.

Do đó, kết quả này cần được xem cùng ACF tại lag 168.

## 9. Phân tích ACF – s3_acf.png

### 9.1. Mục đích

ACF được sử dụng để xác định mức độ liên hệ giữa phụ tải tại thời điểm hiện tại và các thời điểm trước đó.

### 9.2. Công thức

Với độ trễ \(k\):

\[
ACF(k)=
\frac{
\sum_{t=k+1}^{N}(L_t-\overline{L})(L_{t-k}-\overline{L})
}{
\sum_{t=1}^{N}(L_t-\overline{L})^2
}
\]

Trong đó:

- \(L_t\): phụ tải tại thời điểm \(t\).
- \(\overline{L}\): phụ tải trung bình.
- \(k\): số giờ trễ.
- \(N\): tổng số quan sát.

### 9.3. Tham số

Chương trình sử dụng:

```python
from statsmodels.tsa.stattools import acf

values = acf(
    df["load"].to_numpy(),
    nlags=200,
    fft=True,
)
```

Ý nghĩa:

- `nlags=200`: tính ACF đến độ trễ 200 giờ.
- `fft=True`: sử dụng thuật toán dựa trên biến đổi Fourier để tăng hiệu quả tính toán.
- Chuỗi đầu vào: phụ tải gốc, chưa điều chỉnh mùa vụ.

### 9.4. Kết quả

```text
ACF lag 24: 0.9873
ACF lag 168: 0.9640
```

ACF tại lag 24 và 168 đều rất cao.

Kết quả cho thấy chuỗi có sự lặp lại mạnh theo ngày và tuần.

### 9.5. Cách vẽ

Biểu đồ hiển thị các hệ số từ lag 1 đến lag 200 bằng các vạch thẳng đứng.

Hai đường nét đứt đánh dấu:

- Lag 24.
- Lag 168.

Giá trị ACF tương ứng được ghi trong chú thích.

Cách thể hiện này giúp nhận diện các đỉnh lặp lại và làm nổi bật hai chu kỳ quan trọng.

### 9.6. Giới hạn

ACF được tính trên chuỗi gốc nên có thể chịu ảnh hưởng của xu hướng và mùa vụ.

Hệ số cao không chứng minh quan hệ nhân quả và không bảo đảm một mô hình dự báo sẽ đạt sai số thấp.

## 10. Phụ tải theo thời gian – s3_load.png

### 10.1. Phụ tải theo giờ

Vẽ toàn bộ chuỗi `load` theo thời gian.

Mục đích là quan sát:

- Biến động ngắn hạn.
- Các giai đoạn phụ tải cao.
- Những thời điểm giảm đột ngột.
- Sự lặp lại giữa các năm.

Với 26.304 quan sát, đường phụ tải theo giờ có mật độ rất dày.

### 10.2. Phụ tải trung bình ngày

Sử dụng:

```python
daily = df["load"].resample("D").mean()
```

Phương pháp này chuyển chuỗi theo giờ thành chuỗi trung bình ngày.

Nhờ giảm số lượng điểm hiển thị, các đợt tăng và giảm kéo dài trở nên dễ quan sát hơn.

### 10.3. Trung bình trượt 7 ngày

Chương trình sử dụng:

```python
trend = daily.rolling(
    7,
    center=True,
    min_periods=4
).mean()
```

Giải thích:

- `7`: cửa sổ gồm bảy ngày.
- `center=True`: cửa sổ được căn giữa ngày đang xét.
- `min_periods=4`: cần ít nhất bốn giá trị hợp lệ để tính trung bình.

Đường trung bình trượt được sử dụng để làm mượt chuỗi và quan sát xu hướng.

Do sử dụng cửa sổ căn giữa, giá trị trung bình trượt tại một ngày có thể dùng cả dữ liệu trước và sau ngày đó.

Vì vậy, đây chỉ là công cụ trực quan hóa trong EDA, không được sử dụng trực tiếp làm đặc trưng dự báo tương lai nếu chưa điều chỉnh để tránh rò rỉ dữ liệu.

### 10.4. Kết quả

Biểu đồ cho thấy phụ tải có các đợt tăng rõ vào những giai đoạn giữa năm.

Mức phụ tải giữa các năm có khác biệt nhưng hình dạng biến động theo mùa vẫn có sự lặp lại.

Đường trung bình trượt giúp thể hiện các đợt biến động kéo dài rõ hơn so với chuỗi theo giờ.

## 11. Phụ tải và nhiệt độ theo mùa – s3_load_vs_temp.png

### 11.1. Mục đích

Khảo sát quan hệ giữa phụ tải và nhiệt độ, đồng thời so sánh sự khác biệt giữa các mùa.

### 11.2. Phân nhóm mùa

Dựa trên tháng của mỗi quan sát:

| Mùa | Tháng |
|---|---|
| Winter | 12, 1, 2 |
| Spring | 3, 4, 5 |
| Summer | 6, 7, 8 |
| Autumn | 9, 10, 11 |

Mỗi mùa được thể hiện bằng một màu riêng.

### 11.3. Biểu đồ phân tán

Mỗi quan sát là một điểm với:

- Hoành độ: nhiệt độ.
- Tung độ: phụ tải.
- Màu sắc: mùa.

Do số lượng quan sát lớn, chương trình sử dụng điểm có kích thước nhỏ và độ trong suốt cao để hạn chế chồng lấp.

### 11.4. Hồi quy tuyến tính theo mùa

Với từng mùa, đường xu hướng được ước lượng bằng mô hình:

\[
L=\beta_0+\beta_1T+\varepsilon
\]

Trong đó:

- \(L\): phụ tải.
- \(T\): nhiệt độ.
- \(\beta_0\): hệ số chặn.
- \(\beta_1\): hệ số góc.
- \(\varepsilon\): sai số.

Sử dụng:

```python
slope, intercept = np.polyfit(x, y, 1)
```

Hàm `np.polyfit` với bậc 1 thực hiện ước lượng đường thẳng theo phương pháp bình phương tối thiểu.

### 11.5. Giới hạn phạm vi hiển thị đường xu hướng

Đường hồi quy được ước lượng từ toàn bộ dữ liệu hợp lệ của từng mùa.

Tuy nhiên, phạm vi hiển thị được giới hạn trong khoảng phân vị nhiệt độ 5%–95%.

Cách triển khai:

```python
x_low, x_high = np.percentile(x, [5, 95])

x_line = np.linspace(x_low, x_high, 100)
y_line = slope * x_line + intercept
```

Việc giới hạn này chỉ ảnh hưởng đến đoạn đường được vẽ, không loại bỏ quan sát khi ước lượng hệ số.

### 11.6. Vấn đề đã xử lý

Ở phiên bản trước, đường hồi quy mùa hè kéo dài lên vùng phụ tải cao tại các mức nhiệt độ cực trị, dễ khiến người xem hiểu nhầm đường xu hướng là giá trị dự báo chính xác.

Để cải thiện, nhóm giữ phương pháp hồi quy tuyến tính nhưng chỉ hiển thị đường trong khoảng nhiệt độ phổ biến hơn của từng mùa.

Kết quả là đường xu hướng không còn bị kéo quá xa vào vùng ít dữ liệu.

### 11.7. Giới hạn

Biểu đồ phân tán và hồi quy tuyến tính chỉ thể hiện mối liên hệ thống kê.

Tại cùng một mức nhiệt độ, phụ tải vẫn có thể thay đổi đáng kể do giờ trong ngày, thứ trong tuần và những yếu tố khác.

Không sử dụng đường xu hướng này như mô hình dự báo chính thức.

## 12. Tương quan Pearson – s3_corr.png

### 12.1. Mục đích

Đánh giá mức độ liên hệ tuyến tính giữa:

- `load`
- `temperature`
- `humidity`
- `wind_speed`

### 12.2. Công thức

Hệ số Pearson giữa \(X\) và \(Y\):

\[
r_{XY}=
\frac{
\sum_i(X_i-\overline{X})(Y_i-\overline{Y})
}{
\sqrt{\sum_i(X_i-\overline{X})^2}
\sqrt{\sum_i(Y_i-\overline{Y})^2}
}
\]

Giá trị nằm trong khoảng từ -1 đến 1.

### 12.3. Cách tính

Sử dụng:

```python
corr = df[COLS].corr(method="pearson")
```

Sau đó vẽ heatmap:

```python
sns.heatmap(
    corr,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
)
```

Các tham số chính:

- `annot=True`: hiển thị hệ số trên từng ô.
- `fmt=".3f"`: làm tròn ba chữ số thập phân khi hiển thị.
- `center=0`: lấy 0 làm trung tâm thang màu.
- `vmin=-1`, `vmax=1`: cố định giới hạn thang màu.
- `square=True`: hiển thị các ô dạng hình vuông.

### 12.4. Kết quả

```text
               load  temperature  humidity  wind_speed
load         1.0000       0.6559   -0.5277      0.2803
temperature  0.6559       1.0000   -0.5711      0.1963
humidity    -0.5277      -0.5711    1.0000     -0.1773
wind_speed   0.2803       0.1963   -0.1773      1.0000
```

Trong ba biến thời tiết, nhiệt độ có trị tuyệt đối của hệ số Pearson với phụ tải lớn nhất.

Độ ẩm có tương quan âm với phụ tải, trong khi tốc độ gió có tương quan dương yếu hơn.

### 12.5. Giới hạn

Pearson chỉ đo tương quan tuyến tính.

Các hệ số được tính trên toàn bộ dữ liệu và chưa tách riêng tác động của thời gian, mùa vụ hoặc những yếu tố khác.

Vì vậy, không sử dụng Pearson để kết luận quan hệ nhân quả.

## 13. Các vấn đề và điều chỉnh trong quá trình thực hiện

### 13.1. Dữ liệu theo giờ có mật độ lớn

**Vấn đề:** Biểu đồ phụ tải gốc gồm 26.304 điểm nên khó quan sát xu hướng dài hạn.

**Cách xử lý:** Bổ sung chuỗi trung bình ngày và đường trung bình trượt 7 ngày.

**Kết quả:** Biểu đồ thể hiện được cả biến động ngắn hạn và xu hướng theo mùa.

### 13.2. Ngoại lai bị ảnh hưởng bởi chu kỳ

**Vấn đề:** Phụ tải điện có mức trung bình rất khác nhau giữa các giờ.

**Cách xử lý:** Tính phần dư sau khi trừ trung bình theo giờ-trong-tuần, sau đó mới áp dụng IQR và Z-score cho phụ tải.

**Kết quả:** Phân tích được các độ lệch so với mức phụ tải thường gặp ở từng nhóm giờ-trong-tuần.

### 13.3. Ngoại lai tập trung theo mùa

**Vấn đề:** Nhiều điểm ngoại lai dương tập trung vào mùa hè.

**Nguyên nhân có thể:** Phần dư chưa loại bỏ chu kỳ mùa vụ hằng năm.

**Cách xử lý:** Giữ nguyên phương pháp được phân công, bổ sung giải thích giới hạn và không tự động xóa dữ liệu.

**Kết quả:** Tránh diễn giải sai các đợt tăng phụ tải thực tế thành lỗi dữ liệu.

### 13.4. Nhầm lẫn số lượng ngoại lai

**Vấn đề:** Số 1.828 điểm chỉ IQR có thể bị hiểu thành tổng số điểm IQR.

**Cách xử lý:** Tách riêng các nhóm IQR only, Z-score only và Both methods, đồng thời ghi rõ tổng số điểm của từng phương pháp.

**Kết quả:** Bảo đảm quan hệ số liệu nhất quán:

\[
1828+271=2099
\]

### 13.5. Đường hồi quy nhiệt độ kéo dài quá xa

**Vấn đề:** Đường hồi quy có thể kéo dài vào vùng nhiệt độ ít quan sát và gây khó khăn khi diễn giải.

**Cách xử lý:** Chỉ hiển thị đường trong khoảng phân vị 5%–95% của nhiệt độ từng mùa.

**Kết quả:** Biểu đồ tập trung hơn vào vùng dữ liệu phổ biến, trong khi hệ số hồi quy vẫn được ước lượng từ toàn bộ dữ liệu hợp lệ.

### 13.6. Cảnh báo thuộc tính ngày trong tuần

Trong quá trình hoàn thiện, cách truy cập thuộc tính ngày trong tuần được thống nhất sử dụng:

```python
df.index.day_of_week
```

Thay đổi này nhằm tương thích tốt hơn với môi trường pandas đang sử dụng, không làm thay đổi công thức tính nhóm giờ-trong-tuần hoặc kết quả thống kê.

## 14. Kiểm tra kết quả đầu ra

Sau khi chạy chương trình, Terminal hiển thị:

```text
DATASET
Observations: 26,304
Period: 2012-01-01 00:00:00 to 2014-12-31 23:00:00

MISSING VALUES
             Missing  Missing (%)
load               0          0.0
temperature        0          0.0
humidity           0          0.0
wind_speed         0          0.0

OUTLIERS BY VARIABLE
             IQR  IQR (%)  Z-score  Z-score (%)
load           9    0.034        3        0.011
temperature  219    0.833      125        0.475
humidity     127    0.483       54        0.205
wind_speed   321    1.220      173        0.658

LOAD RESIDUAL OUTLIERS
                IQR  IQR (%)  Z-score  Z-score (%)
load_residual  2099     7.98      271         1.03

OUTLIER OVERLAP
IQR only: 1,828
Z-score only: 0
Both methods: 271
IQR total: 2,099
Z-score total: 271

CORRELATION
               load  temperature  humidity  wind_speed
load         1.0000       0.6559   -0.5277      0.2803
temperature  0.6559       1.0000   -0.5711      0.1963
humidity    -0.5277      -0.5711    1.0000     -0.1773
wind_speed   0.2803       0.1963   -0.1773      1.0000

SEASONALITY
Peak hour: 18:00
Lowest hour: 04:00
Peak weekday: Fri

AUTOCORRELATION
ACF lag 24: 0.9873
ACF lag 168: 0.9640
```

Kết quả trên được đối chiếu với các bảng thống kê và nhận xét trong `s3_report.md`.

## 15. Danh sách sản phẩm bàn giao

| STT | Đường dẫn | Nội dung |
|---|---|---|
| 1 | `src/s3_eda.py` | Chương trình phân tích EDA |
| 2 | `docs/lab4/figures/s3_load.png` | Phụ tải theo thời gian |
| 3 | `docs/lab4/figures/s3_outlier.png` | Phân tích ngoại lai |
| 4 | `docs/lab4/figures/s3_chu_ky.png` | Chu kỳ ngày và tuần |
| 5 | `docs/lab4/figures/s3_acf.png` | ACF 200 độ trễ |
| 6 | `docs/lab4/figures/s3_load_vs_temp.png` | Phụ tải và nhiệt độ theo mùa |
| 7 | `docs/lab4/figures/s3_corr.png` | Ma trận tương quan Pearson |
| 8 | `docs/lab4/parts/s3_report.md` | Báo cáo kết quả EDA |
| 9 | `docs/lab4/parts/s3_process.md` | Nhật ký quy trình thực hiện |

## 16. Kết luận

Mục 3 đã thực hiện các nội dung phân tích khám phá dữ liệu gồm kiểm tra missing values, phát hiện ngoại lai, phân tích chu kỳ ngày và tuần, tính ACF 200 độ trễ, khảo sát phụ tải theo thời gian, trực quan hóa quan hệ với nhiệt độ theo mùa và phân tích tương quan Pearson.

Kết quả cho thấy phụ tải điện có tính chu kỳ rõ rệt theo giờ và tuần, đồng thời có biến động theo mùa và mối liên hệ đáng kể với các biến thời tiết.

Phương pháp phát hiện ngoại lai trên phần dư giúp xem xét độ lệch so với mức phụ tải trung bình theo giờ-trong-tuần. Tuy nhiên, kết quả vẫn chịu ảnh hưởng của mùa vụ hằng năm, nên các điểm được đánh dấu không được mặc nhiên xem là dữ liệu sai.

Toàn bộ quá trình được thực hiện trên dữ liệu đã xử lý, không chỉnh sửa CSV, không tự động loại bỏ ngoại lai và không thực hiện các bước thuộc giai đoạn huấn luyện mô hình.

Các kết quả và biểu đồ được lưu riêng để phục vụ tổng hợp báo cáo chung của nhóm.
