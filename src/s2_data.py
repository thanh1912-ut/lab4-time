"""Mục 2: tải dữ liệu điện UCI, gộp theo giờ, ghép thời tiết Open-Meteo.
Người làm: Thành.
Chi tiết: docs/lab4/phan-cong.md, mục 5.1.
"""
from utils import RAW, PROCESSED, FIG, COLS  # noqa: F401

# TODO:
#  - Tải zip UCI ElectricityLoadDiagrams20112014 vào data/raw/, giải nén LD2011_2014.txt
#  - Đọc sep=';', decimal=','; lọc 2012-2014; load = tổng 370 khách hàng, resample('h').mean()
#  - Xử lý giờ đổi giờ (tháng 3 bằng 0, tháng 10 cộng dồn)
#  - Tải thời tiết Lisbon từ Open-Meteo (temperature, humidity, wind_speed), timezone Europe/Lisbon
#  - Ghép theo Datetime, kiểm tra đủ giờ; lưu data/processed/load_weather_hourly.csv
#  - Vẽ s2_4bien.png

if __name__ == "__main__":
    print("[s2_data.py] chưa làm")
