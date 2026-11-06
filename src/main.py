"""Chạy lần lượt toàn bộ Lab 4. Người viết: Thành.

Chạy từ thư mục gốc: python src/main.py
(Bài nâng cao nc_traffic.py nặng nên chạy riêng: python src/nc_traffic.py)
"""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

ORDER = [
    "s2_data.py",           # Thành
    "s3_eda.py",            # Phạm Lê Văn Bảo
    "q1_weather.py",        # Phạm Lê Văn Bảo
    "s4_s5_preprocess.py",  # Bao Dinh
    "m1_var.py",            # Lê Quỳnh
    "q2_var_fail.py",       # Lê Quỳnh
    "m2_xgboost.py",        # Hoàng Thắng
    "q3_xgb_weather.py",    # Hoàng Thắng
    "m3_lstm.py",           # Hoàng Nam
    "q4_q5_lstm.py",        # Hoàng Nam
    "s10_compare.py",       # Bao Dinh
]

if __name__ == "__main__":
    for name in ORDER:
        print(f"\n===== {name} =====")
        runpy.run_path(str(SRC / name), run_name="__main__")
