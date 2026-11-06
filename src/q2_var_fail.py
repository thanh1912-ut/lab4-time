"""Câu hỏi 2: khi nào VAR thất bại.
Người làm: Lê Quỳnh.
Chi tiết: docs/lab4/phan-cong.md, mục 5.4.
"""
from utils import load_data, COLS, L, SEED, PROCESSED, FIG, metrics  # noqa: F401
from features import fit_scale, make_windows, add_rolling, inverse_load, split_index  # noqa: F401
from backtest import walk_forward  # noqa: F401

# TODO:
#  - Sai số theo giờ/thứ
#  - Thử p = 2, 24, 48 (168)
#  - Top 10 giờ sai nhiều nhất
#  - Dự báo 24 giờ liền không cập nhật

if __name__ == "__main__":
    print("[q2_var_fail.py] chưa làm")
