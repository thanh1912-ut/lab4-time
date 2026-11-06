"""Mục 4, 5: chuẩn hóa, chia tập, cửa sổ lag, rolling (chạy minh họa).
Người làm: Bao Dinh.
Chi tiết: docs/lab4/phan-cong.md, mục 5.2.
"""
from utils import load_data, COLS, L, SEED, PROCESSED, FIG, metrics  # noqa: F401
from features import fit_scale, make_windows, add_rolling, split_index  # noqa: F401

# TODO:
#  - fit_scale: in mean/std trước-sau trên train và test
#  - Số mẫu, mốc thời gian train/val/test; s4_split.png
#  - make_windows: X.shape (n, 24, 4); s5_window.png
#  - Bản phẳng 96 cột + 3 cột rolling

if __name__ == "__main__":
    print("[s4_s5_preprocess.py] chưa làm")
