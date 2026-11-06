"""Mục 8: Mô hình LSTM.
Người làm: Hoàng Nam.
Chi tiết: docs/lab4/phan-cong.md, mục 5.6.
"""
from utils import load_data, COLS, L, SEED, PROCESSED, FIG, metrics  # noqa: F401
from features import fit_scale, make_windows, add_rolling, inverse_load, split_index  # noqa: F401
from backtest import walk_forward  # noqa: F401

# TODO:
#  - X (n, 24, 4); LSTM(64) -> LSTM(32) -> Dense(1); MSE, Adam
#  - EarlyStopping trên validation, seed 42
#  - Walk-forward trên test; lưu pred_lstm.csv; m3_lstm_1tuan.png

if __name__ == "__main__":
    print("[m3_lstm.py] chưa làm")
