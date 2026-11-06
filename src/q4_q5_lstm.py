"""Câu hỏi 4 (LSTM học chu kỳ nào) và Câu 5 (có overfitting không).
Người làm: Hoàng Nam.
Chi tiết: docs/lab4/phan-cong.md, mục 5.6.
"""
from utils import load_data, COLS, L, SEED, PROCESSED, FIG, metrics  # noqa: F401
from features import fit_scale, make_windows, add_rolling, inverse_load, split_index  # noqa: F401
from backtest import walk_forward  # noqa: F401

# TODO:
#  - Sai số theo giờ/thứ; thử L = 168 hoặc thêm sin/cos thời gian
#  - Dự báo 48 giờ đệ quy
#  - Loss train/val theo epoch; RMSE train/val/test; thử bỏ EarlyStopping, thêm Dropout

if __name__ == "__main__":
    print("[q4_q5_lstm.py] chưa làm")
