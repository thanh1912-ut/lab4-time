"""Câu hỏi 3: XGBoost học được gì từ thời tiết (và Câu 6: ML vs DL).
Người làm: Hoàng Thắng.
Chi tiết: docs/lab4/phan-cong.md, mục 5.5.
"""
from utils import load_data, COLS, L, SEED, PROCESSED, FIG, metrics  # noqa: F401
from features import fit_scale, make_windows, add_rolling, inverse_load, split_index  # noqa: F401
from backtest import walk_forward  # noqa: F401

# TODO:
#  - Gain theo nhóm biến và theo độ trễ
#  - Ablation: bỏ biến thời tiết
#  - (nên làm) partial dependence temperature
#  - Câu 6: bảng so sánh XGBoost vs LSTM (lấy số của Hoàng Nam)

if __name__ == "__main__":
    print("[q3_xgb_weather.py] chưa làm")
