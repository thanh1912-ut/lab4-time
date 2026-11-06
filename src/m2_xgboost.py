"""Mục 7: Mô hình XGBoost.
Người làm: Hoàng Thắng.
Chi tiết: docs/lab4/phan-cong.md, mục 5.5.
"""
from utils import load_data, COLS, L, SEED, PROCESSED, FIG, metrics  # noqa: F401
from features import fit_scale, make_windows, add_rolling, inverse_load, split_index  # noqa: F401
from backtest import walk_forward  # noqa: F401

# TODO:
#  - X phẳng 96 lag + 3 rolling, y = load(t+1)
#  - Tune max_depth, n_estimators với early stopping trên validation
#  - Walk-forward trên test; lưu pred_xgb.csv
#  - m2_xgb_1tuan.png, m2_learning_curve.png

if __name__ == "__main__":
    print("[m2_xgboost.py] chưa làm")
