"""Mục 6: Mô hình VAR.
Người làm: Lê Quỳnh.
Chi tiết: docs/lab4/phan-cong.md, mục 5.4.
"""
from utils import load_data, COLS, L, SEED, PROCESSED, FIG, metrics  # noqa: F401
from features import fit_scale, make_windows, add_rolling, inverse_load, split_index  # noqa: F401
from backtest import walk_forward  # noqa: F401

# TODO:
#  - ADF từng biến, sai phân nếu cần
#  - select_order(maxlags=48): AIC, BIC
#  - fit, walk-forward trên test, chỉ lấy cột load
#  - Lưu pred_var.csv; m1_var_1tuan.png; (nên làm) Granger

if __name__ == "__main__":
    print("[m1_var.py] chưa làm")
