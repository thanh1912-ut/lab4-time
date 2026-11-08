"""Mục 6 - Mô hình 1: VAR (thống kê). Người làm: Lê Quỳnh.

File làm gì:
  1. ADF test từng biến trên train; sai phân bậc 1 (cùng 1 cách cho cả 4 biến) nếu có biến p > 0,05
  2. Chọn bậc trễ p bằng select_order(maxlags=48) theo AIC/BIC
  3. Huấn luyện VAR(p) trên train, in hệ số phương trình load, kiểm định Granger
  4. Walk-forward trên test bằng backtest.walk_forward (refit mỗi 168 giờ)
  5. Lưu data/processed/pred_var.csv (Datetime, y_true, y_pred), hình m1_var_1tuan.png,
     bản nháp docs/lab4/parts/m1_report.md

Chạy từ thư mục gốc:  python src/m1_var.py
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from utils import COLS, TARGET, PROCESSED, FIG, ROOT, load_data, metrics
from features import fit_scale, inverse_load
from backtest import walk_forward

warnings.filterwarnings("ignore")

PARTS = ROOT / "docs" / "lab4" / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
MAXLAGS = 48
CRITERION = "bic"      # "aic" hoặc "bic" - p dùng để huấn luyện
REFIT_EVERY = 168      # huấn luyện lại mỗi 1 tuần
T = COLS.index(TARGET)


# ------------------------------------------------------------------ tiện ích
def md_table(df: pd.DataFrame, index=True, floatfmt=4) -> str:
    """DataFrame -> bảng Markdown (không cần thư viện tabulate)."""
    d = df.reset_index() if index else df.copy()
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v)
                 for v in r.tolist()]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def adf_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        stat, p, *_ = adfuller(df[c].dropna(), autolag="AIC")
        rows.append({"biến": c, "ADF stat": stat, "p-value": p,
                     "dừng (p<0,05)": "có" if p < 0.05 else "không"})
    return pd.DataFrame(rows)


def make_fit_fn(p: int, diff: bool):
    """fit_fn(history_df) -> VAR results (history là dữ liệu đã chuẩn hóa)."""
    def fit_fn(hist):
        data = hist.diff().dropna() if diff else hist
        return VAR(data).fit(p)
    return fit_fn


def make_predict_fn(diff: bool):
    """predict_fn(res, history_df) -> load(t) đã chuẩn hóa, chỉ dùng dữ liệu < t."""
    def predict_fn(res, hist):
        p = res.k_ar
        if diff:
            last = hist.diff().iloc[-p:].to_numpy()
            return hist[TARGET].iloc[-1] + res.forecast(last, 1)[0, T]   # cộng lại load_t
        return res.forecast(hist.to_numpy()[-p:], 1)[0, T]
    return predict_fn


def run_walk_forward(scaled, scaler, i2, p, diff, refit_every=REFIT_EVERY) -> pd.DataFrame:
    """Walk-forward trên test với VAR(p). Trả về DataFrame y_true, y_pred (thang đo gốc)."""
    pred_z = walk_forward(make_fit_fn(p, diff), make_predict_fn(diff),
                          scaled, i2, len(scaled), refit_every=refit_every)
    out = pd.DataFrame({
        "y_true": inverse_load(scaled[TARGET].iloc[i2:].to_numpy(), scaler),
        "y_pred": inverse_load(pred_z, scaler),
    }, index=scaled.index[i2:])
    out.index.name = "Datetime"
    return out


# ---------------------------------------------------------------------- main
def main():
    df = load_data()
    scaled, scaler, (i1, i2) = fit_scale(df)
    train = scaled.iloc[:i1]
    print(f"train {len(train)} giờ | val {i2 - i1} giờ | test {len(scaled) - i2} giờ")

    # 1. ADF -------------------------------------------------------------
    adf0 = adf_table(train)
    print("\n== ADF trên train (chuỗi gốc) ==\n", adf0.round(4).to_string(index=False))
    diff = bool((adf0["p-value"] > 0.05).any())
    adf1 = None
    if diff:
        adf1 = adf_table(train.diff().dropna())
        print("\n== ADF sau sai phân bậc 1 (cả 4 biến) ==\n", adf1.round(4).to_string(index=False))
    else:
        print("-> Cả 4 biến đều dừng (p < 0,05): không cần sai phân.")
    data = train.diff().dropna() if diff else train

    # 2. Chọn bậc trễ ------------------------------------------------------
    sel = VAR(data).select_order(MAXLAGS)
    ics = pd.DataFrame({k: sel.ics[k] for k in ("aic", "bic")})
    ics.index.name = "p"
    ics.to_csv(PROCESSED / "var_aic_bic.csv")
    p_aic, p_bic = int(sel.selected_orders["aic"]), int(sel.selected_orders["bic"])
    p = max(1, p_aic if CRITERION == "aic" else p_bic)
    print(f"\n== Chọn bậc trễ == p theo AIC = {p_aic}, p theo BIC = {p_bic} -> dùng p = {p} ({CRITERION.upper()})")
    print(ics.iloc[[1, 2, 6, 12, 24, 36, 48]].round(3))

    # 3. Huấn luyện + hệ số load + Granger ----------------------------------
    res = VAR(data).fit(p)
    print("\nVAR ổn định:", res.is_stable())
    coef = pd.DataFrame({"coef": res.params[TARGET], "t": res.tvalues[TARGET]})
    top = coef.reindex(coef["t"].abs().sort_values(ascending=False).index).head(12)
    print("\n== 12 hệ số có |t| lớn nhất trong phương trình load ==\n", top.round(4))

    weather = [c for c in COLS if c != TARGET]
    g_rows = []
    for w in weather + ["cả 3 biến"]:
        causing = weather if w == "cả 3 biến" else [w]
        g = res.test_causality(TARGET, causing, kind="f")
        g_rows.append({"nguyên nhân": w, "F": g.test_statistic, "p-value": g.pvalue,
                       "Granger-cause load": "có" if g.pvalue < 0.05 else "không"})
    granger = pd.DataFrame(g_rows)
    print("\n== Granger -> load ==\n", granger.round(4).to_string(index=False))

    # 4. Walk-forward -------------------------------------------------------
    print(f"\nWalk-forward trên test (refit mỗi {REFIT_EVERY} giờ), có thể mất vài phút...")
    t0 = time.time()
    out = run_walk_forward(scaled, scaler, i2, p, diff)
    secs = time.time() - t0
    m = metrics(out["y_true"], out["y_pred"])
    print("Sai số test (thang gốc, MW):", m, f"| {secs:.0f}s")
    out.to_csv(PROCESSED / "pred_var.csv")

    json.dump({"p": p, "diff": diff, "criterion": CRITERION, "p_aic": p_aic, "p_bic": p_bic},
              open(PROCESSED / "var_order.json", "w"))

    # 5. Hình 1 tuần ---------------------------------------------------------
    w = out.iloc[:168]
    plt.figure(figsize=(12, 4))
    plt.plot(w.index, w["y_true"], label="Thực tế")
    plt.plot(w.index, w["y_pred"], label=f"VAR({p}) dự báo 1 giờ tới", alpha=.85)
    plt.title(f"m1 - VAR({p}): 1 tuần đầu của tập test")
    plt.xlabel("Thời gian")
    plt.ylabel("load (MW)")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIG / "m1_var_1tuan.png", dpi=150)
    plt.close()

    # 6. Bản nháp báo cáo ----------------------------------------------------
    ic_show = ics.iloc[[1, 2, 6, 12, 24, 36, 48]].copy()
    txt = f"""# Mục 6 - Mô hình VAR (Lê Quỳnh)

## Kiểm tra tính dừng (ADF, trên tập train)
{md_table(adf0, index=False)}

{"Biến có p > 0,05 nên lấy sai phân bậc 1 cho cả 4 biến (cùng một cách để dễ đảo ngược). ADF sau sai phân:" if diff else "Cả 4 biến đều dừng nên không sai phân."}

{md_table(adf1, index=False) if adf1 is not None else ""}

## Chọn bậc trễ (select_order, maxlags={MAXLAGS})
{md_table(ic_show)}

p theo AIC = {p_aic}, p theo BIC = {p_bic}. Chọn **p = {p}** theo {CRITERION.upper()} (BIC phạt số tham số mạnh hơn nên thường chọn p nhỏ hơn; AIC dễ chọn p lớn, tốn tham số).

## Kiểm định Granger (tác động lên load)
{md_table(granger, index=False)}

## Sai số trên tập test (walk-forward, refit mỗi {REFIT_EVERY} giờ, thang đo gốc)
| RMSE (MW) | MAE (MW) | MAPE (%) |
|---|---|---|
| {m['RMSE']} | {m['MAE']} | {m['MAPE(%)']} |

Hình: `figures/m1_var_1tuan.png`. Thời gian chạy walk-forward: {secs:.0f} giây.
"""
    (PARTS / "m1_report.md").write_text(txt, encoding="utf-8")
    print("\nĐã lưu: pred_var.csv, m1_var_1tuan.png, parts/m1_report.md")


if __name__ == "__main__":
    main()
