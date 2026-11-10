"""Câu 2 - Khi nào VAR thất bại? Người làm: Lê Quỳnh.

File làm gì (cần chạy m1_var.py trước để có pred_var.csv và var_order.json):
  1. Sai số VAR theo giờ trong ngày và theo thứ trong tuần  -> q2_loi_theo_gio.png
  2. Thử p = 2, 24, 48 (thêm 168 bằng --p168): sai số, số tham số -> q2_theo_p.png
  3. Top 10 giờ sai nhiều nhất: ngày lễ / đổi giờ / nhiệt độ cực đoan?
  4. Dự báo nhiều bước (không cập nhật giá trị thật), hội tụ về trung bình -> q2_nhieu_buoc.png
  5. Ghi docs/lab4/parts/q2_report.md

Chạy từ thư mục gốc:  python src/q2_var_fail.py [--p168]
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf

from utils import COLS, TARGET, PROCESSED, FIG, SEED, load_data, metrics
from features import fit_scale
from m1_var import run_walk_forward, md_table, PARTS

warnings.filterwarnings("ignore")

T = COLS.index(TARGET)
DAYS = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"]
# Ngày lễ quốc gia Bồ Đào Nha và ngày đổi giờ nằm trong tập test (20/07 - 31/12/2014)
HOLIDAYS = {"2014-08-15": "Lễ Đức Mẹ lên trời", "2014-10-05": "Ngày Cộng hòa",
            "2014-11-01": "Lễ Các Thánh", "2014-12-01": "Ngày Khôi phục độc lập",
            "2014-12-08": "Lễ Đức Mẹ vô nhiễm", "2014-12-25": "Giáng sinh"}
DST_DAYS = {"2014-10-26": "Đổi giờ mùa đông"}
H_STAT, H_PLOT = 24, 72          # số bước dự báo nhiều bước để thống kê / để vẽ


def multistep(res, scaled, diff, starts, H):
    """Dự báo H giờ liền từ mỗi mốc t trong starts, KHÔNG cập nhật giá trị thật.
    Trả về (dự báo, thực tế) đều đã chuẩn hóa, shape (len(starts), H)."""
    p = res.k_ar
    F, A = [], []
    for t in starts:
        hist = scaled.iloc[:t]
        if diff:
            fc = res.forecast(hist.diff().iloc[-p:].to_numpy(), H)[:, T]
            fc = hist[TARGET].iloc[-1] + np.cumsum(fc)
        else:
            fc = res.forecast(hist.to_numpy()[-p:], H)[:, T]
        F.append(fc)
        A.append(scaled[TARGET].iloc[t:t + H].to_numpy())
    return np.array(F), np.array(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p168", action="store_true", help="thử thêm p = 168 (nặng)")
    args = ap.parse_args()

    if not (PROCESSED / "pred_var.csv").exists():
        raise SystemExit("Chưa có pred_var.csv - hãy chạy: python src/m1_var.py")
    cfg = json.load(open(PROCESSED / "var_order.json"))
    p0, diff = cfg["p"], cfg["diff"]

    df = load_data()
    scaled, scaler, (i1, i2) = fit_scale(df)
    s0 = scaler.scale_[0]
    d = pd.read_csv(PROCESSED / "pred_var.csv", parse_dates=["Datetime"], index_col="Datetime")
    d["err"] = d["y_true"] - d["y_pred"]
    d["abs_err"] = d["err"].abs()
    d["ape"] = d["abs_err"] / d["y_true"] * 100
    d["hour"], d["dow"] = d.index.hour, d.index.dayofweek

    # ---- 1. Theo giờ và theo thứ -------------------------------------------
    by_hour = d.groupby("hour")[["abs_err", "ape"]].mean().rename(columns={"abs_err": "MAE", "ape": "MAPE"})
    by_dow = d.groupby("dow")[["abs_err", "ape"]].mean().rename(columns={"abs_err": "MAE", "ape": "MAPE"})
    by_dow.index = DAYS
    ac = acf(d["err"], nlags=168)
    print("== 1. MAE theo giờ ==\n", by_hour.round(2).T)
    print("\n== MAE theo thứ ==\n", by_dow.round(2).T)
    print(f"ACF phần dư: lag24 = {ac[24]:.3f}, lag168 = {ac[168]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(by_hour.index, by_hour["MAE"])
    ax[0].set(title=f"q2 - Sai số VAR({p0}) theo giờ trong ngày", xlabel="giờ", ylabel="MAE (MW)")
    ax[1].bar(by_dow.index, by_dow["MAE"], color="tab:orange")
    ax[1].set(title=f"q2 - Sai số VAR({p0}) theo thứ trong tuần", xlabel="thứ", ylabel="MAE (MW)")
    for a in ax:
        a.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "q2_loi_theo_gio.png", dpi=150)
    plt.close(fig)

    # ---- 2. Thử p -------------------------------------------------------------
    ps = [2, 24, 48] + ([168] if args.p168 else [])
    if p0 not in ps:
        ps = sorted(ps + [p0])
    rows = []
    for p in ps:
        t0 = time.time()
        o = d[["y_true", "y_pred"]] if p == p0 else run_walk_forward(scaled, scaler, i2, p, diff)
        m = metrics(o["y_true"], o["y_pred"])
        rows.append({"p": p, "số tham số (4x4xp)": 16 * p, "RMSE": m["RMSE"], "MAE": m["MAE"],
                     "MAPE(%)": m["MAPE(%)"], "thời gian (s)": 0 if p == p0 else round(time.time() - t0)})
        print(f"p={p}: {m}")
    tab_p = pd.DataFrame(rows)
    tab_p.to_csv(PROCESSED / "q2_theo_p.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(tab_p["p"], tab_p["RMSE"], "o-", label="RMSE")
    ax[0].plot(tab_p["p"], tab_p["MAE"], "s-", label="MAE")
    ax[0].set(title="q2 - Sai số test theo bậc trễ p", xlabel="p", ylabel="MW")
    ax[0].legend()
    ax[1].plot(tab_p["p"], tab_p["số tham số (4x4xp)"], "o-", color="tab:red")
    ax[1].set(title="q2 - Số tham số theo p", xlabel="p", ylabel="số tham số hệ số")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "q2_theo_p.png", dpi=150)
    plt.close(fig)

    # ---- 3. Top 10 giờ sai nhiều nhất -----------------------------------------
    temp = df["temperature"]
    lo, hi = temp.quantile(.05), temp.quantile(.95)
    top = d.nlargest(10, "abs_err").copy()
    top["thứ"] = [DAYS[i] for i in top["dow"]]
    top["temperature"] = temp.reindex(top.index).to_numpy()
    day = top.index.strftime("%Y-%m-%d")
    top["ngày lễ"] = [HOLIDAYS.get(x, "") for x in day]
    top["đổi giờ"] = [DST_DAYS.get(x, "") for x in day]
    top["nhiệt độ cực đoan"] = np.where(top["temperature"] >= hi, "nóng (>P95)",
                                        np.where(top["temperature"] <= lo, "lạnh (<P5)", ""))
    top["bất thường"] = ((top["ngày lễ"] != "") | (top["đổi giờ"] != "") |
                         (top["nhiệt độ cực đoan"] != ""))
    top_show = top[["y_true", "y_pred", "err", "thứ", "temperature", "ngày lễ", "đổi giờ",
                    "nhiệt độ cực đoan"]].round(2)
    top_show.index = top_show.index.strftime("%Y-%m-%d %H:%M")
    top_show.index.name = "giờ"
    n_odd = int(top["bất thường"].sum())
    print("\n== 3. Top 10 giờ sai nhiều nhất ==\n", top_show.to_string())
    top_show.to_csv(PROCESSED / "q2_top10.csv")

    # ---- 4. Dự báo nhiều bước ---------------------------------------------------
    res = VAR(scaled.iloc[:i2].diff().dropna() if diff else scaled.iloc[:i2]).fit(p0)
    rng = np.random.default_rng(SEED)
    starts = np.sort(rng.choice(np.arange(i2, len(scaled) - H_PLOT), size=100, replace=False))
    F, A = multistep(res, scaled, diff, starts, H_PLOT)
    err = (F - A) * s0
    rmse_h = np.sqrt((err ** 2).mean(axis=0))
    mae_h = np.abs(err).mean(axis=0)
    # biên độ dao động dự báo ở ngày thứ 3 (h = 49..72) so với thực tế
    amp = np.mean(F[:, 48:].std(axis=1) / A[:, 48:].std(axis=1))
    amp1 = np.mean(F[:, :24].std(axis=1) / A[:, :24].std(axis=1))
    print(f"\n== 4. Nhiều bước: RMSE h=1: {rmse_h[0]:.1f}, h=24: {rmse_h[23]:.1f}, h=72: {rmse_h[-1]:.1f} MW")
    print(f"   Biên độ dao động dự báo / thực tế: ngày 1 = {amp1:.0%}, ngày 3 = {amp:.0%}")

    ex = 0                              # ví dụ: mốc đầu tiên trong starts
    idx = scaled.index[starts[ex]:starts[ex] + H_PLOT]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].plot(idx, A[ex] * s0 + scaler.mean_[0], label="Thực tế")
    ax[0].plot(idx, F[ex] * s0 + scaler.mean_[0], "--", label="VAR dự báo 72 giờ (không cập nhật)")
    ax[0].axhline(scaler.mean_[0], color="gray", ls=":", label="Trung bình train")
    ax[0].set(title=f"q2 - Ví dụ dự báo nhiều bước từ {idx[0]:%d/%m/%Y %H:%M}", xlabel="Thời gian", ylabel="load (MW)")
    ax[0].legend(fontsize=8)
    ax[1].plot(range(1, H_PLOT + 1), rmse_h, label="RMSE")
    ax[1].plot(range(1, H_PLOT + 1), mae_h, label="MAE")
    ax[1].axvline(H_STAT, color="gray", ls=":")
    ax[1].set(title="q2 - Sai số theo số giờ dự báo trước (100 mốc ngẫu nhiên)", xlabel="h (giờ)", ylabel="MW")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "q2_nhieu_buoc.png", dpi=150)
    plt.close(fig)

    # ---- 5. Báo cáo -----------------------------------------------------------------
    wh, bh = by_hour["MAE"].idxmax(), by_hour["MAE"].idxmin()
    wd, bd = by_dow["MAE"].idxmax(), by_dow["MAE"].idxmin()
    r2, rbig = tab_p.iloc[0], tab_p.iloc[-1]
    txt = f"""# Câu 2 - Khi nào VAR thất bại? (Lê Quỳnh)

Mô hình chính: VAR({p0}), {"có sai phân bậc 1" if diff else "không sai phân"}, walk-forward trên test (20/07 - 31/12/2014).

## 1. Sai số theo giờ và theo thứ
{md_table(by_hour.T.round(2))}

{md_table(by_dow.T.round(2))}

Hình: `figures/q2_loi_theo_gio.png`. ACF phần dư: lag 24 = {ac[24]:.3f}, lag 168 = {ac[168]:.3f}.

## 2. Thay đổi bậc trễ p
{md_table(tab_p, index=False, floatfmt=2)}

Hình: `figures/q2_theo_p.png`.

## 3. Top 10 giờ sai nhiều nhất
{md_table(top_show)}

{n_odd}/10 giờ rơi vào ngày lễ, ngày đổi giờ hoặc nhiệt độ cực đoan (<P5 hoặc >P95 của toàn bộ dữ liệu).

## 4. Dự báo nhiều bước (không cập nhật giá trị thật, 100 mốc ngẫu nhiên trong test)
RMSE: h = 1 là {rmse_h[0]:.1f} MW, h = 24 là {rmse_h[23]:.1f} MW, h = 72 là {rmse_h[-1]:.1f} MW.
Biên độ dao động trong ngày của dự báo so với thực tế: ngày 1 là {amp1:.0%}, ngày 3 là {amp:.0%}.
Hình: `figures/q2_nhieu_buoc.png`.

## Trả lời Câu 2
> Các nhận xét sau điền từ số liệu ở trên; đọc lại hình và sửa câu chữ cho khớp trước khi nộp.

1. Chu kỳ tuần: MAE cao nhất vào {wd} ({by_dow.loc[wd, 'MAE']:.1f} MW), thấp nhất vào {bd} ({by_dow.loc[bd, 'MAE']:.1f} MW); ACF phần dư tại lag 168 là {ac[168]:.2f}, nghĩa là còn cấu trúc tuần chưa được mô hình hóa.
2. Theo giờ: sai số lớn nhất lúc {wh}h ({by_hour.loc[wh, 'MAE']:.1f} MW), nhỏ nhất lúc {bh}h ({by_hour.loc[bh, 'MAE']:.1f} MW), tức VAR kém ở các giờ tải đổi nhanh.
3. Tăng p từ {int(r2['p'])} lên {int(rbig['p'])} đưa RMSE từ {r2['RMSE']} xuống {rbig['RMSE']} MW nhưng số tham số tăng từ {int(r2['số tham số (4x4xp)'])} lên {int(rbig['số tham số (4x4xp)'])}: bắt chu kỳ dài cần p rất lớn.
4. Top 10 giờ sai nhiều: {n_odd}/10 là ngày lễ, đổi giờ hoặc nhiệt độ cực đoan, tức VAR không có thông tin về sự kiện đặc biệt.
5. Dự báo nhiều bước: sai số tăng từ {rmse_h[0]:.1f} (h=1) lên {rmse_h[23]:.1f} MW (h=24) và dao động ngày 3 chỉ còn {amp:.0%} thực tế, dự báo dần phẳng về trung bình.
6. Kết luận: VAR thất bại khi quan hệ phi tuyến (load theo nhiệt độ hình chữ U), chu kỳ dài (tuần), chuỗi có gãy cấu trúc hoặc sự kiện đặc biệt, và khi số biến tăng (tham số tăng theo k²p).
"""
    (PARTS / "q2_report.md").write_text(txt, encoding="utf-8")
    print("\nĐã lưu: q2_loi_theo_gio.png, q2_theo_p.png, q2_nhieu_buoc.png, parts/q2_report.md")


if __name__ == "__main__":
    main()
