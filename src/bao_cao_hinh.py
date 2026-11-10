"""Vẽ 18 hình cho báo cáo tóm tắt docs/lab4/bao-cao/bao-cao-lab4.md.

Chạy sau src/main.py (cần các file trong data/processed/):
    python src/bao_cao_hinh.py
Hình được ghi vào docs/lab4/bao-cao/hinh/.
"""
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
P = REPO / "data/processed"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs/lab4/bao-cao/hinh"
OUT.mkdir(parents=True, exist_ok=True)

SURF, PLANE = "#fcfcfb", "#f9f9f7"
INK, SEC, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C = {"VAR": "#2a78d6", "XGB": "#eb6834", "LSTM": "#1baf7a", "ENS": "#6250d6",
     "BASE": "#898781", "BASE2": "#c3c2b7"}
BLUE, BLUE_LIGHT = "#2a78d6", "#9ec5f4"

mpl.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 14, "text.color": INK,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": SEC, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def vn(x, d=2):
    s = f"{x:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def new(title, sub, size=(12, 6.2), ncols=1, top=0.80, **kw):
    fig, ax = plt.subplots(1, ncols, figsize=size, **kw)
    fig.text(0.04, 0.95, title, weight="bold", size=20, va="top")
    fig.text(0.04, 0.875, sub, color=SEC, size=14, va="top")
    fig.subplots_adjust(top=top, left=0.08, right=0.96, bottom=0.12)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, facecolor=SURF)
    plt.close(fig)
    print("saved", name)


def ygrid(ax):
    ax.yaxis.grid(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def xgrid(ax):
    ax.xaxis.grid(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)


# 1. Số liệu chính (stat tiles)
def tiles():
    fig = plt.figure(figsize=(12, 3.4))
    items = [("26.304", "giờ dữ liệu", "01/2012 đến 12/2014"),
             ("4 biến", "điện năng + nhiệt độ,", "độ ẩm, tốc độ gió"),
             ("≈ 6,1 MW", "RMSE của cả VAR,", "XGBoost và LSTM trên test"),
             ("−54%", "sai số so với cách", "“lấy cùng giờ hôm qua”")]
    w, gap = 0.225, 0.02
    for i, (big, l1, l2) in enumerate(items):
        x0 = 0.02 + i * (w + gap)
        ax = fig.add_axes([x0, 0.08, w, 0.84])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(mpl.patches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                     boxstyle="round,pad=0,rounding_size=0.06", fc=PLANE,
                     ec=GRID, lw=1.2, transform=ax.transAxes))
        ax.text(0.08, 0.68, big, size=34, weight="bold", va="center")
        ax.text(0.08, 0.36, l1, size=14, color=SEC, va="center")
        ax.text(0.08, 0.20, l2, size=14, color=SEC, va="center")
    save(fig, "01-so-lieu-chinh.png")


# 2. Dữ liệu và cách chia tập
def split():
    d = pd.read_csv(P / "load_weather_hourly.csv", parse_dates=["Datetime"]).set_index("Datetime")
    day = d["load"].resample("D").mean()
    t1, t2 = pd.Timestamp("2014-02-06 03:00"), pd.Timestamp("2014-07-20 13:00")
    fig, ax = new("Dữ liệu 3 năm, chia theo thời gian 70 / 15 / 15, không xáo trộn",
                  "Điện năng trung bình mỗi ngày (MW) của 318 khách hàng ở Bồ Đào Nha, ghép thời tiết trạm sân bay Lisbon")
    ax.axvspan(t1, t2, color="#f0efec", lw=0)
    ax.axvspan(t2, day.index[-1] + pd.Timedelta("1D"), color=GRID, lw=0)
    ax.plot(day.index, day.values, color=BLUE, lw=1.4)
    ax.set_ylim(80, 320)
    ax.set_xlim(day.index[0], day.index[-1] + pd.Timedelta("1D"))
    for x, lab in [(pd.Timestamp("2012-01-10"), "Train 70%\n18.412 giờ"),
                   (t1 + pd.Timedelta("8D"), "Validation 15%\n3.946 giờ"),
                   (t2 + pd.Timedelta("8D"), "Test 15%\n3.946 giờ")]:
        ax.text(x, 312, lab, size=13, color=INK, va="top", weight="bold")
    ax.set_ylabel("MW")
    ygrid(ax)
    save(fig, "02-chia-tap.png")


# 3. Chu kỳ ngày
def cycle():
    d = pd.read_csv(P / "load_weather_hourly.csv", parse_dates=["Datetime"])
    h = d.groupby(d.Datetime.dt.hour)["load"].mean()
    fig, ax = new("Chu kỳ ngày rất rõ: thấp nhất lúc 4h, cao nhất lúc 18h",
                  "Điện năng trung bình theo giờ trong ngày (MW), 2012–2014 · tự tương quan lag 24 = 0,987, lag 168 = 0,964")
    ax.plot(h.index, h.values, color=BLUE, lw=2.2, marker="o", ms=7, mec=SURF, mew=1.5)
    lo, hi = h.idxmin(), h.idxmax()
    ax.annotate(f"{vn(h[lo], 0)} MW lúc {lo}h", (lo, h[lo]), xytext=(lo + 1.2, h[lo] - 5),
                size=14, weight="bold", va="center")
    ax.annotate(f"{vn(h[hi], 0)} MW lúc {hi}h", (hi, h[hi]), xytext=(hi - 4.2, h[hi] + 20),
                size=14, weight="bold", va="center")
    ax.text(1.0, 215, "Từ 8h đến 10h\nđiện tăng khoảng 90 MW", size=13.5, color=SEC)
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([f"{i}h" for i in range(0, 24, 2)])
    ax.set_ylim(80, 310); ax.set_ylabel("MW")
    ygrid(ax)
    save(fig, "03-chu-ky-ngay.png")


# 4. Câu 1: tương quan thời tiết
def weather_corr():
    fig, (a1, a2) = new("Câu 1. Nhiệt độ ảnh hưởng mạnh nhất, rõ nhất vào mùa hè",
                        "Hệ số tương quan Pearson với điện năng, 26.304 giờ", ncols=2,
                        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.35})
    fig.subplots_adjust(left=0.13, top=0.72)
    names = ["Nhiệt độ", "Độ ẩm", "Tốc độ gió"]
    vals = [0.6559, -0.5277, 0.2803]
    cols = [BLUE, BLUE_LIGHT, BLUE_LIGHT]
    y = np.arange(3)[::-1]
    a1.barh(y, vals, color=cols, height=0.55, edgecolor=SURF, lw=2)
    for yi, v in zip(y, vals):
        a1.text(v + (0.03 if v > 0 else -0.03), yi, vn(v), va="center",
                ha="left" if v > 0 else "right", size=15, weight="bold")
    a1.axvline(0, color=AXIS, lw=1.2)
    a1.set_yticks(y); a1.set_yticklabels(names, size=14, color=INK)
    a1.set_xlim(-1.0, 0.95); a1.set_title("Cả năm", loc="left", size=15, color=SEC, pad=12)
    xgrid(a1); a1.spines["left"].set_visible(False); a1.tick_params(axis="y", length=0)

    seasons = ["Đông", "Xuân", "Hè", "Thu"]
    sv = [0.4337, 0.6653, 0.7203, 0.6154]
    x = np.arange(4)
    a2.bar(x, sv, color=[BLUE_LIGHT, BLUE_LIGHT, BLUE, BLUE_LIGHT], width=0.55, edgecolor=SURF, lw=2)
    for xi, v in zip(x, sv):
        a2.text(xi, v + 0.02, vn(v), ha="center", size=15, weight="bold")
    a2.set_xticks(x); a2.set_xticklabels(seasons, size=14, color=INK)
    a2.set_ylim(0, 0.85); a2.set_title("Riêng nhiệt độ, theo mùa", loc="left", size=15, color=SEC, pad=12)
    ygrid(a2); a2.tick_params(axis="x", length=0)
    save(fig, "04-cau1-thoi-tiet.png")


# 5. VAR: bậc trễ
def var_p():
    q = pd.read_csv(P / "q2_theo_p.csv")
    fig, ax = new("VAR cần nhìn đủ hơn 1 ngày: p = 2 sai gấp 2,7 lần p = 26",
                  "RMSE trên test (MW) theo bậc trễ p, walk-forward học lại mỗi tuần · BIC chọn p = 26, AIC chọn p = 48")
    x = np.arange(len(q))
    cols = [C["VAR"] if p == 26 else BLUE_LIGHT for p in q["p"]]
    ax.bar(x, q["RMSE"], color=cols, width=0.55, edgecolor=SURF, lw=2)
    for xi, v in zip(x, q["RMSE"]):
        ax.text(xi, v + 0.35, vn(v), ha="center", size=16, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"p = {p}\n{vn(k, 0)} tham số" for p, k in zip(q["p"], q.iloc[:, 1])], size=13, color=INK)
    ax.text(2, q["RMSE"].iloc[2] + 1.8, "nhóm chọn\n(theo BIC)", ha="center", size=12.5, color=C["VAR"], weight="bold")
    ax.set_ylim(0, 19); ax.set_ylabel("RMSE (MW)")
    ygrid(ax); ax.tick_params(axis="x", length=0)
    save(fig, "05-var-bac-tre.png")


# 6. VAR: lỗi theo giờ
def var_hour():
    mae = [4.07, 2.14, 2.35, 2.08, 2.02, 1.73, 1.70, 2.22, 4.92, 7.75, 8.18, 6.57,
           4.70, 4.86, 4.47, 3.93, 4.05, 4.40, 4.03, 3.62, 3.73, 3.74, 5.09, 5.07]
    fig, ax = new("Câu 2. VAR sai nhiều nhất lúc điện tăng nhanh buổi sáng",
                  "Sai số tuyệt đối trung bình MAE (MW) của VAR(26) theo giờ trong ngày, tập test")
    ax.axvspan(7.5, 11.5, color="#f0efec", lw=0)
    ax.text(9.5, 9.3, "8h–11h: điện tăng nhanh", ha="center", size=13, color=SEC)
    ax.plot(range(24), mae, color=C["VAR"], lw=2.2, marker="o", ms=7, mec=SURF, mew=1.5)
    ax.annotate("8,18 MW lúc 10h", (10, 8.18), xytext=(12.3, 8.0), size=14, weight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=SEC, lw=1))
    ax.annotate("1,70 MW lúc 6h", (6, 1.70), xytext=(3.2, 0.6), size=14, weight="bold", va="center")
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([f"{i}h" for i in range(0, 24, 2)])
    ax.set_ylim(0, 10); ax.set_ylabel("MAE (MW)")
    ygrid(ax)
    save(fig, "06-var-loi-theo-gio.png")


# 7. VAR: dự báo nhiều bước
def var_h():
    fig, ax = new("Dự báo càng xa VAR càng sai: 72 giờ sai gấp 3,5 lần 1 giờ",
                  "RMSE (MW) khi tự đưa giá trị dự báo vào chuỗi, không sửa bằng giá trị thật · 100 mốc ngẫu nhiên trong test")
    hs, v = ["1 giờ", "24 giờ", "72 giờ"], [5.3, 9.3, 18.4]
    x = np.arange(3)
    ax.bar(x, v, color=C["VAR"], width=0.5, edgecolor=SURF, lw=2)
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 0.35, vn(vi, 1), ha="center", size=16, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"tầm dự báo\n{h}" for h in hs], size=14, color=INK)
    ax.set_ylim(0, 21); ax.set_ylabel("RMSE (MW)")
    ygrid(ax); ax.tick_params(axis="x", length=0)
    save(fig, "07-var-nhieu-buoc.png")


# 8. XGBoost: Gain theo nhóm
def xgb_gain():
    names = ["Điện năng quá khứ (24 lag)", "Rolling và xu hướng (3)", "Nhiệt độ (24 lag)",
             "Độ ẩm (24 lag)", "Tốc độ gió (24 lag)"]
    v = [97.19, 1.02, 0.94, 0.47, 0.39]
    fig, ax = new("Câu 3. XGBoost dựa gần như hoàn toàn vào điện năng quá khứ",
                  "Tỷ lệ Gain (%) theo nhóm đặc trưng · riêng load_lag24 (cùng giờ hôm qua) chiếm 94% · cả 3 biến thời tiết cộng lại chỉ 1,8%")
    fig.subplots_adjust(left=0.27)
    y = np.arange(5)[::-1]
    cols = [C["XGB"], C["BASE2"], C["XGB"], C["XGB"], C["XGB"]]
    ax.barh(y, v, color=cols, height=0.55, edgecolor=SURF, lw=2)
    for yi, vi in zip(y, v):
        ax.text(vi + 1.0, yi, f"{vn(vi)}%", va="center", size=15, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(names, size=14, color=INK)
    ax.set_xlim(0, 110); ax.set_xlabel("% tổng Gain")
    xgrid(ax); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "08-xgb-gain.png")


# 9. XGBoost: ablation
def xgb_ablation():
    cfg = ["Chỉ điện năng\n(24 đặc trưng)", "Điện năng + rolling\n(27, bỏ thời tiết)", "Đầy đủ\n(99, có thời tiết)"]
    val, test = [5.59, 4.71, 4.52], [7.11, 6.16, 6.17]
    fig, ax = new("Thêm thời tiết: validation giảm 4%, test gần như không đổi",
                  "RMSE (MW) của XGBoost khi bỏ bớt nhóm đặc trưng · phần giảm lớn đến từ đặc trưng rolling, không phải thời tiết")
    x = np.arange(3); w = 0.34
    ax.bar(x - w / 2, val, w, color=C["BASE2"], edgecolor=SURF, lw=2, label="Validation")
    ax.bar(x + w / 2, test, w, color=C["XGB"], edgecolor=SURF, lw=2, label="Test")
    for xi, a, b in zip(x, val, test):
        ax.text(xi - w / 2, a + 0.12, vn(a), ha="center", size=14, weight="bold")
        ax.text(xi + w / 2, b + 0.12, vn(b), ha="center", size=14, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cfg, size=13.5, color=INK)
    ax.set_ylim(0, 8.3); ax.set_ylabel("RMSE (MW)")
    ax.legend(loc="upper right", ncol=2, fontsize=13)
    ygrid(ax); ax.tick_params(axis="x", length=0)
    save(fig, "09-xgb-bo-thoi-tiet.png")


# 10. LSTM: xáo trộn
def lstm_shuffle():
    names = ["LSTM, train theo đúng thứ tự thời gian", "Lấy cùng giờ hôm qua (seasonal naive)",
             "LSTM, xáo trộn thứ tự mẫu train mỗi epoch"]
    v = [13.96, 13.42, 6.14]
    fig, ax = new("Cách train quyết định LSTM: RMSE giảm từ 13,96 xuống 6,14",
                  "RMSE trên test (MW) · cùng kiến trúc LSTM(64) → LSTM(32) → Dense(1), chỉ đổi cách đưa mẫu train vào")
    fig.subplots_adjust(left=0.36)
    y = np.arange(3)[::-1]
    ax.barh(y[0], v[0], color=C["LSTM"], alpha=0.45, height=0.55, edgecolor=SURF, lw=2, hatch="//")
    ax.barh(y[1], v[1], color=C["BASE2"], height=0.55, edgecolor=SURF, lw=2)
    ax.barh(y[2], v[2], color=C["LSTM"], height=0.55, edgecolor=SURF, lw=2)
    for yi, vi in zip(y, v):
        ax.text(vi + 0.2, yi, vn(vi), va="center", size=16, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(names, size=14, color=INK)
    ax.set_xlim(0, 16); ax.set_xlabel("RMSE (MW)")
    xgrid(ax); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "10-lstm-xao-tron.png")


# 11. LSTM theo seed
def lstm_seed():
    s = pd.read_csv(P / "benchmark/bm_lstm_seeds.csv")
    fig, ax = new("Chỉ đổi seed, RMSE của LSTM đã dao động 5,89–6,28 MW",
                  "RMSE trên test (MW) · LSTM chạy 5 seed (42–46), VAR và XGBoost chạy lại cho đúng cùng một kết quả")
    fig.subplots_adjust(left=0.16)
    rows = {"VAR(26)": 2, "XGBoost": 1, "LSTM": 0}
    ax.axvspan(s.RMSE.min(), s.RMSE.max(), color="#e6f6ef", lw=0)
    ax.scatter([6.0656], [2], s=160, color=C["VAR"], ec=SURF, lw=2, zorder=3)
    ax.scatter([6.1729], [1], s=160, color=C["XGB"], ec=SURF, lw=2, zorder=3)
    ax.scatter(s.RMSE, [0] * len(s), s=160, color=C["LSTM"], ec=SURF, lw=2, zorder=3)
    ax.text(6.0656, 2.28, "6,07", ha="center", size=14, weight="bold")
    ax.text(6.1729, 1.28, "6,17", ha="center", size=14, weight="bold")
    for _, r in s.iterrows():
        dy = -0.62 if int(r.seed) == 43 else -0.25
        ax.text(r.RMSE, dy, f"seed {int(r.seed)}: {vn(r.RMSE)}", ha="center", va="top", size=12, color=SEC)
    ax.set_yticks(list(rows.values())); ax.set_yticklabels(list(rows.keys()), size=15, color=INK)
    ax.set_ylim(-0.95, 2.95); ax.set_xlim(5.7, 6.45); ax.set_xlabel("RMSE (MW)")
    ax.text(s.RMSE.min() + 0.008, 2.75, "dải kết quả 5 seed của LSTM", size=12.5, color=SEC, va="center")
    xgrid(ax); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "11-lstm-seed.png")


# 12. Câu 4: lịch
def lstm_calendar():
    names = ["LSTM 24 giờ", "LSTM 24 giờ + lịch\n(sin/cos giờ, thứ)", "Cùng giờ hôm qua"]
    v = [6.14, 5.77, 13.42]
    fig, ax = new("Câu 4. Thêm thông tin lịch giúp LSTM giảm 6% sai số",
                  "RMSE trên test (MW) · cửa sổ 24 giờ đã bắt được chu kỳ ngày, nhưng còn thiếu chu kỳ tuần")
    x = np.arange(3)
    ax.bar(x, v, color=[C["LSTM"], C["LSTM"], C["BASE2"]], width=0.5, edgecolor=SURF, lw=2)
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 0.25, vn(vi), ha="center", size=16, weight="bold")
    ax.text(1, 5.77 + 1.25, "−6%", ha="center", size=14, color=SEC, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, size=14, color=INK)
    ax.set_ylim(0, 15.5); ax.set_ylabel("RMSE (MW)")
    ygrid(ax); ax.tick_params(axis="x", length=0)
    save(fig, "12-cau4-lich.png")


# 13. Câu 4: dự báo đệ quy 48 giờ
def lstm_48():
    r = pd.read_csv(P / "lstm/q4_recursive_48.csv", parse_dates=["Datetime"])
    rm1 = np.sqrt(((r.y_true - r.y_pred_one_step) ** 2).mean())
    rmr = np.sqrt(((r.y_true - r.y_pred_recursive) ** 2).mean())
    fig, ax = new("Tự dự báo liền 48 giờ: LSTM giữ được hình dạng ngày, sai số tăng 1,7 lần",
                  f"20–22/07/2014 · dự báo từng giờ có giá trị thật: RMSE {vn(rm1)} MW · tự dự báo liền 48 giờ: RMSE {vn(rmr)} MW")
    ax.plot(r.Datetime, r.y_true, color=INK, lw=2.2, label="Thực tế")
    ax.plot(r.Datetime, r.y_pred_one_step, color=C["LSTM"], lw=2, label="Dự báo 1 bước")
    ax.plot(r.Datetime, r.y_pred_recursive, color=C["ENS"], lw=2, ls="--", label="Tự dự báo 48 giờ")
    ax.legend(loc="upper left", ncol=3, fontsize=13, bbox_to_anchor=(0, 1.02))
    ax.set_ylabel("MW"); ax.set_ylim(80, 420)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m %Hh"))
    ygrid(ax)
    save(fig, "13-cau4-48-gio.png")


# 14. Câu 5: overfitting
def lstm_overfit():
    h = pd.read_csv(P / "lstm/lstm_no_early_stopping_history.csv")
    fig, ax = new("Câu 5. Có overfitting: sau epoch 16 train tiếp tục giảm, validation tăng",
                  "Hàm mất mát MSE (dữ liệu đã chuẩn hóa) khi tắt EarlyStopping và chạy đủ 50 epoch")
    ax.plot(h.epoch, h.loss, color=C["LSTM"], lw=2.2)
    ax.plot(h.epoch, h.val_loss, color=C["ENS"], lw=2.2)
    ax.axvline(16, color=SEC, lw=1.2, ls=":")
    ax.text(16.5, 0.0122, "epoch 16: validation tốt nhất\nEarlyStopping dừng ở đây", size=13, color=SEC, va="top")
    ax.text(50.5, h.loss.iloc[-1], "Train\n−34%", size=14, weight="bold", color=INK, va="center")
    ax.text(50.5, h.val_loss.iloc[-1], "Validation\n+12%", size=14, weight="bold", color=INK, va="center")
    ax.set_xlim(1, 56); ax.set_ylim(0.0035, 0.0125)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(["Train", "Validation"], loc="upper right", ncol=2, fontsize=13, bbox_to_anchor=(0.9, 1.0))
    ygrid(ax)
    save(fig, "14-cau5-overfit.png")


# 15. So sánh chính với CI
def compare_ci():
    m = pd.read_csv(P / "benchmark/bm_metrics.csv").set_index("Model")
    rows = [("Kết hợp 3 mô hình", "Kết hợp 3 mô hình", C["ENS"]), ("VAR(26)", "VAR(26)", C["VAR"]),
            ("LSTM", "LSTM", C["LSTM"]), ("XGBoost", "XGBoost", C["XGB"]),
            ("Seasonal naive 24h", "Cùng giờ hôm qua", C["BASE"])]
    fig, ax = new("Ba mô hình ngang nhau (≈ 6,1 MW), giảm 54% sai số so với baseline",
                  "RMSE trên test (MW), khoảng tin cậy 95% bằng bootstrap khối 168 giờ (2.000 lần lấy mẫu)")
    fig.subplots_adjust(left=0.2)
    for i, (key, lab, col) in enumerate(rows):
        y = len(rows) - 1 - i
        r = m.loc[key]
        ax.plot([r.RMSE_lo, r.RMSE_hi], [y, y], color=col, lw=3, solid_capstyle="round")
        ax.scatter([r.RMSE], [y], s=180, color=col, ec=SURF, lw=2, zorder=3)
        ax.text(r.RMSE_hi + 0.3, y, f"{vn(r.RMSE)}   [{vn(r.RMSE_lo)} – {vn(r.RMSE_hi)}]",
                va="center", size=14, weight="bold")
    ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels([r[1] for r in rows], size=15, color=INK)
    ax.set_xlim(0, 25); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlabel("RMSE (MW)")
    xgrid(ax); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "15-so-sanh-rmse.png")


# 16. Theo loại ngày và khung giờ
def segments():
    s = pd.read_csv(P / "benchmark/bm_segments.csv")
    models = [("VAR(26)", "VAR", C["VAR"]), ("XGBoost", "XGBoost", C["XGB"]), ("LSTM", "LSTM", C["LSTM"])]
    fig, (a1, a2) = new("Không mô hình nào thắng mọi lúc",
                        "RMSE trên test (MW) · thanh thấp hơn là tốt hơn",
                        ncols=2, sharey=True, gridspec_kw={"wspace": 0.08, "width_ratios": [3, 4]})
    fig.subplots_adjust(bottom=0.2, top=0.76)
    for ax, grp, order in [(a1, "Loại ngày", ["Ngày thường", "Cuối tuần", "Lễ / kỳ nghỉ cuối năm"]),
                           (a2, "Khung giờ", ["0–6 giờ", "7–11 giờ", "12–17 giờ", "18–23 giờ"])]:
        sub = s[s["Nhóm"] == grp]
        x = np.arange(len(order)); w = 0.26
        for j, (key, lab, col) in enumerate(models):
            vals = [sub[(sub["Giá trị"] == o) & (sub.Model == key)].RMSE.iloc[0] for o in order]
            ax.bar(x + (j - 1) * w, vals, w, color=col, edgecolor=SURF, lw=2, label=lab)
            for xi, v in zip(x, vals):
                ax.text(xi + (j - 1) * w, v + 0.2, vn(v, 1), ha="center", size=11.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([o.replace("Lễ / kỳ nghỉ cuối năm", "Lễ, nghỉ\ncuối năm") for o in order], size=13, color=INK)
        ax.set_title(grp, loc="left", size=15, color=SEC, pad=10)
        ygrid(ax); ax.tick_params(axis="x", length=0)
    a1.set_ylabel("RMSE (MW)"); a1.set_ylim(0, 16)
    h_, l_ = a1.get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper right", fontsize=13, ncol=3, bbox_to_anchor=(0.96, 0.9))
    save(fig, "16-theo-nhom.png")


# 17. Chi phí
def cost():
    names = ["VAR(26)", "XGBoost", "LSTM"]
    sec = [0.106, 2.87, 145.4]
    notes = ["0,1 giây · 420 tham số", "2,9 giây · 154 cây", "145 giây · 30.113 tham số"]
    fig, ax = new("Câu 6. Sai số ngang nhau, nhưng LSTM học lâu hơn XGBoost khoảng 50 lần",
                  "Thời gian học 1 lần trên CPU (giây, thang log) · chưa tính thời gian chọn cấu hình")
    fig.subplots_adjust(left=0.13)
    y = np.arange(3)[::-1]
    ax.barh(y, sec, color=[C["VAR"], C["XGB"], C["LSTM"]], height=0.5, edgecolor=SURF, lw=2)
    for yi, v, n in zip(y, sec, notes):
        ax.text(v * 1.15, yi, n, va="center", size=15, weight="bold")
    ax.set_xscale("log"); ax.set_xlim(0.05, 5000)
    ax.set_xticks([0.1, 1, 10, 100, 1000]); ax.set_xticklabels(["0,1", "1", "10", "100", "1.000"])
    ax.set_yticks(y); ax.set_yticklabels(names, size=15, color=INK)
    ax.set_xlabel("giây (thang log)")
    xgrid(ax); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "17-cau6-chi-phi.png")


# 18. Nâng cao: giao thông
def traffic():
    t = pd.read_csv(P / "nc_test_metrics.csv")
    fig, ax = new("Nâng cao. GRU tốt hơn baseline, khoảng cách tăng khi dự báo xa",
                  "MAE (dặm/giờ) trên test METR-LA, 207 cảm biến, 12 bước quá khứ (60 phút) · DCRNN là số công bố trong bài báo")
    fig.subplots_adjust(right=0.76)
    hz = [15, 30, 60]
    series = [("Last Value", "Giữ giá trị cuối", C["BASE"], "-"),
              ("Historical Average", "Trung bình lịch sử", C["BASE2"], "-"),
              ("GRU", "GRU (nhóm)", C["LSTM"], "-")]
    for key, lab, col, ls in series:
        v = t[t.Model == key].sort_values("Horizon_min").MAE.values
        ax.plot(hz, v, color=col, lw=2.4, marker="o", ms=9, mec=SURF, mew=1.5, ls=ls)
        ax.text(62, v[-1], f"{lab}  {vn(v[-1])}", va="center", size=13.5, weight="bold")
        if key == "GRU":
            for hx, vv in zip(hz[:2], v[:2]):
                ax.text(hx, vv - 0.18, vn(vv), ha="center", va="top", size=12.5, color=INK)
    dc = [2.77, 3.15, 3.60]
    ax.plot(hz, dc, color=C["ENS"], lw=2, ls="--", marker="o", ms=8, mec=SURF, mew=1.5)
    ax.text(62, dc[-1] - 0.08, f"DCRNN (bài báo)  {vn(dc[-1])}", va="center", size=13.5, weight="bold")
    ax.set_xticks(hz); ax.set_xticklabels(["15 phút", "30 phút", "60 phút"], size=14, color=INK)
    ax.set_xlim(12, 63); ax.set_ylim(2.4, 5.8); ax.set_ylabel("MAE")
    ygrid(ax); ax.tick_params(axis="x", length=0)
    save(fig, "18-nang-cao-giao-thong.png")


for f in [tiles, split, cycle, weather_corr, var_p, var_hour, var_h, xgb_gain, xgb_ablation,
          lstm_shuffle, lstm_seed, lstm_calendar, lstm_48, lstm_overfit, compare_ci, segments, cost, traffic]:
    f()
