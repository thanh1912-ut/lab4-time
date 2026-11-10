"""Benchmark bổ sung theo chuẩn bài báo khoa học: VAR, XGBoost, LSTM cùng một luật.
Người làm: Thành.

Chạy từ gốc repo (sau m1_var.py, m2_xgboost.py, m3_lstm.py):
    python src/benchmark.py

Nội dung:
  1. Cùng luật: học trên train, validation chỉ để chọn cấu hình, không học lại trên test,
     dự báo 1 bước từ lịch sử thật, cùng 3.946 giờ test.
  2. Baseline: naive, seasonal naive 24 giờ và 168 giờ.
  3. Chỉ số: RMSE, MAE, MAPE, MASE (Hyndman & Koehler, 2006), skill so với seasonal naive 24h.
  4. Khoảng tin cậy 95% bằng moving block bootstrap (Künsch, 1989), khối 168 giờ.
  5. Kiểm định Diebold–Mariano (1995) có hiệu chỉnh Harvey–Leybourne–Newbold (1997),
     phương sai HAC Newey–West, hiệu chỉnh đa so sánh Holm (1979).
  6. LSTM 5 seed (độ ổn định), sai số theo nhóm giờ / ngày / tháng / kỳ lễ, chi phí tính toán.
Kết quả: data/processed/benchmark/, docs/lab4/figures/bm_*.png, docs/lab4/parts/bm_*.md
"""
from __future__ import annotations

import itertools
import json
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils import FIG, PROCESSED, ROOT, SEED, load_data

OUT = PROCESSED / "benchmark"
PARTS = ROOT / "docs" / "lab4" / "parts"
LSTM_DIR = PROCESSED / "lstm"
EXTRA_SEEDS = (45, 46)          # thêm vào 3 seed lúc chọn patience -> 5 seed để báo độ ổn định
BLOCK = 168                     # độ dài khối bootstrap (1 tuần) giữ phụ thuộc ngày/tuần
N_BOOT = 2000
HAC_LAG = 24                    # trễ Newey–West chính (1 ngày); kiểm tra lại với 168
HAC_LAG_ROBUST = 168
ALPHA = 0.05
# Ngày lễ chính thức của Bồ Đào Nha trong khoảng test (5/10, 1/11, 1/12 bị tạm bỏ 2013–2015)
HOLIDAYS = pd.to_datetime(["2014-08-15", "2014-12-08", "2014-12-25"])
YEAR_END = (pd.Timestamp("2014-12-24"), pd.Timestamp("2014-12-31"))

MAIN = ["Naive (giờ trước)", "Seasonal naive 24h", "Seasonal naive 168h",
        "VAR(26)", "XGBoost", "LSTM", "Kết hợp 3 mô hình"]
MODELS = ["VAR(26)", "XGBoost", "LSTM", "Kết hợp 3 mô hình"]
DM_SET = ["Seasonal naive 24h", "VAR(26)", "XGBoost", "LSTM", "Kết hợp 3 mô hình"]
# Màu theo thực thể, cố định (đã kiểm tra mù màu): xanh, cam, xanh ngọc, vàng; baseline xám.
COLORS = {"VAR(26)": "#2a78d6", "XGBoost": "#eb6834", "LSTM": "#1baf7a",
          "Kết hợp 3 mô hình": "#eda100"}
GREY = "#8a8a85"
INK, INK2, GRID = "#22221f", "#5f5e58", "#e6e5e0"


# ---------------------------------------------------------------- tiện ích
def vn(x, d=2):
    """Số kiểu Việt: dấu phẩy thập phân."""
    return f"{x:.{d}f}".replace(".", ",")


def vn_int(x):
    """Số nguyên kiểu Việt: dấu chấm ngăn nghìn."""
    return f"{int(x):,}".replace(",", ".")


def md(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False, name=None)]
    return "\n".join([head, sep] + rows)


def score(y, p, scale):
    e = y - p
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return {"RMSE": rmse, "MAE": float(np.mean(np.abs(e))),
            "MAPE": float(np.mean(np.abs(e) / np.abs(y)) * 100), "MASE": float(np.mean(np.abs(e)) / scale)}


def read_pred(path, times, truth):
    f = pd.read_csv(path, parse_dates=["Datetime"]).set_index("Datetime")
    if not f.index.equals(times):
        raise ValueError(f"{path.name}: mốc thời gian khác tập test.")
    if not np.allclose(f["y_true"].to_numpy(), truth, atol=1e-3):
        raise ValueError(f"{path.name}: y_true không khớp dữ liệu gốc.")
    p = f["y_pred"].to_numpy(dtype=float)
    if not np.isfinite(p).all():
        raise ValueError(f"{path.name}: dự báo có NaN/inf.")
    return p


def block_indices(n, block, n_boot, rng):
    """Moving block bootstrap: ghép các khối liên tiếp dài `block`, cắt về n."""
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


def hac_var(d, lag):
    """Phương sai dài hạn Newey–West (kernel Bartlett) của chuỗi d."""
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return v


def dm_test(e1, e2, loss, lag, h=1):
    """Diebold–Mariano, hiệu chỉnh HLN cho h=1, p-value hai phía theo t(n-1)."""
    d = (e1 ** 2 - e2 ** 2) if loss == "squared" else (np.abs(e1) - np.abs(e2))
    n = len(d)
    stat = d.mean() / np.sqrt(hac_var(d, lag) / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(d.mean()), float(stat), float(p)


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------- LSTM nhiều seed
def lstm_runs(times, truth):
    from lstm_common import prepare_data, train_experiment, load_experiment, predict_original
    sel = json.loads((LSTM_DIR / "lstm_selection.json").read_text(encoding="utf-8"))
    patience, main_seed = int(sel["patience"]), int(sel["main_seed"])
    data = prepare_data()
    seeds = sorted({42, 43, 44, *EXTRA_SEEDS})
    preds, meta = {}, {}
    for s in seeds:
        name = f"lstm_p{patience}_s{s}"
        res = train_experiment(name, data, patience=patience, seed=s, verbose=0)
        preds[s] = read_pred(LSTM_DIR / f"{name}_test.csv", times, truth)
        meta[s] = res["metadata"]
    main = load_experiment(f"lstm_p{patience}_s{main_seed}", data)
    t0 = time.perf_counter()
    predict_original(main["model"], data["X_test"], data["scaler"])
    infer = time.perf_counter() - t0
    return {"patience": patience, "main_seed": main_seed, "preds": preds, "meta": meta,
            "infer_seconds": infer, "params": int(main["model"].count_params())}


# ---------------------------------------------------------------- chi phí VAR / XGBoost
def cost_var_xgb(times, truth):
    from statsmodels.tsa.api import VAR
    import xgboost as xgb
    from features import fit_scale, inverse_load
    from m1_var import run_walk_forward
    from m2_xgboost import build_features
    from utils import L
    order = json.loads((PROCESSED / "var_order.json").read_text(encoding="utf-8"))
    df = load_data()
    scaled, scaler, (i1, i2) = fit_scale(df)
    t0 = time.perf_counter()
    res = VAR(scaled.iloc[:i1]).fit(order["p"])
    var_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_walk_forward(scaled, scaler, i2, order["p"], order["diff"], refit_every=None, train_end=i1)
    var_total = time.perf_counter() - t0
    var_params = int(res.params.size)

    flat, y, _, _ = build_features(scaled, L=L)
    tr, va, te = slice(0, i1 - L), slice(i1 - L, i2 - L), slice(i2 - L, len(y))
    model = xgb.XGBRegressor(max_depth=7, n_estimators=2000, learning_rate=0.05, early_stopping_rounds=50,
                             random_state=SEED, n_jobs=-1, eval_metric="rmse")
    t0 = time.perf_counter()
    model.fit(flat.iloc[tr], y[tr], eval_set=[(flat.iloc[tr], y[tr]), (flat.iloc[va], y[va])], verbose=False)
    xgb_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pz = model.predict(flat.iloc[te])
    xgb_infer = time.perf_counter() - t0
    check = inverse_load(pz, scaler)
    trees = model.best_iteration + 1
    leaves = int((model.get_booster().trees_to_dataframe().query("Feature == 'Leaf' and Tree < @trees")).shape[0])
    return {"var_fit": var_fit, "var_total": var_total, "var_params": var_params,
            "xgb_fit": xgb_fit, "xgb_infer": xgb_infer, "xgb_trees": trees, "xgb_leaves": leaves,
            "xgb_check": check}


# ---------------------------------------------------------------- hình
def style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_ci(ci):
    rows = [m for m in MAIN if m != "Naive (giờ trước)"]   # naive ~25 MW kéo giãn trục
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        for i, m in enumerate(rows[::-1]):
            r = ci[(ci.Model == m)].iloc[0]
            c = COLORS.get(m, GREY)
            ax.plot([r[f"{metric}_lo"], r[f"{metric}_hi"]], [i, i], color=c, linewidth=2, solid_capstyle="round")
            ax.plot(r[metric], i, "o", color=c, markersize=8, markeredgecolor="white", markeredgewidth=2)
            ax.text(r[f"{metric}_hi"] + 0.15, i, vn(r[metric]), va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(rows)), rows[::-1])
        ax.set_xlabel(f"{metric} (MW), khoảng tin cậy 95%", color=INK2, fontsize=9)
        style(ax)
    fig.suptitle("Sai số trên 3.946 giờ test, cùng luật huấn luyện (bootstrap khối 168 giờ)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "bm_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_month(month):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for m in MODELS:
        part = month[month.Model == m]
        ax.plot(part["Tháng"], part["RMSE"], marker="o", markersize=8, linewidth=2, color=COLORS[m],
                markeredgecolor="white", markeredgewidth=2, label=m)
    ax.set_ylabel("RMSE (MW)", color=INK2, fontsize=9)
    ax.set_xlabel("Tháng của năm 2014 (tháng 7 chỉ từ ngày 20)", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("RMSE theo tháng trong tập test", color=INK, fontsize=11, loc="left")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(FIG / "bm_thang.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- chương trình chính
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    from features import fit_scale
    _, _, (i1, i2) = fit_scale(df)
    times = df.index[i2:]
    y = df["load"].to_numpy(dtype=float)[i2:]
    train_load = df["load"].iloc[:i1].to_numpy(dtype=float)
    mase_scale = float(np.mean(np.abs(train_load[24:] - train_load[:-24])))

    print("Đọc dự báo và huấn luyện thêm seed LSTM nếu chưa có...", flush=True)
    lstm = lstm_runs(times, y)
    pred = {
        "Naive (giờ trước)": df["load"].shift(1).to_numpy()[i2:],
        "Seasonal naive 24h": df["load"].shift(24).to_numpy()[i2:],
        "Seasonal naive 168h": df["load"].shift(168).to_numpy()[i2:],
        "VAR(26)": read_pred(PROCESSED / "pred_var_fixed.csv", times, y),
        "XGBoost": read_pred(PROCESSED / "pred_xgb_fixed.csv", times, y),
        "LSTM": lstm["preds"][lstm["main_seed"]],
    }
    lstm_main_file = read_pred(PROCESSED / "pred_lstm.csv", times, y)
    if not np.allclose(lstm_main_file, pred["LSTM"], atol=1e-6):
        raise ValueError("pred_lstm.csv khác lần chạy LSTM chính đã chọn.")
    pred["Kết hợp 3 mô hình"] = (pred["VAR(26)"] + pred["XGBoost"] + pred["LSTM"]) / 3
    refit = {"VAR(26), refit mỗi 168 giờ": read_pred(PROCESSED / "pred_var.csv", times, y),
             "XGBoost, refit mỗi 168 giờ": read_pred(PROCESSED / "pred_xgb.csv", times, y)}
    err = {m: y - p for m, p in pred.items()}
    sn = score(y, pred["Seasonal naive 24h"], mase_scale)
    day_all = times.normalize()
    year_end = (day_all >= YEAR_END[0]) & (day_all <= YEAR_END[1])
    sq_sn = err["Seasonal naive 24h"] ** 2
    sn["year_end_share"] = float(sq_sn[year_end].sum() / sq_sn.sum() * 100)

    # 1. Chỉ số + CI bootstrap ---------------------------------------------------------
    rng = np.random.default_rng(SEED)
    idx = block_indices(len(y), BLOCK, N_BOOT, rng)
    rows = []
    for m in MAIN:
        e = err[m]
        boot_rmse = np.sqrt((e ** 2)[idx].mean(axis=1))
        boot_mae = np.abs(e)[idx].mean(axis=1)
        boot_mape = (np.abs(e) / y)[idx].mean(axis=1) * 100
        s = score(y, pred[m], mase_scale)
        rows.append({"Model": m, **s, "Skill_RMSE_vs_SN24": (1 - s["RMSE"] / sn["RMSE"]) * 100,
                     "RMSE_lo": np.percentile(boot_rmse, 2.5), "RMSE_hi": np.percentile(boot_rmse, 97.5),
                     "MAE_lo": np.percentile(boot_mae, 2.5), "MAE_hi": np.percentile(boot_mae, 97.5),
                     "MAPE_lo": np.percentile(boot_mape, 2.5), "MAPE_hi": np.percentile(boot_mape, 97.5)})
    ci = pd.DataFrame(rows)
    ci.round(4).to_csv(OUT / "bm_metrics.csv", index=False)
    refit_rows = [{"Model": m, **score(y, p, mase_scale)} for m, p in refit.items()]
    pd.DataFrame(refit_rows).round(4).to_csv(OUT / "bm_metrics_refit.csv", index=False)

    # 2. Diebold–Mariano --------------------------------------------------------------
    dm_rows = []
    pairs = list(itertools.combinations(DM_SET, 2))
    for loss in ["squared", "absolute"]:
        for lag in [HAC_LAG, HAC_LAG_ROBUST]:
            res = [dm_test(err[a], err[b], loss, lag) for a, b in pairs]
            adj = holm([r[2] for r in res])
            for (a, b), (dbar, stat, p), pa in zip(pairs, res, adj):
                dm_rows.append({"A": a, "B": b, "loss": loss, "hac_lag": lag, "mean_diff": dbar,
                                "DM": stat, "p": p, "p_holm": pa})
    dm = pd.DataFrame(dm_rows)
    dm.round(6).to_csv(OUT / "bm_dm.csv", index=False)

    # 3. LSTM nhiều seed ----------------------------------------------------------------
    seed_rows = []
    for s, p in lstm["preds"].items():
        sc = score(y, p, mase_scale)
        mt = lstm["meta"][s]
        seed_rows.append({"seed": s, "epochs_run": mt["epochs_run"], "selected_epoch": mt["selected_epoch"],
                          "val_RMSE": mt["split_metrics"]["validation"]["RMSE"], **sc,
                          "train_seconds": mt["train_seconds"]})
    seeds = pd.DataFrame(seed_rows)
    seeds.round(4).to_csv(OUT / "bm_lstm_seeds.csv", index=False)

    # 4. Theo nhóm ----------------------------------------------------------------------
    hour = times.hour
    band = np.select([hour <= 6, hour <= 11, hour <= 17], ["0–6 giờ", "7–11 giờ", "12–17 giờ"], "18–23 giờ")
    day = times.normalize()
    is_holiday = day.isin(HOLIDAYS) | ((day >= YEAR_END[0]) & (day <= YEAR_END[1]))
    groups = {
        "Khung giờ": band,
        "Loại ngày": np.where(is_holiday, "Lễ / kỳ nghỉ cuối năm",
                              np.where(times.dayofweek >= 5, "Cuối tuần", "Ngày thường")),
    }
    seg_rows = []
    for gname, labels in groups.items():
        for g in pd.unique(labels):
            mask = labels == g
            for m in ["Seasonal naive 24h"] + MODELS:
                sc = score(y[mask], pred[m][mask], mase_scale)
                seg_rows.append({"Nhóm": gname, "Giá trị": g, "Số giờ": int(mask.sum()), "Model": m,
                                 "RMSE": sc["RMSE"], "MAE": sc["MAE"]})
    seg = pd.DataFrame(seg_rows)
    seg.round(4).to_csv(OUT / "bm_segments.csv", index=False)
    month_rows = []
    for mo in sorted(pd.unique(times.month)):
        mask = times.month == mo
        for m in MODELS:
            sc = score(y[mask], pred[m][mask], mase_scale)
            month_rows.append({"Tháng": int(mo), "Model": m, "Số giờ": int(mask.sum()), **sc})
    month = pd.DataFrame(month_rows)
    month.round(4).to_csv(OUT / "bm_month.csv", index=False)

    # 5. Chi phí ------------------------------------------------------------------------
    print("Đo thời gian VAR và XGBoost...", flush=True)
    c = cost_var_xgb(times, y)
    if not np.allclose(c["xgb_check"], pred["XGBoost"], atol=2e-3):
        raise ValueError("XGBoost huấn luyện lại không ra đúng pred_xgb_fixed.csv.")
    cost = pd.DataFrame([
        {"Model": "VAR(26)", "Tham số": vn_int(c["var_params"]), "Học (giây)": c["var_fit"],
         "Học + dự báo 3.946 giờ (giây)": c["var_total"], "Chọn cấu hình": "p theo BIC, 1–48 trễ, trên train"},
        {"Model": "XGBoost", "Tham số": f"{c['xgb_trees']} cây, {vn_int(c['xgb_leaves'])} lá", "Học (giây)": c["xgb_fit"],
         "Học + dự báo 3.946 giờ (giây)": c["xgb_fit"] + c["xgb_infer"],
         "Chọn cấu hình": "max_depth trong {3, 5, 7, 9}, theo validation"},
        {"Model": "LSTM", "Tham số": vn_int(lstm["params"]), "Học (giây)": float(seeds.train_seconds.median()),
         "Học + dự báo 3.946 giờ (giây)": float(seeds.train_seconds.median()) + lstm["infer_seconds"],
         "Chọn cấu hình": "patience trong {5, 10} × 3 seed, theo validation"},
    ])
    cost.to_csv(OUT / "bm_cost.csv", index=False)

    import statsmodels, xgboost, sklearn
    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "statsmodels": statsmodels.__version__, "xgboost": xgboost.__version__,
           "scikit-learn": sklearn.__version__, "platform": platform.platform(),
           "data_sha256": lstm["meta"][lstm["main_seed"]]["config"]["data_sha256"],
           "tensorflow": lstm["meta"][lstm["main_seed"]]["config"]["tensorflow"],
           "bootstrap": {"block": BLOCK, "n_boot": N_BOOT, "seed": SEED},
           "dm": {"hac_lag": HAC_LAG, "hac_lag_robust": HAC_LAG_ROBUST, "correction": "HLN 1997", "multiple": "Holm"}}
    (OUT / "bm_environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_ci(ci)
    plot_month(month)
    write_report(ci, dm, seeds, seg, month, cost, refit_rows, lstm, env, sn, mase_scale, times, i1,
                 str(df.index[i1 - 1]))
    print(ci[["Model", "RMSE", "RMSE_lo", "RMSE_hi", "MAE", "MAPE", "MASE"]].round(3).to_string(index=False))
    print("Đã lưu data/processed/benchmark/, figures/bm_*.png, parts/bm_report.md, parts/bm_process.md")


# ---------------------------------------------------------------- báo cáo
def write_report(ci, dm, seeds, seg, month, cost, refit_rows, lstm, env, sn, mase_scale, times, i1, train_end):
    def with_ci(r, k, d=2):
        return f"{vn(r[k], d)} [{vn(r[k + '_lo'], d)}; {vn(r[k + '_hi'], d)}]"

    main_tab = pd.DataFrame([{
        "Mô hình": r.Model, "RMSE (MW)": with_ci(r, "RMSE"), "MAE (MW)": with_ci(r, "MAE"),
        "MAPE (%)": with_ci(r, "MAPE"), "MASE": vn(r.MASE, 3),
        "Skill RMSE so với SN24 (%)": vn(r.Skill_RMSE_vs_SN24, 1)} for _, r in ci.iterrows()])
    models_ci = ci[ci.Model.isin(MODELS[:3])].set_index("Model")
    best_rmse = models_ci.RMSE.idxmin()
    best_mae = models_ci.MAE.idxmin()
    ens = ci.set_index("Model").loc["Kết hợp 3 mô hình"]

    def dm_table(loss, lag):
        part = dm[(dm.loss == loss) & (dm.hac_lag == lag)]
        out = []
        for r in part.itertuples():
            if r.p_holm < ALPHA:
                verdict = f"{r.A if r.DM < 0 else r.B} tốt hơn"
            else:
                verdict = "không khác biệt có ý nghĩa"
            out.append({"A": r.A, "B": r.B, "Chênh lệch loss TB (A − B)": vn(r.mean_diff, 2),
                        "DM": vn(r.DM, 2), "p (Holm)": f"{r.p_holm:.3g}".replace(".", ","), "Kết luận ở mức 5%": verdict})
        return pd.DataFrame(out)

    def verdicts(loss, lag):
        part = dm[(dm.loss == loss) & (dm.hac_lag == lag)]
        return {(r.A, r.B): (r.p_holm < ALPHA, np.sign(r.DM)) for r in part.itertuples()}

    changed_pairs = [(l, k) for l in ["squared", "absolute"] for k in verdicts(l, HAC_LAG)
                     if verdicts(l, HAC_LAG)[k] != verdicts(l, HAC_LAG_ROBUST)[k]]
    changed = len(changed_pairs)
    loss_vn = {"squared": "bình phương", "absolute": "tuyệt đối"}
    changed_text = "; ".join(f"{a} – {b} (loss {loss_vn[l]})" for l, (a, b) in changed_pairs) or "không có"
    sn_lost = any(l == "squared" and "Seasonal naive 24h" in k for l, k in changed_pairs)
    sn_note = ("Với loss bình phương, ưu thế so với seasonal naive 24 giờ không còn có ý nghĩa ở trễ 168, "
               "dù RMSE chênh hơn gấp đôi. Lý do là sai số bình phương của seasonal naive dồn vào vài tuần "
               f"({vn(sn['year_end_share'], 0)}% tổng sai số bình phương nằm trong 24–31/12, chỉ 192 trên 3.946 giờ), "
               "nên phương sai dài hạn rất lớn; với loss tuyệt đối, ưu thế vẫn rõ ở cả hai mức trễ."
               ) if sn_lost else ""
    sq = dm[(dm.loss == "squared") & (dm.hac_lag == HAC_LAG)]
    trio = sq[sq.A.isin(MODELS[:3]) & sq.B.isin(MODELS[:3])]
    trio_sig = trio[trio.p_holm < ALPHA]
    ab = dm[(dm.loss == "absolute") & (dm.hac_lag == HAC_LAG)]
    trio_abs = ab[ab.A.isin(MODELS[:3]) & ab.B.isin(MODELS[:3])]
    trio_abs_sig = trio_abs[trio_abs.p_holm < ALPHA]

    def pair_list(frame):
        if frame.empty:
            return "không cặp nào"
        return "; ".join(f"{(r.A if r.DM < 0 else r.B)} tốt hơn {(r.B if r.DM < 0 else r.A)} (p Holm = "
                         f"{r.p_holm:.2g})".replace(".", ",") for r in frame.itertuples())

    ens_vs = sq[(sq.B == "Kết hợp 3 mô hình") & sq.A.isin(MODELS[:3])]
    ens_all_sig = bool((ens_vs.p_holm < ALPHA).all() and (ens_vs.DM > 0).all())

    seed_tab = seeds.assign(**{c: seeds[c].map(vn) for c in ["val_RMSE", "RMSE", "MAE", "MAPE"]})[
        ["seed", "epochs_run", "selected_epoch", "val_RMSE", "RMSE", "MAE", "MAPE"]]
    seed_tab.columns = ["Seed", "Epoch chạy", "Epoch chọn", "Val RMSE", "Test RMSE", "Test MAE", "Test MAPE (%)"]
    s_mean, s_std = seeds.RMSE.mean(), seeds.RMSE.std(ddof=1)
    s_lo, s_hi = seeds.RMSE.min(), seeds.RMSE.max()
    other = models_ci.drop(index="LSTM")
    inside = [m for m in other.index if s_lo <= other.loc[m, "RMSE"] <= s_hi]

    seg_tab = seg.pivot_table(index=["Nhóm", "Giá trị", "Số giờ"], columns="Model", values="MAE").reset_index()
    seg_tab = seg_tab[["Nhóm", "Giá trị", "Số giờ", "Seasonal naive 24h"] + MODELS]
    order = {"0–6 giờ": 0, "7–11 giờ": 1, "12–17 giờ": 2, "18–23 giờ": 3,
             "Ngày thường": 4, "Cuối tuần": 5, "Lễ / kỳ nghỉ cuối năm": 6}
    seg_tab = seg_tab.sort_values("Giá trị", key=lambda s: s.map(order))
    for col in ["Seasonal naive 24h"] + MODELS:
        seg_tab[col] = seg_tab[col].map(vn)
    hol = seg[(seg["Giá trị"] == "Lễ / kỳ nghỉ cuối năm")].set_index("Model").MAE
    bands = seg[seg["Nhóm"] == "Khung giờ"]
    worst_band = {m: bands[bands.Model == m].set_index("Giá trị").MAE.idxmax() for m in MODELS[:3]}
    wkd = seg[(seg["Giá trị"] == "Ngày thường")].set_index("Model").MAE

    month_tab = month.pivot_table(index="Tháng", columns="Model", values="RMSE")[MODELS].reset_index()
    nh = month.drop_duplicates("Tháng").set_index("Tháng")["Số giờ"]
    month_tab.insert(1, "Số giờ", month_tab["Tháng"].map(nh))
    for col in MODELS:
        month_tab[col] = month_tab[col].map(vn)
    month_best = month.loc[month[month.Model.isin(MODELS[:3])].groupby("Tháng").RMSE.idxmin()]
    wins = month_best.Model.value_counts().to_dict()

    cost_tab = cost.copy()
    for col in ["Học (giây)", "Học + dự báo 3.946 giờ (giây)"]:
        cost_tab[col] = cost_tab[col].map(lambda v: vn(v, 1))
    refit_tab = pd.DataFrame([{"Mô hình": r["Model"], "RMSE": vn(r["RMSE"]), "MAE": vn(r["MAE"]),
                               "MAPE (%)": vn(r["MAPE"]), "MASE": vn(r["MASE"], 3)} for r in refit_rows])
    t_xgb = float(cost.loc[cost.Model == "XGBoost", "Học (giây)"].iloc[0])
    t_lstm = float(cost.loc[cost.Model == "LSTM", "Học (giây)"].iloc[0])

    report = f"""# Benchmark bổ sung: so sánh VAR, XGBoost và LSTM theo chuẩn nghiên cứu — Thành

## Tóm tắt

Trên 3.946 giờ test (20/07–31/12/2014), khi cả ba mô hình được huấn luyện theo cùng một luật, sai số của chúng gần như ngang nhau:

- RMSE: VAR(26) {vn(models_ci.loc['VAR(26)', 'RMSE'])}, XGBoost {vn(models_ci.loc['XGBoost', 'RMSE'])}, LSTM {vn(models_ci.loc['LSTM', 'RMSE'])} MW.
- Khoảng tin cậy 95% của ba mô hình chồng lên nhau.
- Kiểm định Diebold–Mariano giữa ba mô hình, sau hiệu chỉnh Holm:
  - Với sai số bình phương (ứng với RMSE), các cặp khác biệt có ý nghĩa: {pair_list(trio_sig)}.
  - Với sai số tuyệt đối (ứng với MAE): {pair_list(trio_abs_sig)}.
- Cả ba đều tốt hơn rất rõ so với seasonal naive 24 giờ (RMSE {vn(sn['RMSE'])} MW).
- Lấy trung bình đơn giản của ba dự báo cho RMSE {vn(ens.RMSE)} MW, {'tốt hơn có ý nghĩa so với từng mô hình' if ens_all_sig else 'thấp nhất nhưng không phải cặp nào cũng khác biệt có ý nghĩa'}.

## 1. Thiết kế thí nghiệm

| Hạng mục | Thiết lập |
|---|---|
| Dữ liệu | `load_weather_hourly.csv`, 26.304 giờ, 4 biến; SHA-256 `{env['data_sha256'][:16]}…` |
| Chia tập | Theo thời gian 70/15/15: train {vn_int(i1)} giờ đến {train_end}, validation 3.946 giờ, test 3.946 giờ |
| Bài toán | Dự báo load giờ $t$ từ dữ liệu thật đến giờ $t-1$ (1 bước) |
| Luật chung | Học trên train; validation chỉ để chọn cấu hình hoặc dừng sớm; **không học lại trên test** |
| Chuẩn hóa | StandardScaler fit trên train, dùng chung cho cả ba mô hình |
| Đầu vào | 24 giờ × 4 biến (VAR dùng p = 26 theo BIC; XGBoost thêm 3 đặc trưng rolling của load) |
| Baseline | Naive (giờ trước), seasonal naive 24 giờ (cùng giờ hôm trước), seasonal naive 168 giờ (cùng giờ tuần trước) |
| Kết hợp | Trung bình đều VAR, XGBoost, LSTM; không ước lượng trọng số nên không dùng thông tin test |
| LSTM | Kiến trúc theo đề; patience {lstm['patience']} chọn bằng validation; mô hình chính seed {lstm['main_seed']}; độ ổn định báo trên {len(seeds)} seed |

Chỉ số:
- RMSE, MAE, MAPE tính trên MW.
- MASE chia MAE cho MAE trong mẫu của seasonal naive 24 giờ trên train ({vn(mase_scale)} MW) (Hyndman & Koehler, 2006). MASE < 1 nghĩa là tốt hơn dự báo "cùng giờ hôm trước" tính trên train.
- Skill = 1 − RMSE / RMSE(seasonal naive 24h).

Suy luận thống kê:
- **Khoảng tin cậy 95%:** moving block bootstrap (Künsch, 1989), khối {BLOCK} giờ, {N_BOOT} lần lặp, cùng chỉ số khối cho mọi mô hình. Khối 1 tuần giữ được tương quan ngày và tuần của sai số.
- **Kiểm định Diebold–Mariano** (Diebold & Mariano, 1995) cho từng cặp mô hình:
  - Loss là sai số bình phương (ứng với RMSE) và sai số tuyệt đối (ứng với MAE).
  - Phương sai dài hạn Newey–West, trễ {HAC_LAG} giờ.
  - Hiệu chỉnh mẫu nhỏ Harvey–Leybourne–Newbold (1997), so với phân phối t.
  - Hiệu chỉnh đa so sánh Holm (1979) trên {len(list(itertools.combinations(DM_SET, 2)))} cặp của mỗi loại loss.
  - Kiểm tra độ vững với trễ {HAC_LAG_ROBUST} giờ.

## 2. Kết quả chính

Giá trị là ước lượng điểm, trong ngoặc là khoảng tin cậy 95%.

{md(main_tab)}

![Khoảng tin cậy](../figures/bm_ci.png)

- **RMSE thấp nhất** trong ba mô hình là {best_rmse}; **MAE thấp nhất** là {best_mae}.
- Khoảng tin cậy RMSE của ba mô hình chồng nhau rộng, nên không thể xếp hạng chỉ bằng ước lượng điểm.
- Seasonal naive 168 giờ ({vn(ci.set_index('Model').loc['Seasonal naive 168h', 'RMSE'])} MW) kém hơn seasonal naive 24 giờ, vì vậy baseline mạnh nhất là "cùng giờ hôm trước".

## 3. Kiểm định Diebold–Mariano

DM < 0 nghĩa là A có loss trung bình nhỏ hơn B. Đơn vị chênh lệch là MW² với loss bình phương và MW với loss tuyệt đối.

**Loss bình phương (ứng với RMSE), trễ HAC {HAC_LAG} giờ**

{md(dm_table('squared', HAC_LAG))}

**Loss tuyệt đối (ứng với MAE), trễ HAC {HAC_LAG} giờ**

{md(dm_table('absolute', HAC_LAG))}

Độ vững: khi tăng trễ HAC lên {HAC_LAG_ROBUST} giờ, có **{changed}** trên {2 * len(list(itertools.combinations(DM_SET, 2)))} kết luận chuyển thành "không khác biệt có ý nghĩa": {changed_text}.
{sn_note}
Các kết luận "không khác biệt" giữa VAR, XGBoost và LSTM với loss bình phương không đổi. Bảng đầy đủ ở `data/processed/benchmark/bm_dm.csv`.

## 4. Độ ổn định của LSTM theo seed

{md(seed_tab)}

- Test RMSE qua {len(seeds)} seed: **{vn(s_mean)} ± {vn(s_std)} MW**, dao động từ {vn(s_lo)} đến {vn(s_hi)} MW.
- {('RMSE của ' + ' và '.join(inside) + ' nằm trong khoảng dao động này') if inside else 'RMSE của VAR và XGBoost nằm ngoài khoảng dao động này'}, nên {'chỉ đổi seed cũng đủ làm thay đổi thứ hạng giữa LSTM và ' + ' / '.join(inside) if inside else 'thứ hạng không phụ thuộc vào seed'}.
- VAR và XGBoost (không dùng subsample) cho cùng kết quả mỗi lần chạy.

## 5. Sai số theo nhóm (MAE, MW)

{md(seg_tab)}

- Khung giờ có MAE cao nhất: {'; '.join(f"{m} {worst_band[m]}" for m in MODELS[:3])}.
- Vào ngày lễ và kỳ nghỉ cuối năm, MAE tăng so với ngày thường ở cả ba mô hình: VAR {vn(wkd['VAR(26)'])} → {vn(hol['VAR(26)'])}, XGBoost {vn(wkd['XGBoost'])} → {vn(hol['XGBoost'])}, LSTM {vn(wkd['LSTM'])} → {vn(hol['LSTM'])} MW.
- Không mô hình nào có biến lịch lễ, nên đây là điểm yếu chung.

{md(month_tab)}

![RMSE theo tháng](../figures/bm_thang.png)

Số tháng mỗi mô hình có RMSE thấp nhất (trong ba mô hình): {', '.join(f'{k} {v}' for k, v in wins.items())}. Thứ hạng đổi theo tháng, vì vậy kết luận "mô hình nào tốt nhất" phụ thuộc vào giai đoạn đánh giá.

## 6. Chi phí tính toán

{md(cost_tab)}

- Thời gian đo trên cùng máy (CPU, {env['platform'].split('-')[0]}).
- Thời gian học của LSTM là trung vị của {len(seeds)} seed; mỗi seed học riêng và có thể chạy song song với tác vụ khác.
- Một lần học XGBoost nhanh hơn LSTM khoảng {vn(t_lstm / t_xgb, 0)} lần.

Bảng phụ, bản có học lại mỗi 168 giờ (mục 6, 7; không thuộc luật chung):

{md(refit_tab)}

## 7. Giới hạn của kết quả

1. **Một giai đoạn test duy nhất:** nửa cuối 2014, một vùng (318 khách hàng ở Bồ Đào Nha). Chưa đánh giá trên nhiều mốc gốc hay nhiều năm.
2. **Công sức tinh chỉnh không hoàn toàn bằng nhau:**
   - VAR chọn p theo BIC trên 48 giá trị.
   - XGBoost thử 4 giá trị depth.
   - LSTM thử 2 giá trị patience × 3 seed, với kiến trúc cố định theo đề.
   - Kết quả phản ánh mức tinh chỉnh vừa phải, chưa phải giới hạn tốt nhất của từng loại mô hình.
3. **Dữ liệu thời tiết:** vài giờ trống đã được nội suy từ hai phía khi tạo dữ liệu chung (mục 2), nên có rò rỉ rất nhỏ từ tương lai. Ngoài ra, thời tiết là giá trị quan trắc, không phải dự báo thời tiết.
4. **Giả định của kiểm định DM:** chuỗi chênh lệch loss dừng. Phép thử độ vững với trễ {HAC_LAG_ROBUST} giờ được dùng để kiểm tra độ nhạy của kết luận.
5. **LSTM** chỉ có {len(seeds)} seed, nên độ lệch chuẩn còn thô.

## 8. Tái lập

```bash
python src/m1_var.py       # pred_var_fixed.csv
python src/m2_xgboost.py   # pred_xgb_fixed.csv
python src/m3_lstm.py      # chọn patience, pred_lstm.csv
python src/benchmark.py    # huấn luyện thêm seed {', '.join(map(str, EXTRA_SEEDS))} nếu chưa có, rồi tính toàn bộ
```

Môi trường: Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, statsmodels {env['statsmodels']}, xgboost {env['xgboost']}, TensorFlow {env['tensorflow']}. Bootstrap và mọi mô hình dùng seed {SEED}. Chi tiết ở `data/processed/benchmark/bm_environment.json`.

## Tài liệu tham khảo

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
"""
    process = f"""# Quy trình benchmark bổ sung — Thành

1. **Dự báo đầu vào**, cùng luật (học trên train, không học lại trên test):
   - `pred_var_fixed.csv` (mục 6).
   - `pred_xgb_fixed.csv` (mục 7).
   - `lstm_p{lstm['patience']}_s*_test.csv` (mục 8).
   - Kiểm tra mỗi file có đúng 3.946 mốc test và `y_true` khớp dữ liệu gốc (≤ 1e-3 MW).
   - Kiểm tra `pred_lstm.csv` trùng lần chạy seed {lstm['main_seed']}.
2. **Thêm seed LSTM** {', '.join(map(str, EXTRA_SEEDS))}: dùng `lstm_common.train_experiment` với patience đã chọn, có cache. Seed thêm chỉ để đo độ ổn định, không dùng để chọn cấu hình.
3. **Baseline** tính trực tiếp trên load gốc: `shift(1)`, `shift(24)`, `shift(168)`. **Kết hợp** là trung bình đều ba dự báo.
4. **Chỉ số:** RMSE, MAE, MAPE, MASE (thang = MAE của seasonal naive 24 giờ trên train, {vn(mase_scale)} MW), skill so với seasonal naive 24 giờ.
5. **Bootstrap:** `block_indices()` chọn ngẫu nhiên điểm đầu của các khối {BLOCK} giờ (seed {SEED}), ghép rồi cắt về 3.946 giờ, {N_BOOT} lần. Dùng cùng chỉ số khối cho mọi mô hình. Khoảng tin cậy lấy phân vị 2,5% và 97,5%.
6. **Diebold–Mariano** (`dm_test()`):
   - $d_t = L(e_{{A,t}}) - L(e_{{B,t}})$.
   - Thống kê $\\bar d / \\sqrt{{\\hat\\sigma^2_{{NW}}/n}}$, trong đó Newey–West dùng kernel Bartlett, trễ {HAC_LAG}.
   - Nhân hệ số HLN, tính p hai phía theo $t_{{n-1}}$.
   - Hiệu chỉnh Holm trên {len(list(itertools.combinations(DM_SET, 2)))} cặp, làm riêng cho từng loại loss.
   - Lặp lại với trễ {HAC_LAG_ROBUST}.
7. **Nhóm:**
   - Khung giờ: 0–6, 7–11, 12–17, 18–23.
   - Loại ngày: ngày thường, cuối tuần, lễ / kỳ nghỉ cuối năm. Lễ gồm 15/08, 08/12, 25/12, cộng thêm 24–31/12. Các ngày 5/10, 1/11, 1/12 không còn là ngày lễ chính thức ở Bồ Đào Nha trong giai đoạn 2013–2015.
   - Tháng.
8. **Chi phí:**
   - Học lại VAR trên train và đo thời gian.
   - Học lại XGBoost (depth 7, early stopping theo validation), đo thời gian và kiểm tra ra đúng `pred_xgb_fixed.csv`. Đếm số cây và số lá.
   - LSTM lấy thời gian học từ metadata (trung vị các seed), đo thời gian dự báo của mô hình chính.
9. **Lưu kết quả:**
   - `data/processed/benchmark/`: `bm_metrics.csv`, `bm_metrics_refit.csv`, `bm_dm.csv`, `bm_lstm_seeds.csv`, `bm_segments.csv`, `bm_month.csv`, `bm_cost.csv`, `bm_environment.json`.
   - Hình `bm_ci.png`, `bm_thang.png`.
   - `parts/bm_report.md`, `parts/bm_process.md`.

Hình dùng một màu cố định cho mỗi mô hình (VAR xanh dương, XGBoost cam, LSTM xanh ngọc, kết hợp vàng); bảng màu đã được kiểm tra cho người mù màu. Mọi giá trị trên hình đều có trong bảng.
"""
    (PARTS / "bm_report.md").write_text(report, encoding="utf-8")
    (PARTS / "bm_process.md").write_text(process, encoding="utf-8")


if __name__ == "__main__":
    main()
