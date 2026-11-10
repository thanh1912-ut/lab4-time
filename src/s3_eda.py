
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.tsa.stattools import acf
from utils import load_data, FIG, COLS


DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]

COLORS = {
    "load": "#4267A9",
    "trend": "#E07B39",
    "iqr": "#E9A23B",
    "zscore": "#D94B4B",
    "both": "#8B3A83",
    "residual": "#777777",
}

SEASON_COLORS = {
    "Winter": "#278A78",
    "Spring": "#E38B53",
    "Summer": "#4A70B8",
    "Autumn": "#B35A91",
}


def save_figure(name):
    plt.tight_layout(pad=1.5)
    plt.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close()


def get_outlier_masks(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    iqr_mask = (
        data.lt(q1 - 1.5 * iqr)
        | data.gt(q3 + 1.5 * iqr)
    )

    std = data.std(ddof=0).replace(0, np.nan)
    z_scores = (data - data.mean()) / std
    z_mask = z_scores.abs().gt(3).fillna(False)

    summary = pd.DataFrame({
        "IQR": iqr_mask.sum(),
        "IQR (%)": (iqr_mask.mean() * 100).round(3),
        "Z-score": z_mask.sum(),
        "Z-score (%)": (z_mask.mean() * 100).round(3),
    })

    return summary, iqr_mask, z_mask


def analyze_residual(df):
    # Loại chu kỳ trung bình theo giờ-trong-tuần (168 nhóm).
    # Chưa loại mùa vụ hằng năm.
    hour_of_week = df.index.day_of_week * 24 + df.index.hour

    expected = df["load"].groupby(hour_of_week).transform("mean")
    residual = df["load"] - expected

    summary, iqr_mask, z_mask = get_outlier_masks(
        residual.to_frame("load_residual")
    )

    return (
        residual,
        summary,
        iqr_mask["load_residual"],
        z_mask["load_residual"],
    )


def format_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def plot_load(df):
    daily = df["load"].resample("D").mean()
    trend = daily.rolling(
        7, center=True, min_periods=4
    ).mean()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1]},
    )

    axes[0].plot(
        df.index,
        df["load"],
        color=COLORS["load"],
        linewidth=0.45,
        alpha=0.65,
        rasterized=True,
    )

    axes[0].set(
        title="Hourly electricity load (2012–2014)",
        ylabel="Load",
    )

    axes[1].plot(
        daily.index,
        daily,
        color=COLORS["load"],
        linewidth=0.8,
        alpha=0.5,
        label="Daily mean",
    )

    axes[1].plot(
        trend.index,
        trend,
        color=COLORS["trend"],
        linewidth=2,
        label="7-day moving average",
    )

    axes[1].set(
        title="Daily load and smoothed trend",
        xlabel="Time",
        ylabel="Average load",
    )

    axes[1].legend(loc="upper right")
    format_time_axis(axes[1])

    save_figure("s3_load.png")


def plot_load_vs_temperature(df):
    months = df.index.month

    season = np.select(
        [
            months.isin([12, 1, 2]),
            months.isin([3, 4, 5]),
            months.isin([6, 7, 8]),
        ],
        ["Winter", "Spring", "Summer"],
        default="Autumn",
    )

    plot_data = df[["temperature", "load"]].copy()
    plot_data["season"] = season

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Vẽ scatter trước để tránh các đường xu hướng bị che.
    for name in SEASONS:
        subset = plot_data.loc[
            plot_data["season"] == name,
            ["temperature", "load"],
        ].dropna()

        ax.scatter(
            subset["temperature"],
            subset["load"],
            s=9,
            alpha=0.12,
            color=SEASON_COLORS[name],
            linewidths=0,
            rasterized=True,
            label=name,
        )

    # Hồi quy tuyến tính trên toàn bộ dữ liệu từng mùa.
    # Chỉ giới hạn phạm vi HIỂN THỊ đường ở P5–P95.
    # Không cắt dữ liệu khi ước lượng hệ số hồi quy.
    for name in SEASONS:
        subset = plot_data.loc[
            plot_data["season"] == name,
            ["temperature", "load"],
        ].dropna()

        if subset["temperature"].nunique() < 2:
            continue

        x = subset["temperature"].to_numpy()
        y = subset["load"].to_numpy()

        slope, intercept = np.polyfit(x, y, 1)

        x_low, x_high = np.percentile(x, [5, 95])
        x_line = np.linspace(x_low, x_high, 100)
        y_line = slope * x_line + intercept

        ax.plot(
            x_line,
            y_line,
            color=SEASON_COLORS[name],
            linewidth=2.5,
            label=f"{name} linear trend",
            zorder=5,
        )

    ax.set(
        title="Electricity load versus temperature by season",
        xlabel="Temperature",
        ylabel="Electricity load",
    )

    ax.legend(
        title="Season and linear trends",
        loc="upper left",
        fontsize=9,
        ncol=2,
        frameon=True,
        markerscale=2,
    )

    # Giải thích rõ đường xu hướng chỉ hiển thị ở vùng
    # nhiệt độ phổ biến, không phải mô hình phi tuyến.
    fig.text(
        0.5,
        0.005,
        "Linear fits use all observations; trend lines are "
        "displayed over each season's 5th–95th temperature percentiles.",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    save_figure("s3_load_vs_temp.png")


def plot_correlation(df):
    corr = df[COLS].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(7.5, 6))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r", "shrink": 0.85},
        ax=ax,
    )

    ax.set_title("Pearson correlation matrix", pad=14)

    save_figure("s3_corr.png")
    return corr


def plot_outliers(df, residual, iqr_mask, z_mask):
    # Ba nhóm loại trừ nhau để không vẽ trùng điểm.
    iqr_only = iqr_mask & ~z_mask
    z_only = z_mask & ~iqr_mask
    both = iqr_mask & z_mask

    iqr_total = int(iqr_mask.sum())
    z_total = int(z_mask.sum())
    overlap_total = int(both.sum())

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    axes[0].plot(
        df.index,
        df["load"],
        color=COLORS["load"],
        linewidth=0.5,
        alpha=0.75,
        rasterized=True,
    )

    axes[0].set(
        title="Original hourly electricity load",
        ylabel="Load",
    )

    axes[1].plot(
        residual.index,
        residual,
        color=COLORS["residual"],
        linewidth=0.45,
        alpha=0.65,
        rasterized=True,
    )

    groups = [
        (iqr_only, COLORS["iqr"], "IQR only"),
        (z_only, COLORS["zscore"], "Z-score only"),
        (both, COLORS["both"], "Both methods"),
    ]

    for mask, color, label in groups:
        count = int(mask.sum())

        if count == 0:
            continue

        axes[1].scatter(
            residual.index[mask],
            residual[mask],
            color=color,
            s=12,
            alpha=0.85,
            label=f"{label}: {count:,}",
            rasterized=True,
            zorder=3,
        )

    axes[1].axhline(
        0,
        color="#333333",
        linewidth=0.8,
    )

    axes[1].set(
        title="Load residual after hour-of-week adjustment",
        xlabel="Time",
        ylabel="Load residual",
    )

    axes[1].legend(
        loc="upper right",
        title=(
            f"IQR total: {iqr_total:,} | "
            f"Z-score total: {z_total:,}"
        ),
        fontsize=9,
        title_fontsize=9,
    )

    format_time_axis(axes[1])

    fig.text(
        0.5,
        0.005,
        "IQR total = IQR only + Both methods. "
        "Statistical outliers are not necessarily data errors; "
        "annual seasonality may remain.",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    save_figure("s3_outlier.png")

    return {
        "IQR only": int(iqr_only.sum()),
        "Z-score only": int(z_only.sum()),
        "Both methods": overlap_total,
        "IQR total": iqr_total,
        "Z-score total": z_total,
    }


def plot_seasonality(df):
    hourly = df["load"].groupby(df.index.hour).mean()

    weekly = (
        df["load"]
        .groupby(df.index.day_of_week)
        .mean()
        .reindex(range(7))
    )

    peak_hour = int(hourly.idxmax())
    low_hour = int(hourly.idxmin())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        hourly.index,
        hourly.values,
        color=COLORS["load"],
        marker="o",
        markersize=4,
        linewidth=2,
    )

    axes[0].scatter(
        [peak_hour, low_hour],
        [hourly.max(), hourly.min()],
        color=[COLORS["zscore"], COLORS["trend"]],
        s=65,
        zorder=5,
    )

    axes[0].annotate(
        f"Peak: {peak_hour:02d}:00",
        xy=(peak_hour, hourly.max()),
        xytext=(-85, -35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    axes[0].annotate(
        f"Lowest: {low_hour:02d}:00",
        xy=(low_hour, hourly.min()),
        xytext=(25, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    axes[0].set(
        title="Average load by hour of day",
        xlabel="Hour",
        ylabel="Average load",
        xticks=range(0, 24, 3),
    )

    axes[1].bar(
        range(7),
        weekly.values,
        color=COLORS["load"],
        alpha=0.85,
        width=0.7,
    )

    for i, value in enumerate(weekly.values):
        axes[1].text(
            i,
            value + 2,
            f"{value:.1f}",
            ha="center",
            fontsize=9,
        )

    axes[1].set(
        title="Average load by day of week",
        xlabel="Day",
        ylabel="Average load",
        xticks=range(7),
        xticklabels=DAYS,
    )

    axes[1].set_ylim(0, weekly.max() * 1.14)

    save_figure("s3_chu_ky.png")
    return hourly, weekly


def plot_acf(df):
    values = acf(
        df["load"].to_numpy(),
        nlags=200,
        fft=True,
    )

    lags = np.arange(1, 201)

    fig, ax = plt.subplots(figsize=(14, 4.8))

    ax.vlines(
        lags,
        0,
        values[1:],
        color=COLORS["load"],
        linewidth=0.85,
        alpha=0.8,
    )

    ax.axhline(
        0,
        color="#333333",
        linewidth=0.9,
    )

    for lag, color in [
        (24, COLORS["trend"]),
        (168, COLORS["zscore"]),
    ]:
        ax.axvline(
            lag,
            color=color,
            linestyle="--",
            linewidth=1.7,
            label=f"Lag {lag}: ACF = {values[lag]:.4f}",
        )

        ax.scatter(
            lag,
            values[lag],
            color=color,
            s=35,
            zorder=5,
        )

    ax.set(
        title="Autocorrelation of electricity load (200 lags)",
        xlabel="Lag (hours)",
        ylabel="ACF",
        xlim=(0, 200),
    )

    ax.legend(loc="upper right")
    save_figure("s3_acf.png")

    return values


def main():
    FIG.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )

    df = load_data().sort_index()

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps detected.")

    if len(df) <= 200:
        raise ValueError(
            "Dataset requires more than 200 observations."
        )

    missing = pd.DataFrame({
        "Missing": df.isna().sum(),
        "Missing (%)": df.isna().mean() * 100,
    })

    print("\nDATASET")
    print(f"Observations: {len(df):,}")
    print(
        f"Period: {df.index.min()} to {df.index.max()}"
    )

    print("\nMISSING VALUES")
    print(missing.to_string())

    if df[COLS].isna().any().any():
        raise ValueError(
            "Processed data contains missing values."
        )

    outlier_summary, _, _ = get_outlier_masks(
        df[COLS]
    )

    print("\nOUTLIERS BY VARIABLE")
    print(outlier_summary.to_string())

    residual, residual_summary, iqr_mask, z_mask = (
        analyze_residual(df)
    )

    print("\nLOAD RESIDUAL OUTLIERS")
    print(residual_summary.to_string())

    plot_load(df)
    plot_load_vs_temperature(df)
    corr = plot_correlation(df)

    overlap = plot_outliers(
        df, residual, iqr_mask, z_mask
    )

    hourly, weekly = plot_seasonality(df)
    acf_values = plot_acf(df)

    print("\nOUTLIER OVERLAP")
    for label, count in overlap.items():
        print(f"{label}: {count:,}")

    print("\nCORRELATION")
    print(corr.round(4).to_string())

    print("\nSEASONALITY")
    print(f"Peak hour: {hourly.idxmax():02d}:00")
    print(f"Lowest hour: {hourly.idxmin():02d}:00")
    print(f"Peak weekday: {DAYS[weekly.idxmax()]}")

    print("\nAUTOCORRELATION")
    print(f"ACF lag 24: {acf_values[24]:.4f}")
    print(f"ACF lag 168: {acf_values[168]:.4f}")

    print("\nEDA completed. Six figures saved to:")
    print(FIG)


if __name__ == "__main__":
    main()
