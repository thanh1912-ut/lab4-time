
"""Câu 1: Phân tích tương quan giữa thời tiết và phụ tải điện."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import FIG, load_data


WEATHER_COLS = ["temperature", "humidity", "wind_speed"]

SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
}


def correlation_table(df):
    result = pd.DataFrame({
        "Pearson": df[WEATHER_COLS].corrwith(df["load"]),
        "Spearman": df[WEATHER_COLS].corrwith(
            df["load"], method="spearman"
        ),
    })

    result.index.name = "Variable"
    result["|Pearson|"] = result["Pearson"].abs()

    return result.sort_values("|Pearson|", ascending=False)


def seasonal_correlations(df):
    rows = []

    for season, months in SEASONS.items():
        subset = df[df.index.month.isin(months)]

        row = {
            "Season": season,
            "Observations": len(subset),
        }

        for variable in WEATHER_COLS:
            row[variable] = subset["load"].corr(subset[variable])

        rows.append(row)

    return pd.DataFrame(rows).set_index("Season")


def temperature_deviation(df):
    deviation = (df["temperature"] - 18).abs()

    return pd.Series({
        "Pearson": df["load"].corr(deviation),
        "Spearman": df["load"].corr(
            deviation, method="spearman"
        ),
    }, name="|temperature - 18|")


def lag_correlations(df, max_lag=24):
    rows = []

    for lag in range(max_lag + 1):
        past_temperature = df["temperature"].shift(lag)

        rows.append({
            "Lag": lag,
            "Pearson": df["load"].corr(past_temperature),
            "Spearman": df["load"].corr(
                past_temperature, method="spearman"
            ),
        })

    return pd.DataFrame(rows).set_index("Lag")


def adjusted_correlations(df):
    load_hour_mean = df.groupby(
        df.index.hour
    )["load"].transform("mean")

    temp_hour_mean = df.groupby(
        df.index.hour
    )["temperature"].transform("mean")

    load_residual = df["load"] - load_hour_mean
    temp_residual = df["temperature"] - temp_hour_mean

    return pd.Series({
        "Raw Pearson": df["load"].corr(df["temperature"]),
        "Adjusted Pearson": load_residual.corr(temp_residual),
        "Raw Spearman": df["load"].corr(
            df["temperature"], method="spearman"
        ),
        "Adjusted Spearman": load_residual.corr(
            temp_residual, method="spearman"
        ),
    })


def plot_lag_correlations(lag_table):
    FIG.mkdir(parents=True, exist_ok=True)

    best_lag = int(lag_table["Pearson"].abs().idxmax())
    best_corr = lag_table.loc[best_lag, "Pearson"]
    lag24_corr = lag_table.loc[24, "Pearson"]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        lag_table.index,
        lag_table["Pearson"],
        color="#2563eb",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Pearson",
    )

    ax.plot(
        lag_table.index,
        lag_table["Spearman"],
        color="#e68a2e",
        marker="s",
        markersize=3,
        linewidth=1.8,
        label="Spearman",
    )

    ax.axhline(
        0,
        color="#64748b",
        linewidth=0.9,
        alpha=0.7,
    )

    ax.axvline(
        best_lag,
        color="#dc2626",
        linestyle="--",
        linewidth=1.1,
        alpha=0.65,
    )

    ax.axvline(
        24,
        color="#059669",
        linestyle=":",
        linewidth=1.1,
        alpha=0.65,
    )

    ax.scatter(
        [best_lag, 24],
        [best_corr, lag24_corr],
        color=["#dc2626", "#059669"],
        s=48,
        zorder=5,
    )

    ax.annotate(
        f"Lag {best_lag}h: r = {best_corr:.4f}",
        xy=(best_lag, best_corr),
        xytext=(12, -24),
        textcoords="offset points",
        fontsize=9,
        color="#b91c1c",
    )

    ax.annotate(
        f"Lag 24h: r = {lag24_corr:.4f}",
        xy=(24, lag24_corr),
        xytext=(-112, -24),
        textcoords="offset points",
        fontsize=9,
        color="#047857",
    )

    ax.set_title(
        "Temperature–Load Correlation by Time Lag",
        fontsize=13,
        pad=12,
    )

    ax.set_xlabel(
        "Temperature lag k (hours): corr(load[t], temp[t-k])"
    )
    ax.set_ylabel("Correlation coefficient")
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(-0.5, 24.5)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper center", fontsize=9)

    fig.tight_layout()

    output = FIG / "q1_lag_corr.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return output


def main():
    df = load_data().sort_index()

    print("=" * 65)
    print("QUESTION 1: WEATHER VARIABLES AND ELECTRICITY LOAD")
    print("=" * 65)

    print("\nDATASET")
    print(f"Observations: {len(df):,}")
    print(f"Period: {df.index.min()} to {df.index.max()}")

    print("\n1. PEARSON AND SPEARMAN CORRELATIONS")
    corr = correlation_table(df)
    print(corr.round(4).to_string())

    print("\n2. PEARSON CORRELATIONS BY SEASON")
    seasonal = seasonal_correlations(df)
    print(seasonal.round(4).to_string())

    print("\n3. TEMPERATURE DEVIATION FROM 18°C")
    deviation = temperature_deviation(df)
    print(deviation.round(4).to_string())

    print("\n4. TEMPERATURE–LOAD LAG CORRELATIONS (0–24 HOURS)")
    lag_table = lag_correlations(df)
    print(lag_table.round(4).to_string())

    best_lag = int(lag_table["Pearson"].abs().idxmax())
    best_corr = lag_table.loc[best_lag, "Pearson"]

    print(
        f"\nStrongest absolute Pearson correlation: "
        f"lag {best_lag}h, r = {best_corr:.4f}"
    )

    print("\n5. CORRELATION AFTER REMOVING DAILY CYCLE")
    adjusted = adjusted_correlations(df)
    print(adjusted.round(4).to_string())

    strongest = corr["|Pearson|"].idxmax()

    print("\n6. SUMMARY")
    print(f"Strongest weather variable by |Pearson|: {strongest}")
    print(
        f"Pearson coefficient: "
        f"{corr.loc[strongest, 'Pearson']:.4f}"
    )
    print(
        f"Spearman coefficient: "
        f"{corr.loc[strongest, 'Spearman']:.4f}"
    )

    output = plot_lag_correlations(lag_table)

    print(f"\nFigure saved: {output}")
    print("\nQuestion 1 analysis completed.")


if __name__ == "__main__":
    main()
