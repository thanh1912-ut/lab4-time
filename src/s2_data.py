"""Mục 2: Chuẩn bị dữ liệu điện năng + thời tiết theo giờ.
Người làm: Thành.
Chi tiết: docs/lab4/phan-cong.md, mục 5.1.

1. Tải dữ liệu điện UCI ElectricityLoadDiagrams20112014 (370 khách hàng, 15 phút/lần, Bồ Đào Nha).
2. Tải số đo thời tiết thật của trạm sân bay Lisbon (NOAA ISD, trạm 085360, mã ICAO LPPT).
3. Gộp điện về theo giờ, sửa các giờ đổi giờ, tính độ ẩm từ nhiệt độ và điểm sương.
4. Ghép thành data/processed/load_weather_hourly.csv (Datetime, load, temperature, humidity, wind_speed).
"""
import gzip
import hashlib
import io
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import RAW, PROCESSED, FIG, COLS

DPI = 150
START, END = "2012-01-01 00:00", "2014-12-31 23:00"   # 3 năm đầy đủ, bỏ 2011 vì nhiều khách hàng chưa đăng ký

UCI_TXT = RAW / "LD2011_2014.txt"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
# Bản sao file gốc trên GitHub, dùng khi không vào được UCI. Mã băm trùng ở nhiều kho độc lập.
UCI_MIRROR_URL = "https://media.githubusercontent.com/media/kkkkim157/pyraformer/HEAD/data/LD2011_2014.txt"
UCI_SHA256 = "d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b"

ISD_DIR = RAW / "isd"
ISD_STATION = "085360-99999"   # Lisboa/Portela (LPPT), vĩ độ 38,767, kinh độ -9,133
ISD_URL = "https://noaa-isd-pds.s3.amazonaws.com/data/{year}/{station}-{year}.gz"
ISD_BAD_QC = set("2367")       # mã chất lượng ISD bị coi là nghi ngờ / sai


# ---------------------------------------------------------------- tải dữ liệu
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    print("Tải", url)
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def get_uci():
    if not UCI_TXT.exists():
        try:
            zpath = RAW / "electricityloaddiagrams20112014.zip"
            download(UCI_ZIP_URL, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extract("LD2011_2014.txt", RAW)
        except Exception as e:  # UCI bị chặn hoặc lỗi mạng -> dùng bản sao
            print("Không tải được từ UCI:", e, "-> dùng bản sao trên GitHub")
            download(UCI_MIRROR_URL, UCI_TXT)
    digest = sha256(UCI_TXT)
    print("SHA-256 LD2011_2014.txt:", digest, "(khớp bản gốc)" if digest == UCI_SHA256 else "(KHÁC bản đã kiểm tra!)")


def get_isd(years=(2012, 2013, 2014)):
    ISD_DIR.mkdir(parents=True, exist_ok=True)
    for y in years:
        dest = ISD_DIR / f"{ISD_STATION}-{y}.gz"
        if not dest.exists():
            download(ISD_URL.format(year=y, station=ISD_STATION), dest)


# ---------------------------------------------------------------- điện năng
def dst_days(year):
    """Ngày đổi giờ ở Bồ Đào Nha: Chủ nhật cuối tháng 3 và cuối tháng 10."""
    out = []
    for month in (3, 10):
        last = pd.Timestamp(year=year, month=month, day=31)
        out.append((last - pd.Timedelta(days=(last.dayofweek + 1) % 7)).normalize())
    return out


def build_load():
    raw = pd.read_csv(UCI_TXT, sep=";", decimal=",", index_col=0, parse_dates=True)
    print(f"UCI gốc: {raw.shape[0]} mốc 15 phút x {raw.shape[1]} khách hàng, "
          f"{raw.index.min()} đến {raw.index.max()}")

    # Mỗi mốc là cuối của khoảng 15 phút: (00:00, 00:15] ghi ở 00:15. Lấy 2012 đến hết 2014.
    d = raw.loc["2012-01-01 00:15":"2015-01-01 00:00"]

    # Chỉ giữ khách hàng có dữ liệu suốt 3 năm, để tổng không bị nhảy bậc khi có khách hàng mới
    first = d.ne(0).idxmax()
    last = d[::-1].ne(0).idxmax()
    keep = first[(first <= pd.Timestamp("2012-01-02")) & (last >= pd.Timestamp("2014-12-30"))].index
    dropped = sorted(set(d.columns) - set(keep))
    print(f"Giữ {len(keep)} khách hàng có dữ liệu suốt 2012-2014, bỏ {len(dropped)}")

    # Tổng các khách hàng (kW), rồi trung bình 4 mốc 15 phút trong giờ.
    # Giờ H gồm (H:00, H+1:00], ghi nhãn H. Đổi sang MW.
    total = d[keep].sum(axis=1)
    load = total.resample("h", closed="right", label="left").mean() / 1000
    load = load.loc[START:END]

    # Giờ đổi giờ (theo mô tả của UCI): tháng 3 giờ 1:00-2:00 bằng 0, tháng 10 giờ 1:00-2:00 cộng dồn 2 giờ
    dst_hours = [day + pd.Timedelta(hours=1) for y in (2012, 2013, 2014) for day in dst_days(y)]
    print("Giờ đổi giờ bị sai, đặt NaN rồi nội suy:")
    for t in dst_hours:
        prev, nxt = load[t - pd.Timedelta(hours=1)], load[t + pd.Timedelta(hours=1)]
        print(f"   {t}: {load[t]:.1f} MW (giờ trước {prev:.1f}, giờ sau {nxt:.1f})")
    load_before = load.copy()
    load[dst_hours] = np.nan
    load = load.interpolate(method="time")
    return load.rename("load"), load_before, dst_hours, len(keep), dropped


# ---------------------------------------------------------------- thời tiết
def parse_isd(path):
    """Đọc file ISD dạng cố định độ rộng: thời gian (UTC), tốc độ gió, nhiệt độ, điểm sương."""
    rows = []
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            t = pd.Timestamp(line[15:23] + line[23:27])
            wind, wind_q = int(line[65:69]), line[69]
            temp, temp_q = int(line[87:92]), line[92]
            dew, dew_q = int(line[93:98]), line[98]
            rows.append((
                t,
                wind / 10 if wind != 9999 and wind_q not in ISD_BAD_QC else np.nan,
                temp / 10 if temp != 9999 and temp_q not in ISD_BAD_QC else np.nan,
                dew / 10 if dew != 9999 and dew_q not in ISD_BAD_QC else np.nan,
                line[41:46],
            ))
    return pd.DataFrame(rows, columns=["utc", "wind_speed", "temperature", "dew_point", "report_type"])


def relative_humidity(t, td):
    """Độ ẩm tương đối (%) từ nhiệt độ và điểm sương (°C), công thức Magnus."""
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td / (b + td)) / np.exp(a * t / (b + t))


def build_weather():
    obs = pd.concat([parse_isd(p) for p in sorted(ISD_DIR.glob(f"{ISD_STATION}-*.gz"))], ignore_index=True)
    print(f"ISD: {len(obs)} bản tin, loại bản tin: {obs['report_type'].value_counts().to_dict()}")
    print("Số giá trị thiếu/bị loại theo mã chất lượng:", obs[["wind_speed", "temperature", "dew_point"]].isna().sum().to_dict())

    # UTC -> giờ địa phương Lisbon (cùng giờ với dữ liệu điện), rồi trung bình theo giờ
    obs["local"] = (pd.DatetimeIndex(obs["utc"]).tz_localize("UTC")
                    .tz_convert("Europe/Lisbon").tz_localize(None))
    obs["humidity"] = relative_humidity(obs["temperature"], obs["dew_point"])
    hourly = (obs.set_index("local")[["temperature", "humidity", "wind_speed"]]
              .groupby(lambda t: t.floor("h")).mean())
    hourly = hourly.reindex(pd.date_range(START, END, freq="h"))
    return hourly


def gap_lengths(s):
    """Độ dài đoạn thiếu chứa mỗi vị trí (0 nếu không thiếu)."""
    isna = s.isna()
    groups = (isna != isna.shift()).cumsum()
    return isna.groupby(groups).transform("sum").where(isna, 0)


def fill_weather(w, short=3):
    """Đoạn thiếu ngắn (<= 3 giờ): nội suy tuyến tính.
    Đoạn thiếu dài (10 đến 24 giờ): nội suy tuyến tính sẽ xóa mất chu kỳ ngày, nên lấy trung bình
    cùng giờ của ngày hôm trước và ngày hôm sau, rồi mới nội suy phần còn lại."""
    w = w.copy()
    for c in w.columns:
        long_gap = gap_lengths(w[c]) > short
        same_hour = pd.concat([w[c].shift(24), w[c].shift(-24)], axis=1).mean(axis=1)
        w.loc[long_gap, c] = same_hour[long_gap]
        w[c] = w[c].interpolate(method="time", limit_direction="both")
    return w


def longest_gap(s):
    isna = s.isna().astype(int)
    groups = (isna.diff() != 0).cumsum()
    return int(isna.groupby(groups).sum().max())


# ---------------------------------------------------------------- chạy
def main():
    get_uci()
    get_isd()

    load, load_before, dst_hours, n_keep, dropped = build_load()
    weather = build_weather()

    print("\nGiờ thiếu thời tiết trước khi nội suy:")
    gaps = {}
    for c in weather.columns:
        gaps[c] = (int(weather[c].isna().sum()), longest_gap(weather[c]))
        print(f"   {c}: {gaps[c][0]} giờ, đoạn thiếu dài nhất {gaps[c][1]} giờ")
    weather = fill_weather(weather)

    df = pd.concat([load, weather], axis=1)[COLS]
    df.index.name = "Datetime"
    assert len(df) == len(pd.date_range(START, END, freq="h")) and df.notna().all().all()
    out = PROCESSED / "load_weather_hourly.csv"
    df.round(3).to_csv(out)
    print(f"\nĐã lưu {out}: {len(df)} giờ, {df.index.min()} đến {df.index.max()}")
    print(df.describe().round(2))
    print("Tương quan nhanh với load:", df.corr()["load"].round(3).to_dict())

    # ---- Hình 1: 4 biến theo thời gian (trung bình ngày cho dễ nhìn)
    daily = df.resample("D").mean()
    units = {"load": "MW", "temperature": "°C", "humidity": "%", "wind_speed": "m/s"}
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, c, color in zip(axes, COLS, ["C0", "C3", "C2", "C4"]):
        ax.plot(df.index, df[c], lw=0.2, alpha=0.35, color=color, label="Theo giờ")
        ax.plot(daily.index, daily[c], lw=1.0, color=color, label="Trung bình ngày")
        ax.set_ylabel(f"{c} ({units[c]})")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", ncol=2)
    axes[-1].set_xlabel("Thời gian")
    fig.suptitle("Dữ liệu sau khi ghép: điện năng (UCI, 318 khách hàng) và thời tiết Lisbon (NOAA ISD), 2012-2014")
    fig.tight_layout()
    fig.savefig(FIG / "s2_4bien.png", dpi=DPI)
    plt.close(fig)

    # ---- Hình 2: sửa giờ đổi giờ
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, t, title in [(axes[0], dst_hours[4], "Đổi sang giờ mùa hè (30/03/2014)"),
                         (axes[1], dst_hours[5], "Đổi về giờ mùa đông (26/10/2014)")]:
        win = slice(t - pd.Timedelta(hours=12), t + pd.Timedelta(hours=12))
        ax.plot(load_before[win].index, load_before[win].values, "o--", color="C3", ms=3, label="Trước khi sửa")
        ax.plot(load[win].index, load[win].values, "o-", color="C0", ms=3, label="Sau khi sửa (nội suy)")
        ax.axvline(t, color="gray", ls=":")
        ax.set(title=title, xlabel="Giờ", ylabel="load (MW)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "s2_dst.png", dpi=DPI)
    plt.close(fig)
    print("Đã lưu hình s2_4bien.png, s2_dst.png")


if __name__ == "__main__":
    main()
