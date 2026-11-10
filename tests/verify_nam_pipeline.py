"""Numerical checks for Nam's local LSTM submission; never trains or writes.

Run before training: python tests/verify_nam_pipeline.py --data-only
Run after both scripts: python tests/verify_nam_pipeline.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from features import fit_scale, inverse_load
from lstm_common import (
    MODEL_DIR, RESULTS_DIR, load_experiment, predict_original, prepare_data,
)
from utils import COLS, PROCESSED, load_data, metrics

SPLITS = {"train": "train", "validation": "val", "test": "test"}
# Mô hình chính là lần chạy patience đã chọn trên validation, seed 42 (lstm_selection.json).
MAIN_NAME = json.loads((ROOT / "data" / "processed" / "lstm" / "lstm_selection.json")
                       .read_text(encoding="utf-8"))["main_name"]
EXPERIMENTS = [
    MAIN_NAME, "lstm_calendar", "lstm_no_early_stopping", "lstm_dropout",
]


def check_data():
    data = prepare_data()
    calendar = prepare_data(calendar=True)
    df = load_data()
    scaled, scaler, (i1, i2) = fit_scale(df)
    assert (len(df), i1, i2) == (26304, 18412, 22358)
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert not df.isna().any().any()
    np.testing.assert_allclose(scaler.mean_, df.iloc[:i1][COLS].mean(), rtol=1e-12)
    np.testing.assert_allclose(scaler.scale_, df.iloc[:i1][COLS].std(ddof=0), rtol=1e-12)
    for split, n in [("train", 18388), ("val", 3946), ("test", 3946)]:
        X, y, times = data[f"X_{split}"], data[f"y_{split}"], data[f"times_{split}"]
        assert X.shape == (n, 24, 4), (split, X.shape)
        assert len(y) == len(times) == n
        assert np.isfinite(X).all() and np.isfinite(y).all()
        np.testing.assert_allclose(
            inverse_load(y, data["scaler"]), df.loc[times, "load"], rtol=0, atol=5e-5,
        )
        assert calendar[f"X_{split}"].shape == (n, 24, 8)
        assert calendar[f"times_{split}"].equals(times)
        np.testing.assert_array_equal(calendar[f"X_{split}"][:, :, :4], X)
        np.testing.assert_array_equal(calendar[f"y_{split}"], y)
        for k in [0, len(X)//2, len(X)-1]:
            target_position = df.index.get_loc(times[k])
            history_times = df.index[target_position-24:target_position]
            expected = scaled.loc[history_times, COLS].to_numpy()
            np.testing.assert_allclose(X[k], expected, atol=1e-6, rtol=1e-6)
            hour = history_times.hour.to_numpy()
            weekday = history_times.dayofweek.to_numpy()
            expected_calendar = np.column_stack([
                np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7),
            ])
            np.testing.assert_allclose(
                calendar[f"X_{split}"][k, :, 4:], expected_calendar, atol=1e-6, rtol=1e-6,
            )
            assert history_times[-1] < times[k]
    assert str(data["times_test"][0]) == "2014-07-20 14:00:00"
    assert str(data["times_test"][-1]) == "2014-12-31 23:00:00"
    assert data["times_train"][-1] < data["times_val"][0] < data["times_test"][0]
    print("PASS data: target splits, train-only scaler, past-only windows, original MW, calendar timestamps")
    return data, calendar


def read_prediction(path, data, split):
    frame = pd.read_csv(path, parse_dates=["Datetime"])
    assert list(frame) == ["Datetime", "y_true", "y_pred"], str(path)
    times = data[f"times_{SPLITS[split]}"]
    assert len(frame) == len(times), str(path)
    assert frame["Datetime"].is_unique, str(path)
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(frame["Datetime"]), pd.DatetimeIndex(times), check_names=False,
    )
    assert np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all(), str(path)
    # Published truth must retain original CSV precision, not inverse(float32 label).
    np.testing.assert_array_equal(frame["y_true"], data["df"].loc[times, "load"])
    return frame


def check_artifact_consistency(data):
    main_metadata = None
    for name in EXPERIMENTS:
        meta = json.loads((RESULTS_DIR / f"{name}_metadata.json").read_text(encoding="utf-8"))
        assert meta["name"] == name and meta["units"] == "MW"
        assert meta["config"]["data_sha256"] == data["data_sha256"]
        assert meta["config"]["seed"] == 42
        assert meta["config"]["window"] == 24
        assert meta["config"]["calendar"] == (name == "lstm_calendar")
        assert meta["config"]["shuffle"] is True
        assert meta["parameters"] == (31137 if name == "lstm_calendar" else 30113)
        assert (MODEL_DIR / f"{name}.keras").is_file()
        history = pd.read_csv(RESULTS_DIR / f"{name}_history.csv")
        np.testing.assert_array_equal(history["epoch"], np.arange(1, len(history)+1))
        assert np.isfinite(history[["loss", "val_loss"]].to_numpy()).all()
        assert len(history) == meta["epochs_run"] <= meta["config"]["epochs"]
        assert meta["best_epoch"] == int(history["val_loss"].argmin())+1
        if meta["config"]["early_stopping"]:
            assert meta["selected_epoch"] == meta["best_epoch"]
        else:
            assert meta["epochs_run"] == meta["config"]["epochs"]
            assert meta["selected_epoch"] == meta["epochs_run"]
        for split in SPLITS:
            frame = read_prediction(RESULTS_DIR / f"{name}_{split}.csv", data, split)
            assert meta["split_metrics"][split] == metrics(frame.y_true, frame.y_pred)
            assert meta["splits"][split]["samples"] == len(frame)
            assert meta["splits"][split]["start"] == str(frame.Datetime.iloc[0])
            assert meta["splits"][split]["end"] == str(frame.Datetime.iloc[-1])
        if name == MAIN_NAME:
            main_metadata = meta
        else:
            assert meta["config"]["source_sha256"] == main_metadata["config"]["source_sha256"]
    official = read_prediction(PROCESSED / "pred_lstm.csv", data, "test")
    cached = read_prediction(RESULTS_DIR / f"{MAIN_NAME}_test.csv", data, "test")
    np.testing.assert_allclose(official["y_pred"], cached["y_pred"], atol=1e-10, rtol=0)
    assert main_metadata["split_metrics"]["test"] == metrics(official.y_true, official.y_pred)
    baseline = pd.read_csv(RESULTS_DIR / "m3_baselines.csv").set_index("Model")
    for label, lag in [("Naive", 1), ("Seasonal naive (24h)", 24)]:
        expected = metrics(official.y_true, data["df"].load.shift(lag).loc[data["times_test"]])
        assert baseline.loc[label].to_dict() == expected
    print("PASS artifacts: all four runs use identical target dates; original y_true and recomputed MW metrics match")
    return official


def check_model_reload(data, official):
    result = load_experiment(MAIN_NAME, data)
    model = result["model"]
    assert model.count_params() == 30113
    first = predict_original(model, data["X_test"][:12], data["scaler"])
    np.testing.assert_allclose(first, official["y_pred"].iloc[:12], atol=0.005, rtol=1e-5)
    individual = np.asarray([
        inverse_load(
            np.asarray(model(data["X_test"][i:i+1], training=False)).reshape(-1), data["scaler"],
        )[0]
        for i in range(12)
    ])
    np.testing.assert_allclose(individual, first, atol=0.005, rtol=1e-5)
    print("PASS model: Keras reload, official architecture, batch versus individual fixed-weight one-step predictions")
    print(json.dumps({"test_metrics": metrics(official.y_true, official.y_pred),
                      "parameters": model.count_params()}, ensure_ascii=False))


def check_deliverables():
    for name in ["m3", "q4", "q5"]:
        for suffix in ["report", "process"]:
            path = ROOT / "docs" / "lab4" / "parts" / f"{name}_{suffix}.md"
            assert path.is_file() and path.stat().st_size > 300, str(path)
            text = path.read_text(encoding="utf-8")
            assert "Hoàng Nam" in text, str(path)
            for line in text.splitlines():
                if line.startswith("![") and "](" in line:
                    relative = line.split("](", 1)[1].split(")", 1)[0]
                    assert (path.parent / relative).is_file(), (str(path), relative)
    for name in ["m3_lstm_1tuan.png", "q4_loi_theo_thu.png", "q5_loss.png"]:
        path = ROOT / "docs" / "lab4" / "figures" / name
        assert path.is_file() and path.stat().st_size > 10000, str(path)
    print("PASS deliverables: six Markdown parts, valid image links, required figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="Check splits and features before training")
    args = parser.parse_args()
    data, _calendar = check_data()
    if not args.data_only:
        official = check_artifact_consistency(data)
        check_model_reload(data, official)
        check_deliverables()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
