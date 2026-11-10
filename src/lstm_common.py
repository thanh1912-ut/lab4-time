"""Hỗ trợ mục 8, câu 4 và 5. Người làm: Hoàng Nam.

Dùng scaler/cửa sổ của nhóm; chia mẫu theo thời điểm của y.
Train xáo trộn thứ tự các cửa sổ mỗi epoch (validation/test giữ thứ tự thời gian);
patience chọn trên validation, trung bình 3 seed (xem tune_main).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
_LOCAL_CACHE = Path(__file__).resolve().parents[1] / ".venv" / "nam_cache"
os.environ.setdefault("KERAS_HOME", str(_LOCAL_CACHE / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_CACHE / "matplotlib"))

import numpy as np
import pandas as pd
from features import fit_scale, inverse_load, make_windows
from utils import COLS, L, PROCESSED, SEED, load_data, metrics

RESULTS_DIR = PROCESSED / "lstm"
MODEL_DIR = RESULTS_DIR / "models"
PATIENCE_CANDIDATES = (5, 10)   # thử trên validation, cùng công sức với 4 depth của XGBoost
TUNE_SEEDS = (42, 43, 44)       # seed 42 là mô hình chính; 3 seed để đo độ ổn định


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare_data(calendar=False):
    """Fit scaler trên train; xếp cửa sổ theo mốc y, không chia lại len(X)."""
    df = load_data()
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("Mốc thời gian phải duy nhất và tăng dần.")
    if not np.isfinite(df[COLS].to_numpy()).all():
        raise ValueError("Dữ liệu có NaN/inf.")
    if not df.index.equals(pd.date_range(df.index[0], df.index[-1], freq="h")):
        raise ValueError("Dữ liệu phải liên tục từng giờ.")
    scaled, scaler, (i1, i2) = fit_scale(df)
    X, y, times = make_windows(scaled, L=L)
    X, y = X.astype("float32"), y.astype("float32")
    if calendar:
        hour, day = df.index.hour.to_numpy(), df.index.dayofweek.to_numpy()
        extra = np.column_stack([
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 7), np.cos(2 * np.pi * day / 7),
        ]).astype("float32")
        windows = np.stack([extra[i-L:i] for i in range(L, len(extra))])
        X = np.concatenate([X, windows], axis=-1)
    masks = {"train": times < df.index[i1],
             "val": (times >= df.index[i1]) & (times < df.index[i2]),
             "test": times >= df.index[i2]}
    data = {"df": df, "scaled": scaled, "scaler": scaler, "window": L,
            "calendar": bool(calendar), "split_indices": [int(i1), int(i2)],
            "data_sha256": hashlib.sha256((PROCESSED / "load_weather_hourly.csv").read_bytes()).hexdigest()}
    for split, mask in masks.items():
        data[f"X_{split}"] = np.ascontiguousarray(X[mask])
        data[f"y_{split}"] = y[mask]
        data[f"times_{split}"] = times[mask]
        if len(data[f"y_{split}"]) == 0:
            raise ValueError(f"Tập {split} rỗng.")
    return data


def tensorflow_runtime():
    import tensorflow as tf
    if not getattr(tensorflow_runtime, "configured", False):
        try:
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass
        tf.config.experimental.enable_op_determinism()
        tensorflow_runtime.configured = True
    return tf


def _dataset(X, y=None, batch_size=64, shuffle_seed=None):
    """shuffle_seed khác None: xáo trộn thứ tự cửa sổ mỗi epoch (chỉ dùng cho train)."""
    tf = tensorflow_runtime()
    ds = tf.data.Dataset.from_tensor_slices(X if y is None else (X, y))
    if shuffle_seed is not None:
        ds = ds.shuffle(len(X), seed=int(shuffle_seed), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 2
    options.experimental_deterministic = True
    return ds.with_options(options).prefetch(1)


def predict_original(model, X, scaler):
    z = model.predict(_dataset(X, batch_size=256), verbose=0).reshape(-1)
    pred = np.asarray(inverse_load(z, scaler), dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Mô hình sinh NaN/inf.")
    return pred


def _validate_name(name):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("Tên thí nghiệm không hợp lệ.")


def _source_digest():
    h = hashlib.sha256()
    for p in [Path(__file__), Path(__file__).with_name("features.py"), Path(__file__).with_name("utils.py")]:
        h.update(p.read_bytes())
    return h.hexdigest()


def _check_data_cache(metadata, data):
    cfg = metadata["config"]
    if cfg["data_sha256"] != data["data_sha256"] or cfg["calendar"] != data["calendar"]:
        raise ValueError("Cache không khớp dữ liệu/đặc trưng; chạy lại với --force.")
    if cfg.get("source_sha256") != _source_digest():
        raise ValueError("Mã nguồn đã đổi; chạy lại với --force.")


def load_experiment(name, data):
    _validate_name(name)
    metadata = json.loads((RESULTS_DIR / f"{name}_metadata.json").read_text(encoding="utf-8"))
    _check_data_cache(metadata, data)
    tf = tensorflow_runtime()
    model = tf.keras.models.load_model(MODEL_DIR / f"{name}.keras")
    frame = pd.read_csv(RESULTS_DIR / f"{name}_history.csv")
    history = {col: frame[col].tolist() for col in frame if col != "epoch"}
    predictions = {}
    for split, key in [("train", "train"), ("validation", "val"), ("test", "test")]:
        frame = pd.read_csv(RESULTS_DIR / f"{name}_{split}.csv", parse_dates=["Datetime"])
        if not pd.DatetimeIndex(frame["Datetime"]).equals(data[f"times_{key}"]):
            raise ValueError("Mốc thời gian cache không khớp.")
        truth = data["df"].loc[data[f"times_{key}"], "load"].to_numpy()
        if not np.allclose(frame["y_true"], truth, atol=1e-8):
            raise ValueError("Giá trị thật cache không khớp.")
        predictions[split] = frame["y_pred"].to_numpy()
        if not np.isfinite(predictions[split]).all():
            raise ValueError("Cache chứa dự báo không hợp lệ.")
    return {"model": model, "history": history, "metadata": metadata, "predictions": predictions}


def train_experiment(name, data, epochs=50, batch_size=64, patience=5,
                     dropout=0.0, early_stopping=True, verbose=2, force=False,
                     shuffle=True, seed=SEED):
    _validate_name(name)
    if epochs < 1 or batch_size < 1 or patience < 0 or not 0 <= dropout < 1:
        raise ValueError("Cấu hình huấn luyện không hợp lệ.")
    tf = tensorflow_runtime()
    config = {"epochs": int(epochs), "batch_size": int(batch_size), "patience": int(patience),
              "dropout": float(dropout), "early_stopping": bool(early_stopping), "seed": int(seed),
              "window": data["window"], "calendar": data["calendar"], "shuffle": bool(shuffle),
              "learning_rate": 0.001, "data_sha256": data["data_sha256"],
              "source_sha256": _source_digest(), "tensorflow": tf.__version__,
              "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version()}
    meta_path = RESULTS_DIR / f"{name}_metadata.json"
    if meta_path.exists() and not force:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("config") == config:
            print(f"[{name}] Dùng kết quả đã lưu, cấu hình và dữ liệu khớp.", flush=True)
            return load_experiment(name, data)
        print(f"[{name}] Cấu hình thay đổi; huấn luyện lại.", flush=True)
    # Không để metadata cũ xác nhận bộ file bị ghi dở nếu chạy lại bị ngắt.
    meta_path.unlink(missing_ok=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    layers = [tf.keras.Input(shape=data["X_train"].shape[1:]), tf.keras.layers.LSTM(64, return_sequences=True)]
    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.LSTM(32))
    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
    print(f"[{name}] train={data['X_train'].shape}, val={data['X_val'].shape}, test={data['X_test'].shape}", flush=True)
    started = time.perf_counter()
    fitted = model.fit(_dataset(data["X_train"], data["y_train"], batch_size,
                                shuffle_seed=int(seed) if shuffle else None),
                       validation_data=_dataset(data["X_val"], data["y_val"], batch_size),
                       epochs=epochs, callbacks=callbacks, verbose=verbose,
                       shuffle=False)  # Keras bỏ qua tham số này với tf.data; xáo trộn nằm trong _dataset
    elapsed = time.perf_counter() - started
    history = {k: [float(v) for v in values] for k, values in fitted.history.items()}
    if not np.isfinite(np.asarray(list(history.values()))).all():
        raise RuntimeError("Loss có NaN/inf; không xuất kết quả.")
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    metadata = {"name": name, "config": config, "epochs_run": len(history["loss"]),
                "best_epoch": best_epoch, "selected_epoch": best_epoch if early_stopping else len(history["loss"]),
                "train_seconds": round(elapsed, 3), "parameters": int(model.count_params()),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "devices": [d.name for d in tf.config.list_physical_devices()],
                "units": "MW", "split_metrics": {}, "splits": {}}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for split, key in [("train", "train"), ("validation", "val"), ("test", "test")]:
        times = data[f"times_{key}"]
        truth = data["df"].loc[times, "load"].to_numpy()
        pred = predict_original(model, data[f"X_{key}"], data["scaler"])
        predictions[split] = pred
        metadata["split_metrics"][split] = metrics(truth, pred)
        metadata["splits"][split] = {"samples": len(times), "start": str(times[0]), "end": str(times[-1])}
        pd.DataFrame({"Datetime": times, "y_true": truth, "y_pred": pred}).to_csv(RESULTS_DIR / f"{name}_{split}.csv", index=False)
    hist = pd.DataFrame(history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(RESULTS_DIR / f"{name}_history.csv", index=False)
    model.save(MODEL_DIR / f"{name}.keras")
    # Metadata ghi cuối: không nhận cache dở nếu quá trình chạy bị ngắt.
    save_json(meta_path, metadata)
    print(f"[{name}] Test: {metadata['split_metrics']['test']}; train {elapsed:.1f}s", flush=True)
    return {"model": model, "history": history, "metadata": metadata, "predictions": predictions}


def tune_main(data, epochs=50, batch_size=64, force=False, verbose=2):
    """Chọn patience cho mô hình chính chỉ bằng validation.

    Mỗi patience trong PATIENCE_CANDIDATES chạy với 3 seed; chọn patience có RMSE
    validation trung bình thấp nhất. Mô hình chính = patience đã chọn, seed 42.
    Cột test chỉ để báo cáo độ ổn định, không tham gia chọn.
    """
    runs, rows = {}, []
    for patience in PATIENCE_CANDIDATES:
        for seed in TUNE_SEEDS:
            result = train_experiment(f"lstm_p{patience}_s{seed}", data, epochs=epochs,
                                      batch_size=batch_size, patience=patience, seed=seed,
                                      verbose=verbose, force=force)
            runs[(patience, seed)] = result
            meta, sm = result["metadata"], result["metadata"]["split_metrics"]
            rows.append({"patience": patience, "seed": seed, "epochs_run": meta["epochs_run"],
                         "selected_epoch": meta["selected_epoch"], "train_seconds": meta["train_seconds"],
                         "train_RMSE": sm["train"]["RMSE"], "val_RMSE": sm["validation"]["RMSE"],
                         "test_RMSE": sm["test"]["RMSE"], "test_MAE": sm["test"]["MAE"],
                         "test_MAPE": sm["test"]["MAPE(%)"]})
    table = pd.DataFrame(rows)
    mean_val = table.groupby("patience")["val_RMSE"].mean()
    best = int(mean_val.idxmin())
    chosen = table[table.patience == best]
    selection = {"patience": best, "main_seed": SEED, "main_name": f"lstm_p{best}_s{SEED}",
                 "rule": "patience có RMSE validation trung bình 3 seed thấp nhất; test không dùng để chọn",
                 "mean_validation_rmse": {str(k): round(float(v), 4) for k, v in mean_val.items()},
                 "test_rmse_mean": round(float(chosen.test_RMSE.mean()), 2),
                 "test_rmse_std": round(float(chosen.test_RMSE.std(ddof=1)), 2)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "lstm_tuning.csv", index=False)
    save_json(RESULTS_DIR / "lstm_selection.json", selection)
    print(f"[tune] patience chọn = {best} (val RMSE trung bình {selection['mean_validation_rmse']})", flush=True)
    return {"selection": selection, "table": table, "main": runs[(best, SEED)], "runs": runs}
