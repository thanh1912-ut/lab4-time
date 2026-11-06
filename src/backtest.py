"""Walk-forward validation dùng chung. Người viết: Bao Dinh. (Khung gợi ý, Bao Dinh kiểm tra và hoàn thiện.)"""
import numpy as np

def walk_forward(fit_fn, predict_fn, scaled, start, end, refit_every=None):
    """Dự báo lần lượt từng giờ t trong [start, end) chỉ dùng dữ liệu trước t.
    fit_fn(history_df) -> model ; predict_fn(model, history_df) -> load(t) đã chuẩn hóa.
    Sau mỗi bước, giá trị THẬT của giờ t được thêm vào lịch sử (đúng như đề:
    train -> predict t+1 -> thêm vào chuỗi -> predict t+2 ...).
    refit_every: số giờ giữa 2 lần huấn luyện lại (None = không huấn luyện lại)."""
    model = fit_fn(scaled.iloc[:start])
    preds = []
    for k, t in enumerate(range(start, end)):
        if refit_every and k > 0 and k % refit_every == 0:
            model = fit_fn(scaled.iloc[:t])
        preds.append(predict_fn(model, scaled.iloc[:t]))
    return np.array(preds)
