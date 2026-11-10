# Mục 6 - Mô hình VAR (Lê Quỳnh)

## Kiểm tra tính dừng (ADF, trên tập train)
| biến | ADF stat | p-value | dừng (p<0,05) |
|---|---|---|---|
| load | -3.2583 | 0.0168 | có |
| temperature | -4.6565 | 0.0001 | có |
| humidity | -7.9147 | 0.0000 | có |
| wind_speed | -11.6509 | 0.0000 | có |

Cả 4 biến đều dừng nên không sai phân.



## Chọn bậc trễ (select_order, maxlags=48)
| p | aic | bic |
|---|---|---|
| 1.0000 | -11.3850 | -11.3765 |
| 2.0000 | -12.6006 | -12.5853 |
| 6.0000 | -12.8567 | -12.8141 |
| 12.0000 | -13.1133 | -13.0298 |
| 24.0000 | -14.1276 | -13.9624 |
| 36.0000 | -14.7761 | -14.5292 |
| 48.0000 | -14.8172 | -14.4885 |

p theo AIC = 48, p theo BIC = 26. Chọn **p = 26** theo BIC (BIC phạt số tham số mạnh hơn nên thường chọn p nhỏ hơn; AIC dễ chọn p lớn, tốn tham số).

## Kiểm định Granger (tác động lên load)
| nguyên nhân | F | p-value | Granger-cause load |
|---|---|---|---|
| temperature | 10.4700 | 0.0000 | có |
| humidity | 2.3993 | 0.0001 | có |
| wind_speed | 1.3416 | 0.1143 | không |
| cả 3 biến | 4.8707 | 0.0000 | có |

## Sai số trên tập test (walk-forward, refit mỗi 168 giờ, thang đo gốc)
| RMSE (MW) | MAE (MW) | MAPE (%) |
|---|---|---|
| 6.05 | 4.06 | 2.03 |

Hình: `figures/m1_var_1tuan.png`. Thời gian chạy walk-forward: 7 giây.

## Bản dùng cho so sánh công bằng (mục 10)
Học VAR(26) **một lần trên train**, không refit trên test; mỗi giờ test vẫn dự báo 1 bước từ 26 giờ thật trước đó.
Cùng luật với XGBoost và LSTM (validation chỉ để chọn cấu hình). File `data/processed/pred_var_fixed.csv`.

| RMSE (MW) | MAE (MW) | MAPE (%) |
|---|---|---|
| 6.07 | 4.11 | 2.06 |

Refit mỗi tuần chỉ đổi RMSE -0.02 MW so với bản học một lần.
