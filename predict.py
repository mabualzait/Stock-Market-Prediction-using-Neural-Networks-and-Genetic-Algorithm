import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


DATE_COL = "Date"
PRICE_COLS = ["Open", "High", "Low", "Close"]


def read_csv_flexible(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read CSV to numpy arrays handling quoted single-column rows.

    Returns arrays: dates (str), prices (float) with shape (N, 4) for Open,High,Low,Close.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    # Detect header
    has_header = lines[0].lower().startswith("date,") or lines[0].startswith("\"Date,")
    if has_header:
        lines = lines[1:]

    dates = []
    prices = []
    for line in lines:
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            # Skip malformed
            continue
        dates.append(parts[0])
        try:
            prices.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        except ValueError:
            continue
    return np.array(dates), np.array(prices, dtype=float)


def ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[:] = np.nan
    if len(values) == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def compute_indicators(dates: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # prices columns: Open, High, Low, Close
    close = prices[:, 3]
    sma10 = np.convolve(close, np.ones(10) / 10.0, mode="full")[: len(close)]
    sma10[:9] = np.nan
    sma50 = np.convolve(close, np.ones(50) / 50.0, mode="full")[: len(close)]
    sma50[:49] = np.nan
    ema10 = ema(close, 10)
    ema50 = ema(close, 50)

    # Stack features
    feats = np.column_stack([prices, sma10, sma50, ema10, ema50])
    # Target next-day close
    target_next = np.roll(close, -1)
    ret_next = target_next / close - 1.0

    # Drop rows with NaNs and the last row (no next target)
    mask = (
        ~np.isnan(feats).any(axis=1)
        & ~np.isnan(target_next)
    )
    mask[-1] = False
    return feats[mask], np.column_stack([target_next[mask], ret_next[mask]])


def prepare_features(feats_and_targets: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats, targets = feats_and_targets
    X = feats.astype(float)
    y_next_close = targets[:, 0].astype(float)
    ret_next = targets[:, 1].astype(float)
    return X, y_next_close, ret_next


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    # Add bias term
    Xb = np.column_stack([X, np.ones(len(X))])
    # OLS closed-form: w = (X'X)^-1 X'y, with small ridge for stability
    ridge = 1e-6 * np.eye(Xb.shape[1])
    w = np.linalg.pinv(Xb.T @ Xb + ridge) @ (Xb.T @ y)
    weights, bias = w[:-1], w[-1]
    return weights, bias


def predict_linear_regression(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return X @ weights + bias


def generate_signals(close_today: np.ndarray, y_pred_next: np.ndarray, up_thresh: float, down_thresh: float) -> np.ndarray:
    pred_return = y_pred_next / close_today - 1.0
    return np.where(
        pred_return >= up_thresh, "BUY",
        np.where(pred_return <= -down_thresh, "SELL", "HOLD")
    )


def run(train_csv: Path, test_csv: Path, out_csv: Path, up_thresh: float, down_thresh: float) -> None:
    _, train_prices = read_csv_flexible(train_csv)
    test_dates, test_prices = read_csv_flexible(test_csv)

    X_train_full, targets_train = compute_indicators(np.array([]), train_prices)
    X_test_full, targets_test = compute_indicators(test_dates, test_prices)

    # For signals we need today's close for corresponding rows
    close_today_test = X_test_full[:, 3]

    # Fit simple linear regression
    weights, bias = fit_linear_regression(X_train_full, targets_train[:, 0])
    y_pred_next = predict_linear_regression(X_test_full, weights, bias)
    signals = generate_signals(close_today_test, y_pred_next, up_thresh, down_thresh)

    # Build output csv
    predicted_return_next = y_pred_next / close_today_test - 1.0
    ret_next_true = targets_test[:, 1]

    # Write CSV manually to avoid pandas dependency
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Date,Close,Return_next,Predicted_Close_next,Predicted_Return_next,Signal\n")
        for i in range(len(close_today_test)):
            f.write(
                f"{test_dates[i]},{close_today_test[i]:.6f},{ret_next_true[i]:.6f},{y_pred_next[i]:.6f},{predicted_return_next[i]:.6f},{signals[i]}\n"
            )

    # Basic metrics
    mae = float(np.mean(np.abs(targets_test[:, 0] - y_pred_next)))
    rmse = float(np.sqrt(np.mean((targets_test[:, 0] - y_pred_next) ** 2)))
    print({"MAE": mae, "RMSE": rmse})
    print(f"Saved predictions and signals to: {out_csv}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stock prediction and signal generator")
    p.add_argument("--train_csv", type=Path, default=Path("stock_market_train.csv"))
    p.add_argument("--test_csv", type=Path, default=Path("stock_market_test.csv"))
    p.add_argument("--out_csv", type=Path, default=Path("signals.csv"))
    p.add_argument("--up_thresh", type=float, default=0.005, help="BUY threshold on predicted next-day return")
    p.add_argument("--down_thresh", type=float, default=0.005, help="SELL threshold on predicted next-day return magnitude")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args.train_csv, args.test_csv, args.out_csv, args.up_thresh, args.down_thresh)


if __name__ == "__main__":
    main()

