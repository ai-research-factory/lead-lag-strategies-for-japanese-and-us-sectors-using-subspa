"""Validation script for the PCASub model using real market data from the ARF Data API.

Fetches U.S. Select Sector SPDR ETF and TOPIX-17 sector ETF data,
computes returns, and verifies that PCASub.fit() and .predict() execute
correctly and produce outputs of the expected shape.
"""

import sys
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pca_sub import PCASub

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"

US_TICKERS = ["XLE", "XLF", "XLU", "XLI", "XLK", "XLV", "XLY", "XLP", "XLB", "XLRE", "XLC"]
JP_TICKERS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
    "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
    "1631.T", "1632.T", "1633.T",
]

DATA_DIR = PROJECT_ROOT / "data"


def fetch_ohlcv(ticker: str, interval: str = "1d", period: str = "5y") -> pd.DataFrame:
    """Fetch OHLCV data from ARF Data API with local caching."""
    DATA_DIR.mkdir(exist_ok=True)
    safe_name = ticker.replace("/", "_").replace(".", "_")
    cache_path = DATA_DIR / f"{safe_name}_{interval}_{period}.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")
        return df

    url = f"{API_BASE}?ticker={ticker}&interval={interval}&period={period}"
    print(f"  Fetching {ticker} from API...")
    df = pd.read_csv(url, parse_dates=["timestamp"], index_col="timestamp")
    df.to_csv(cache_path)
    return df


def compute_close_returns(tickers: list[str], label: str) -> pd.DataFrame:
    """Fetch data for multiple tickers and compute close-to-close returns."""
    closes = {}
    for t in tickers:
        try:
            ohlcv = fetch_ohlcv(t)
            closes[t] = ohlcv["close"]
        except Exception as e:
            print(f"  WARNING: Could not fetch {t}: {e}")
    close_df = pd.DataFrame(closes).sort_index()
    returns = close_df.pct_change().dropna()
    print(f"  {label}: {returns.shape[0]} days x {returns.shape[1]} tickers")
    return returns


def main():
    print("=" * 60)
    print("PCASub Model Validation (Real Data)")
    print("=" * 60)

    # --- Fetch data ---
    print("\n[1/4] Fetching U.S. sector ETF data...")
    us_returns = compute_close_returns(US_TICKERS, "US")

    print("\n[2/4] Fetching Japanese sector ETF data...")
    jp_returns = compute_close_returns(JP_TICKERS, "JP")

    # Align dates: use intersection of available trading dates
    common_dates = us_returns.index.intersection(jp_returns.index)
    # For the lead-lag setup, US day t predicts JP day t+1.
    # Use US returns shifted by 1 day relative to JP returns.
    us_aligned = us_returns.loc[common_dates].iloc[:-1]
    jp_aligned = jp_returns.loc[common_dates].iloc[1:]

    X_us = us_aligned.values
    Y_jp = jp_aligned.values
    n_samples, n_us = X_us.shape
    _, n_jp = Y_jp.shape

    print(f"\n  Aligned data: {n_samples} samples, {n_us} US sectors, {n_jp} JP sectors")

    # --- Split into train / test ---
    train_size = min(200, n_samples - 50)
    test_size = min(50, n_samples - train_size)
    X_train, X_test = X_us[:train_size], X_us[train_size:train_size + test_size]
    Y_train, Y_test = Y_jp[:train_size], Y_jp[train_size:train_size + test_size]

    # --- Fit model ---
    print("\n[3/4] Fitting PCASub model (K=3, L=60, lambda=0.9)...")
    model = PCASub(K=3, L=60, lambda_decay=0.9)
    model.fit(X_train, Y_train)

    print(f"  Eigenvectors shape: {model.eigvecs_.shape}  (expected: ({n_us}, 3))")
    print(f"  Beta shape:         {model.beta_.shape}  (expected: (3, {n_jp}))")
    print(f"  Intercept shape:    {model.intercept_.shape}  (expected: ({n_jp},))")

    # --- Predict ---
    print("\n[4/4] Running predictions...")
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    print(f"  Train predictions shape: {Y_pred_train.shape}  (expected: ({train_size}, {n_jp}))")
    print(f"  Test predictions shape:  {Y_pred_test.shape}  (expected: ({test_size}, {n_jp}))")

    # --- Validation checks ---
    print("\n" + "=" * 60)
    print("Validation Checks")
    print("=" * 60)

    checks = {}

    # Check 1: Shapes
    shape_ok = (
        model.eigvecs_.shape == (n_us, 3)
        and model.beta_.shape == (3, n_jp)
        and model.intercept_.shape == (n_jp,)
        and Y_pred_train.shape == (train_size, n_jp)
        and Y_pred_test.shape == (test_size, n_jp)
    )
    checks["shapes_correct"] = bool(shape_ok)
    print(f"  Shapes correct:           {'PASS' if shape_ok else 'FAIL'}")

    # Check 2: No NaN/Inf in outputs
    no_nan = (
        not np.any(np.isnan(Y_pred_train))
        and not np.any(np.isnan(Y_pred_test))
        and not np.any(np.isinf(Y_pred_train))
        and not np.any(np.isinf(Y_pred_test))
    )
    checks["no_nan_inf"] = bool(no_nan)
    print(f"  No NaN/Inf in preds:      {'PASS' if no_nan else 'FAIL'}")

    # Check 3: Predictions are non-trivial (not all zeros or constant)
    pred_std = np.std(Y_pred_test)
    nontrivial = pred_std > 1e-12
    checks["nontrivial_predictions"] = bool(nontrivial)
    print(f"  Non-trivial predictions:  {'PASS' if nontrivial else 'FAIL'} (std={pred_std:.6e})")

    # Check 4: Direction accuracy on test set
    sign_match = np.sign(Y_pred_test) == np.sign(Y_test)
    direction_accuracy = sign_match.mean()
    checks["direction_accuracy"] = float(direction_accuracy)
    print(f"  Direction accuracy:       {direction_accuracy:.4f}")

    # Check 5: Correlation per JP sector
    correlations = []
    for j in range(n_jp):
        if np.std(Y_test[:, j]) > 0 and np.std(Y_pred_test[:, j]) > 0:
            corr = np.corrcoef(Y_pred_test[:, j], Y_test[:, j])[0, 1]
            correlations.append(float(corr))
    mean_corr = float(np.mean(correlations)) if correlations else 0.0
    checks["mean_correlation"] = mean_corr
    print(f"  Mean pred-actual corr:    {mean_corr:.4f}")

    all_passed = checks["shapes_correct"] and checks["no_nan_inf"] and checks["nontrivial_predictions"]
    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")

    # --- Collect metrics ---
    metrics = {
        "phase": 1,
        "model": "PCASub",
        "parameters": {"K": 3, "L": 60, "lambda_decay": 0.9},
        "data": {
            "n_us_sectors": n_us,
            "n_jp_sectors": n_jp,
            "n_samples_aligned": n_samples,
            "train_size": train_size,
            "test_size": test_size,
        },
        "validation": checks,
        "status": "PASS" if all_passed else "FAIL",
    }

    return metrics


if __name__ == "__main__":
    metrics = main()

    # Save metrics
    reports_dir = PROJECT_ROOT / "reports" / "cycle_1"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
