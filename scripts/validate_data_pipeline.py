"""Validate the Phase 2 data pipeline.

Verifies that:
1. All 11 U.S. and 17 Japanese ETFs are fetched successfully.
2. U.S. returns are close-to-close, JP returns are open-to-close.
3. Lead-lag alignment is correct (each U.S. day paired with next JP trading day).
4. No lookahead bias — JP date always strictly after U.S. date.
5. Returns are within reasonable ranges (no data corruption).
6. The aligned dataset integrates correctly with PCASub model.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import DataPipeline, US_TICKERS, JP_TICKERS
from src.models.pca_sub import PCASub


def main():
    print("=" * 60)
    print("Phase 2: Data Pipeline Validation")
    print("=" * 60)

    pipeline = DataPipeline(data_dir=PROJECT_ROOT / "data")
    checks = {}

    # --- Step 1: Fetch and compute returns ---
    print("\n[1/6] Computing U.S. close-to-close returns...")
    us_returns = pipeline.compute_us_returns()
    us_ok = us_returns.shape[1] == len(US_TICKERS)
    checks["us_all_tickers_fetched"] = bool(us_ok)
    print(f"  Shape: {us_returns.shape} — {'PASS' if us_ok else 'FAIL'}")

    print("\n[2/6] Computing JP open-to-close returns...")
    jp_returns = pipeline.compute_jp_returns()
    jp_ok = jp_returns.shape[1] == len(JP_TICKERS)
    checks["jp_all_tickers_fetched"] = bool(jp_ok)
    print(f"  Shape: {jp_returns.shape} — {'PASS' if jp_ok else 'FAIL'}")

    # --- Step 2: Verify return type correctness ---
    print("\n[3/6] Verifying return calculations...")

    # Spot-check: U.S. close-to-close
    us_ohlcv = pipeline.fetch_ohlcv("XLE")
    us_close = us_ohlcv["close"]
    expected_ret = (us_close.iloc[1] - us_close.iloc[0]) / us_close.iloc[0]
    actual_ret = us_returns["XLE"].iloc[0]
    us_ret_match = abs(expected_ret - actual_ret) < 1e-10
    checks["us_close_to_close_correct"] = bool(us_ret_match)
    print(f"  U.S. close-to-close spot check: {'PASS' if us_ret_match else 'FAIL'}")
    print(f"    Expected: {expected_ret:.8f}, Got: {actual_ret:.8f}")

    # Spot-check: JP open-to-close
    jp_ohlcv = pipeline.fetch_ohlcv("1617.T")
    jp_open = jp_ohlcv["open"]
    jp_close = jp_ohlcv["close"]
    # Find first date present in jp_returns
    first_jp_date = jp_returns.index[0]
    jp_ohlcv_norm = jp_ohlcv.copy()
    jp_ohlcv_norm.index = jp_ohlcv_norm.index.normalize()
    if first_jp_date in jp_ohlcv_norm.index:
        row = jp_ohlcv_norm.loc[first_jp_date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        expected_otc = (row["close"] - row["open"]) / row["open"]
        actual_otc = jp_returns["1617.T"].iloc[0]
        jp_ret_match = abs(expected_otc - actual_otc) < 1e-10
    else:
        jp_ret_match = False
        expected_otc = actual_otc = float("nan")
    checks["jp_open_to_close_correct"] = bool(jp_ret_match)
    print(f"  JP open-to-close spot check: {'PASS' if jp_ret_match else 'FAIL'}")
    print(f"    Expected: {expected_otc:.8f}, Got: {actual_otc:.8f}")

    # --- Step 3: Lead-lag alignment ---
    print("\n[4/6] Verifying lead-lag alignment...")
    dataset = pipeline.align_lead_lag(us_returns, jp_returns)

    # Check all JP dates are strictly after their paired US dates
    no_lookahead = all(
        jp_d > us_d
        for us_d, jp_d in zip(dataset.dates_us, dataset.dates_jp)
    )
    checks["no_lookahead_bias"] = bool(no_lookahead)
    print(f"  No lookahead (JP date > US date): {'PASS' if no_lookahead else 'FAIL'}")

    # Check typical lag is 1 calendar day (most pairs should be 1-3 days apart)
    lags = [(jp_d - us_d).days for us_d, jp_d in zip(dataset.dates_us, dataset.dates_jp)]
    avg_lag = np.mean(lags)
    max_lag = max(lags)
    min_lag = min(lags)
    lag_reasonable = min_lag >= 1 and max_lag <= 7
    checks["lag_range_reasonable"] = bool(lag_reasonable)
    print(f"  Lag range: {min_lag}-{max_lag} days (avg {avg_lag:.1f}) — {'PASS' if lag_reasonable else 'FAIL'}")

    # Show lag distribution
    from collections import Counter
    lag_dist = Counter(lags)
    print(f"  Lag distribution: {dict(sorted(lag_dist.items()))}")

    # --- Step 4: Data quality ---
    print("\n[5/6] Checking data quality...")
    X_us, Y_jp = dataset.X_us, dataset.Y_jp

    no_nan = not np.any(np.isnan(X_us)) and not np.any(np.isnan(Y_jp))
    no_inf = not np.any(np.isinf(X_us)) and not np.any(np.isinf(Y_jp))
    checks["no_nan"] = bool(no_nan)
    checks["no_inf"] = bool(no_inf)
    print(f"  No NaN values: {'PASS' if no_nan else 'FAIL'}")
    print(f"  No Inf values: {'PASS' if no_inf else 'FAIL'}")

    # Returns should be within reasonable range (< 20% daily)
    us_max = np.abs(X_us).max()
    jp_max = np.abs(Y_jp).max()
    returns_reasonable = us_max < 0.25 and jp_max < 0.25
    checks["returns_in_range"] = bool(returns_reasonable)
    print(f"  US max |return|: {us_max:.4f}, JP max |return|: {jp_max:.4f} — {'PASS' if returns_reasonable else 'WARN'}")

    # Summary stats
    print(f"\n  Dataset shape: X_us={X_us.shape}, Y_jp={Y_jp.shape}")
    print(f"  US return stats: mean={X_us.mean():.6f}, std={X_us.std():.4f}")
    print(f"  JP return stats: mean={Y_jp.mean():.6f}, std={Y_jp.std():.4f}")
    print(f"  Date range: {dataset.dates_us[0].date()} to {dataset.dates_us[-1].date()}")

    # --- Step 5: Integration with PCASub ---
    print("\n[6/6] Testing integration with PCASub model...")
    train_size = min(200, X_us.shape[0] - 50)
    test_size = 50
    X_train, X_test = X_us[:train_size], X_us[train_size:train_size + test_size]
    Y_train, Y_test = Y_jp[:train_size], Y_jp[train_size:train_size + test_size]

    model = PCASub(K=3, L=60, lambda_decay=0.9)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    pred_shape_ok = Y_pred.shape == (test_size, len(JP_TICKERS))
    pred_finite = np.all(np.isfinite(Y_pred))
    checks["model_integration_shape"] = bool(pred_shape_ok)
    checks["model_integration_finite"] = bool(pred_finite)
    print(f"  Prediction shape correct: {'PASS' if pred_shape_ok else 'FAIL'}")
    print(f"  Predictions finite: {'PASS' if pred_finite else 'FAIL'}")

    # Direction accuracy with proper open-to-close returns
    sign_match = np.sign(Y_pred) == np.sign(Y_test)
    direction_acc = sign_match.mean()
    checks["direction_accuracy"] = float(direction_acc)
    print(f"  Direction accuracy: {direction_acc:.4f}")

    # Per-sector correlations
    correlations = []
    for j in range(Y_jp.shape[1]):
        if np.std(Y_test[:, j]) > 0 and np.std(Y_pred[:, j]) > 0:
            corr = np.corrcoef(Y_pred[:, j], Y_test[:, j])[0, 1]
            correlations.append(float(corr))
    mean_corr = float(np.mean(correlations)) if correlations else 0.0
    checks["mean_correlation"] = mean_corr
    print(f"  Mean pred-actual correlation: {mean_corr:.4f}")

    # --- Overall ---
    critical_checks = [
        checks["us_all_tickers_fetched"],
        checks["jp_all_tickers_fetched"],
        checks["us_close_to_close_correct"],
        checks["jp_open_to_close_correct"],
        checks["no_lookahead_bias"],
        checks["no_nan"],
        checks["no_inf"],
        checks["model_integration_shape"],
        checks["model_integration_finite"],
    ]
    all_passed = all(critical_checks)

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL CRITICAL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 60)

    metrics = {
        "phase": 2,
        "description": "Real data pipeline with proper return calculations and lead-lag alignment",
        "data": {
            "n_us_sectors": int(X_us.shape[1]),
            "n_jp_sectors": int(Y_jp.shape[1]),
            "n_aligned_pairs": int(X_us.shape[0]),
            "date_range_start": str(dataset.dates_us[0].date()),
            "date_range_end": str(dataset.dates_us[-1].date()),
            "us_return_type": "close-to-close",
            "jp_return_type": "open-to-close",
            "lag_days_avg": float(avg_lag),
            "lag_days_min": int(min_lag),
            "lag_days_max": int(max_lag),
        },
        "return_stats": {
            "us_mean": float(X_us.mean()),
            "us_std": float(X_us.std()),
            "jp_mean": float(Y_jp.mean()),
            "jp_std": float(Y_jp.std()),
        },
        "model_integration": {
            "direction_accuracy": float(direction_acc),
            "mean_correlation": mean_corr,
        },
        "validation": checks,
        "status": "PASS" if all_passed else "FAIL",
    }

    return metrics


if __name__ == "__main__":
    metrics = main()

    reports_dir = PROJECT_ROOT / "reports" / "cycle_3"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
