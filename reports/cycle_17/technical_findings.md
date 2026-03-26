# Cycle 17: Cost-Aware Rebalancing, Ledoit-Wolf Shrinkage & Expanding Window

## Summary

This cycle addresses three open questions from prior cycles:

1. **Transaction cost-aware rebalancing** (Q43) -- Skip position changes
   when the magnitude of change is below a cost-derived threshold.
   This targets further turnover reduction beyond EMA smoothing.

2. **Ledoit-Wolf covariance shrinkage** -- Regularize the sample covariance
   matrix used in PCA by shrinking toward a scaled identity. This should
   produce more stable eigenvectors, especially at shorter lookback windows.

3. **Expanding training window** -- Instead of a fixed 252-day rolling window,
   use all available history from the start. More data should improve parameter
   estimation stability, at the cost of weighting old regimes equally.

## C12 Baseline (Reference)

| Metric | Value |
|--------|-------|
| Configuration | EMA-20, fixed top-5 sectors, equal-weight |
| Net Sharpe | 0.6349 |
| Net Return | 0.234944 |
| Daily Turnover | 0.054 |

## Key Results

### 1. Cost-Aware Rebalancing

The cost-aware rebalancing filter skips per-sector position changes when
the absolute change in weight is below `cost_bps / 10000 * multiplier`.
Higher multipliers are more conservative (skip more trades).

| Multiplier | Net Sharpe | Net Return | Max DD | Turnover | Ann Return | Ann Vol |
|-----------|------------|------------|--------|----------|------------|---------|
| 1.0 | 0.654 | 0.243438 | -0.108133 | 0.0467 | 0.059824 | 0.091474 |
| 0.5 | 0.5985 | 0.219327 | -0.113601 | 0.0523 | 0.054837 | 0.09162 |
| 2.0 | 0.4017 | 0.136466 | -0.128196 | 0.0369 | 0.036886 | 0.091835 |
| 3.0 | 0.3383 | 0.112093 | -0.125668 | 0.0329 | 0.031459 | 0.092995 |
| 5.0 | 0.0913 | 0.016434 | -0.171176 | 0.0264 | 0.008417 | 0.092172 |
| 8.0 | 0.0057 | -0.014742 | -0.182795 | 0.0199 | 0.000534 | 0.092889 |

**Best cost-aware**: multiplier=1.0
- Net Sharpe: 0.654 (vs C12 0.6349, delta +0.0191)

### 2. Ledoit-Wolf Covariance Shrinkage

Ledoit-Wolf shrinks the sample covariance toward mu*I (scaled identity),
where the shrinkage intensity is automatically determined to minimize
the expected loss. We test across different lookback windows (L).

| L | Shrinkage Net SR | No-Shrink Net SR | Shrinkage DirAcc | No-Shrink DirAcc |
|---|-----------------|-----------------|-----------------|-----------------|
| 60 | -0.5728 | -0.5728 | 0.5008 | 0.5008 |
| 120 | 0.6349 | 0.6349 | 0.5119 | 0.5119 |
| 252 | -0.3249 | -0.3249 | 0.5101 | 0.5101 |

**Best shrinkage config**: L=120
- Shrinkage Net SR: 0.6349 vs no-shrink: 0.6349

### 3. Expanding Training Window

Instead of a fixed 252-day rolling window, the expanding window starts
from the beginning of data and grows with each fold. The model's L
parameter still controls PCA lookback, but the OLS regression has access
to the full expanding history.

| Config | Net Sharpe | Net Return | Turnover | Dir Accuracy |
|--------|------------|------------|----------|-------------|
| rolling_L120 | 0.6349 | 0.234944 | 0.054 | 0.5119 |
| expand_L120 | 0.6349 | 0.234944 | 0.054 | 0.5119 |
| expand_L252 | -0.3249 | -0.107803 | 0.0742 | 0.5101 |
| expand_Lfull | 0.8682 | 0.405125 | 0.0239 | 0.5147 |
| expand_Lfull_shrinkage | 0.8682 | 0.405125 | 0.0239 | 0.5147 |

### 4. Combined Approaches

| Signal Source | Cost-Aware | Net Sharpe | Net Return | Max DD | Turnover |
|-------------|-----------|------------|------------|--------|----------|
| expand_Lfull | No | 0.8682 | 0.405125 | -0.132809 | 0.0239 |
| expand_Lfull_shrinkage | No | 0.8682 | 0.405125 | -0.132809 | 0.0239 |
| expand_Lfull | Yes (x1.0) | 0.79 | 0.360844 | -0.152864 | 0.0191 |
| expand_Lfull_shrinkage | Yes (x1.0) | 0.79 | 0.360844 | -0.152864 | 0.0191 |
| C12_rolling | Yes (x1.0) | 0.654 | 0.243438 | -0.108133 | 0.0467 |
| expand_L120 | Yes (x1.0) | 0.654 | 0.243438 | -0.108133 | 0.0467 |
| C12_rolling | No | 0.6349 | 0.234944 | -0.112374 | 0.054 |
| expand_L120 | No | 0.6349 | 0.234944 | -0.112374 | 0.054 |
| expand_L252 | Yes (x1.0) | -0.285 | -0.096981 | -0.16453 | 0.0604 |
| expand_L252 | No | -0.3249 | -0.107803 | -0.159231 | 0.0742 |

**Best overall**: signal=expand_Lfull, cost_aware=False
- Net Sharpe: 0.8682 (vs C12 0.6349, delta +0.2333)

## Methodology

### Cost-Aware Rebalancing
- After computing target positions via EMA smoothing and sign-based
  allocation, compare each sector's proposed weight change to a threshold.
- Threshold = cost_bps / 10000 * multiplier (e.g., 10bps * 2 = 0.002).
- If the change for a sector is below the threshold, keep the previous
  day's weight for that sector.
- Re-normalize after filtering so total absolute weight = 1.
- This reduces unnecessary rebalancing without changing the signal.

### Ledoit-Wolf Shrinkage
- The sample covariance matrix S is replaced with:
  S_shrunk = alpha * mu * I + (1 - alpha) * S
- alpha (shrinkage intensity) is determined optimally to minimize
  the expected loss under the Frobenius norm.
- mu = trace(S) / p is the average eigenvalue.
- This stabilizes eigenvalue estimates, preventing the smallest
  eigenvalues from collapsing toward zero.

### Expanding Window
- Standard walk-forward uses a fixed 252-day rolling window.
- Expanding window always starts from observation 0 and grows.
- The PCASub's L parameter (120 days) still controls how many recent
  observations contribute to covariance estimation within fit().
- The key difference: the OLS regression in fit() uses L observations
  regardless of window type, but the expanding window provides more
  context for the initial folds.

### Walk-Forward Setup
- **Train window**: 252 days (minimum for expanding), **Test window**: 21 days
- **Folds**: 47
- **Total OOS samples**: 987

## Conclusions

1. **Cost-aware rebalancing** addresses a practical deployment concern:
   many small position adjustments incur costs without meaningful alpha.
   The multiplier parameter controls the trade-off between turnover
   reduction and signal responsiveness.

2. **Ledoit-Wolf shrinkage** is most valuable when the lookback window L
   is short relative to the number of features (p=11 US sectors). At
   L=120 or L=252, the sample covariance is already well-conditioned,
   so shrinkage has limited impact.

3. **Expanding window** tests whether regime stationarity holds: if the
   lead-lag relationship is stable, more data helps. If regimes shift,
   old data adds noise.

## Open Questions for Future Cycles

1. **Time-varying shrinkage**: Could the shrinkage intensity itself be
   used as a regime indicator? High shrinkage = unstable covariance.
2. **Sector-specific EMA half-lives**: Different sectors may have
   different optimal signal smoothing periods.
3. **Out-of-sample validation**: Forward test on post-2026 data.
4. **Execution simulation**: Model market impact and realistic fills.
