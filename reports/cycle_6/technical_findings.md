# Phase 6: Hyperparameter Optimization — Technical Findings

## Overview

Implemented nested walk-forward hyperparameter optimization to search for optimal
PCA_SUB parameters (K, L, λ) without test-set leakage. The nested scheme uses an
inner walk-forward loop within each outer training window to select the best
parameters, which are then evaluated on the outer test fold.

## Configuration

- **Parameter grid**: 125 combinations
  - K (principal components): [1, 2, 3, 4, 5]
  - L (lookback window): [20, 40, 60, 80, 120]
  - λ (decay rate): [0.8, 0.85, 0.9, 0.95, 1.0]
- **Outer loop**: train=504d (~2yr), test=21d (~1mo)
- **Inner loop**: train=252d (~1yr), test=21d (~1mo)
- **Selection metric**: Inner Sharpe ratio (gross)
- **Total outer folds**: 35
- **Total OOS test days**: 735

## Results: Optimized vs Baseline

| Metric | Optimized | Baseline (K=3, L=60, λ=0.9) |
|--------|-----------|------------------------------|
| Direction Accuracy | 0.5117 ± 0.0411 | 0.5028 ± 0.0376 |
| Sharpe Ratio (gross) | 1.3907 | 0.8644 |
| Ann. Return (gross) | 0.102876 | 0.065115 |
| Max Drawdown | -0.117368 | -0.061381 |
| Correlation (mean) | 0.0573 | 0.0197 |

## Parameter Selection Analysis

### Most Frequently Selected Parameters
- **K**: 5 (selected 17/35 folds)
- **L**: 120 (selected 29/35 folds)
- **λ**: 1.0 (selected 33/35 folds)

### Selection Counts
- K values: {2: 4, 3: 6, 4: 8, 5: 17}
- L values: {20: 1, 40: 5, 120: 29}
- λ values: {0.85: 1, 0.95: 1, 1.0: 33}

### Parameter Stability
- **Unique parameter combinations selected**: 8 out of 35 folds
- High stability: few unique combos relative to folds

## Parameter Sensitivity Analysis

Average inner-loop Sharpe by parameter value (averaged across all folds and other param settings):

### K (Number of Principal Components)
- K=1: inner Sharpe = 0.1161 ± 0.8685
- K=2: inner Sharpe = 0.3310 ± 0.8367
- K=3: inner Sharpe = 0.6085 ± 1.0588
- K=4: inner Sharpe = 0.7292 ± 1.3832
- K=5: inner Sharpe = 0.9779 ± 1.1996

### L (Lookback Window)
- L=20: inner Sharpe = 0.5021 ± 1.1431
- L=40: inner Sharpe = 0.5458 ± 1.0404
- L=60: inner Sharpe = 0.4824 ± 1.1199
- L=80: inner Sharpe = 0.5541 ± 1.1476
- L=120: inner Sharpe = 0.6783 ± 1.1827

### λ (Decay Rate)
- λ=0.8: inner Sharpe = 0.2034 ± 0.9836
- λ=0.85: inner Sharpe = 0.2976 ± 1.1023
- λ=0.9: inner Sharpe = 0.4710 ± 1.0485
- λ=0.95: inner Sharpe = 0.8777 ± 1.0478
- λ=1.0: inner Sharpe = 0.9129 ± 1.2544

## Key Observations

1. **Optimization impact**: The nested walk-forward optimization improved the gross Sharpe ratio compared to the default parameters.

2. **Parameter stability**: 8 unique parameter combinations were selected across 35 folds, indicating that optimal parameters are relatively stable over time.

3. **Overfitting risk**: The nested walk-forward design guards against overfitting by never using test data for parameter selection. The inner-loop Sharpe is used purely for ranking, and OOS performance is measured independently.

4. **Default parameters robustness**: The paper's default parameters (K=3, L=60, λ=0.9) provide a suboptimal baseline, suggesting room for improvement through optimization.

## Implications for Strategy Design

- Dynamic parameter selection (re-optimizing periodically) may offer marginal improvement over fixed parameters for this dataset.
- The parameter sensitivity analysis can guide which parameters are most worth tuning vs. fixing.
- All results are gross of transaction costs — net performance evaluation is needed in Phase 4.
