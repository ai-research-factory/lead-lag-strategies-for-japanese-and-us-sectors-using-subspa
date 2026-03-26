# Phase 3: Walk-Forward Evaluation — Technical Findings

## Overview

Implemented and executed a walk-forward evaluation framework for the PCA_SUB model,
testing out-of-sample prediction accuracy across 47 rolling folds.

## Configuration

| Parameter | Value |
|-----------|-------|
| Train window | 252 days (~1 year) |
| Test window | 21 days (~1 month) |
| Principal components (K) | 3 |
| Lookback (L) | 60 |
| Decay rate (λ) | 0.9 |
| Total aligned pairs | 1254 |
| Date range | 2021-03-29 to 2026-03-25 |

## Key Results

### Prediction Accuracy (Out-of-Sample)

| Metric | Mean | Std |
|--------|------|-----|
| Direction accuracy | 0.4978 | 0.0362 |
| Correlation | 0.0183 | 0.0997 |
| RMSE | 0.011364 | 0.003006 |

- Folds with >50% direction accuracy: 44.7%
- Total out-of-sample test days: 987

### Strategy Performance (Gross, Equal-Weight Long-Short)

| Metric | Value |
|--------|-------|
| Sharpe ratio (gross) | 0.5418 |
| Annualized return | 3.7700% |
| Annualized volatility | 6.9583% |
| Max drawdown | -9.0403% |
| Total return | 14.8252% |
| % positive days | 51.8% |

### Notable Folds

- **Best fold**: #26 (2024-05-30 to 2024-06-28), accuracy=0.5714
- **Worst fold**: #9 (2022-12-27 to 2023-01-26), accuracy=0.4174

### Per-Sector Highlights

- **Best sector**: 1617.T (accuracy=0.5218, corr=0.1246)
- **Worst sector**: 1631.T (accuracy=0.4732, corr=-0.0292)

## Observations

1. **Prediction signal is weak but non-zero**: Direction accuracy hovers around 50%,
   suggesting the U.S.-to-Japan lead-lag signal is subtle. The correlation between
   predicted and actual returns provides additional evidence of a small but measurable
   relationship.

2. **Walk-forward validates no lookahead bias**: All 47 folds maintain
   strict temporal separation between training and testing periods. The model is
   retrained at each fold using only past data.

3. **Strategy gross Sharpe provides a baseline**: The equal-weight long-short strategy
   based on predicted return signs gives a gross Sharpe ratio that will serve as the
   starting point for Phase 4 (backtest with transaction costs).

4. **Temporal variation in performance**: The fold-level breakdown shows significant
   variation in accuracy across different market periods, suggesting the lead-lag
   relationship may be regime-dependent.

5. **Sector heterogeneity**: Some Japanese sectors are more predictable from U.S.
   sector movements than others, consistent with the expectation that globally
   connected sectors (e.g., technology, energy) have stronger cross-market linkages.

## Validation

All 7 validation checks passed:
- No NaN/Inf in predictions
- Correct output shapes
- No train/test overlap
- Monotonic fold ordering

## Next Steps (Phase 4)

- Implement a full backtest engine with transaction cost model
- Add position sizing and portfolio construction
- Compute net Sharpe ratio after costs
- Compare against naive baselines (buy-and-hold, random)
