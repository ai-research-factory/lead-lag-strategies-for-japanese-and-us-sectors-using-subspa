# Phase 1: Core Algorithm (PCA_SUB) Implementation — Technical Findings

## Summary

Implemented the PCA_SUB (subspace regularization PCA) model for predicting Japanese sector ETF returns from U.S. sector ETF returns. The model was validated against real market data fetched from the ARF Data API.

## Implementation Details

### PCASub Class (`src/models/pca_sub.py`)

The model follows the paper's proposed algorithm:

1. **Exponential decay weighting**: Recent observations receive higher weight via decay factor `lambda_decay`. Weights are normalized to sum to 1.
2. **Weighted covariance estimation**: Computes the decay-weighted covariance matrix of U.S. sector returns over the lookback window `L`.
3. **PCA extraction**: Eigen-decomposition of the covariance matrix; top-`K` eigenvectors define the principal component subspace.
4. **Subspace projection**: U.S. returns are projected onto the K-dimensional PC subspace, producing factor scores.
5. **Weighted OLS regression**: Factor scores are regressed against Japanese returns using the same decay weights, yielding regression coefficients (beta) and intercept.

### Parameters Used
- `K = 3` (number of principal components)
- `L = 60` (lookback window)
- `lambda_decay = 0.9` (exponential decay rate)

## Data

- **U.S. sectors**: 11 Select Sector SPDR ETFs (XLE, XLF, XLU, XLI, XLK, XLV, XLY, XLP, XLB, XLRE, XLC)
- **Japanese sectors**: 17 TOPIX-17 sector ETFs (1617.T–1633.T)
- **Aligned samples**: 1,175 trading days with overlapping dates
- **Lead-lag alignment**: U.S. returns on day *t* predict Japanese returns on day *t+1*
- **Train/test split**: 200 / 50 samples (simple temporal split for validation only)

## Validation Results

| Check | Result |
|-------|--------|
| Output shapes correct | PASS |
| No NaN/Inf in predictions | PASS |
| Non-trivial predictions (std > 0) | PASS (std=8.00e-03) |
| Direction accuracy (sign match) | 52.7% |
| Mean prediction-actual correlation | 0.105 |

All acceptance criteria are met:
- `fit()` and `predict()` execute without errors on real data.
- `predict()` output shape is correct: (n_samples, 17).

## Observations

- The 52.7% direction accuracy and 0.105 mean correlation are modest but expected for a single-window fit without walk-forward optimization. The model captures a weak but non-zero signal.
- The current validation uses a simple train/test split; proper walk-forward evaluation is deferred to Phase 3.
- The covariance matrix is computed over `L=60` days with exponential decay, giving an effective lookback of roughly 10–15 days (since `0.9^60 ≈ 0.002`).

## Output Files

- `src/models/pca_sub.py` — PCASub model class
- `scripts/run_synthetic_test.py` — Validation script using ARF Data API
- `reports/cycle_1/metrics.json` — Structured validation metrics
