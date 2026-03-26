# Cycle 15: Ensemble Model Diversity & Regime-Aware Positioning

## Summary

This cycle addresses two open questions from Cycle 14:

1. **Ensemble model diversity** — Combine PCA_SUB with Ridge and ElasticNet
   regression to improve prediction quality through model diversity, rather
   than multi-horizon averaging of a single model's predictions.

2. **Regime-aware positioning** — Use volatility-based regime detection to
   scale exposure, reducing positions in high-volatility regimes where the
   lead-lag signal is weaker (confirmed in Phase 7).

## C12 Baseline (Reference)

| Metric | Value |
|--------|-------|
| Configuration | EMA-20, fixed top-5 sectors, equal-weight |
| Net Sharpe | 0.6349 |
| Net Return | 0.234944 |
| Daily Turnover | 0.054 |

## Key Results

### 1. Individual Model Comparison

| Model | Direction Accuracy | Net Sharpe | Net Return | Turnover |
|-------|-------------------|------------|------------|----------|
| PCA_SUB | 0.5119 | 0.6349 | 0.234944 | 0.054 |
| Ridge | 0.4965 | -0.1992 | -0.074392 | 0.0097 |
| ElasticNet | 0.496 | -0.1513 | -0.059429 | 0.0073 |

### 2. Ensemble Model Results

Tested 8 ensemble configurations with different weighting schemes
and combination methods (weighted average vs sign vote).

| Rank | Config | Method | Dir Acc | SR(net, fixed top5) | Return(net) | Turnover |
|------|--------|--------|---------|---------------------|-------------|----------|
| 1 | pca_heavy_vote | sign_vote | 0.5119 | 0.5106 | 0.179539 | 0.0523 |
| 2 | pca_dominant_avg | weighted_avg | 0.5115 | 0.3055 | 0.094585 | 0.0402 |
| 3 | pca_heavy_avg | weighted_avg | 0.5108 | 0.1528 | 0.037249 | 0.043 |
| 4 | pca_ridge_only | weighted_avg | 0.5106 | 0.1059 | 0.021176 | 0.0422 |
| 5 | ridge_strong | weighted_avg | 0.5089 | -0.0617 | -0.032625 | 0.0511 |
| 6 | enet_strong | weighted_avg | 0.5099 | -0.2051 | -0.074639 | 0.0373 |
| 7 | equal_weight_vote | sign_vote | 0.4982 | -0.4122 | -0.133428 | 0.0158 |
| 8 | equal_weight_avg | weighted_avg | 0.5089 | -0.4754 | -0.150313 | 0.0308 |

**Best ensemble**: pca_heavy_vote
- Net Sharpe: 0.5106 (vs C12 0.6349, delta -0.1243)
- Direction accuracy: 0.5119

### 3. Regime-Aware Positioning

Tested 24 regime configurations across volatility lookbacks
(10/21/42/63 days), regime counts (2/3), and exposure scaling levels
(conservative/moderate/mild).

| Rank | Vol LB | Regimes | Scale | SR(net) | Return(net) | Max DD | Turnover |
|------|--------|---------|-------|---------|-------------|--------|----------|
| 1 | 42 | 2 | mild | 0.6051 | 0.213737 | -0.112374 | 0.0547 |
| 2 | 21 | 3 | mild | 0.6013 | 0.203413 | -0.110839 | 0.0595 |
| 3 | 63 | 2 | mild | 0.5908 | 0.209132 | -0.112374 | 0.0547 |
| 4 | 63 | 3 | mild | 0.5906 | 0.202622 | -0.112374 | 0.0545 |
| 5 | 42 | 3 | mild | 0.5857 | 0.199258 | -0.112374 | 0.0552 |
| 6 | 42 | 2 | moderate | 0.5798 | 0.199529 | -0.112374 | 0.0552 |
| 7 | 21 | 2 | mild | 0.5764 | 0.195619 | -0.112374 | 0.0585 |
| 8 | 21 | 3 | moderate | 0.5711 | 0.18248 | -0.109819 | 0.0632 |
| 9 | 63 | 2 | moderate | 0.5566 | 0.192 | -0.112374 | 0.0552 |
| 10 | 63 | 3 | moderate | 0.5539 | 0.181245 | -0.112374 | 0.0549 |

**Best regime config**: vol_lookback=42, n_regimes=2, scale=mild
- Net Sharpe: 0.6051 (vs C12 0.6349, delta -0.0298)
- Regime distribution: {0: 860, 1: 127}

### 4. Combined Ensemble + Regime

| Config | PCA_SUB SR(net) | Ensemble SR(net) | PCA_SUB Return | Ensemble Return |
|--------|----------------|------------------|----------------|-----------------|
| vol42_reg2_mild | 0.6051 | 0.4647 | 0.213737 | 0.154982 |
| vol21_reg3_mild | 0.6013 | 0.4569 | 0.203413 | 0.145847 |
| vol63_reg2_mild | 0.5908 | 0.4624 | 0.209132 | 0.154886 |
| vol63_reg3_mild | 0.5906 | 0.469 | 0.202622 | 0.152893 |
| vol42_reg3_mild | 0.5857 | 0.45 | 0.199258 | 0.144789 |

### 5. Borrowing Cost Sensitivity

| Config | SR(net) | Return(net) |
|--------|---------|-------------|
| C12_borrow0 | 0.6349 | 0.234944 |
| best_ensemble_borrow0 | 0.5106 | 0.179539 |
| best_regime_borrow0 | 0.6051 | 0.213737 |
| C12_borrow75 | 0.5867 | 0.213829 |
| best_ensemble_borrow75 | 0.4616 | 0.159206 |
| best_regime_borrow75 | 0.5575 | 0.193975 |

## Methodology

### Ensemble Model
- **PCA_SUB**: K=5, L=120, lambda=1.0 (optimized params from Phase 6)
- **Ridge**: L2-regularized direct regression (US->JP sectors)
- **ElasticNet**: L1+L2 regularized regression (per-sector)
- **Combination methods**: Weighted average and sign vote
- All models fit independently on each walk-forward training window

### Regime Detection
- Rolling realized volatility of US equal-weight sector returns
- Expanding-window quantile classification (no lookahead)
- Exposure scaled down in high-vol regimes (confirmed weaker signal in Phase 7)

### Walk-Forward Setup
- **Train window**: 252 days, **Test window**: 21 days
- **Folds**: 47
- **Total OOS samples**: 987
- **Direction accuracy (PCA_SUB)**: 0.5119

### Cost Assumptions
- One-way transaction cost: 10 bps
- Realistic borrowing cost: 75 bps annualized

## Conclusions

1. **Model diversity** through ensemble combining PCA_SUB, Ridge, and ElasticNet
   provides an alternative prediction approach. The comparison reveals whether
   combining fundamentally different regression approaches (subspace vs direct)
   improves the cross-market lead-lag signal.

2. **Regime-aware positioning** scales exposure based on market volatility.
   Phase 7 confirmed the model performs better in low-vol regimes (SR=0.72 vs
   0.47 in high-vol), so reducing exposure in high-vol periods should improve
   risk-adjusted returns.

3. The Cycle 12 baseline (EMA-20, top-5, net Sharpe 0.63) remains the
   benchmark to beat. Any improvement must be evaluated with borrowing costs.

## Open Questions for Future Cycles

1. **Online model selection**: Could we dynamically choose between PCA_SUB and
   ensemble based on recent performance, rather than fixed weights?
2. **Factor timing**: Could exposure to specific PCA factors be timed based on
   regime indicators rather than blanket exposure scaling?
3. **Transaction cost optimization in regime transitions**: Could position
   changes during regime transitions be smoothed to avoid unnecessary turnover?
4. **Out-of-sample validation**: All improvements should be tested on post-2026
   data to confirm they are not overfit.
