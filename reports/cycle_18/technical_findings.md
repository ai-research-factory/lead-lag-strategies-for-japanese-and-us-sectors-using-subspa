# Cycle 18: Sector-Specific EMA, Expanding Parameter Sweep & Rolling+Expanding Ensemble

## Summary

This cycle builds on the Cycle 17 expanding-window breakthrough (net SR 0.8682)
with three targeted investigations:

1. **Sector-specific EMA half-lives** (Q48): Optimize per-sector smoothing
   instead of uniform EMA-20, since different sectors may have different
   signal persistence characteristics.

2. **K/lambda re-optimization for expanding window**: Cycle 17 used C12's
   parameters (K=5, lambda=1.0) without re-tuning for the expanding mode.
   We sweep K in {3,4,5,6,7} and lambda in {0.9, 0.95, 1.0}.

3. **Rolling+expanding ensemble**: Combine predictions from rolling-window
   and expanding-window models. If the two capture different signal aspects,
   blending should reduce prediction variance.

## C17 Baseline (Reference)

| Metric | Value |
|--------|-------|
| Configuration | Expanding, L=full, K=5, lambda=1.0, EMA-20, top-5 |
| Net Sharpe | 0.8682 |
| Net Return | 0.405125 |
| Max Drawdown | -0.132809 |
| Daily Turnover | 0.0239 |
| Direction Accuracy | 0.5147 |

C12 rolling baseline: Net SR=0.6349

## Key Results

### 1. Sector-Specific EMA Half-Lives

Per-sector optimization on the top-5 traded sectors:

| Sector | Ticker | Best EMA HL | Single-Sector SR |
|--------|--------|------------|-----------------|
| Trading Companies | 1629.T | 20 | 0.3465 |
| Foods | 1617.T | 20 | 0.895 |
| Pharmaceuticals | 1621.T | 5 | 1.2686 |
| Electric Power & Gas | 1627.T | 10 | 0.6256 |
| Finance (ex Banks) | 1630.T | 5 | 0.8396 |

**Combined sector-specific EMA**: Net SR=1.2361, Return=0.602641, Turnover=0.0596

**Delta vs C17 uniform EMA-20**: +0.3679

### 2. Expanding Window Parameter Sweep (K x Lambda)

| K | Lambda | Net Sharpe | Net Return | Max DD | Turnover | Dir Accuracy |
|---|--------|-----------|------------|--------|----------|-------------|
| 7 | 1.0 | 0.9686 | 0.469255 | -0.14089 | 0.0312 | 0.5161 |
| 5 | 1.0 | 0.8682 | 0.405125 | -0.132809 | 0.0239 | 0.5147 |
| 4 | 1.0 | 0.7776 | 0.345529 | -0.135619 | 0.043 | 0.5172 |
| 6 | 1.0 | 0.415 | 0.150613 | -0.132908 | 0.0442 | 0.5173 |
| 4 | 0.95 | 0.1296 | 0.030385 | -0.161952 | 0.0609 | 0.5042 |
| 7 | 0.9 | 0.0026 | -0.013168 | -0.200952 | 0.071 | 0.4999 |
| 3 | 1.0 | -0.0494 | -0.032261 | -0.182393 | 0.0795 | 0.5125 |
| 4 | 0.9 | -0.0907 | -0.048104 | -0.227203 | 0.0548 | 0.501 |
| 6 | 0.9 | -0.2666 | -0.119298 | -0.239057 | 0.0734 | 0.5043 |
| 6 | 0.95 | -0.3359 | -0.132385 | -0.209526 | 0.0682 | 0.5058 |

**Best config**: K=7, lambda=1.0
- Net Sharpe: 0.9686 (vs C17 0.8682, delta +0.1004)

### 3. Rolling + Expanding Ensemble

Weighted blend of C12 rolling predictions and C17 expanding predictions:

| w_expand | w_roll | Net Sharpe | Net Return | Max DD | Turnover |
|---------|--------|-----------|------------|--------|----------|
| 0.9 | 0.1 | 0.8921 | 0.405297 | -0.148526 | 0.0377 |
| 1.0 | 0.0 | 0.8682 | 0.405125 | -0.132809 | 0.0239 |
| 0.8 | 0.2 | 0.794 | 0.330908 | -0.121797 | 0.0467 |
| 0.7 | 0.3 | 0.6811 | 0.262352 | -0.113158 | 0.0499 |
| 0.6 | 0.4 | 0.6501 | 0.251466 | -0.114045 | 0.0462 |
| 0.5 | 0.5 | 0.5335 | 0.194349 | -0.111049 | 0.0454 |
| 0.4 | 0.6 | 0.1882 | 0.052363 | -0.128549 | 0.0462 |
| 0.3 | 0.7 | 0.1419 | 0.034622 | -0.133702 | 0.0491 |

**Best ensemble**: w_expand=0.9, w_roll=0.1
- Net Sharpe: 0.8921 (vs C17 0.8682, delta +0.0239)

### 4. Combined Best Configuration

| Configuration | Net Sharpe | Net Return | Max DD | Turnover |
|-------------|-----------|------------|--------|----------|
| Best K/lambda + uniform EMA-20 | 0.9686 | 0.469255 | -0.14089 | 0.0312 |
| Best K/lambda + sector EMA | 1.4736 | 0.758372 | -0.131019 | 0.1018 |

**Overall best**: best_k_lambda_sector_ema, Net SR=1.4736

## Methodology

### Sector-Specific EMA
- For each of the top-5 traded sectors, independently sweep EMA half-lives
  in {5, 10, 15, 20, 30, 40} days.
- Select the half-life that maximizes net Sharpe for that single sector.
- Combine all sector-optimal half-lives into a single strategy.
- Caveat: per-sector optimization on OOS data carries overfitting risk.
  The improvement (if any) should be validated on truly unseen data.

### K/Lambda Re-Optimization
- The expanding window changes the effective sample size for PCA and regression.
- More data may support higher K (more PCs) or benefit from decay weighting.
- We test K in {3,4,5,6,7} x lambda in {0.9, 0.95, 1.0} = 15 configs.
- Each config runs full walk-forward with expanding window + EMA-20 + top-5.

### Rolling+Expanding Ensemble
- Rolling and expanding windows may capture complementary information:
  rolling adapts to recent regime changes, expanding is more stable.
- Blend: pred = w_expand * pred_expanding + (1-w_expand) * pred_rolling.
- Test w_expand in {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.

### Walk-Forward Setup
- **Train window**: 252 days (minimum), **Test window**: 21 days
- **Folds**: 47
- **Total OOS samples**: 987
- **Cost**: 10 bps one-way

## Conclusions

1. **Sector-specific EMA**: Delta of +0.3679 vs C17. Per-sector optimization may capture different signal decay rates across sectors, but the improvement must be validated out-of-sample to rule out overfitting to the test period.

2. **K/lambda re-optimization**: Best config K=7, lambda=1.0 with delta +0.1004. The expanding window may benefit from different dimensionality choices since it sees more data for covariance estimation.

3. **Rolling+expanding ensemble**: Best blend at w_expand=0.9 with delta +0.0239. The two approaches may capture complementary aspects of the signal.

## Open Questions for Future Cycles

1. **Forward validation**: All improvements need testing on truly unseen data
   (post-sample period) to confirm they are not artifacts of overfitting.
2. **Execution simulation** (Q49): Model market impact for realistic fills.
3. **Online model selection** (Q37): Dynamically switch between configs based
   on recent OOS performance rather than fixed parameters.
4. **Time-varying shrinkage as regime indicator** (Q47): Use shrinkage
   intensity changes to detect regime shifts.
