# Cycle 16: Factor Timing, Risk-Parity & Long-Bias Strategy

## Summary

This cycle addresses three open questions from prior cycles:

1. **Factor timing** (Q38) — Weight individual PCA factors by their recent
   predictive accuracy instead of treating all K=5 components equally.
   Different PCs may capture different economic dynamics (market vs rotation
   vs sector-specific), and their predictive power may vary over time.

2. **Risk-parity position sizing** — Replace equal-weight positions with
   inverse-volatility weighting. Lower-volatility sectors receive larger
   allocations, balancing risk contributions across the portfolio.

3. **Long-biased strategy** — Phase 7 showed the lead-lag signal has strong
   directional asymmetry (up-market SR=1.31 vs down-market SR=-0.18). A
   strategy that reduces short exposure may improve risk-adjusted returns
   while also reducing borrowing costs.

## C12 Baseline (Reference)

| Metric | Value |
|--------|-------|
| Configuration | EMA-20, fixed top-5 sectors, equal-weight |
| Net Sharpe | 0.6349 |
| Net Return | 0.234944 |
| Daily Turnover | 0.054 |

## Key Results

### 1. Factor Timing

Per-PC weight adaptation tracks which principal components are currently
predictive and upweights them dynamically. After each walk-forward fold,
per-factor directional accuracy is measured and weights are updated via
softmax scaling.

| Factor EMA HL | Dir Accuracy | Net Sharpe | Net Return | Max DD | Turnover | Final PC Weights |
|--------------|-------------|------------|------------|--------|----------|-----------------|
| 21 | 0.5028 | -0.1819 | -0.073838 | -0.186298 | 0.0183 | [0.188, 0.234, 0.207, 0.143, 0.228] |
| 42 | 0.5028 | -0.1819 | -0.073838 | -0.186298 | 0.0183 | [0.188, 0.234, 0.207, 0.143, 0.228] |
| 63 | 0.5028 | -0.1819 | -0.073838 | -0.186298 | 0.0183 | [0.188, 0.234, 0.207, 0.143, 0.228] |
| 126 | 0.5028 | -0.1819 | -0.073838 | -0.186298 | 0.0183 | [0.188, 0.234, 0.207, 0.143, 0.228] |

**Best factor timing**: EMA half-life=21
- Net Sharpe: -0.1819 (vs C12 0.6349, delta -0.8168)

### 2. Risk-Parity Position Sizing

Inverse-volatility weighting sizes positions so that lower-volatility
sectors receive larger allocations, balancing risk contributions.

| RP Lookback | Net Sharpe | Net Return | Max DD | Turnover | Ann Return | Ann Vol |
|------------|------------|------------|--------|----------|------------|---------|
| 126 | 0.5668 | 0.193507 | -0.117842 | 0.058 | 0.048893 | 0.086265 |
| 63 | 0.5041 | 0.168006 | -0.121171 | 0.0635 | 0.043356 | 0.086011 |
| 42 | 0.4901 | 0.162132 | -0.118636 | 0.0688 | 0.042051 | 0.085802 |
| 21 | 0.4366 | 0.142003 | -0.121359 | 0.0857 | 0.03762 | 0.086161 |

**Best risk-parity**: lookback=126
- Net Sharpe: 0.5668 (vs C12 0.6349, delta -0.0681)

### 3. Long-Biased Strategy

Phase 7 confirmed the lead-lag signal is asymmetric: stronger for long
positions. We test scaling down short exposure by a long_bias factor.
long_bias=0 is standard long-short; long_bias=1.0 is long-only.

| Long Bias | Net Sharpe | Net Return | Max DD | Turnover | % Positive Days |
|-----------|------------|------------|--------|----------|----------------|
| 0.0 | 0.6349 | 0.234944 | -0.112374 | 0.054 | 53.7 |
| 0.25 | 0.5224 | 0.192306 | -0.119255 | 0.0607 | 52.68 |
| 0.5 | 0.3567 | 0.128067 | -0.13819 | 0.0703 | 52.38 |
| 0.75 | 0.0985 | 0.018717 | -0.165914 | 0.0859 | 50.86 |
| 1.0 | -0.3637 | -0.220582 | -0.256525 | 0.097 | 41.74 |

**Best long bias**: 0.0
- Net Sharpe: 0.6349 (vs C12 0.6349, delta +0.0000)

### 4. Combined Approaches

| Config | Net Sharpe | Net Return | Max DD | Turnover |
|--------|------------|------------|--------|----------|
| RP(lb=126)+LongBias(0.0) | 0.5668 | 0.193507 | -0.117842 | 0.058 |
| RP(lb=126)+LongBias(0.25) | 0.4482 | 0.150212 | -0.116796 | 0.0641 |
| RP(lb=126)+LongBias(0.5) | 0.2686 | 0.084677 | -0.133658 | 0.0736 |
| RP(lb=126)+LongBias(0.75) | -0.0092 | -0.025415 | -0.171272 | 0.0892 |
| FT(hl=21)+RP(lb=126) | -0.1451 | -0.058525 | -0.175793 | 0.0246 |
| FT(hl=21) | -0.1819 | -0.073838 | -0.186298 | 0.0183 |
| FT(hl=21)+LB(0.5) | -0.4106 | -0.163956 | -0.234295 | 0.0236 |
| FT(hl=21)+RP(lb=126)+LB(0.5) | -0.4233 | -0.15597 | -0.229548 | 0.03 |

**Best combined**: RP(lb=126)+LongBias(0.0)
- Net Sharpe: 0.5668 (vs C12 0.6349, delta -0.0681)

### 5. Borrowing Cost Sensitivity

| Config | SR(net) | Return(net) |
|--------|---------|-------------|
| C12_borrow0 | 0.6349 | 0.234944 |
| best_RP_borrow0 | 0.5668 | 0.193507 |
| best_LB(0.0)_borrow0 | 0.6349 | 0.234944 |
| C12_borrow75 | 0.5867 | 0.213829 |
| best_RP_borrow75 | 0.5139 | 0.172369 |
| best_LB(0.0)_borrow75 | 0.5867 | 0.213829 |

## Methodology

### Factor Timing
- PCA_SUB decomposes predictions into per-PC contributions: each PC's
  score times its regression coefficient.
- After each walk-forward fold, per-PC directional accuracy is measured.
- Factor weights are updated via softmax of excess accuracy (accuracy - 0.5),
  so PCs near 50% accuracy receive low weight.
- Weights carry over between folds, providing momentum in factor selection.

### Risk-Parity
- For each day, compute each sector's realized volatility over a lookback window.
- Position sizes are proportional to 1/volatility (inverse-vol weighting).
- Positions are then normalized to sum of absolute weights = 1.
- This balances risk contributions across sectors rather than capital allocation.

### Long Bias
- Short positions are scaled by (1 - long_bias).
- long_bias=0.0: standard long-short (50% long, 50% short).
- long_bias=0.5: partial long bias (shorts halved).
- long_bias=1.0: long-only (all shorts eliminated).
- Reduces borrowing costs proportionally to reduced short exposure.

### Walk-Forward Setup
- **Train window**: 252 days, **Test window**: 21 days
- **Folds**: 47
- **Total OOS samples**: 987

## Conclusions

1. **Factor timing** reveals whether the 5 PCA components have heterogeneous
   and time-varying predictive power, addressing the hypothesis that PC1 (market)
   may dominate while sector-rotation factors (PC2-5) are noisier.

2. **Risk-parity** tests whether balancing risk contributions across the top-5
   sectors — rather than equal capital allocation — improves the Sharpe ratio.
   Steel & Nonferrous (highest accuracy) also tends to be high-volatility,
   so equal-weight may over-allocate risk to this sector.

3. **Long bias** exploits the known signal asymmetry from Phase 7. If the
   lead-lag signal primarily predicts positive returns (rather than negative),
   reducing short exposure improves returns and reduces borrowing costs.

## Open Questions for Future Cycles

1. **Sector-specific long bias**: Could each sector have a different optimal
   long/short asymmetry based on its individual signal characteristics?
2. **Factor timing with regime conditioning**: Could factor weights be
   conditioned on volatility regimes rather than just recent accuracy?
3. **Transaction cost-aware rebalancing**: Could the strategy skip small
   position changes when the expected alpha is below the trading cost?
4. **Out-of-sample validation**: All improvements need forward validation
   on post-2026 data to confirm robustness.
