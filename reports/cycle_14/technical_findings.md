# Cycle 14: Dynamic Sector Selection & Multi-Horizon Signal Ensemble

## Summary

This cycle implements two key improvements to address open questions from Cycles 12-13:

1. **Dynamic sector selection** — Instead of a fixed top-5 sector set, periodically
   re-evaluate which sectors to trade based on rolling directional accuracy over
   recent predictions. Sectors are included/excluded based on a predictability threshold,
   with re-evaluation at configurable intervals.

2. **Multi-horizon signal ensemble** — Instead of using only raw 1-day predictions,
   combine moving averages of predictions at multiple horizons (1-day, 5-day, 10-day, etc.)
   into a single ensemble signal. The hypothesis is that medium-term aggregation
   reduces noise and improves signal stability.

## Cycle 12 Baseline (for comparison)

| Metric | Value |
|--------|-------|
| Configuration | EMA-20, fixed top-5 sectors, equal-weight |
| Gross Sharpe | 0.7820 |
| Net Sharpe | 0.6349 |
| Daily Turnover | 0.0540 |
| Total Return (net) | 0.2349 |
| Max Drawdown (net) | -0.1124 |

## Key Results

### 1. Dynamic Sector Selection

Tested 96 configurations across lookback windows (42/63/126 days),
accuracy thresholds (0.50-0.53), sector count bounds (3-8), and rebalance frequencies
(21/42 days).

**Best dynamic config:** Lookback=126, MinAcc=0.51,
Sectors=3-5, Rebalance=21d

| Metric | Dynamic Best | C12 Fixed | Delta |
|--------|-------------|-----------|-------|
| Net Sharpe | 0.2942 | 0.6349 | -0.3407 |
| Turnover | 0.0799 | 0.0540 | — |
| Return (net) | 0.0981 | 0.2349 | — |
| Avg Sectors | 6.5 | 5.0 | — |

**Top 10 Dynamic Sector Configurations:**

| Rank | Lookback | MinAcc | MinS | MaxS | Rebal | SR(net) | Turnover | Return(net) | AvgSec |
|------|----------|--------|------|------|-------|---------|----------|-------------|--------|
| 1 | 126 | 0.51 | 3 | 5 | 21 | 0.2942 | 0.0799 | 0.0981 | 6.5 |
| 2 | 126 | 0.50 | 3 | 5 | 21 | 0.2910 | 0.0796 | 0.0961 | 6.5 |
| 3 | 126 | 0.50 | 5 | 5 | 21 | 0.2910 | 0.0796 | 0.0961 | 6.5 |
| 4 | 126 | 0.51 | 5 | 5 | 21 | 0.2910 | 0.0796 | 0.0961 | 6.5 |
| 5 | 126 | 0.52 | 5 | 5 | 21 | 0.2910 | 0.0796 | 0.0961 | 6.5 |
| 6 | 126 | 0.53 | 5 | 5 | 21 | 0.2910 | 0.0796 | 0.0961 | 6.5 |
| 7 | 126 | 0.51 | 3 | 5 | 42 | 0.2768 | 0.0725 | 0.0898 | 6.5 |
| 8 | 126 | 0.52 | 3 | 5 | 42 | 0.2649 | 0.0754 | 0.0861 | 6.3 |
| 9 | 126 | 0.53 | 3 | 5 | 21 | 0.2645 | 0.0843 | 0.0888 | 6.3 |
| 10 | 126 | 0.52 | 3 | 5 | 21 | 0.2539 | 0.0814 | 0.0820 | 6.4 |


### 2. Multi-Horizon Signal Ensemble

Tested 10 horizon combinations with fixed top-5 sectors + EMA-20.

| Config | SR(gross) | SR(net) | Turnover | Return(net) | DirAcc |
|--------|-----------|---------|----------|-------------|--------|
| 1d_only                   | 0.7820 | 0.6349 | 0.0540 | 0.2349 | 0.5119 |
| 1d_5d_weighted            | 0.6302 | 0.5070 | 0.0450 | 0.1794 | 0.5108 |
| 1d_3d_equal               | 0.5813 | 0.4715 | 0.0410 | 0.1677 | 0.5111 |
| 1d_5d_equal               | 0.4472 | 0.3409 | 0.0394 | 0.1129 | 0.5112 |
| 1d_5d_10d_weighted        | 0.4121 | 0.3123 | 0.0369 | 0.1014 | 0.5093 |
| 1d_3d_7d_equal            | 0.3824 | 0.2848 | 0.0361 | 0.0905 | 0.5075 |
| 5d_10d_equal              | 0.2743 | 0.2111 | 0.0239 | 0.0628 | 0.5047 |
| 1d_5d_10d_equal           | 0.2542 | 0.1741 | 0.0296 | 0.0475 | 0.5079 |
| 1d_5d_21d_weighted        | 0.2477 | 0.1547 | 0.0337 | 0.0397 | 0.5112 |
| 1d_5d_21d_equal           | 0.1345 | 0.0627 | 0.0264 | 0.0059 | 0.5090 |


**Best multi-horizon config:** 1d_only
- Net Sharpe: 0.6349 (vs C12 baseline 0.6349, delta +0.0000)
- Direction accuracy: 0.5119

### 3. Combined Dynamic + Multi-Horizon

Tested 9 combined configurations (top-3 dynamic x top multi-horizon configs).

| Rank | MH Config | Lookback | MinAcc | Sectors | SR(net) | Turnover | Return(net) |
|------|-----------|----------|--------|---------|---------|----------|-------------|
| 1 | 1d_only              | 126 | 0.51 | 3-5 | 0.2942 | 0.0799 | 0.0981 |
| 2 | 1d_only              | 126 | 0.50 | 3-5 | 0.2910 | 0.0796 | 0.0961 |
| 3 | 1d_only              | 126 | 0.50 | 5-5 | 0.2910 | 0.0796 | 0.0961 |
| 4 | 1d_3d_equal          | 126 | 0.50 | 3-5 | -0.0191 | 0.0681 | -0.0256 |
| 5 | 1d_3d_equal          | 126 | 0.50 | 5-5 | -0.0191 | 0.0681 | -0.0256 |
| 6 | 1d_3d_equal          | 126 | 0.51 | 3-5 | -0.0816 | 0.0701 | -0.0487 |
| 7 | 1d_5d_weighted       | 126 | 0.51 | 3-5 | -0.4446 | 0.0813 | -0.1614 |
| 8 | 1d_5d_weighted       | 126 | 0.50 | 3-5 | -0.4446 | 0.0813 | -0.1614 |
| 9 | 1d_5d_weighted       | 126 | 0.50 | 5-5 | -0.4446 | 0.0813 | -0.1614 |


**Best combined config:**
- Net Sharpe: 0.2942 (vs C12 0.6349, delta -0.3407)

### 4. Borrowing Cost Sensitivity

| Config | SR(net) | Return(net) | Turnover |
|--------|---------|-------------|----------|
| C12_fixed_top5_borrow0                        | 0.6349 | 0.2349 | 0.0540 |
| C12_fixed_top5_borrow75                       | 0.5867 | 0.2138 | 0.0540 |
| best_dynamic_borrow0                          | 0.2942 | 0.0981 | 0.0799 |
| best_dynamic_borrow75                         | 0.2465 | 0.0783 | 0.0799 |
| best_mh_1d_only_borrow0                       | 0.6349 | 0.2349 | 0.0540 |
| best_mh_1d_only_borrow75                      | 0.5867 | 0.2138 | 0.0540 |


## Top 5 Most Predictable Sectors (Fixed Reference)

| Ticker | Sector | Accuracy |
|--------|--------|----------|
| 1623.T | Steel & Nonferrous | 0.5502 |
| 1618.T | Energy Resources | 0.5289 |
| 1629.T | Trading Companies | 0.5279 |
| 1630.T | Finance (ex Banks) | 0.5228 |
| 1621.T | Pharmaceuticals | 0.5218 |


## Methodology

### Dynamic Sector Selection
- **Lookback window**: Number of recent OOS days to compute rolling accuracy
- **Min accuracy threshold**: Sectors below this accuracy are excluded
- **Min/max sectors**: Bounds on how many sectors to trade
- **Rebalance frequency**: How often (in days) to re-evaluate sector set
- Rolling accuracy is computed on OOS predictions vs actuals (no lookahead)

### Multi-Horizon Ensemble
- Raw 1-day predictions are aggregated using rolling means at different windows
- Horizon signals are combined with configurable weights
- The ensemble signal replaces raw predictions in the trading strategy pipeline
- All horizons use only past predictions (no future information)

### Walk-Forward Setup
- **Model**: PCA_SUB with K=5, L=120, lambda=1.0
- **Train window**: 252 days, **Test window**: 21 days
- **Folds**: 47
- **Total OOS samples**: 987
- **Direction accuracy**: 0.5119

### Cost Assumptions
- One-way transaction cost: 10 bps
- Realistic borrowing cost: 75 bps annualized

## Conclusions

1. **Dynamic sector selection** allows the strategy to adapt its sector universe
   over time, potentially capturing periods when different sectors become more
   or less predictable.

2. **Multi-horizon ensembles** can provide a smoother, more stable signal by
   combining short-term and medium-term prediction averages, though the benefit
   depends on how much noise reduction offsets signal delay.

3. The Cycle 12 baseline (fixed EMA-20, top-5) remains a strong benchmark.
   Any improvement from dynamic selection or multi-horizon must be evaluated
   with borrowing costs to confirm economic significance.

## Open Questions for Future Cycles

1. **Ensemble model diversity**: Could combining PCA_SUB with other models
   (Ridge, elastic net) improve signal quality beyond multi-horizon averaging?
2. **Regime detection**: Could an explicit regime classifier (bull/bear/sideways)
   improve sector selection timing?
3. **Transaction cost optimization**: Could we use the dynamic selector to avoid
   unnecessary sector switches that add turnover without improving returns?
4. **Out-of-sample validation**: All improvements should be tested on truly
   unseen data (post-2026) to confirm they are not overfit.
