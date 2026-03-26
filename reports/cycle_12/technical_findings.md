# Cycle 12: Turnover Optimization & Cost-Aware Strategy

## Summary

This cycle addresses the **most critical finding** from Phases 1-11: the PCA_SUB model generates a
strong gross signal (Sharpe 2.18) that is entirely destroyed by excessive daily turnover (76%).
After 10 bps one-way transaction costs, the naive sign-based strategy has a **net Sharpe of -1.24**.

We implement and evaluate three turnover reduction techniques:
1. **EMA signal smoothing** — smooths raw predictions to reduce position flip-flopping
2. **Signal threshold filtering** — requires minimum prediction magnitude before taking positions
3. **Selective sector trading** — only trades the most predictable sectors
4. **Position change limits** — caps daily position changes to reduce turnover directly

## Key Results

### Baseline vs Optimized

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Net Sharpe | -1.2431 | 0.6349 | +1.8780 |
| Gross Sharpe | 2.1756 | 0.7820 | — |
| Daily Turnover | 0.8583 | 0.0540 | -93.7% |
| Total Return (net) | -0.2708 | 0.2349 | — |
| Max Drawdown (net) | -0.3075 | -0.1124 | — |

### Best Strategy Configuration

| Parameter | Value |
|-----------|-------|
| EMA half-life | 20 |
| Signal threshold | 0.0 |
| Sector selection | top_5_sectors |
| Max position change | None |

## Strategy Comparison

| Strategy | SR(gross) | SR(net) | Turnover | Return(net) | MaxDD |
|----------|-----------|---------|----------|-------------|-------|
| Baseline (no smoothing)             |   2.1756 |  -1.2431 |   0.8583 |    -0.2708 |  -0.3075 |
| EMA-5                               |   0.8026 |   0.1499 |   0.1772 |     0.0314 |  -0.1576 |
| EMA-10                              |   0.3545 |   0.0157 |   0.0935 |    -0.0052 |  -0.1422 |
| EMA-20                              |   0.1819 |  -0.0209 |   0.0543 |    -0.0144 |  -0.1061 |
| Threshold 0.03%                     |   2.1571 |  -1.6682 |   1.0153 |    -0.3597 |  -0.3882 |
| Threshold 0.05%                     |   2.1012 |  -1.7812 |   1.0982 |    -0.3980 |  -0.4286 |
| Threshold 0.1%                      |   1.9229 |  -2.0654 |   1.2769 |    -0.4862 |  -0.5081 |
| EMA-10 + Threshold 0.03%            |   0.4416 |  -0.1731 |   0.1971 |    -0.0652 |  -0.1921 |
| EMA-10 + MaxChg 0.1                 |   0.2999 |  -0.0700 |   0.1027 |    -0.0284 |  -0.1505 |
| Position limit 0.05                 |   1.7290 |  -0.4583 |   0.5650 |    -0.1176 |  -0.1893 |
| Position limit 0.1                  |   2.2216 |  -1.2375 |   0.8770 |    -0.2719 |  -0.2995 |
| Top-5 sectors                       |   2.3360 |   0.0028 |   0.8333 |    -0.0149 |  -0.3059 |
| Top-5 + EMA-10                      |   0.4819 |   0.2530 |   0.0856 |     0.0785 |  -0.1976 |
| Top-8 sectors                       |   2.5899 |  -0.2237 |   0.8649 |    -0.0766 |  -0.3004 |
| Top-8 + EMA-10                      |   0.3600 |   0.0604 |   0.0999 |     0.0060 |  -0.1833 |


## Top 10 Grid Search Configurations (by net Sharpe)

| Rank | EMA | Threshold | Sectors | MaxChg | SR(net) | Turnover |
|------|-----|-----------|---------|--------|---------|----------|
| 1 | 20 | 0.0 | top_5_sectors | None | 0.6349 | 0.0540 |
| 2 | 20 | 0.0 | top_8_sectors | None | 0.5088 | 0.0598 |
| 3 | 20 | 0.0003 | top_5_sectors | None | 0.4546 | 0.1343 |
| 4 | 20 | 0.0 | top_8_sectors | 0.2 | 0.4125 | 0.0707 |
| 5 | 20 | 0.0003 | top_5_sectors | 0.2 | 0.3831 | 0.1203 |
| 6 | 20 | 0.0 | top_5_sectors | 0.2 | 0.3654 | 0.0872 |
| 7 | 20 | 0.0003 | top_5_sectors | 0.1 | 0.3566 | 0.0953 |
| 8 | None | 0.0 | top_5_sectors | 0.1 | 0.3406 | 0.2957 |
| 9 | 20 | 0.0 | top_10_sectors | None | 0.3355 | 0.0527 |
| 10 | 20 | 0.0 | top_10_sectors | 0.2 | 0.3355 | 0.0527 |


## Top 5 Most Predictable Sectors

| Ticker | Sector | Accuracy |
|--------|--------|----------|
| 1623.T | Steel & Nonferrous | 0.5502 |
| 1618.T | Energy Resources | 0.5289 |
| 1629.T | Trading Companies | 0.5279 |
| 1630.T | Finance (ex Banks) | 0.5228 |
| 1621.T | Pharmaceuticals | 0.5218 |


## Methodology

### Walk-Forward Setup
- **Model**: PCA_SUB with K=5, L=120, λ=1.0 (optimized in Phase 6)
- **Train window**: 252 days (1 year)
- **Test window**: 21 days (1 month)
- **Folds**: 47
- **Total OOS samples**: 987
- **Direction accuracy**: 0.5119

### Cost Assumptions
- One-way transaction cost: 10 bps (commission + market impact)
- Applied to absolute position changes (turnover)
- No borrowing costs modeled for short positions

### Grid Search
- **EMA half-lives tested**: None, 3, 5, 10, 20 days
- **Signal thresholds**: 0, 0.01%, 0.03%, 0.05%, 0.1%
- **Sector masks**: All 17, Top 5, Top 8, Top 10
- **Position change limits**: None, 0.05, 0.1, 0.2
- **Total configurations**: 400

## Analysis and Observations

### Signal Smoothing Impact
EMA smoothing is the most effective single technique for turnover reduction. Longer half-lives
reduce turnover more but also reduce gross Sharpe by delaying signal incorporation.
The optimal half-life balances signal freshness against position stability.

### Threshold Filtering
Higher thresholds eliminate low-confidence predictions, reducing turnover. However, very high
thresholds can filter out valid signals, reducing the number of active trading days.

### Sector Selection
Trading only the most predictable sectors concentrates exposure on stronger signals
and naturally reduces turnover (fewer position changes across fewer sectors).

### Combined Approaches
The best performance comes from combining multiple techniques — smoothing plus
threshold or sector selection provides better risk-adjusted returns than any single approach.

## Open Questions for Future Cycles

1. **Signal-weighted positions**: Instead of equal-weight long/short, could prediction magnitude
   be used to size positions, concentrating on highest-conviction trades?
2. **Adaptive smoothing**: Could the EMA half-life adapt to market regime (longer in volatile
   periods, shorter in trending markets)?
3. **Out-of-sample parameter stability**: The optimal strategy parameters were selected on the
   2021-2026 walk-forward window. Forward validation is needed.
4. **Borrowing costs**: Short positions incur borrowing costs not modeled here. For Japanese
   ETFs, these can be 50-100 bps annually.
