# Cycle 13: Advanced Strategy Enhancements

## Summary

This cycle implements three key improvements to the cost-aware trading strategy from Cycle 12:

1. **Signal-weighted position sizing** — Allocates capital proportional to prediction
   magnitude instead of equal-weight long/short. Higher conviction trades receive
   larger positions.
2. **Regime-adaptive EMA smoothing** — Dynamically adjusts the EMA half-life based on
   realized market volatility. In high-volatility regimes, smoothing increases to
   avoid whipsaws; in low-volatility regimes, smoothing decreases for faster signals.
3. **Borrowing cost modeling** — Adds annualized short-selling costs (50-100 bps for
   Japanese ETFs) to the net return calculation, providing a more realistic P&L.

## Cycle 12 Baseline (for comparison)

| Metric | Value |
|--------|-------|
| Configuration | EMA-20, top-5 sectors, equal-weight |
| Gross Sharpe | 0.7820 |
| Net Sharpe | 0.6349 |
| Daily Turnover | 0.0540 |
| Total Return (net) | 0.2349 |
| Max Drawdown (net) | -0.1124 |

## Key Results

### 1. Signal-Weighted vs Equal-Weight Positions

| Config | SR(gross) | SR(net) | Turnover | Return(net) |
|--------|-----------|---------|----------|-------------|
| EMA-10_all_ew                  |   0.3545 |   0.0157 |   0.0935 |    -0.0052 |
| EMA-10_all_sw                  |   0.3000 |  -0.1623 |   0.1649 |    -0.0704 |
| EMA-10_top_5_ew                |   0.4819 |   0.2530 |   0.0856 |     0.0785 |
| EMA-10_top_5_sw                |   0.2868 |  -0.0768 |   0.1642 |    -0.0578 |
| EMA-20_all_ew                  |   0.1819 |  -0.0209 |   0.0543 |    -0.0144 |
| EMA-20_all_sw                  |   0.0608 |  -0.2184 |   0.0985 |    -0.0874 |
| EMA-20_top_5_ew                |   0.7820 |   0.6349 |   0.0540 |     0.2349 |
| EMA-20_top_5_sw                |   0.2821 |   0.0516 |   0.1014 |    -0.0017 |
| EMA-30_all_ew                  |   0.1372 |  -0.0175 |   0.0401 |    -0.0128 |
| EMA-30_all_sw                  |   0.0096 |  -0.2022 |   0.0735 |    -0.0808 |
| EMA-30_top_5_ew                |   0.1544 |   0.0495 |   0.0361 |     0.0021 |
| EMA-30_top_5_sw                |   0.1983 |   0.0215 |   0.0760 |    -0.0138 |


**Finding**: With EMA-20 and top-5 sectors, signal-weighted achieves net Sharpe
0.0516 vs equal-weight 0.6349. Signal-weighted turnover is 0.1014
vs equal-weight 0.054.

### 2. Regime-Adaptive EMA (Top 10 Configurations)

| Config | SR(net) | Turnover | Return(net) |
|--------|---------|----------|-------------|
| AdaEMA-20_vw10_top_5_ew                       |   0.6836 |   0.0613 |     0.2748 |
| AdaEMA-20_vw42_top_5_ew                       |   0.6307 |   0.0540 |     0.2338 |
| AdaEMA-20_vw21_top_5_ew                       |   0.6022 |   0.0548 |     0.2257 |
| AdaEMA-10_vw21_top_5_ew                       |   0.5888 |   0.0913 |     0.2303 |
| AdaEMA-10_vw10_top_5_ew                       |   0.4262 |   0.1002 |     0.1565 |
| AdaEMA-10_vw42_top_5_ew                       |   0.3385 |   0.0888 |     0.1140 |
| AdaEMA-30_vw10_top_5_ew                       |   0.2333 |   0.0483 |     0.0710 |
| AdaEMA-20_vw21_top_5_sw                       |   0.1706 |   0.1034 |     0.0517 |
| AdaEMA-30_vw21_top_5_sw                       |   0.1359 |   0.0761 |     0.0355 |
| AdaEMA-20_vw42_top_5_sw                       |   0.1260 |   0.1001 |     0.0311 |


**Finding**: Adaptive EMA adjusts smoothing strength to volatility regime. In high-vol
periods, the effective half-life increases (up to 2x base), reducing noise-driven
trading. In calm periods, half-life decreases (down to 0.5x) for faster signal capture.

### 3. Borrowing Cost Impact

| Config | SR(net) | Turnover | Return(net) | MaxDD |
|--------|---------|----------|-------------|-------|
| borrow0_ew_ada                      |   0.6022 |   0.0548 |     0.2257 |    -0.1342 |
| borrow0_ew_fix                      |   0.6349 |   0.0540 |     0.2349 |    -0.1124 |
| borrow0_sw_ada                      |   0.1706 |   0.1034 |     0.0517 |    -0.1844 |
| borrow0_sw_fix                      |   0.0516 |   0.1014 |    -0.0017 |    -0.2227 |
| borrow100_ew_ada                    |   0.5404 |   0.0548 |     0.1983 |    -0.1413 |
| borrow100_ew_fix                    |   0.5707 |   0.0540 |     0.2069 |    -0.1157 |
| borrow100_sw_ada                    |   0.1200 |   0.1034 |     0.0284 |    -0.1915 |
| borrow100_sw_fix                    |  -0.0011 |   0.1014 |    -0.0243 |    -0.2305 |
| borrow25_ew_ada                     |   0.5868 |   0.0548 |     0.2188 |    -0.1357 |
| borrow25_ew_fix                     |   0.6188 |   0.0540 |     0.2279 |    -0.1132 |
| borrow25_sw_ada                     |   0.1580 |   0.1034 |     0.0458 |    -0.1861 |
| borrow25_sw_fix                     |   0.0384 |   0.1014 |    -0.0074 |    -0.2246 |
| borrow50_ew_ada                     |   0.5713 |   0.0548 |     0.2119 |    -0.1373 |
| borrow50_ew_fix                     |   0.6028 |   0.0540 |     0.2208 |    -0.1140 |
| borrow50_sw_ada                     |   0.1453 |   0.1034 |     0.0400 |    -0.1879 |
| borrow50_sw_fix                     |   0.0253 |   0.1014 |    -0.0130 |    -0.2266 |
| borrow75_ew_ada                     |   0.5559 |   0.0548 |     0.2051 |    -0.1393 |
| borrow75_ew_fix                     |   0.5867 |   0.0540 |     0.2138 |    -0.1149 |
| borrow75_sw_ada                     |   0.1327 |   0.1034 |     0.0342 |    -0.1897 |
| borrow75_sw_fix                     |   0.0121 |   0.1014 |    -0.0187 |    -0.2285 |


**Finding**: At 75 bps annualized borrowing cost (realistic for Japanese ETFs),
the net Sharpe degrades moderately. The short exposure is roughly 50% of the
portfolio, so the daily borrow cost impact is ~0.15 bps/day.

### 4. Best Overall Configuration

**Without borrowing costs:**

| Parameter | Value |
|-----------|-------|
| EMA half-life | 20 |
| Signal-weighted | No |
| Adaptive EMA | No |
| Borrow cost | 0 bps |
| Net Sharpe | 0.6349 |
| Turnover | 0.0540 |
| Total Return (net) | 0.2349 |
| Max Drawdown (net) | -0.1124 |

**With realistic borrowing costs (75 bps):**

| Parameter | Value |
|-----------|-------|
| EMA half-life | 20 |
| Signal-weighted | No |
| Adaptive EMA | No |
| Borrow cost | 75 bps |
| Net Sharpe | 0.5867 |
| Turnover | 0.0540 |
| Total Return (net) | 0.2138 |
| Max Drawdown (net) | -0.1149 |

### 5. Top 10 Overall Configurations

| Rank | EMA | SW | Adaptive | Borrow(bps) | SR(net) | Turnover | Return(net) |
|------|-----|-----|----------|-------------|---------|----------|-------------|
| 1 | 20 | No | No | 0 | 0.6349 | 0.0540 | 0.2349 |
| 2 | 20 | No | Yes | 0 | 0.6022 | 0.0548 | 0.2257 |
| 3 | 10 | No | Yes | 0 | 0.5888 | 0.0913 | 0.2303 |
| 4 | 20 | No | No | 75 | 0.5867 | 0.0540 | 0.2138 |
| 5 | 20 | No | Yes | 75 | 0.5559 | 0.0548 | 0.2051 |
| 6 | 10 | No | Yes | 75 | 0.5464 | 0.0913 | 0.2104 |
| 7 | 10 | No | No | 0 | 0.2530 | 0.0856 | 0.0785 |
| 8 | 10 | No | No | 75 | 0.2073 | 0.0856 | 0.0606 |
| 9 | 20 | Yes | Yes | 0 | 0.1706 | 0.1034 | 0.0517 |
| 10 | 30 | Yes | Yes | 0 | 0.1359 | 0.0761 | 0.0355 |


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
- **Model**: PCA_SUB with K=5, L=120, lambda=1.0
- **Train window**: 252 days (1 year)
- **Test window**: 21 days (1 month)
- **Folds**: 47
- **Total OOS samples**: 987
- **Direction accuracy**: 0.5119

### Cost Assumptions
- One-way transaction cost: 10 bps (commission + market impact)
- Borrowing costs tested: [0, 25, 50, 75, 100] bps annualized
- Realistic borrowing cost for Japanese ETFs: ~75 bps annualized

### Signal-Weighted Sizing
Positions are proportional to |prediction|, normalized so sum of |weights| = 1.
This concentrates capital on highest-conviction predictions while maintaining
the same gross exposure.

### Adaptive EMA
The EMA half-life scales with the ratio of short-term to long-term realized
volatility. Scale factor is clamped to [0.5x, 2.0x] of the base half-life.
- High vol: longer smoothing (up to 2x base HL) reduces whipsaw losses
- Low vol: shorter smoothing (down to 0.5x base HL) captures signals faster

### Borrowing Cost Model
Short positions incur daily borrowing costs computed as:
`daily_cost = short_exposure * (borrow_bps / 10000 / 252)`
Applied to the absolute value of negative weight exposures.

## Open Questions for Future Cycles

1. **Dynamic sector selection**: Could the set of traded sectors change over time
   based on rolling predictability scores?
2. **Multi-horizon signals**: Could combining predictions at different horizons
   (1-day, 5-day, 21-day) improve the signal quality?
3. **Intraday execution**: Could splitting orders across the trading day reduce
   market impact and improve fill prices?
4. **Factor-timing overlay**: Could timing exposure to specific PCA factors based
   on regime indicators improve returns?
