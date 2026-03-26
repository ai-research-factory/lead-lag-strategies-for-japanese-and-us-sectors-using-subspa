# Phase 2: Real Data Pipeline — Technical Findings

## Overview

Implemented a production-quality data pipeline (`src/data/pipeline.py`) that fetches U.S. and Japanese sector ETF data from the ARF Data API, computes the correct return types as specified in the paper, and aligns trading days with a proper lead-lag structure.

## Key Implementation Details

### Return Calculations
- **U.S. sectors (11 ETFs)**: Close-to-close returns via `pct_change()` on closing prices.
- **Japanese sectors (17 ETFs)**: Open-to-close returns `(close - open) / open`, as specified by the paper. This captures the intraday movement on the Japanese market following U.S. trading activity.

### Lead-Lag Alignment
The paper's hypothesis is that U.S. market close on day t predicts Japanese market behavior on the next trading day. The alignment logic:
- For each U.S. trading day, find the **next** Japanese trading day (strictly after).
- This handles weekends, holidays, and market calendar differences between the two markets.

**Lag distribution** (calendar days between paired US/JP dates):
- 1 day: 957 pairs (76.3%) — Normal weekday-to-weekday
- 2 days: 21 pairs (1.7%) — JP holiday on next day
- 3 days: 219 pairs (17.5%) — Friday US → Monday JP
- 4-7 days: 57 pairs (4.5%) — Extended holidays

### Data Quality
- **1,254 aligned pairs** from 2021-03-29 to 2026-03-25
- Zero NaN/Inf values after alignment
- U.S. returns: mean=0.000457, std=0.0125, max|return|=0.1343
- JP returns: mean=-0.000320, std=0.0109, max|return|=0.1540
- All returns within reasonable bounds (<25% daily)

## Validation Results

| Check | Result |
|-------|--------|
| All 11 US tickers fetched | PASS |
| All 17 JP tickers fetched | PASS |
| US close-to-close calculation correct | PASS |
| JP open-to-close calculation correct | PASS |
| No lookahead bias (JP date > US date) | PASS |
| No NaN values | PASS |
| No Inf values | PASS |
| Returns within reasonable range | PASS |
| PCASub integration — shape | PASS |
| PCASub integration — finite predictions | PASS |

### Model Integration Test
Using the aligned open-to-close returns with PCASub (K=3, L=60, lambda=0.9):
- Direction accuracy: 49.9% (comparable to Phase 1's 52.7% with close-to-close)
- Mean prediction-actual correlation: 0.136 (slightly improved from Phase 1's 0.105)

The slight change in direction accuracy is expected: open-to-close returns have different statistical properties than close-to-close, as they exclude overnight gaps.

## Changes from Phase 1

1. **Correct JP returns**: Phase 1 used close-to-close for both markets. Phase 2 uses open-to-close for Japan as specified in the paper.
2. **Proper calendar alignment**: Phase 1 used date intersection and simple shift. Phase 2 explicitly finds the next Japanese trading day for each U.S. date, correctly handling holidays.
3. **Modular architecture**: Data pipeline is now a reusable `DataPipeline` class in `src/data/pipeline.py`, separate from the validation script.
4. **AlignedDataset named tuple**: Structured output with metadata (dates, ticker lists, full DataFrames) for downstream analysis.

## Open Questions Addressed
- "Return calculation" (from Phase 1 open_questions.md): Now resolved — JP uses open-to-close.
- "Lead-lag alignment" (from Phase 1): Now uses proper next-trading-day lookup instead of naive date shifting.
