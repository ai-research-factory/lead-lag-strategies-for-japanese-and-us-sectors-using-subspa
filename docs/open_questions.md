# Open Questions

## Phase 1

1. **Covariance estimation period (Cfull)**: The paper uses a fixed period (2010–2014) for covariance estimation. How does this compare to the rolling window approach implemented here? This is the focus of Phase 5.

2. ~~**Open-to-close vs close-to-close returns for Japan**~~: **Resolved in Phase 2** — JP returns now use open-to-close calculation `(close - open) / open`.

3. ~~**Lead-lag alignment**~~: **Resolved in Phase 2** — Now uses proper next-trading-day lookup for each U.S. date, correctly handling holidays and market calendar differences.

4. **Regularization of covariance matrix**: With 11 features and L=60, the covariance matrix is well-conditioned. However, with shorter windows or more features, regularization (shrinkage) may be needed.

5. **Intercept term**: The current implementation includes an intercept in the regression. The paper's formulation should be checked to confirm whether an intercept is used.

## Phase 2

6. **Open-to-close vs overnight gap**: Open-to-close returns exclude the overnight gap (previous close to today's open). The paper may implicitly assume this gap captures different information. Consider whether close-to-open returns could add signal.

7. **Weekend/holiday lag effects**: The lag distribution shows ~18% of pairs span 3+ calendar days (weekends/holidays). Whether the lead-lag signal decays over longer gaps could affect strategy performance.

8. **JP ETF liquidity**: Some TOPIX-17 ETFs show low volume (e.g., single-digit shares on some days). This may affect the reliability of open prices and thus open-to-close returns.

## Phase 3

9. **Weak but non-zero signal**: Walk-forward direction accuracy (~49.8%) is near coin-flip level, but the gross long-short Sharpe ratio of 0.54 suggests a small exploitable signal. Whether this survives transaction costs is the key question for Phase 4.

10. **Regime dependence**: Per-fold accuracy varies widely (41.7%–57.1%), suggesting the U.S.-to-Japan lead-lag relationship is regime-dependent. Periods of high cross-market correlation may offer stronger signal.

11. **Sector heterogeneity in predictability**: Foods (1617.T, acc=52.2%) and Finance ex-Banks (1630.T, acc=51.9%) show the strongest predictability. Construction (1619.T) and Real Estate (1631.T) are hardest to predict. This may relate to global vs. domestic sector exposure.

12. **Train window sensitivity**: The current 252-day train window is fixed. Shorter windows might capture regime changes faster; longer windows might provide more stable estimates. This needs systematic testing in Phase 6.

13. **Equal-weight strategy limitation**: The current strategy uses equal weights across all sectors. A volatility-weighted or signal-strength-weighted approach could improve the Sharpe ratio.
