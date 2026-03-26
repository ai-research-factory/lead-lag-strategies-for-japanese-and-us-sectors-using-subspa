# Open Questions

## Phase 1

1. ~~**Covariance estimation period (Cfull)**~~: **Resolved in Phase 5** — Compared fixed Cfull (126d, 252d, 504d) vs rolling covariance. See reports/cycle_5/ for detailed results and analysis.

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

12. ~~**Train window sensitivity**~~: **Resolved in Phase 6** — Nested walk-forward optimization tested L ∈ {20, 40, 60, 80, 120}. L=120 was selected in 29/35 folds; longer lookbacks provide more stable covariance estimates and better inner-loop Sharpe. The sensitivity analysis shows a monotonic improvement from L=20 to L=120.

13. **Equal-weight strategy limitation**: The current strategy uses equal weights across all sectors. A volatility-weighted or signal-strength-weighted approach could improve the Sharpe ratio.

## Phase 6

14. **Optimal parameters favor more components and no decay**: The optimizer consistently selected K=5 (17/35 folds), L=120 (29/35), λ=1.0 (33/35). λ=1.0 means equal weighting (no exponential decay), suggesting the covariance structure is stable enough that recent-weighting hurts more than it helps. This contradicts the paper's default λ=0.9.

15. **In-sample vs OOS gap in optimization**: Inner-loop Sharpe ratios reach 3–4+ in later folds while OOS accuracy stays around 51%. This gap suggests the inner-loop metric may overfit to specific market regimes, though the nested design prevents this from contaminating OOS evaluation.

16. **Parameter non-stationarity**: Early folds (2023) selected K=4, while later folds (2024–2026) shifted to K=5. This time-varying optimal dimensionality suggests the cross-market factor structure evolves, supporting periodic re-optimization.

## Phase 7

17. ~~**Parameter sensitivity magnitude**~~: **Resolved** — One-at-a-time sensitivity shows K has the largest impact on Sharpe (spread ~0.9 across K=1..7), λ has moderate impact (spread ~0.7), and L has smallest impact (spread ~0.08). Model is most sensitive to dimensionality choice.

18. ~~**Regime dependence confirmed**~~: **Resolved** — Model performs better in low-volatility regimes (SR=0.72 vs 0.47 in high-vol) and up-market days (SR=1.31 vs -0.18 in down-market). The lead-lag signal is directionally asymmetric — stronger when JP market goes up.

19. ~~**Temporal stability**~~: **Resolved** — Slight positive accuracy trend over time (slope +0.0007/fold). However, max consecutive losing streak (7 folds below 50%) indicates sustained drawdown periods are expected. Sub-period analysis shows 2022–2023 were poor (negative Sharpe) while 2024 was strong (SR=1.22).

20. ~~**Sector-level robustness**~~: **Resolved** — Foods (1617.T, acc=52.2%, SR=1.22) and Finance ex-Banks (1630.T, acc=51.9%, SR=1.21) are consistently the most predictable sectors. Real Estate (1631.T) and Construction (1619.T) remain near or below 50%. Selective sector trading on top-predictable sectors is a promising avenue.

21. ~~**Statistical significance**~~: **Resolved** — Block bootstrap (2000 resamples) gives baseline Sharpe 90% CI of [-0.45, 1.40] with P(SR>0)=82%. Optimized params show stronger significance. The wide CI reflects regime dependence; the signal exists but is noisy.

22. **Optimized vs baseline gap**: Optimized params (K=5, L=120, λ=1.0) deliver SR=2.18 vs baseline SR=0.54, but this may reflect in-sample fitting to the 2021–2026 period. True out-of-sample validation on new data is needed to confirm the improvement is genuine.

23. **Selective sector strategy**: Given consistent sector-level variation, a strategy that only trades the most predictable sectors (Foods, Finance ex-Banks, Energy Resources, Trading Companies) could improve risk-adjusted returns and reduce noise from unpredictable sectors.
