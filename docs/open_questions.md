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

### Phase 8: PC Interpretability
8. **RESOLVED**: What do the principal components represent economically?
   - PC1: Broad market factor (explains largest variance share)
   - PC2-3: Sector rotation factors (risk-on/risk-off, growth/value)
   - PC4-5: Finer sector-specific dynamics (in K=5 model)
   - Loading stability confirmed via high cosine similarity across folds

9. **OPEN**: Could factor rotation (e.g., varimax) improve interpretability?
   - Current PCA extracts orthogonal components; rotated factors might align
     more clearly with economic themes but would change the regression structure.

10. **OPEN**: Do transmission channel strengths predict strategy performance?
    - Periods with stronger, more stable transmission may correspond to
      higher strategy returns — worth investigating in future cycles.

### Phase 9: Baseline Comparison

11. **RESOLVED**: Does PCA_SUB outperform simpler baselines?
    - On gross metrics, optimized PCA_SUB (K=5, L=120, λ=1.0) achieves the highest
      Sharpe (2.18) and direction accuracy (51.2%), outperforming Direct OLS (1.95),
      Simple PCA (0.92), and Ridge regression (0.14).
    - All models have negative net Sharpe due to ~76% daily turnover from the
      naive sign-based strategy construction.

12. **RESOLVED**: Does exponential decay weighting add value?
    - Simple PCA (no decay) performs comparably to PCA_SUB with decay (λ=0.9).
      The difference is not statistically significant (p=0.49). The value of
      PCA_SUB comes from PCA dimensionality reduction, not decay weighting.

13. **OPEN**: Can signal smoothing or threshold-based positioning reduce turnover?
    - The 76% daily turnover destroys all gross alpha. EMA smoothing on predictions,
      minimum signal thresholds, or position change limits could dramatically
      improve net performance. This is the single most important improvement needed.

14. **OPEN**: Would sector-selective trading improve net Sharpe?
    - Steel & Nonferrous (55% accuracy), Energy Resources (53%), and Pharmaceuticals
      (52%) show strongest signals. Trading only high-signal sectors would reduce
      turnover and concentrate on the most predictable cross-market relationships.

### Phase 10: Final Report and Visualization

15. **RESOLVED**: Can all research phases be consolidated into a reproducible report?
    - Yes. Phase 10 generates 9 publication-quality visualizations and an integrated
      summary. All charts are saved as PNG files in reports/cycle_10/ and the full
      metrics are consolidated into a single metrics.json covering all phases.

16. **OPEN**: Can the gross-to-net gap be closed?
    - The optimized PCA_SUB achieves 2.18 gross Sharpe but -1.24 net Sharpe. The
      76% daily turnover is the sole cause. Three approaches to investigate:
      (a) EMA smoothing on predictions (target <10% turnover)
      (b) Minimum signal threshold for position changes
      (c) Sector-selective trading on top 4-5 predictable sectors

17. **OPEN**: Is the optimized parameter set (K=5, L=120, λ=1.0) stable going forward?
    - Inner-loop Sharpe has been trending upward (from 1.5 to 4.4) over 2023-2026
      folds, but this may reflect regime-specific fitting. True forward validation
      on post-2026 data is needed to confirm parameter stability.

### Phase 11: Executive Summary and Code Quality

18. **RESOLVED**: Is the project accessible to non-technical stakeholders?
    - Yes. An executive summary (`docs/executive_summary.md`) was created covering
      model explanation, key results, the critical turnover challenge, and recommended
      next steps in plain language.

19. **RESOLVED**: Is test coverage adequate for the core modules?
    - 83 unit tests now cover the 4 most critical modules: PCA_SUB model, 6 baseline
      models, walk-forward evaluator, and data pipeline. All tests pass in <1 second
      with no external API dependencies. Tests verify mathematical properties
      (orthonormality, positive semi-definiteness), interface contracts, output shapes,
      edge cases, and lookahead-bias prevention.

20. **OPEN**: Should higher-level evaluation modules also have dedicated unit tests?
    - Modules like `robustness_analyzer.py`, `hyperparam_optimizer.py`, and
      `pc_interpreter.py` are indirectly tested through their dependencies on
      the core modules. Dedicated tests would increase coverage but require
      longer-running synthetic walk-forward evaluations.
