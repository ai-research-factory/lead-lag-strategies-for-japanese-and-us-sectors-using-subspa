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

### Cycle 12: Turnover Optimization & Cost-Aware Strategy

21. **RESOLVED**: Can signal smoothing or threshold-based positioning reduce turnover?
    - Yes. EMA smoothing with half-life=20 days on top-5 predictable sectors achieves
      the best net Sharpe of 0.63, reducing daily turnover from 86% to 5.4%.
    - EMA smoothing alone (half-life=5) improves net Sharpe from -1.24 to +0.15.
    - Sector selection to top-5 alone improves net Sharpe from -1.24 to ~0.00.
    - Combined EMA + sector selection delivers the strongest improvement.

22. **RESOLVED**: Can the gross-to-net gap be closed?
    - Partially. The best configuration (EMA-20, top-5 sectors) achieves net Sharpe 0.63
      with total return +23.5% over 2022-2026 and max drawdown -11.2%. The 76% daily
      turnover was reduced to 5.4%, preserving most of the sector-concentrated signal.
    - The gross Sharpe drops from 2.18 to 0.78 with smoothing (signal delay cost),
      but the net Sharpe improves dramatically (+1.88 improvement).

23. **RESOLVED**: Would sector-selective trading improve net Sharpe?
    - Yes. Top-5 sectors (Steel & Nonferrous 55%, Energy Resources 53%, Trading Companies
      53%, Pharmaceuticals 52%, Finance ex-Banks 52%) concentrate the strongest signals.
    - Combined with EMA smoothing, sector selection is the most effective approach.

24. **RESOLVED**: Signal-weighted positions — could prediction magnitude size positions instead
    of equal-weight long/short?
    - No improvement. Signal-weighted positions (EMA-20, top-5) achieve net Sharpe 0.05 vs
      equal-weight 0.63. The magnitude-based sizing increases turnover from 5.4% to 10.1%
      without proportional signal improvement. The PCA_SUB model's directional signal is
      stronger than its magnitude signal — knowing the sign matters more than the size.

25. **RESOLVED**: Adaptive smoothing — could EMA half-life adapt to market regime?
    - Modest improvement in specific configurations. Adaptive EMA-20 with vol_window=10
      achieves net Sharpe 0.68 vs fixed EMA-20 at 0.63 (+0.05). The adaptive approach
      lengthens smoothing in high-vol periods (reducing whipsaw losses) and shortens it
      in calm periods (faster signal capture). However, the improvement is small and
      the best non-adaptive config (EMA-20, top-5, equal-weight) remains competitive.

26. **OPEN**: Out-of-sample parameter stability — the optimal strategy parameters (EMA-20,
    top-5 sectors) were selected on the 2021-2026 walk-forward window. Forward validation
    on post-2026 data is needed to confirm stability.

27. **RESOLVED**: Borrowing costs — how much do short-selling costs reduce net returns?
    - At 75 bps annualized (realistic for Japanese ETFs), net Sharpe drops from 0.63 to
      0.59 (a reduction of 0.05). Total return drops from +23.5% to +21.4%. The strategy
      remains profitable even at 100 bps borrow costs (net Sharpe 0.57). Short exposure
      is roughly 50% of portfolio, so daily borrow impact is ~0.15 bps/day — modest
      relative to the 10 bps one-way transaction cost.

### Cycle 13: Advanced Strategy Enhancements

28. **RESOLVED**: Does the Cycle 12 optimal config (EMA-20, top-5, equal-weight) remain
    best after adding new strategy dimensions?
    - Yes. Across 24 configurations tested (EMA x signal-weighted x adaptive x borrow),
      the original EMA-20, top-5, equal-weight, fixed-EMA remains the top performer
      with net Sharpe 0.63 (0.59 with 75 bps borrow costs). Signal-weighting and adaptive
      EMA do not improve upon the simple configuration.

29. **RESOLVED**: Dynamic sector selection — could the set of traded sectors change over time
    based on rolling predictability scores instead of fixed top-5?
    - No improvement. Best dynamic config (lookback=126d, minAcc=0.51, 3-5 sectors, monthly
      rebalance) achieves net Sharpe 0.29 vs fixed top-5 at 0.63. Dynamic selection increases
      turnover (0.08 vs 0.05) and reduces return (+9.8% vs +23.5%). The fixed top-5 sectors
      maintain stable predictability across the full evaluation period; re-ranking adds noise
      and missed opportunities from premature sector exclusion.

30. **RESOLVED**: Multi-horizon signals — could combining predictions at different horizons
    (1-day, 5-day, 21-day) improve signal quality?
    - No improvement. All multi-horizon ensembles underperform raw 1-day predictions.
      Best ensemble (1d+5d weighted 70/30) achieves net Sharpe 0.51 vs 1d-only at 0.63.
      Longer horizons (10d, 21d) further degrade performance. The PCA_SUB signal is strongest
      at the 1-day horizon; moving-average aggregation delays signal capture without
      meaningful noise reduction. The EMA-20 smoothing already provides optimal noise
      filtering — stacking additional averaging on top over-smooths the signal.

31. **OPEN**: Factor-timing overlay — could timing exposure to specific PCA factors based
    on regime indicators improve returns?

32. **OPEN**: Intraday execution — could splitting orders across the trading day reduce
    market impact and improve fill prices?

### Cycle 14: Dynamic Sector Selection & Multi-Horizon Signals

33. **RESOLVED**: Ensemble model diversity — could combining PCA_SUB with other model types
    (Ridge, elastic net) improve signal quality beyond single-model multi-horizon averaging?
    - No improvement over PCA_SUB alone. Best ensemble (PCA-heavy sign vote, 60/20/20 weights)
      achieves net Sharpe 0.51 vs C12 baseline 0.63. Ridge and ElasticNet individually perform
      worse (net SR -0.20 and -0.15) with direction accuracy below 50%. PCA_SUB's dimensionality
      reduction captures the cross-market signal better than direct regression approaches.
      Adding weaker models dilutes the PCA_SUB signal rather than complementing it.

34. **RESOLVED**: Regime detection — could an explicit regime classifier (bull/bear/sideways)
    improve sector selection timing instead of rolling accuracy?
    - No improvement. Best regime-aware config (vol lookback=42d, 2 regimes, mild scaling)
      achieves net Sharpe 0.61 vs C12 0.63. Volatility-based exposure scaling modestly reduces
      drawdowns but also reduces returns proportionally. The lead-lag signal, while weaker in
      high-vol regimes (Phase 7), still contributes enough alpha that reducing exposure hurts
      more than it helps on a Sharpe basis.

35. **OPEN**: Transaction cost optimization — could dynamic selection be modified to penalize
    unnecessary sector switches, reducing turnover while maintaining adaptivity?

36. **OPEN**: Out-of-sample validation — all improvements (Cycles 12-15) should be tested on
    truly unseen data (post-2026) to confirm they are not overfit to the 2021-2026 period.

### Cycle 15: Ensemble Model Diversity & Regime-Aware Positioning

37. **OPEN**: Online model selection — could the strategy dynamically switch between PCA_SUB
    and ensemble models based on recent OOS performance, rather than using fixed weights?

38. **RESOLVED**: Factor timing — could exposure to specific PCA factors (PC1 market, PC2
    rotation) be timed based on recent predictive accuracy?
    - No improvement. Factor-timed predictions (softmax weighting by per-PC directional accuracy)
      achieve net Sharpe -0.18 vs C12 baseline 0.63. Weighting PCs unequally disrupts the
      balanced regression structure. Direction accuracy drops from 51.19% to 50.28%.
      All 5 PCs contribute complementary information; selective weighting loses signal.

39. **OPEN**: Regime transition smoothing — could position changes during regime transitions
    be smoothed to avoid unnecessary turnover from sudden exposure changes?

### Cycle 16: Factor Timing, Risk-Parity & Long-Bias Strategy

40. **RESOLVED**: Risk-parity position sizing — could inverse-volatility weighting improve
    risk-adjusted returns by balancing risk contributions across sectors?
    - No improvement. Best risk-parity config (126-day lookback) achieves net Sharpe 0.57
      vs C12 0.63. Increased turnover (5.8% vs 5.4%) from changing volatility-weighted
      positions offsets the diversification benefit. Equal-weight sizing remains optimal
      for this portfolio of 5 pre-selected sectors.

41. **RESOLVED**: Long-biased strategy — could reducing short exposure exploit the signal
    asymmetry documented in Phase 7 (up-market SR=1.31 vs down-market SR=-0.18)?
    - No improvement. All long-bias levels (0.25, 0.5, 0.75, 1.0) strictly underperform
      standard long-short (bias=0.0). Long-only (bias=1.0) achieves net SR=-0.36.
      Despite the asymmetry in signal quality, the strategy profits from both correct long
      AND correct short calls. Removing shorts eliminates half the alpha while the signal
      asymmetry doesn't compensate. The Phase 7 finding describes conditional performance
      differences, not a reason to eliminate short exposure.

42. **OPEN**: Sector-specific long bias — could each sector have a different optimal
    long/short asymmetry based on its individual signal characteristics?

43. **OPEN**: Transaction cost-aware rebalancing — could the strategy skip small position
    changes when the expected alpha is below the trading cost?

44. **OPEN**: Out-of-sample validation — all improvements (Cycles 12-16) should be tested
    on truly unseen data (post-2026) to confirm they are not overfit to the 2021-2026 period.
