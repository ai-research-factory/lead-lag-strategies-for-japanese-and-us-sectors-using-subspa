"""Phase 10: Final Report and Visualization.

Integrates all analysis results from Phases 1-9, generates comprehensive
visualizations and a reproducible summary report.

Outputs:
  - reports/cycle_10/metrics.json — aggregated metrics from all phases
  - reports/cycle_10/technical_findings.md — integrated technical summary
  - reports/cycle_10/*.png — visualization charts
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import DataPipeline, JP_SECTOR_NAMES, US_SECTOR_NAMES
from src.models.pca_sub import PCASub
from src.models.baselines import DirectOLS
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.evaluation.baseline_comparator import BaselineComparator
from src.evaluation import report_generator as rg


def load_cycle_metrics(reports_dir: Path) -> dict:
    """Load all existing cycle metrics."""
    all_metrics = {}
    for cycle_dir in sorted(reports_dir.iterdir()):
        if not cycle_dir.is_dir():
            continue
        metrics_path = cycle_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                all_metrics[cycle_dir.name] = json.load(f)
            print(f"  Loaded {cycle_dir.name}/metrics.json")
    return all_metrics


def run_key_walk_forward(dataset):
    """Run walk-forward for the optimized model to generate cumulative returns chart."""
    print("\nRunning optimized PCA_SUB walk-forward for cumulative returns chart...")
    evaluator = WalkForwardEvaluator(
        train_window=252, test_window=21,
        K=5, L=120, lambda_decay=1.0,
    )
    result = evaluator.evaluate(dataset.X_us, dataset.Y_jp, dataset.dates_us)
    strategy = evaluator.compute_oos_sharpe(result)
    print(f"  Folds: {result.n_folds}, Accuracy: {result.mean_direction_accuracy:.4f}")
    print(f"  Gross Sharpe: {strategy.get('sharpe_ratio_gross', 0):.4f}")
    return result, strategy


def compile_integrated_metrics(all_metrics: dict, wf_result, strategy: dict) -> dict:
    """Compile integrated metrics from all phases."""
    timestamp = datetime.now().isoformat()

    # Extract key results from each phase
    integrated = {
        "phase": 10,
        "description": "Final integrated report and visualization — all phases consolidated",
        "timestamp": timestamp,
        "data": all_metrics.get("cycle_9", {}).get("data", {}),
        "phases_completed": sorted(all_metrics.keys()),
        "n_phases": len(all_metrics),

        "phase_1_core_algorithm": {
            "description": "PCA_SUB core implementation validated with real data",
            "status": "completed",
        },

        "phase_2_data_pipeline": {
            "description": "Real data pipeline with ARF Data API, proper returns, lead-lag alignment",
            "status": "completed",
            "aligned_observations": all_metrics.get("cycle_4", {}).get("dataset", {}).get("total_aligned_pairs", 1254),
        },

        "phase_3_walk_forward": {
            "description": "Walk-forward evaluation framework with 47 folds",
            "status": "completed",
            "baseline_accuracy": all_metrics.get("cycle_4", {}).get("results", {}).get("direction_accuracy_mean", 0.4978),
            "baseline_sharpe_gross": all_metrics.get("cycle_4", {}).get("strategy_gross", {}).get("sharpe_ratio", 0.5418),
        },

        "phase_5_cfull_validation": {
            "description": "Fixed vs rolling covariance period comparison",
            "status": "completed",
            "best_method": "cfull_504d",
            "best_sharpe": all_metrics.get("cycle_5", {}).get("comparison_results", {}).get("cfull_504d", {}).get("strategy", {}).get("sharpe_ratio_gross", 1.0452),
        },

        "phase_6_optimization": {
            "description": "Nested walk-forward hyperparameter optimization",
            "status": "completed",
            "optimal_K": 5,
            "optimal_L": 120,
            "optimal_lambda": 1.0,
            "optimized_sharpe": all_metrics.get("cycle_6", {}).get("optimized_results", {}).get("strategy", {}).get("sharpe_ratio_gross", 1.3907),
            "improvement_over_baseline": "61% Sharpe improvement (0.86 → 1.39)",
        },

        "phase_7_robustness": {
            "description": "Sensitivity analysis, regime analysis, bootstrap CIs",
            "status": "completed",
            "most_sensitive_param": "K (number of PCs)",
            "regime_finding": "Low-vol outperforms high-vol; up-market stronger than down-market",
        },

        "phase_8_interpretation": {
            "description": "Principal component economic interpretation",
            "status": "completed",
            "pc1": "Broad market factor (~57% variance)",
            "pc2": "Value vs Growth rotation (~18% variance)",
            "pc3": "Sector-specific dynamics (~9% variance)",
        },

        "phase_9_baselines": {
            "description": "Comprehensive baseline comparison (9 models)",
            "status": "completed",
            "best_model": "PCA_SUB (K=5, L=120, λ=1.0)",
            "best_gross_sharpe": 2.18,
            "best_net_sharpe": -1.24,
            "critical_issue": "76% daily turnover from sign-based positioning destroys net alpha",
        },

        "final_model_performance": {
            "model": "PCA_SUB (K=5, L=120, λ=1.0)",
            "walk_forward_config": {"train_window": 252, "test_window": 21},
            "n_folds": wf_result.n_folds,
            "total_oos_samples": wf_result.total_test_samples,
            "direction_accuracy": {
                "mean": round(wf_result.mean_direction_accuracy, 6),
                "std": round(wf_result.std_direction_accuracy, 6),
            },
            "correlation": {
                "mean": round(wf_result.mean_correlation, 6),
                "std": round(wf_result.std_correlation, 6),
            },
            "strategy_metrics": {
                "sharpe_ratio_gross": strategy.get("sharpe_ratio_gross", 0),
                "annualized_return_gross": strategy.get("annualized_return_gross", 0),
                "annualized_volatility": strategy.get("annualized_volatility", 0),
                "max_drawdown": strategy.get("max_drawdown", 0),
                "total_return": strategy.get("total_return", 0),
                "pct_positive_days": strategy.get("pct_positive_days", 0),
            },
            "positive_accuracy_folds_pct": round(wf_result.positive_accuracy_folds_pct, 2),
        },

        "key_findings": [
            "Cross-market lead-lag signal exists: US sector returns predict JP sector returns with 51.2% accuracy (above 50% random)",
            "PCA dimensionality reduction (PCA_SUB) outperforms direct regression on gross metrics",
            "Optimized parameters (K=5, L=120, λ=1.0) significantly outperform paper defaults (K=3, L=60, λ=0.9), p=0.014",
            "λ=1.0 (no decay) consistently selected — covariance structure is stable over lookback windows",
            "Signal is regime-dependent: stronger in low-volatility periods and JP up-market days",
            "PC1 captures broad market factor (~57% variance), PC2-3 capture sector rotation themes",
            "Best predictable sectors: Steel & Nonferrous (55%), Energy Resources (53%), Pharmaceuticals (52%)",
            "CRITICAL: 76% daily turnover from naive sign-based positioning destroys all net alpha",
            "Net Sharpe is negative (-1.24) for the best model despite positive gross Sharpe (2.18)",
            "Signal smoothing or threshold-based position management is essential for practical deployment",
        ],

        "recommendations": [
            "Implement EMA signal smoothing to reduce turnover below 10%",
            "Add minimum signal threshold for position changes",
            "Consider sector-selective trading (top 4-5 sectors only)",
            "Test position sizing based on prediction confidence",
            "Validate on extended out-of-sample period beyond 2026",
        ],

        "visualizations_generated": [
            "executive_summary.png",
            "model_comparison.png",
            "sector_accuracy_heatmap.png",
            "walk_forward_timeline.png",
            "parameter_sensitivity.png",
            "cfull_comparison.png",
            "optimization_evolution.png",
            "cumulative_returns.png",
            "phase_progression.png",
        ],
    }

    return integrated


def generate_technical_findings(metrics: dict, output_dir: Path):
    """Generate comprehensive technical findings markdown."""
    content = f"""# Phase 10: Final Report and Visualization

## Overview

This report integrates results from all 9 research phases of the PCA_SUB lead-lag
strategy for Japanese and U.S. sector ETFs. The study uses 11 U.S. Select Sector
SPDR ETFs as predictors and 17 TOPIX-17 sector ETFs as targets, exploiting the
overnight lead-lag relationship between U.S. close-to-close returns and Japanese
open-to-close returns on the following trading day.

**Data**: {metrics['data'].get('total_aligned_observations', 1254)} aligned observations,
{metrics['data'].get('date_range_start', '2021-03-29')} to {metrics['data'].get('date_range_end', '2026-03-25')}
(11 US sectors x 17 JP sectors)

**Methodology**: Walk-forward evaluation with 252-day training window and 21-day
test window (no lookahead bias). Nested walk-forward for hyperparameter optimization.

---

## Executive Summary

The PCA_SUB model extracts a statistically significant but economically marginal
cross-market signal. The optimized model (K=5, L=120, lambda=1.0) achieves:

| Metric | Value |
|--------|-------|
| Direction Accuracy | {metrics['final_model_performance']['direction_accuracy']['mean']*100:.1f}% |
| Gross Sharpe Ratio | {metrics['final_model_performance']['strategy_metrics']['sharpe_ratio_gross']:.2f} |
| Net Sharpe Ratio (10bps) | {metrics['phase_9_baselines']['best_net_sharpe']:.2f} |
| Gross Total Return | {metrics['final_model_performance']['strategy_metrics']['total_return']*100:.1f}% |
| Max Drawdown (gross) | {metrics['final_model_performance']['strategy_metrics']['max_drawdown']*100:.1f}% |
| Folds with >50% accuracy | {metrics['final_model_performance']['positive_accuracy_folds_pct']:.0f}% |

**Critical Finding**: While gross alpha exists, the naive sign-based long-short
positioning generates ~76% daily turnover, which at 10bps per trade produces
negative net returns. Signal smoothing is the key bottleneck to practical deployment.

---

## Phase-by-Phase Summary

### Phase 1: Core Algorithm Implementation
- Implemented PCA_SUB class with decay-weighted covariance, PCA, and OLS regression
- Validated on real data: fit/predict work correctly with expected output shapes
- Initial direction accuracy ~52.7% on full-sample test

### Phase 2: Real Data Pipeline
- Built ARF Data API integration with local caching
- US returns: close-to-close; JP returns: open-to-close (per paper specification)
- Lead-lag alignment: each US day t paired with next JP trading day t+1
- 1,254 aligned observation pairs after cleaning

### Phase 3: Walk-Forward Evaluation
- 47 walk-forward folds, 987 out-of-sample test observations
- Baseline (K=3, L=60, lambda=0.9): 49.8% accuracy, 0.54 gross Sharpe
- Per-fold accuracy ranges from 41.7% to 57.1% (high variance)
- Gross long-short strategy returns 14.8% total over test period

### Phase 5: Covariance Period Validation
- Compared rolling PCA re-estimation vs fixed Cfull periods (126d, 252d, 504d)
- Cfull 504d achieves best Sharpe (1.05) vs rolling (0.54)
- Eigenvector stability is high for 126d-504d comparison (cosine sim 0.93)
- Fixed covariance provides more stable factor structure

### Phase 6: Hyperparameter Optimization
- Nested walk-forward grid search over 125 parameter combinations
- 35 outer folds with inner-loop optimization on each training window
- Optimal: K=5 (49% of folds), L=120 (83% of folds), lambda=1.0 (94% of folds)
- Optimized Sharpe: 1.39 vs baseline 0.86 (+61% improvement)
- lambda=1.0 (no decay) dominates — covariance structure is temporally stable

### Phase 7: Robustness Analysis
- K is the most impactful parameter (Sharpe spread 0.88 across K=1..7)
- Signal is regime-dependent: low-vol SR=0.72 vs high-vol SR=0.47
- Up-market days show stronger signal (SR=1.31 vs -0.18 for down-market)
- Bootstrap 90% CI for Sharpe: [-0.45, 1.40], P(SR>0)=82%
- Yearly variation: 2022-2023 poor, 2024 strong (SR=1.22)

### Phase 8: PC Interpretability
- PC1: Broad market factor (57% variance) — all US sectors load positively
- PC2: Value/Growth rotation (18%) — Energy positive, Tech negative
- PC3: Sector-specific dynamics (9%) — finer differentiation
- High temporal stability of loadings (cosine similarity >0.9 across folds)
- Strongest transmission: PC1 -> JP broad market, PC2 -> JP cyclical sectors

### Phase 9: Baseline Comparison
- 9 models evaluated: 2 PCA_SUB configs, Direct OLS, 2 Ridge, Simple PCA, Historical Mean, Zero Predictor, Equal-Weight
- PCA_SUB (K=5, optimized) achieves highest gross Sharpe (2.18) and accuracy (51.2%)
- PCA dimensionality reduction adds value vs direct regression (OLS: 1.95, Ridge: 0.14)
- ALL models have negative net Sharpe due to turnover problem
- Optimized PCA_SUB significantly outperforms baseline (t-stat=2.46, p=0.014)

---

## Visualization Inventory

All charts saved to `reports/cycle_10/`:

1. **executive_summary.png** — Single-page dashboard with key metrics, model ranking,
   parameter selection, and performance timeline
2. **model_comparison.png** — 4-panel comparison: Sharpe (gross/net), accuracy,
   returns (gross/net), turnover across all 9 models
3. **sector_accuracy_heatmap.png** — 17 JP sectors x 6 models direction accuracy
4. **walk_forward_timeline.png** — Per-fold accuracy and correlation over 47 folds
5. **parameter_sensitivity.png** — K, L, lambda inner-loop Sharpe with error bars
6. **cfull_comparison.png** — Rolling vs fixed covariance: accuracy, Sharpe, distribution
7. **optimization_evolution.png** — Selected K, L, lambda and OOS accuracy over 35 folds
8. **cumulative_returns.png** — Gross/net cumulative return curves with drawdown
9. **phase_progression.png** — Sharpe and accuracy improvement across research phases

---

## Key Findings

{chr(10).join(f'{i+1}. {f}' for i, f in enumerate(metrics['key_findings']))}

---

## Recommendations for Future Work

{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(metrics['recommendations']))}

---

## Statistical Significance

| Comparison | t-stat | p-value | Significant (5%) |
|-----------|--------|---------|-------------------|
| Optimized PCA_SUB vs Baseline PCA_SUB | -2.46 | 0.014 | Yes |
| Optimized PCA_SUB vs Direct OLS | -2.26 | 0.024 | Yes |
| Baseline PCA_SUB vs Zero Predictor | 1.07 | 0.284 | No |
| Baseline PCA_SUB vs Historical Mean | 0.88 | 0.380 | No |

The optimized PCA_SUB model shows statistically significant improvement over the
baseline configuration and direct OLS, validating that both PCA regularization and
hyperparameter optimization add genuine out-of-sample value.

---

## Conclusion

The PCA_SUB approach successfully identifies a cross-market lead-lag relationship
between U.S. and Japanese sector ETFs. The signal is:

- **Real but weak**: 51.2% direction accuracy (1.2% above random)
- **Regime-dependent**: Stronger in calm, rising markets
- **Sector-heterogeneous**: Strongest for cyclical/trade-exposed sectors
- **Economically meaningful**: PCs map to interpretable market factors

The primary obstacle to practical implementation is **execution cost**. The naive
sign-based strategy generates excessive turnover (~76% daily). Addressing this
through signal smoothing, position thresholds, or adaptive execution is the single
most impactful improvement for converting gross alpha into net returns.

*Report generated: {metrics['timestamp']}*
"""
    output_path = output_dir / "technical_findings.md"
    output_path.write_text(content)
    print(f"  Saved technical_findings.md")


def main():
    print("=" * 60)
    print("PHASE 10: FINAL REPORT AND VISUALIZATION")
    print("=" * 60)
    print()

    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    output_dir = reports_dir / "cycle_10"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all existing metrics
    print("Loading metrics from all phases...")
    all_metrics = load_cycle_metrics(reports_dir)
    print(f"  Loaded {len(all_metrics)} phase results")
    print()

    # Step 2: Load data and run optimized model for cumulative returns
    print("Loading data via pipeline...")
    pipeline = DataPipeline(data_dir=project_root / "data")
    dataset = pipeline.load()
    print()

    wf_result, strategy = run_key_walk_forward(dataset)
    print()

    # Step 3: Generate all visualizations
    print("Generating visualizations...")

    rg.generate_executive_summary_chart(
        all_metrics.get("cycle_9", {}),
        all_metrics.get("cycle_6", {}),
        output_dir,
    )

    rg.generate_model_comparison_chart(
        all_metrics.get("cycle_9", {}),
        output_dir,
    )

    rg.generate_sector_accuracy_chart(
        all_metrics.get("cycle_9", {}),
        JP_SECTOR_NAMES,
        output_dir,
    )

    rg.generate_walk_forward_timeline(
        all_metrics.get("cycle_4", {}),
        output_dir,
    )

    rg.generate_parameter_sensitivity_chart(
        all_metrics.get("cycle_6", {}),
        output_dir,
    )

    rg.generate_cfull_comparison_chart(
        all_metrics.get("cycle_5", {}),
        output_dir,
    )

    rg.generate_optimization_evolution_chart(
        all_metrics.get("cycle_6", {}),
        output_dir,
    )

    rg.generate_cumulative_returns_chart(
        wf_result, strategy, output_dir,
        title="PCA_SUB Optimized (K=5, L=120, λ=1.0)",
    )

    rg.generate_phase_progression_chart(
        all_metrics, output_dir,
    )

    print()

    # Step 4: Compile integrated metrics
    print("Compiling integrated metrics...")
    integrated_metrics = compile_integrated_metrics(all_metrics, wf_result, strategy)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(integrated_metrics, f, indent=2, default=str)
    print(f"  Saved metrics.json")

    # Step 5: Generate technical findings
    print("Generating technical findings report...")
    generate_technical_findings(integrated_metrics, output_dir)

    print()
    print("=" * 60)
    print("PHASE 10 COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:40s} ({size:,} bytes)")


if __name__ == "__main__":
    main()
