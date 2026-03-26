"""Phase 9: Baseline model comparison.

Evaluates PCA_SUB against multiple baseline models using walk-forward
methodology to demonstrate the value of the subspace regularization approach.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import DataPipeline, JP_SECTOR_NAMES, US_SECTOR_NAMES
from src.models.pca_sub import PCASub
from src.models.baselines import (
    ZeroPredictor,
    HistoricalMeanPredictor,
    DirectOLS,
    DirectRidge,
    SimplePCA,
    SectorMomentum,
)
from src.evaluation.baseline_comparator import BaselineComparator


def create_pca_sub_wrapper(K=3, L=60, lambda_decay=0.9):
    """Create a PCASub instance with a name attribute for comparison."""
    model = PCASub(K=K, L=L, lambda_decay=lambda_decay)
    model.name = f"PCA_SUB (K={K}, L={L}, λ={lambda_decay})"
    return model


def run_comparison():
    print("=" * 70)
    print("Phase 9: Baseline Model Comparison")
    print("=" * 70)

    # Load data
    pipeline = DataPipeline()
    dataset = pipeline.load()

    X_us = dataset.X_us
    Y_jp = dataset.Y_jp
    dates_us = dataset.dates_us

    print(f"\nDataset: {X_us.shape[0]} aligned observations")
    print(f"  U.S. sectors: {X_us.shape[1]}")
    print(f"  JP sectors: {Y_jp.shape[1]}")
    print(f"  Date range: {dates_us[0].date()} to {dates_us[-1].date()}")

    # Define models to compare
    models = [
        create_pca_sub_wrapper(K=3, L=60, lambda_decay=0.9),    # Paper default
        create_pca_sub_wrapper(K=5, L=120, lambda_decay=1.0),   # Optimized (Phase 6)
        ZeroPredictor(),
        HistoricalMeanPredictor(),
        DirectOLS(),
        DirectRidge(alpha=1.0),
        DirectRidge(alpha=10.0),
        SimplePCA(K=3, L=60),
        SectorMomentum(),
    ]

    # Give Ridge variants distinct names
    models[5].name = "Ridge (α=1.0)"
    models[6].name = "Ridge (α=10.0)"

    # Run walk-forward comparison
    comparator = BaselineComparator(train_window=252, test_window=21)

    results = {}
    for model in models:
        name = getattr(model, "name", model.__class__.__name__)
        print(f"\nEvaluating: {name}")
        result = comparator.evaluate_model(model, X_us, Y_jp, dates_us)
        results[name] = result
        wf = result.wf_result
        strat = result.strategy_metrics
        print(f"  Direction accuracy: {wf.mean_direction_accuracy:.4f} ± {wf.std_direction_accuracy:.4f}")
        print(f"  Correlation:        {wf.mean_correlation:.4f} ± {wf.std_correlation:.4f}")
        print(f"  RMSE:               {wf.mean_rmse:.6f} ± {wf.std_rmse:.6f}")
        if strat:
            print(f"  Sharpe (gross):     {strat['sharpe_ratio_gross']:.4f}")
            print(f"  Sharpe (net):       {strat['sharpe_ratio_net']:.4f}")
            print(f"  Ann. Return (net):  {strat['annualized_return_net']:.4%}")
            print(f"  Max Drawdown:       {strat['max_drawdown_net']:.4%}")

    # Statistical tests: PCA_SUB default vs each baseline
    print("\n" + "=" * 70)
    print("Statistical Significance Tests (PCA_SUB default vs baselines)")
    print("=" * 70)

    pca_default_name = "PCA_SUB (K=3, L=60, λ=0.9)"
    pca_default = results[pca_default_name]
    pca_daily = pca_default.strategy_metrics.get("_daily_returns", None)

    # Reconstruct daily returns for statistical tests
    def get_daily_returns(result):
        wf = result.wf_result
        Y_pred = wf.all_predictions
        Y_actual = wf.all_actuals
        if Y_pred is None or Y_actual is None:
            return np.array([])
        positions = np.sign(Y_pred)
        n_active = np.abs(positions).sum(axis=1, keepdims=True)
        n_active = np.where(n_active == 0, 1, n_active)
        weights = positions / n_active
        return (weights * Y_actual).sum(axis=1)

    pca_daily = get_daily_returns(pca_default)
    stat_tests = {}

    for name, result in results.items():
        if name == pca_default_name:
            continue
        baseline_daily = get_daily_returns(result)
        if len(pca_daily) == len(baseline_daily) and len(pca_daily) > 0:
            test = comparator.paired_sharpe_test(pca_daily, baseline_daily)
            stat_tests[name] = test
            sig = "***" if test["significant_5pct"] else ""
            print(f"  vs {name}: t={test['t_stat']:.3f}, p={test['p_value']:.4f} {sig}")

    # Per-sector accuracy comparison
    print("\n" + "=" * 70)
    print("Per-Sector Direction Accuracy (aggregated across all folds)")
    print("=" * 70)

    jp_tickers = dataset.jp_tickers
    sector_accuracy = {}
    for name, result in results.items():
        wf = result.wf_result
        if wf.all_predictions is not None:
            signs_true = np.sign(wf.all_actuals)
            signs_pred = np.sign(wf.all_predictions)
            correct = (signs_true == signs_pred)
            sector_accuracy[name] = correct.mean(axis=0)

    # Print header
    header = f"{'Sector':<30}"
    short_names = {}
    for i, (name, _) in enumerate(results.items()):
        short = name[:12]
        short_names[name] = short
        header += f" {short:>12}"
    print(header)
    print("-" * len(header))

    for j, ticker in enumerate(jp_tickers):
        sector_name = JP_SECTOR_NAMES.get(ticker, ticker)
        row = f"{sector_name:<30}"
        for name in results:
            if name in sector_accuracy:
                row += f" {sector_accuracy[name][j]:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Build metrics output
    print("\n\nSaving results...")

    report_dir = Path("reports/cycle_9")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison summary
    comparison_table = []
    for name, result in results.items():
        wf = result.wf_result
        strat = result.strategy_metrics
        entry = {
            "model": name,
            "direction_accuracy_mean": round(wf.mean_direction_accuracy, 6),
            "direction_accuracy_std": round(wf.std_direction_accuracy, 6),
            "correlation_mean": round(wf.mean_correlation, 6),
            "correlation_std": round(wf.std_correlation, 6),
            "rmse_mean": round(wf.mean_rmse, 8),
            "rmse_std": round(wf.std_rmse, 8),
            "positive_accuracy_folds_pct": round(wf.positive_accuracy_folds_pct, 2),
            "n_folds": wf.n_folds,
            "total_test_samples": wf.total_test_samples,
        }
        if strat:
            entry.update({
                "sharpe_ratio_gross": strat["sharpe_ratio_gross"],
                "sharpe_ratio_net": strat["sharpe_ratio_net"],
                "annualized_return_gross": strat["annualized_return_gross"],
                "annualized_return_net": strat["annualized_return_net"],
                "annualized_volatility": strat["annualized_volatility"],
                "max_drawdown_gross": strat["max_drawdown_gross"],
                "max_drawdown_net": strat["max_drawdown_net"],
                "total_return_gross": strat["total_return_gross"],
                "total_return_net": strat["total_return_net"],
                "pct_positive_days": strat["pct_positive_days"],
                "mean_daily_turnover": strat["mean_daily_turnover"],
            })
        comparison_table.append(entry)

    # Per-sector breakdown for key models
    per_sector_breakdown = {}
    for name in results:
        if name in sector_accuracy:
            per_sector_breakdown[name] = {
                jp_tickers[j]: round(float(sector_accuracy[name][j]), 6)
                for j in range(len(jp_tickers))
            }

    metrics = {
        "phase": 9,
        "description": "Baseline model comparison - PCA_SUB vs simpler alternatives",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "total_aligned_observations": int(X_us.shape[0]),
            "date_range_start": str(dates_us[0].date()),
            "date_range_end": str(dates_us[-1].date()),
            "n_us_sectors": int(X_us.shape[1]),
            "n_jp_sectors": int(Y_jp.shape[1]),
        },
        "walk_forward_config": {
            "train_window": 252,
            "test_window": 21,
        },
        "models_evaluated": [m.name if hasattr(m, "name") else m.__class__.__name__ for m in models],
        "comparison_table": comparison_table,
        "statistical_tests_vs_pca_sub_default": stat_tests,
        "per_sector_direction_accuracy": per_sector_breakdown,
    }

    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved metrics to {report_dir / 'metrics.json'}")

    # Generate technical findings markdown
    generate_findings(report_dir, metrics, results, jp_tickers, stat_tests)
    print(f"  Saved findings to {report_dir / 'technical_findings.md'}")

    print("\nPhase 9 complete.")


def generate_findings(report_dir, metrics, results, jp_tickers, stat_tests):
    """Generate the technical findings markdown report."""

    comparison = metrics["comparison_table"]

    # Build comparison table
    table_header = "| Model | Dir. Acc. | Correlation | RMSE | Sharpe (gross) | Sharpe (net) | Ann. Return (net) | Max DD (net) |"
    table_sep = "|-------|-----------|-------------|------|----------------|--------------|-------------------|-------------|"
    table_rows = []
    for entry in comparison:
        name = entry["model"]
        acc = f"{entry['direction_accuracy_mean']:.4f}"
        corr = f"{entry['correlation_mean']:.4f}"
        rmse = f"{entry['rmse_mean']:.6f}"
        sharpe_g = f"{entry.get('sharpe_ratio_gross', 'N/A')}"
        sharpe_n = f"{entry.get('sharpe_ratio_net', 'N/A')}"
        ann_ret = f"{entry.get('annualized_return_net', 0):.4%}" if 'annualized_return_net' in entry else "N/A"
        max_dd = f"{entry.get('max_drawdown_net', 0):.4%}" if 'max_drawdown_net' in entry else "N/A"
        table_rows.append(f"| {name} | {acc} | {corr} | {rmse} | {sharpe_g} | {sharpe_n} | {ann_ret} | {max_dd} |")

    comparison_table_md = "\n".join([table_header, table_sep] + table_rows)

    # Build stat test table
    stat_rows = []
    for name, test in stat_tests.items():
        sig = "Yes" if test["significant_5pct"] else "No"
        stat_rows.append(f"| {name} | {test['t_stat']:.4f} | {test['p_value']:.6f} | {sig} |")
    stat_table_md = "\n".join([
        "| Baseline | t-statistic | p-value | Significant (5%) |",
        "|----------|-------------|---------|------------------|",
    ] + stat_rows) if stat_rows else "No statistical tests performed."

    # Per-sector table for top models
    per_sector_md = ""
    key_models = [c["model"] for c in comparison[:4]]  # PCA_SUB variants + zero predictor + hist mean
    per_sector_header = f"| Sector |" + " | ".join(f" {m[:20]} " for m in key_models) + " |"
    per_sector_sep = "|--------|" + " | ".join(["---" for _ in key_models]) + " |"
    per_sector_rows = []
    ps_data = metrics["per_sector_direction_accuracy"]
    for j, ticker in enumerate(jp_tickers):
        sector_name = JP_SECTOR_NAMES.get(ticker, ticker)
        vals = []
        for m in key_models:
            if m in ps_data and ticker in ps_data[m]:
                vals.append(f"{ps_data[m][ticker]:.4f}")
            else:
                vals.append("N/A")
        per_sector_rows.append(f"| {sector_name} | " + " | ".join(vals) + " |")
    per_sector_md = "\n".join([per_sector_header, per_sector_sep] + per_sector_rows)

    # Find best model
    sharpe_models = [(c["model"], c.get("sharpe_ratio_net", -999)) for c in comparison]
    best_model = max(sharpe_models, key=lambda x: x[1])

    # Determine PCA_SUB improvement over baselines
    pca_default = next(c for c in comparison if "K=3" in c["model"])
    baseline_sharpes = [(c["model"], c.get("sharpe_ratio_net", 0)) for c in comparison
                        if "PCA_SUB" not in c["model"]]
    best_baseline = max(baseline_sharpes, key=lambda x: x[1]) if baseline_sharpes else ("N/A", 0)

    findings_md = f"""# Phase 9: Baseline Model Comparison

## Overview

This analysis compares the PCA_SUB model against multiple baseline models using
identical walk-forward evaluation methodology. The goal is to demonstrate whether
the subspace regularization PCA approach provides genuine predictive value over
simpler alternatives for the U.S.-to-Japan sector lead-lag strategy.

## Data Summary

- **Aligned observations**: {metrics['data']['total_aligned_observations']}
- **Date range**: {metrics['data']['date_range_start']} to {metrics['data']['date_range_end']}
- **U.S. sectors**: {metrics['data']['n_us_sectors']} (Select Sector SPDR ETFs)
- **JP sectors**: {metrics['data']['n_jp_sectors']} (TOPIX-17 ETFs)
- **Walk-forward**: train={metrics['walk_forward_config']['train_window']}d, test={metrics['walk_forward_config']['test_window']}d

## Models Evaluated

1. **PCA_SUB (K=3, L=60, λ=0.9)**: Paper's default configuration
2. **PCA_SUB (K=5, L=120, λ=1.0)**: Optimized parameters from Phase 6
3. **Zero Predictor**: Always predicts zero return (random walk hypothesis)
4. **Historical Mean**: Predicts training-window average return per sector
5. **Direct OLS**: Direct 11→17 regression without PCA dimensionality reduction
6. **Ridge (α=1.0)**: L2-regularized direct regression (mild regularization)
7. **Ridge (α=10.0)**: L2-regularized direct regression (stronger regularization)
8. **Simple PCA (no decay)**: Standard PCA + OLS without exponential decay weighting
9. **Equal-Weight Market Signal**: Average U.S. return as single predictor

## Comparison Results

{comparison_table_md}

## Statistical Significance (PCA_SUB default vs baselines)

Paired t-test on daily strategy return differences:

{stat_table_md}

## Per-Sector Direction Accuracy

{per_sector_md}

## Key Findings

### 1. Overall Model Ranking
- **Best model by net Sharpe**: {best_model[0]} ({best_model[1]:.4f})
- **Best non-PCA_SUB baseline**: {best_baseline[0]} ({best_baseline[1]:.4f})
- **PCA_SUB default net Sharpe**: {pca_default.get('sharpe_ratio_net', 'N/A')}

### 2. Value of PCA Dimensionality Reduction
Comparing PCA_SUB against Direct OLS tests whether reducing U.S. returns to
principal components improves cross-market prediction. The direction accuracy
and Sharpe ratio differences quantify this.

### 3. Value of Decay Weighting
Comparing PCA_SUB (with decay) against Simple PCA (no decay) isolates the
contribution of exponential decay weighting in covariance estimation.

### 4. Value of Cross-Market Signal
Comparing any regression-based model against the Zero Predictor establishes
whether U.S. sector returns contain any predictive information for Japanese
sector returns at all.

### 5. Regularization Approach
Comparing PCA_SUB against Ridge regression tests whether PCA-based
dimensionality reduction is superior to standard L2 regularization for
handling collinearity in U.S. sector returns.

## Recommendations

1. Use net-of-cost Sharpe ratios for all model selection decisions
2. Consider ensemble approaches combining top-performing models
3. The statistical significance tests indicate whether observed differences
   are reliable or could be due to sampling variation
4. Per-sector accuracy reveals which sectors benefit most from the lead-lag signal
"""

    with open(report_dir / "technical_findings.md", "w") as f:
        f.write(findings_md)


if __name__ == "__main__":
    run_comparison()
