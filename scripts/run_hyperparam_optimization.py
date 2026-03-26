"""Phase 6: Hyperparameter optimization via nested walk-forward validation.

Searches over K (principal components), L (lookback window), and lambda_decay
(exponential decay rate) using a nested walk-forward scheme to avoid test-set
leakage in parameter selection.

Compares optimized results against the Phase 3 default-parameter baseline.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import DataPipeline
from src.evaluation.hyperparam_optimizer import HyperparamOptimizer
from src.evaluation.walk_forward import WalkForwardEvaluator


def main():
    print("=" * 70)
    print("Phase 6: Hyperparameter Optimization (Nested Walk-Forward)")
    print("=" * 70)
    print()

    # --- Load data ---
    print("Step 1: Loading aligned dataset...")
    pipeline = DataPipeline()
    dataset = pipeline.load()
    X_us = dataset.X_us
    Y_jp = dataset.Y_jp
    dates_us = dataset.dates_us
    print(f"  Dataset: {X_us.shape[0]} obs, {X_us.shape[1]} US sectors, {Y_jp.shape[1]} JP sectors")
    print()

    # --- Run nested walk-forward optimization ---
    print("Step 2: Running nested walk-forward hyperparameter optimization...")
    print()

    param_grid = {
        "K": [1, 2, 3, 4, 5],
        "L": [20, 40, 60, 80, 120],
        "lambda_decay": [0.8, 0.85, 0.9, 0.95, 1.0],
    }

    optimizer = HyperparamOptimizer(
        param_grid=param_grid,
        outer_train_window=504,   # ~2 years outer training
        outer_test_window=21,     # ~1 month outer test
        inner_train_window=252,   # ~1 year inner training
        inner_test_window=21,     # ~1 month inner test
        selection_metric="sharpe",
    )

    opt_result = optimizer.optimize(X_us, Y_jp, dates_us)
    print()
    print(f"  Optimization complete: {opt_result.n_outer_folds} outer folds")
    print(f"  OOS direction accuracy: {opt_result.mean_direction_accuracy:.4f} ± {opt_result.std_direction_accuracy:.4f}")
    print(f"  OOS Sharpe (gross): {opt_result.strategy_metrics.get('sharpe_ratio_gross', 'N/A')}")
    print()

    # --- Baseline comparison (default params K=3, L=60, lambda=0.9) ---
    print("Step 3: Running baseline with default parameters (K=3, L=60, λ=0.9)...")
    # Use same outer windows for fair comparison
    baseline_eval = WalkForwardEvaluator(
        train_window=504, test_window=21, K=3, L=60, lambda_decay=0.9
    )
    baseline_result = baseline_eval.evaluate(X_us, Y_jp, dates_us)
    baseline_strategy = baseline_eval.compute_oos_sharpe(baseline_result)

    print(f"  Baseline direction accuracy: {baseline_result.mean_direction_accuracy:.4f} ± {baseline_result.std_direction_accuracy:.4f}")
    print(f"  Baseline Sharpe (gross): {baseline_strategy.get('sharpe_ratio_gross', 'N/A')}")
    print()

    # --- Per-param analysis: average inner score by each param value ---
    print("Step 4: Analyzing parameter sensitivity...")
    param_scores = {
        "K": {k: [] for k in param_grid["K"]},
        "L": {l: [] for l in param_grid["L"]},
        "lambda_decay": {ld: [] for ld in param_grid["lambda_decay"]},
    }
    for fold in opt_result.outer_folds:
        for pr in fold.all_param_results:
            param_scores["K"][pr.K].append(pr.inner_sharpe)
            param_scores["L"][pr.L].append(pr.inner_sharpe)
            param_scores["lambda_decay"][pr.lambda_decay].append(pr.inner_sharpe)

    param_sensitivity = {}
    for param_name, val_scores in param_scores.items():
        sensitivity = {}
        for val, scores in val_scores.items():
            sensitivity[str(val)] = {
                "mean_inner_sharpe": round(float(np.mean(scores)), 4),
                "std_inner_sharpe": round(float(np.std(scores)), 4),
            }
        param_sensitivity[param_name] = sensitivity
        print(f"  {param_name}:")
        for val, stats in sensitivity.items():
            print(f"    {val}: inner Sharpe = {stats['mean_inner_sharpe']:.4f} ± {stats['std_inner_sharpe']:.4f}")
        print()

    # --- Param stability analysis ---
    print("Step 5: Analyzing parameter selection stability...")
    selected_params_over_time = []
    for fold in opt_result.outer_folds:
        selected_params_over_time.append({
            "fold_id": fold.fold_id,
            "test_period": f"{fold.test_start_date} to {fold.test_end_date}",
            "K": fold.best_K,
            "L": fold.best_L,
            "lambda_decay": fold.best_lambda_decay,
            "inner_sharpe": round(fold.best_inner_sharpe, 4),
            "oos_accuracy": round(fold.test_direction_accuracy, 4),
        })

    summary = opt_result.param_selection_summary
    print(f"  Most frequently selected: K={summary['most_frequent_K']}, "
          f"L={summary['most_frequent_L']}, λ={summary['most_frequent_lambda_decay']}")
    print(f"  K selection counts: {summary['K_selection_counts']}")
    print(f"  L selection counts: {summary['L_selection_counts']}")
    print(f"  λ selection counts: {summary['lambda_decay_selection_counts']}")
    print()

    # --- Save results ---
    print("Step 6: Saving results...")
    report_dir = Path("reports/cycle_6")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Remove daily_returns array from baseline strategy for JSON serialization
    baseline_strategy_clean = {
        k: v for k, v in baseline_strategy.items() if k != "daily_returns"
    }

    metrics = {
        "phase": 6,
        "description": "Hyperparameter optimization via nested walk-forward validation",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "total_aligned_observations": int(X_us.shape[0]),
            "date_range_start": str(dates_us[0].date()),
            "date_range_end": str(dates_us[-1].date()),
            "n_us_sectors": int(X_us.shape[1]),
            "n_jp_sectors": int(Y_jp.shape[1]),
        },
        "optimization_config": {
            "param_grid": param_grid,
            "outer_train_window": 504,
            "outer_test_window": 21,
            "inner_train_window": 252,
            "inner_test_window": 21,
            "selection_metric": "sharpe",
            "n_param_combinations": len(param_grid["K"]) * len(param_grid["L"]) * len(param_grid["lambda_decay"]),
        },
        "optimized_results": {
            "n_outer_folds": opt_result.n_outer_folds,
            "total_test_samples": opt_result.total_test_samples,
            "direction_accuracy": {
                "mean": round(opt_result.mean_direction_accuracy, 6),
                "std": round(opt_result.std_direction_accuracy, 6),
            },
            "correlation_mean": round(opt_result.mean_correlation, 6),
            "rmse_mean": round(opt_result.mean_rmse, 6),
            "strategy": opt_result.strategy_metrics,
        },
        "baseline_results": {
            "params": {"K": 3, "L": 60, "lambda_decay": 0.9},
            "n_folds": baseline_result.n_folds,
            "total_test_samples": baseline_result.total_test_samples,
            "direction_accuracy": {
                "mean": round(baseline_result.mean_direction_accuracy, 6),
                "std": round(baseline_result.std_direction_accuracy, 6),
            },
            "correlation_mean": round(baseline_result.mean_correlation, 6),
            "rmse_mean": round(baseline_result.mean_rmse, 6),
            "strategy": baseline_strategy_clean,
        },
        "param_selection_summary": summary,
        "param_sensitivity": param_sensitivity,
        "selected_params_per_fold": selected_params_over_time,
    }

    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved metrics to {report_dir / 'metrics.json'}")

    # --- Generate technical findings ---
    opt_sharpe = opt_result.strategy_metrics.get("sharpe_ratio_gross", "N/A")
    bsl_sharpe = baseline_strategy_clean.get("sharpe_ratio_gross", "N/A")
    opt_ret = opt_result.strategy_metrics.get("annualized_return_gross", "N/A")
    bsl_ret = baseline_strategy_clean.get("annualized_return_gross", "N/A")
    opt_dd = opt_result.strategy_metrics.get("max_drawdown", "N/A")
    bsl_dd = baseline_strategy_clean.get("max_drawdown", "N/A")

    # Param stability: compute how many unique combos were selected
    unique_combos = set()
    for fold in opt_result.outer_folds:
        unique_combos.add((fold.best_K, fold.best_L, fold.best_lambda_decay))
    n_unique = len(unique_combos)

    findings = f"""# Phase 6: Hyperparameter Optimization — Technical Findings

## Overview

Implemented nested walk-forward hyperparameter optimization to search for optimal
PCA_SUB parameters (K, L, λ) without test-set leakage. The nested scheme uses an
inner walk-forward loop within each outer training window to select the best
parameters, which are then evaluated on the outer test fold.

## Configuration

- **Parameter grid**: {len(param_grid['K']) * len(param_grid['L']) * len(param_grid['lambda_decay'])} combinations
  - K (principal components): {param_grid['K']}
  - L (lookback window): {param_grid['L']}
  - λ (decay rate): {param_grid['lambda_decay']}
- **Outer loop**: train={optimizer.outer_train_window}d (~2yr), test={optimizer.outer_test_window}d (~1mo)
- **Inner loop**: train={optimizer.inner_train_window}d (~1yr), test={optimizer.inner_test_window}d (~1mo)
- **Selection metric**: Inner Sharpe ratio (gross)
- **Total outer folds**: {opt_result.n_outer_folds}
- **Total OOS test days**: {opt_result.total_test_samples}

## Results: Optimized vs Baseline

| Metric | Optimized | Baseline (K=3, L=60, λ=0.9) |
|--------|-----------|------------------------------|
| Direction Accuracy | {opt_result.mean_direction_accuracy:.4f} ± {opt_result.std_direction_accuracy:.4f} | {baseline_result.mean_direction_accuracy:.4f} ± {baseline_result.std_direction_accuracy:.4f} |
| Sharpe Ratio (gross) | {opt_sharpe} | {bsl_sharpe} |
| Ann. Return (gross) | {opt_ret} | {bsl_ret} |
| Max Drawdown | {opt_dd} | {bsl_dd} |
| Correlation (mean) | {opt_result.mean_correlation:.4f} | {baseline_result.mean_correlation:.4f} |

## Parameter Selection Analysis

### Most Frequently Selected Parameters
- **K**: {summary['most_frequent_K']} (selected {summary['K_selection_counts'].get(summary['most_frequent_K'], 0)}/{opt_result.n_outer_folds} folds)
- **L**: {summary['most_frequent_L']} (selected {summary['L_selection_counts'].get(summary['most_frequent_L'], 0)}/{opt_result.n_outer_folds} folds)
- **λ**: {summary['most_frequent_lambda_decay']} (selected {summary['lambda_decay_selection_counts'].get(summary['most_frequent_lambda_decay'], 0)}/{opt_result.n_outer_folds} folds)

### Selection Counts
- K values: {summary['K_selection_counts']}
- L values: {summary['L_selection_counts']}
- λ values: {summary['lambda_decay_selection_counts']}

### Parameter Stability
- **Unique parameter combinations selected**: {n_unique} out of {opt_result.n_outer_folds} folds
- {"High stability: few unique combos relative to folds" if n_unique <= opt_result.n_outer_folds * 0.3 else "Moderate-to-low stability: many different combos selected across folds"}

## Parameter Sensitivity Analysis

Average inner-loop Sharpe by parameter value (averaged across all folds and other param settings):

### K (Number of Principal Components)
"""
    for val in param_grid["K"]:
        s = param_sensitivity["K"][str(val)]
        findings += f"- K={val}: inner Sharpe = {s['mean_inner_sharpe']:.4f} ± {s['std_inner_sharpe']:.4f}\n"

    findings += "\n### L (Lookback Window)\n"
    for val in param_grid["L"]:
        s = param_sensitivity["L"][str(val)]
        findings += f"- L={val}: inner Sharpe = {s['mean_inner_sharpe']:.4f} ± {s['std_inner_sharpe']:.4f}\n"

    findings += "\n### λ (Decay Rate)\n"
    for val in param_grid["lambda_decay"]:
        s = param_sensitivity["lambda_decay"][str(val)]
        findings += f"- λ={val}: inner Sharpe = {s['mean_inner_sharpe']:.4f} ± {s['std_inner_sharpe']:.4f}\n"

    findings += f"""
## Key Observations

1. **Optimization impact**: The nested walk-forward optimization {"improved" if (isinstance(opt_sharpe, (int, float)) and isinstance(bsl_sharpe, (int, float)) and opt_sharpe > bsl_sharpe) else "did not substantially improve"} the gross Sharpe ratio compared to the default parameters.

2. **Parameter stability**: {n_unique} unique parameter combinations were selected across {opt_result.n_outer_folds} folds, {"indicating that optimal parameters are relatively stable over time" if n_unique <= opt_result.n_outer_folds * 0.3 else "suggesting that optimal parameters shift over time — the lead-lag relationship is non-stationary"}.

3. **Overfitting risk**: The nested walk-forward design guards against overfitting by never using test data for parameter selection. The inner-loop Sharpe is used purely for ranking, and OOS performance is measured independently.

4. **Default parameters robustness**: The paper's default parameters (K=3, L=60, λ=0.9) provide a {"competitive" if abs(float(opt_sharpe) - float(bsl_sharpe)) < 0.2 else "suboptimal"} baseline, {"suggesting they are well-chosen for this dataset" if abs(float(opt_sharpe) - float(bsl_sharpe)) < 0.2 else "suggesting room for improvement through optimization"}.

## Implications for Strategy Design

- Dynamic parameter selection (re-optimizing periodically) {"may offer marginal improvement" if float(opt_sharpe) > float(bsl_sharpe) else "does not appear beneficial"} over fixed parameters for this dataset.
- The parameter sensitivity analysis can guide which parameters are most worth tuning vs. fixing.
- All results are gross of transaction costs — net performance evaluation is needed in Phase 4.
"""

    with open(report_dir / "technical_findings.md", "w") as f:
        f.write(findings)
    print(f"  Saved findings to {report_dir / 'technical_findings.md'}")

    print()
    print("=" * 70)
    print("Phase 6 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
