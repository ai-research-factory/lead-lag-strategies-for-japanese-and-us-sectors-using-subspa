"""Phase 5: Validate Cfull covariance estimation period.

Compares the paper's fixed-period covariance estimation (Cfull) approach
with the standard rolling covariance estimation used in Phase 3.

The paper uses Cfull=2010-2014 for PCA eigenvector estimation. Since our
data spans 2021-2026, we test multiple fixed window sizes (6mo, 1yr, 2yr)
to understand the impact of fixing the covariance estimation period.

For each Cfull window size, we:
1. Compute eigenvectors once from the first N observations (fixed Cfull)
2. Run walk-forward on the remaining data with fixed eigenvectors
3. Compare against rolling walk-forward on the same remaining data
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.pipeline import DataPipeline, JP_SECTOR_NAMES, US_SECTOR_NAMES
from src.evaluation.cfull_validator import CfullValidator


def main():
    print("=" * 70)
    print("Phase 5: Cfull Covariance Estimation Period Validation")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    pipeline = DataPipeline()
    dataset = pipeline.load()
    X_us = dataset.X_us
    Y_jp = dataset.Y_jp
    dates_us = dataset.dates_us
    print(f"Total aligned observations: {X_us.shape[0]}")
    print(f"Date range: {dates_us[0].date()} to {dates_us[-1].date()}")
    print()

    # Cfull windows to test:
    # 126 (~6 months), 252 (~1 year), 504 (~2 years)
    cfull_windows = [126, 252, 504]

    validator = CfullValidator(
        train_window=252,
        test_window=21,
        K=3,
        L=60,
        lambda_decay=0.9,
        cfull_windows=cfull_windows,
    )

    # Run full comparison
    results = validator.run_full_comparison(X_us, Y_jp, dates_us)

    # Analyze eigenvector stability across different Cfull windows
    print("\n" + "=" * 70)
    print("Eigenvector Stability Analysis")
    print("=" * 70)
    eigvec_analysis = analyze_eigenvector_stability(X_us, cfull_windows, K=3)

    # Build metrics dict
    metrics = build_metrics(results, eigvec_analysis, dataset)

    # Print summary
    print_summary(results, eigvec_analysis)

    # Save outputs
    reports_dir = project_root / "reports" / "cycle_5"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    findings_path = reports_dir / "technical_findings.md"
    write_technical_findings(findings_path, results, eigvec_analysis, dataset, metrics)
    print(f"Technical findings saved to {findings_path}")

    # Update open questions
    update_open_questions(project_root / "docs" / "open_questions.md")

    print("\nPhase 5 complete.")


def analyze_eigenvector_stability(
    X_us: np.ndarray, cfull_windows: list[int], K: int = 3
) -> dict:
    """Analyze how stable eigenvectors are across different estimation periods."""
    from src.models.pca_sub import PCASub

    analysis = {}

    # Compute eigenvectors for each Cfull window
    eigvecs_by_window = {}
    for cw in cfull_windows:
        if cw > X_us.shape[0]:
            continue
        ev = PCASub.compute_cfull_eigvecs(X_us[:cw], K=K)
        eigvecs_by_window[cw] = ev

    # Also compute rolling eigenvectors at different time points
    rolling_points = [252, 504, 756, 1000]
    rolling_eigvecs = {}
    for pt in rolling_points:
        if pt > X_us.shape[0]:
            continue
        start = max(0, pt - 60)  # L=60 lookback
        ev = PCASub.compute_cfull_eigvecs(X_us[start:pt], K=K)
        rolling_eigvecs[pt] = ev

    # Compare: principal angles between subspaces
    # Use absolute cosine similarity (eigenvectors can flip sign)
    pairwise_stability = {}
    windows = sorted(eigvecs_by_window.keys())
    for i, w1 in enumerate(windows):
        for w2 in windows[i + 1:]:
            ev1 = eigvecs_by_window[w1]
            ev2 = eigvecs_by_window[w2]
            # Cosine similarity matrix between PC sets
            cos_sim = np.abs(ev1.T @ ev2)  # (K, K)
            # Diagonal = alignment of corresponding PCs
            diag_sim = np.diag(cos_sim)
            # Best match for each PC (handles sign flips and reordering)
            best_match = cos_sim.max(axis=1)
            pairwise_stability[f"{w1}d_vs_{w2}d"] = {
                "diagonal_similarity": diag_sim.tolist(),
                "best_match_similarity": best_match.tolist(),
                "mean_best_match": float(best_match.mean()),
            }

    # Compare fixed vs rolling at different time points
    fixed_vs_rolling = {}
    for cw in windows:
        for pt in sorted(rolling_eigvecs.keys()):
            if pt <= cw:
                continue
            ev_fixed = eigvecs_by_window[cw]
            ev_rolling = rolling_eigvecs[pt]
            cos_sim = np.abs(ev_fixed.T @ ev_rolling)
            best_match = cos_sim.max(axis=1)
            fixed_vs_rolling[f"cfull_{cw}d_vs_rolling_at_{pt}"] = {
                "best_match_similarity": best_match.tolist(),
                "mean_best_match": float(best_match.mean()),
            }

    # Variance explained by Cfull eigenvectors at different time points
    variance_explained = {}
    for cw in windows:
        ev_fixed = eigvecs_by_window[cw]
        for pt in [252, 504, 756, 1000]:
            if pt > X_us.shape[0]:
                continue
            # Compute covariance at time point
            start = max(0, pt - 60)
            X_window = X_us[start:pt]
            cov = np.cov(X_window.T)
            total_var = np.trace(cov)
            # Variance captured by fixed eigenvectors
            projected_var = np.trace(ev_fixed.T @ cov @ ev_fixed)
            ratio = projected_var / total_var if total_var > 0 else 0
            variance_explained[f"cfull_{cw}d_at_t{pt}"] = round(float(ratio), 4)

    analysis["pairwise_stability"] = pairwise_stability
    analysis["fixed_vs_rolling"] = fixed_vs_rolling
    analysis["variance_explained"] = variance_explained

    return analysis


def build_metrics(results, eigvec_analysis, dataset) -> dict:
    """Build structured metrics dict for JSON output."""
    metrics = {
        "phase": 5,
        "description": "Cfull covariance estimation period validation",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "total_aligned_observations": int(dataset.X_us.shape[0]),
            "date_range_start": str(dataset.dates_us[0].date()),
            "date_range_end": str(dataset.dates_us[-1].date()),
            "n_us_sectors": int(dataset.X_us.shape[1]),
            "n_jp_sectors": int(dataset.Y_jp.shape[1]),
        },
        "comparison_results": {},
        "eigenvector_stability": eigvec_analysis,
    }

    for r in results:
        entry = {
            "method": r.method,
            "description": r.description,
            "cfull_window_days": r.cfull_window_days,
            "n_folds": r.wf_result.n_folds,
            "total_test_samples": r.wf_result.total_test_samples,
            "direction_accuracy": {
                "mean": round(r.wf_result.mean_direction_accuracy, 4),
                "std": round(r.wf_result.std_direction_accuracy, 4),
            },
            "correlation": {
                "mean": round(r.wf_result.mean_correlation, 4),
                "std": round(r.wf_result.std_correlation, 4),
            },
            "rmse": {
                "mean": round(r.wf_result.mean_rmse, 6),
                "std": round(r.wf_result.std_rmse, 6),
            },
            "positive_accuracy_folds_pct": round(r.wf_result.positive_accuracy_folds_pct, 2),
        }

        # Strategy metrics (exclude non-serializable daily_returns)
        strat = {k: v for k, v in r.strategy_metrics.items()
                 if k != "daily_returns"}
        entry["strategy"] = strat

        # Per-fold accuracy for temporal analysis
        entry["per_fold_accuracy"] = [
            round(f.direction_accuracy, 4) for f in r.wf_result.folds
        ]

        metrics["comparison_results"][r.method] = entry

    return metrics


def print_summary(results, eigvec_analysis):
    """Print comparison summary table."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Folds':>6} {'Acc':>8} {'Corr':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 70)

    for r in results:
        sr = r.strategy_metrics.get("sharpe_ratio_gross", 0)
        mdd = r.strategy_metrics.get("max_drawdown", 0)
        print(
            f"{r.method:<30} "
            f"{r.wf_result.n_folds:>6} "
            f"{r.wf_result.mean_direction_accuracy:>8.4f} "
            f"{r.wf_result.mean_correlation:>8.4f} "
            f"{sr:>8.4f} "
            f"{mdd:>8.4f}"
        )

    # Print eigenvector stability summary
    print("\n" + "-" * 70)
    print("EIGENVECTOR STABILITY (cosine similarity between Cfull windows)")
    print("-" * 70)
    for key, val in eigvec_analysis.get("pairwise_stability", {}).items():
        print(f"  {key}: mean best-match = {val['mean_best_match']:.4f}, "
              f"per-PC = {[f'{v:.3f}' for v in val['best_match_similarity']]}")

    print("\n" + "-" * 70)
    print("VARIANCE EXPLAINED by Cfull eigenvectors at different time points")
    print("-" * 70)
    for key, val in eigvec_analysis.get("variance_explained", {}).items():
        print(f"  {key}: {val:.4f}")


def write_technical_findings(
    filepath: Path, results, eigvec_analysis, dataset, metrics
):
    """Write technical findings report."""
    lines = []
    lines.append("# Phase 5: Cfull Covariance Estimation Period Validation")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("Validate the paper's fixed covariance estimation period (Cfull) approach ")
    lines.append("against rolling covariance estimation. The paper proposes computing PCA ")
    lines.append("eigenvectors once from a fixed historical period (2010-2014) and keeping ")
    lines.append("them constant throughout the walk-forward evaluation. This phase tests ")
    lines.append("whether fixing the subspace improves or degrades out-of-sample performance.")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("### Data")
    lines.append(f"- Total aligned observations: {dataset.X_us.shape[0]}")
    lines.append(f"- Date range: {dataset.dates_us[0].date()} to {dataset.dates_us[-1].date()}")
    lines.append(f"- U.S. sectors: {dataset.X_us.shape[1]}, JP sectors: {dataset.Y_jp.shape[1]}")
    lines.append("")

    lines.append("### Approach")
    lines.append("")
    lines.append("Since our data spans 2021-2026 (not the paper's 2010-2014), we adapt by:")
    lines.append("")
    lines.append("1. **Fixed Cfull windows**: Use the first N observations (N = 126, 252, 504 days) ")
    lines.append("   to compute PCA eigenvectors once, then run walk-forward on the remaining data ")
    lines.append("   with these fixed eigenvectors (only regression coefficients are re-estimated).")
    lines.append("2. **Matched rolling**: Run standard rolling walk-forward on the same post-Cfull ")
    lines.append("   data for fair comparison.")
    lines.append("3. **Eigenvector stability analysis**: Measure how PCA subspaces change over time ")
    lines.append("   using cosine similarity between eigenvectors.")
    lines.append("")

    lines.append("### Parameters")
    lines.append("- Train window: 252 days (~1 year)")
    lines.append("- Test window: 21 days (~1 month)")
    lines.append("- K=3 components, L=60 lookback, lambda=0.9 decay")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append("### Performance Comparison")
    lines.append("")
    lines.append("| Method | Folds | Direction Acc | Correlation | Sharpe (gross) | Max Drawdown |")
    lines.append("|--------|------:|:------------:|:-----------:|:--------------:|:------------:|")

    for r in results:
        sr = r.strategy_metrics.get("sharpe_ratio_gross", 0)
        mdd = r.strategy_metrics.get("max_drawdown", 0)
        ann_ret = r.strategy_metrics.get("annualized_return_gross", 0)
        lines.append(
            f"| {r.method} | {r.wf_result.n_folds} | "
            f"{r.wf_result.mean_direction_accuracy:.4f} +/- {r.wf_result.std_direction_accuracy:.4f} | "
            f"{r.wf_result.mean_correlation:.4f} | "
            f"{sr:.4f} | {mdd:.4f} |"
        )
    lines.append("")

    # Eigenvector stability
    lines.append("### Eigenvector Stability")
    lines.append("")
    lines.append("Cosine similarity between principal components from different Cfull windows:")
    lines.append("")

    for key, val in eigvec_analysis.get("pairwise_stability", {}).items():
        lines.append(f"- **{key}**: mean best-match similarity = {val['mean_best_match']:.4f}")
        for i, sim in enumerate(val["best_match_similarity"]):
            lines.append(f"  - PC{i+1}: {sim:.4f}")
    lines.append("")

    # Variance explained
    lines.append("### Variance Explained by Fixed Eigenvectors Over Time")
    lines.append("")
    lines.append("Fraction of total variance captured by Cfull eigenvectors at different points:")
    lines.append("")
    lines.append("| Cfull Window | At t=252 | At t=504 | At t=756 | At t=1000 |")
    lines.append("|-------------|:--------:|:--------:|:--------:|:---------:|")

    var_exp = eigvec_analysis.get("variance_explained", {})
    for cw in [126, 252, 504]:
        vals = []
        for pt in [252, 504, 756, 1000]:
            key = f"cfull_{cw}d_at_t{pt}"
            v = var_exp.get(key, "N/A")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(f"| {cw}d | {' | '.join(vals)} |")
    lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")

    # Determine best method
    rolling_results = [r for r in results if "rolling" in r.method and "matched" not in r.method]
    cfull_results = [r for r in results if r.method.startswith("cfull_")]

    if rolling_results and cfull_results:
        best_rolling_sharpe = max(
            r.strategy_metrics.get("sharpe_ratio_gross", 0) for r in rolling_results
        )
        best_cfull = max(cfull_results, key=lambda r: r.strategy_metrics.get("sharpe_ratio_gross", 0))
        best_cfull_sharpe = best_cfull.strategy_metrics.get("sharpe_ratio_gross", 0)

        if best_cfull_sharpe > best_rolling_sharpe:
            lines.append(f"The fixed Cfull approach ({best_cfull.method}) outperforms rolling "
                        f"covariance in Sharpe ratio ({best_cfull_sharpe:.4f} vs {best_rolling_sharpe:.4f}). "
                        f"This supports the paper's Cfull approach: the PCA subspace is stable enough "
                        f"that fixing it reduces estimation noise and improves performance.")
        else:
            lines.append(f"The rolling covariance approach outperforms or matches the fixed Cfull "
                        f"approach (Sharpe: {best_rolling_sharpe:.4f} vs best Cfull {best_cfull_sharpe:.4f}). "
                        f"This suggests that the PCA subspace shifts over time and benefits from "
                        f"re-estimation with recent data.")
    lines.append("")

    # Stability interpretation
    lines.append("### Subspace Stability Interpretation")
    lines.append("")
    pairwise = eigvec_analysis.get("pairwise_stability", {})
    mean_stabilities = [v["mean_best_match"] for v in pairwise.values()]
    if mean_stabilities:
        avg_stability = np.mean(mean_stabilities)
        if avg_stability > 0.9:
            lines.append(f"The average subspace similarity across Cfull windows is high ({avg_stability:.4f}), "
                        f"indicating that the U.S. sector covariance structure is relatively stable over "
                        f"our 5-year sample. The dominant factors (market, value/growth, etc.) persist.")
        elif avg_stability > 0.7:
            lines.append(f"The average subspace similarity ({avg_stability:.4f}) is moderate, "
                        f"suggesting the covariance structure evolves but retains some persistent features. "
                        f"The paper's Cfull approach works if the estimation period captures the main factors.")
        else:
            lines.append(f"The average subspace similarity is low ({avg_stability:.4f}), "
                        f"indicating significant structural change in U.S. sector dynamics over this period. "
                        f"A fixed Cfull approach may miss important regime changes.")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Cfull vs Rolling**: The comparison quantifies the trade-off between "
                "estimation stability (Cfull) and adaptability (rolling).")
    lines.append("2. **Subspace Stability**: The cosine similarity analysis reveals how much "
                "the dominant U.S. sector factors change over time.")
    lines.append("3. **Variance Decay**: The variance-explained metric shows how quickly "
                "fixed eigenvectors lose explanatory power as markets evolve.")
    lines.append("4. **Practical Implication**: For this dataset, the results inform whether "
                "a static or adaptive PCA approach is more appropriate for the lead-lag strategy.")
    lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def update_open_questions(filepath: Path):
    """Update open questions with Phase 5 resolution."""
    if not filepath.exists():
        return

    content = filepath.read_text()

    # Mark question 1 as resolved
    old = ("1. **Covariance estimation period (Cfull)**: The paper uses a fixed period "
           "(2010–2014) for covariance estimation. How does this compare to the rolling "
           "window approach implemented here? This is the focus of Phase 5.")
    new = ("1. ~~**Covariance estimation period (Cfull)**~~: **Resolved in Phase 5** — "
           "Compared fixed Cfull (126d, 252d, 504d) vs rolling covariance. "
           "See reports/cycle_5/ for detailed results and analysis.")

    if old in content:
        content = content.replace(old, new)
        filepath.write_text(content)
        print(f"\nUpdated {filepath}")


if __name__ == "__main__":
    main()
