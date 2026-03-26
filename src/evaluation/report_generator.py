"""Final report generator with integrated visualizations.

Aggregates results from all phases (1-9) and generates:
1. Comprehensive matplotlib charts saved as PNG files
2. Integrated metrics.json
3. Technical findings summary
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# Consistent styling
COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "tertiary": "#059669",
    "quaternary": "#D97706",
    "gray": "#6B7280",
    "light_blue": "#93C5FD",
    "light_red": "#FCA5A5",
    "light_green": "#6EE7B7",
}
MODEL_COLORS = {
    "PCA_SUB (K=3)": "#2563EB",
    "PCA_SUB (K=5)": "#DC2626",
    "Direct OLS": "#059669",
    "Ridge (α=1.0)": "#D97706",
    "Ridge (α=10.0)": "#7C3AED",
    "Simple PCA": "#DB2777",
    "Historical Mean": "#6B7280",
    "Equal-Weight": "#0891B2",
    "Zero Predictor": "#9CA3AF",
}


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    })


def generate_model_comparison_chart(cycle9_metrics: dict, output_dir: Path):
    """Bar chart comparing all models on key metrics."""
    _setup_style()

    table = cycle9_metrics["comparison_table"]
    models = [m["model"] for m in table]
    short_names = []
    for m in models:
        if "K=3" in m:
            short_names.append("PCA_SUB\n(K=3)")
        elif "K=5" in m:
            short_names.append("PCA_SUB\n(K=5, opt)")
        elif "Zero" in m:
            short_names.append("Zero\n(RW)")
        elif "Historical" in m:
            short_names.append("Hist.\nMean")
        elif "Direct OLS" in m:
            short_names.append("Direct\nOLS")
        elif "α=1.0" in m:
            short_names.append("Ridge\n(α=1)")
        elif "α=10.0" in m:
            short_names.append("Ridge\n(α=10)")
        elif "Simple PCA" in m:
            short_names.append("Simple\nPCA")
        elif "Equal" in m:
            short_names.append("Equal\nWeight")
        else:
            short_names.append(m[:10])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison: PCA_SUB vs Baselines (Phase 9)", fontsize=14, fontweight="bold")

    x = np.arange(len(models))
    width = 0.6

    # 1. Sharpe Ratio (gross vs net)
    ax = axes[0, 0]
    gross_sharpe = [m["sharpe_ratio_gross"] for m in table]
    net_sharpe = [m["sharpe_ratio_net"] for m in table]
    bars1 = ax.bar(x - 0.15, gross_sharpe, 0.3, label="Gross", color=COLORS["primary"], alpha=0.8)
    bars2 = ax.bar(x + 0.15, net_sharpe, 0.3, label="Net (10bps)", color=COLORS["secondary"], alpha=0.8)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio (Gross vs Net)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7)
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # 2. Direction Accuracy
    ax = axes[0, 1]
    accs = [m["direction_accuracy_mean"] * 100 for m in table]
    colors = [COLORS["tertiary"] if a > 50 else COLORS["gray"] for a in accs]
    ax.bar(x, accs, width, color=colors, alpha=0.8)
    ax.axhline(y=50, color=COLORS["secondary"], linestyle="--", linewidth=1, label="50% baseline")
    ax.set_ylabel("Direction Accuracy (%)")
    ax.set_title("Direction Accuracy (Mean Across Folds)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7)
    ax.set_ylim(0, 55)
    ax.legend(fontsize=8)

    # 3. Total Return (gross vs net)
    ax = axes[1, 0]
    gross_ret = [m["total_return_gross"] * 100 for m in table]
    net_ret = [m["total_return_net"] * 100 for m in table]
    ax.bar(x - 0.15, gross_ret, 0.3, label="Gross", color=COLORS["primary"], alpha=0.8)
    ax.bar(x + 0.15, net_ret, 0.3, label="Net (10bps)", color=COLORS["secondary"], alpha=0.8)
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Cumulative Return Over Test Period")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7)
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # 4. Daily Turnover
    ax = axes[1, 1]
    turnover = [m["mean_daily_turnover"] * 100 for m in table]
    ax.bar(x, turnover, width, color=COLORS["quaternary"], alpha=0.8)
    ax.set_ylabel("Mean Daily Turnover (%)")
    ax.set_title("Daily Turnover (Driver of Net Costs)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved model_comparison.png")


def generate_sector_accuracy_chart(cycle9_metrics: dict, jp_sector_names: dict, output_dir: Path):
    """Per-sector direction accuracy heatmap for key models."""
    _setup_style()

    per_sector = cycle9_metrics["per_sector_direction_accuracy"]
    models_to_show = [
        "PCA_SUB (K=3, L=60, λ=0.9)",
        "PCA_SUB (K=5, L=120, λ=1.0)",
        "Direct OLS",
        "Ridge (α=1.0)",
        "Simple PCA (no decay)",
        "Historical Mean",
    ]
    models_to_show = [m for m in models_to_show if m in per_sector]

    tickers = list(per_sector[models_to_show[0]].keys())
    sector_labels = [jp_sector_names.get(t, t) for t in tickers]

    data = np.array([[per_sector[m][t] * 100 for t in tickers] for m in models_to_show])

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=46, vmax=56)
    ax.set_xticks(np.arange(len(tickers)))
    ax.set_xticklabels(sector_labels, rotation=45, ha="right", fontsize=8)
    short_model_names = []
    for m in models_to_show:
        if "K=3" in m:
            short_model_names.append("PCA_SUB (K=3)")
        elif "K=5" in m:
            short_model_names.append("PCA_SUB (K=5, opt)")
        elif "Direct" in m:
            short_model_names.append("Direct OLS")
        elif "α=1.0" in m:
            short_model_names.append("Ridge (α=1)")
        elif "Simple" in m:
            short_model_names.append("Simple PCA")
        else:
            short_model_names.append(m[:15])
    ax.set_yticks(np.arange(len(models_to_show)))
    ax.set_yticklabels(short_model_names, fontsize=9)

    # Add text annotations
    for i in range(len(models_to_show)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] < 48 or data[i, j] > 54 else "black")

    plt.colorbar(im, ax=ax, label="Direction Accuracy (%)", shrink=0.8)
    ax.set_title("Per-Sector Direction Accuracy by Model (%)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "sector_accuracy_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved sector_accuracy_heatmap.png")


def generate_walk_forward_timeline(cycle4_metrics: dict, output_dir: Path):
    """Walk-forward fold accuracy over time."""
    _setup_style()

    folds = cycle4_metrics["per_fold"]
    dates = [f["test_period"].split(" to ")[0] if "test_period" in f
             else f.get("train_period", "").split(" to ")[-1] for f in folds]
    accs = [f["direction_accuracy"] * 100 for f in folds]
    corrs = [f["mean_correlation"] for f in folds]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Walk-Forward Performance Over Time (Baseline K=3, L=60, λ=0.9)", fontsize=13, fontweight="bold")

    # Accuracy
    ax1.plot(range(len(accs)), accs, "o-", color=COLORS["primary"], markersize=4, linewidth=1.2, label="Fold Accuracy")
    # Rolling 6-fold average
    if len(accs) >= 6:
        rolling = pd.Series(accs).rolling(6).mean()
        ax1.plot(range(len(accs)), rolling, "-", color=COLORS["secondary"], linewidth=2, label="6-Fold Rolling Avg")
    ax1.axhline(y=50, color=COLORS["gray"], linestyle="--", linewidth=1, alpha=0.7, label="50% baseline")
    ax1.set_ylabel("Direction Accuracy (%)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(38, 60)

    # Show fold dates on x-axis
    tick_positions = list(range(0, len(folds), 6))
    tick_labels = [dates[i] if i < len(dates) else "" for i in tick_positions]

    # Correlation
    ax2.bar(range(len(corrs)), corrs, color=[COLORS["tertiary"] if c > 0 else COLORS["secondary"] for c in corrs], alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("Mean Correlation")
    ax2.set_xlabel("Walk-Forward Fold")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "walk_forward_timeline.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved walk_forward_timeline.png")


def generate_parameter_sensitivity_chart(cycle6_metrics: dict, output_dir: Path):
    """Parameter sensitivity from hyperparameter optimization."""
    _setup_style()

    sensitivity = cycle6_metrics["param_sensitivity"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Parameter Sensitivity: Inner-Loop Sharpe Ratio (Phase 6)", fontsize=13, fontweight="bold")

    # K
    ax = axes[0]
    k_vals = sorted(sensitivity["K"].keys(), key=int)
    k_sharpes = [sensitivity["K"][k]["mean_inner_sharpe"] for k in k_vals]
    k_stds = [sensitivity["K"][k]["std_inner_sharpe"] for k in k_vals]
    ax.errorbar([int(k) for k in k_vals], k_sharpes, yerr=k_stds, fmt="o-",
                color=COLORS["primary"], capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel("K (Number of PCs)")
    ax.set_ylabel("Mean Inner Sharpe")
    ax.set_title("K Sensitivity")

    # L
    ax = axes[1]
    l_vals = sorted(sensitivity["L"].keys(), key=int)
    l_sharpes = [sensitivity["L"][lv]["mean_inner_sharpe"] for lv in l_vals]
    l_stds = [sensitivity["L"][lv]["std_inner_sharpe"] for lv in l_vals]
    ax.errorbar([int(lv) for lv in l_vals], l_sharpes, yerr=l_stds, fmt="s-",
                color=COLORS["secondary"], capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel("L (Lookback Window)")
    ax.set_ylabel("Mean Inner Sharpe")
    ax.set_title("L Sensitivity")

    # Lambda
    ax = axes[2]
    lam_vals = sorted(sensitivity["lambda_decay"].keys(), key=float)
    lam_sharpes = [sensitivity["lambda_decay"][lv]["mean_inner_sharpe"] for lv in lam_vals]
    lam_stds = [sensitivity["lambda_decay"][lv]["std_inner_sharpe"] for lv in lam_vals]
    ax.errorbar([float(lv) for lv in lam_vals], lam_sharpes, yerr=lam_stds, fmt="D-",
                color=COLORS["tertiary"], capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel("λ (Decay Rate)")
    ax.set_ylabel("Mean Inner Sharpe")
    ax.set_title("λ Sensitivity")

    plt.tight_layout()
    fig.savefig(output_dir / "parameter_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved parameter_sensitivity.png")


def generate_cfull_comparison_chart(cycle5_metrics: dict, output_dir: Path):
    """Cfull vs rolling covariance comparison."""
    _setup_style()

    results = cycle5_metrics["comparison_results"]
    methods = ["rolling", "cfull_126d", "cfull_252d", "cfull_504d"]
    labels = ["Rolling", "Cfull 126d", "Cfull 252d", "Cfull 504d"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Covariance Estimation: Rolling vs Fixed Period (Phase 5)", fontsize=13, fontweight="bold")

    x = np.arange(len(methods))
    colors = [COLORS["primary"], COLORS["quaternary"], COLORS["tertiary"], COLORS["secondary"]]

    # Direction Accuracy
    ax = axes[0]
    accs = [results[m]["direction_accuracy"]["mean"] * 100 for m in methods]
    ax.bar(x, accs, color=colors, alpha=0.8)
    ax.axhline(y=50, color="black", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Direction Accuracy (%)")
    ax.set_title("Direction Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(49, 51)

    # Sharpe Ratio
    ax = axes[1]
    sharpes = [results[m]["strategy"]["sharpe_ratio_gross"] for m in methods]
    ax.bar(x, sharpes, color=colors, alpha=0.8)
    ax.set_ylabel("Sharpe Ratio (Gross)")
    ax.set_title("Sharpe Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    # Fold-level accuracy distribution
    ax = axes[2]
    for i, m in enumerate(methods):
        fold_accs = [a * 100 for a in results[m]["per_fold_accuracy"]]
        ax.boxplot(fold_accs, positions=[i], widths=0.5,
                   patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.5))
    ax.axhline(y=50, color="black", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Direction Accuracy (%)")
    ax.set_title("Fold Accuracy Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "cfull_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved cfull_comparison.png")


def generate_optimization_evolution_chart(cycle6_metrics: dict, output_dir: Path):
    """Selected parameter evolution over walk-forward folds."""
    _setup_style()

    folds = cycle6_metrics["selected_params_per_fold"]
    fold_ids = [f["fold_id"] for f in folds]
    ks = [f["K"] for f in folds]
    ls = [f["L"] for f in folds]
    lams = [f["lambda_decay"] for f in folds]
    oos_accs = [f["oos_accuracy"] * 100 for f in folds]
    test_periods = [f["test_period"].split(" to ")[0] for f in folds]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Nested Walk-Forward: Selected Parameters Over Time (Phase 6)", fontsize=13, fontweight="bold")

    axes[0].step(fold_ids, ks, where="mid", color=COLORS["primary"], linewidth=2)
    axes[0].set_ylabel("K")
    axes[0].set_title("Selected K (Number of PCs)")
    axes[0].set_yticks([1, 2, 3, 4, 5])

    axes[1].step(fold_ids, ls, where="mid", color=COLORS["secondary"], linewidth=2)
    axes[1].set_ylabel("L")
    axes[1].set_title("Selected L (Lookback Window)")

    axes[2].step(fold_ids, lams, where="mid", color=COLORS["tertiary"], linewidth=2)
    axes[2].set_ylabel("λ")
    axes[2].set_title("Selected λ (Decay Rate)")

    axes[3].bar(fold_ids, oos_accs,
                color=[COLORS["tertiary"] if a > 50 else COLORS["secondary"] for a in oos_accs], alpha=0.7)
    axes[3].axhline(y=50, color="black", linestyle="--", linewidth=0.5)
    axes[3].set_ylabel("OOS Accuracy (%)")
    axes[3].set_title("Out-of-Sample Direction Accuracy")
    axes[3].set_xlabel("Fold")

    tick_positions = list(range(0, len(fold_ids), 5))
    tick_labels = [test_periods[i] if i < len(test_periods) else "" for i in tick_positions]
    axes[3].set_xticks(tick_positions)
    axes[3].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "optimization_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved optimization_evolution.png")


def generate_cumulative_returns_chart(
    wf_result,
    strategy_metrics: dict,
    output_dir: Path,
    title: str = "PCA_SUB Optimized (K=5, L=120, λ=1.0)",
):
    """Cumulative return curve from walk-forward results."""
    _setup_style()

    Y_pred = wf_result.all_predictions
    Y_actual = wf_result.all_actuals
    dates = wf_result.all_dates_us

    if Y_pred is None or Y_actual is None:
        print("  Skipped cumulative_returns.png (no data)")
        return

    # Compute daily strategy returns
    positions = np.sign(Y_pred)
    n_active = np.abs(positions).sum(axis=1, keepdims=True)
    n_active = np.where(n_active == 0, 1, n_active)
    weights = positions / n_active
    daily_returns = (weights * Y_actual).sum(axis=1)

    # Compute turnover
    n_days = len(daily_returns)
    turnover_daily = np.zeros(n_days)
    if n_days > 1:
        tc = np.abs(np.diff(positions, axis=0)).sum(axis=1) / positions.shape[1]
        turnover_daily[1:] = tc
    cost_per_trade = 0.0010
    net_daily_returns = daily_returns - turnover_daily * cost_per_trade

    cum_gross = np.cumprod(1 + daily_returns)
    cum_net = np.cumprod(1 + net_daily_returns)

    # JP equal-weight buy-and-hold benchmark
    jp_market = Y_actual.mean(axis=1)
    cum_benchmark = np.cumprod(1 + jp_market)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Cumulative Returns: {title}", fontsize=13, fontweight="bold")

    x = range(len(daily_returns))
    if dates is not None:
        x_dates = pd.DatetimeIndex(dates)
        ax1.plot(x_dates, cum_gross, color=COLORS["primary"], linewidth=1.5, label="Strategy (Gross)")
        ax1.plot(x_dates, cum_net, color=COLORS["secondary"], linewidth=1.5, label="Strategy (Net 10bps)")
        ax1.plot(x_dates, cum_benchmark, color=COLORS["gray"], linewidth=1, linestyle="--", label="JP EW Benchmark")
    else:
        ax1.plot(x, cum_gross, color=COLORS["primary"], linewidth=1.5, label="Strategy (Gross)")
        ax1.plot(x, cum_net, color=COLORS["secondary"], linewidth=1.5, label="Strategy (Net 10bps)")
        ax1.plot(x, cum_benchmark, color=COLORS["gray"], linewidth=1, linestyle="--", label="JP EW Benchmark")

    ax1.set_ylabel("Cumulative Return")
    ax1.legend(fontsize=9)
    ax1.axhline(y=1, color="black", linewidth=0.3)

    # Drawdown
    dd_gross = cum_gross / np.maximum.accumulate(cum_gross) - 1
    dd_net = cum_net / np.maximum.accumulate(cum_net) - 1
    if dates is not None:
        ax2.fill_between(x_dates, dd_gross * 100, 0, color=COLORS["primary"], alpha=0.3, label="Gross DD")
        ax2.fill_between(x_dates, dd_net * 100, 0, color=COLORS["secondary"], alpha=0.3, label="Net DD")
    else:
        ax2.fill_between(x, dd_gross * 100, 0, color=COLORS["primary"], alpha=0.3, label="Gross DD")
        ax2.fill_between(x, dd_net * 100, 0, color=COLORS["secondary"], alpha=0.3, label="Net DD")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved cumulative_returns.png")


def generate_phase_progression_chart(all_metrics: dict, output_dir: Path):
    """Summary chart showing how key metrics evolved across phases."""
    _setup_style()

    phases = []
    sharpes = []
    accuracies = []
    labels = []

    # Phase 3/4: Baseline walk-forward
    if "cycle_4" in all_metrics:
        phases.append(3)
        sharpes.append(all_metrics["cycle_4"]["strategy_gross"]["sharpe_ratio"])
        accuracies.append(all_metrics["cycle_4"]["results"]["direction_accuracy_mean"] * 100)
        labels.append("Phase 3\nBaseline WF")

    # Phase 5: Best Cfull
    if "cycle_5" in all_metrics:
        phases.append(5)
        sharpes.append(all_metrics["cycle_5"]["comparison_results"]["cfull_504d"]["strategy"]["sharpe_ratio_gross"])
        accuracies.append(all_metrics["cycle_5"]["comparison_results"]["cfull_504d"]["direction_accuracy"]["mean"] * 100)
        labels.append("Phase 5\nCfull 504d")

    # Phase 6: Optimized
    if "cycle_6" in all_metrics:
        phases.append(6)
        sharpes.append(all_metrics["cycle_6"]["optimized_results"]["strategy"]["sharpe_ratio_gross"])
        accuracies.append(all_metrics["cycle_6"]["optimized_results"]["direction_accuracy"]["mean"] * 100)
        labels.append("Phase 6\nOptimized")

    # Phase 9: Final comparison - best model
    if "cycle_9" in all_metrics:
        table = all_metrics["cycle_9"]["comparison_table"]
        best = max(table, key=lambda m: m["sharpe_ratio_gross"])
        phases.append(9)
        sharpes.append(best["sharpe_ratio_gross"])
        accuracies.append(best["direction_accuracy_mean"] * 100)
        labels.append("Phase 9\nBest Model")

    if not phases:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Performance Progression Across Research Phases", fontsize=13, fontweight="bold")

    x = np.arange(len(phases))
    ax1.bar(x, sharpes, color=[COLORS["primary"], COLORS["quaternary"], COLORS["tertiary"], COLORS["secondary"]][:len(x)], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Gross Sharpe Ratio")
    ax1.set_title("Sharpe Ratio Improvement")
    for i, v in enumerate(sharpes):
        ax1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    ax2.bar(x, accuracies, color=[COLORS["primary"], COLORS["quaternary"], COLORS["tertiary"], COLORS["secondary"]][:len(x)], alpha=0.8)
    ax2.axhline(y=50, color="black", linestyle="--", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Direction Accuracy (%)")
    ax2.set_title("Accuracy Improvement")
    for i, v in enumerate(accuracies):
        ax2.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "phase_progression.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved phase_progression.png")


def generate_executive_summary_chart(cycle9_metrics: dict, cycle6_metrics: dict, output_dir: Path):
    """Single-page executive summary dashboard."""
    _setup_style()

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("PCA_SUB Lead-Lag Strategy: Executive Summary", fontsize=16, fontweight="bold", y=0.98)

    table = cycle9_metrics["comparison_table"]

    # --- Panel 1: Key metrics summary (text) ---
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    best = next(m for m in table if "K=5" in m["model"])
    baseline = next(m for m in table if "K=3" in m["model"])

    text = (
        f"Optimized PCA_SUB (K=5, L=120, λ=1.0)\n"
        f"{'─' * 35}\n"
        f"Direction Accuracy:  {best['direction_accuracy_mean']*100:.1f}%\n"
        f"Gross Sharpe:        {best['sharpe_ratio_gross']:.2f}\n"
        f"Net Sharpe (10bps):  {best['sharpe_ratio_net']:.2f}\n"
        f"Total Return (gross):{best['total_return_gross']*100:.1f}%\n"
        f"Max Drawdown (gross):{best['max_drawdown_gross']*100:.1f}%\n"
        f"Daily Turnover:      {best['mean_daily_turnover']*100:.0f}%\n"
        f"{'─' * 35}\n"
        f"Data: {cycle9_metrics['data']['date_range_start']}\n"
        f"   to {cycle9_metrics['data']['date_range_end']}\n"
        f"OOS samples: {best['total_test_samples']}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontfamily="monospace",
            fontsize=9, verticalalignment="top")
    ax.set_title("Key Metrics", fontsize=11, fontweight="bold")

    # --- Panel 2: Sharpe comparison ---
    ax = fig.add_subplot(gs[0, 1])
    models_short = ["PCA_SUB\n(K=3)", "PCA_SUB\n(K=5)", "OLS", "Ridge", "Simple\nPCA", "Hist\nMean"]
    models_full = [
        "PCA_SUB (K=3, L=60, λ=0.9)", "PCA_SUB (K=5, L=120, λ=1.0)",
        "Direct OLS", "Ridge (α=1.0)", "Simple PCA (no decay)", "Historical Mean"
    ]
    sharpes_gross = []
    for mf in models_full:
        m = next((t for t in table if t["model"] == mf), None)
        sharpes_gross.append(m["sharpe_ratio_gross"] if m else 0)
    colors_bar = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
                  COLORS["quaternary"], "#DB2777", COLORS["gray"]]
    ax.barh(range(len(models_short)), sharpes_gross, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(models_short)))
    ax.set_yticklabels(models_short, fontsize=8)
    ax.set_xlabel("Gross Sharpe Ratio")
    ax.set_title("Model Ranking", fontsize=11, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)

    # --- Panel 3: Param selection frequency ---
    ax = fig.add_subplot(gs[0, 2])
    param_summary = cycle6_metrics["param_selection_summary"]
    k_counts = param_summary["K_selection_counts"]
    k_labels = [f"K={k}" for k in sorted(k_counts.keys(), key=int)]
    k_vals = [k_counts[k] for k in sorted(k_counts.keys(), key=int)]
    ax.pie(k_vals, labels=k_labels, autopct="%1.0f%%", startangle=90,
           colors=[COLORS["light_blue"], COLORS["light_green"], COLORS["primary"], COLORS["tertiary"], COLORS["secondary"]][:len(k_vals)])
    ax.set_title("K Selection Frequency\n(35 Folds)", fontsize=11, fontweight="bold")

    # --- Panel 4: Accuracy over folds ---
    ax = fig.add_subplot(gs[1, :])
    folds_data = cycle6_metrics["selected_params_per_fold"]
    oos_accs = [f["oos_accuracy"] * 100 for f in folds_data]
    periods = [f["test_period"].split(" to ")[0] for f in folds_data]
    x = range(len(oos_accs))
    ax.bar(x, oos_accs, color=[COLORS["tertiary"] if a > 50 else COLORS["secondary"] for a in oos_accs], alpha=0.7)
    ax.axhline(y=50, color="black", linestyle="--", linewidth=0.8, label="50% baseline")
    if len(oos_accs) >= 6:
        rolling = pd.Series(oos_accs).rolling(6).mean()
        ax.plot(x, rolling, color=COLORS["primary"], linewidth=2, label="6-Fold Rolling Avg")
    ax.set_ylabel("OOS Direction Accuracy (%)")
    ax.set_title("Optimized Walk-Forward Performance Over Time", fontsize=11, fontweight="bold")
    tick_pos = list(range(0, len(folds_data), 5))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([periods[i] for i in tick_pos], rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(40, 65)

    # --- Panel 5: Gross vs Net problem ---
    ax = fig.add_subplot(gs[2, 0:2])
    model_names = ["PCA_SUB\n(K=5)", "Direct\nOLS", "Simple\nPCA", "Ridge\n(α=1)", "Hist\nMean"]
    model_keys = [
        "PCA_SUB (K=5, L=120, λ=1.0)", "Direct OLS", "Simple PCA (no decay)",
        "Ridge (α=1.0)", "Historical Mean"
    ]
    gross_rets = []
    net_rets = []
    for mk in model_keys:
        m = next((t for t in table if t["model"] == mk), None)
        gross_rets.append((m["annualized_return_gross"] * 100) if m else 0)
        net_rets.append((m["annualized_return_net"] * 100) if m else 0)
    xp = np.arange(len(model_names))
    ax.bar(xp - 0.15, gross_rets, 0.3, label="Gross", color=COLORS["primary"], alpha=0.8)
    ax.bar(xp + 0.15, net_rets, 0.3, label="Net (10bps)", color=COLORS["secondary"], alpha=0.8)
    ax.set_xticks(xp)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("The Turnover Problem: Gross vs Net Returns", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)

    # --- Panel 6: Key conclusion ---
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    conclusion = (
        "KEY FINDINGS\n"
        "═══════════════════════\n\n"
        "1. Cross-market signal EXISTS\n"
        "   (51.2% accuracy, SR=2.18)\n\n"
        "2. PCA dimensionality reduction\n"
        "   outperforms direct regression\n\n"
        "3. Optimal: K=5, L=120, λ=1.0\n"
        "   (no decay weighting needed)\n\n"
        "4. CRITICAL BLOCKER:\n"
        "   76% daily turnover destroys\n"
        "   all net alpha\n\n"
        "5. NEXT STEP: Signal smoothing\n"
        "   or threshold-based positions"
    )
    ax.text(0.05, 0.95, conclusion, transform=ax.transAxes, fontfamily="monospace",
            fontsize=9, verticalalignment="top")

    fig.savefig(output_dir / "executive_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved executive_summary.png")
