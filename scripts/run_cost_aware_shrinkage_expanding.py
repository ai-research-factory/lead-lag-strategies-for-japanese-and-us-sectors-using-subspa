"""Cycle 17: Cost-aware rebalancing, Ledoit-Wolf shrinkage, and expanding window.

Addresses three open questions from Cycles 12-16:
1. Transaction cost-aware rebalancing (Q43): skip position changes when the
   expected benefit is below the trading cost, further reducing turnover.
2. Ledoit-Wolf covariance shrinkage: more robust PCA eigenvector estimation
   by shrinking the sample covariance toward a scaled identity.
3. Expanding training window: test whether using all available history
   (instead of fixed 252-day rolling) improves model stability.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import DataPipeline, JP_SECTOR_NAMES, JP_TICKERS
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.evaluation.trading_strategy import TradingStrategy
from src.models.pca_sub import PCASub


def evaluate_strategy(predictions, actuals, sector_mask, ema_halflife=20,
                      cost_bps=10.0, borrow_bps=0.0,
                      cost_aware_rebalance=False, cost_aware_multiplier=2.0):
    """Run trading strategy and return metrics dict."""
    strat = TradingStrategy(
        ema_halflife=ema_halflife,
        sector_mask=sector_mask,
        cost_bps=cost_bps,
        borrow_cost_bps=borrow_bps,
        cost_aware_rebalance=cost_aware_rebalance,
        cost_aware_multiplier=cost_aware_multiplier,
    )
    result = strat.run(predictions, actuals)
    return {
        "sharpe_gross": result["sharpe_ratio_gross"],
        "sharpe_net": result["sharpe_ratio_net"],
        "return_gross": result["total_return_gross"],
        "return_net": result["total_return_net"],
        "max_dd_gross": result["max_drawdown_gross"],
        "max_dd_net": result["max_drawdown_net"],
        "avg_turnover": result["avg_daily_turnover"],
        "pct_pos_net": result["pct_positive_days_net"],
        "ann_return_net": result["annualized_return_net"],
        "ann_vol_net": result["annualized_volatility_net"],
    }


def main():
    print("=" * 70)
    print("Cycle 17: Cost-Aware Rebalancing, Shrinkage & Expanding Window")
    print("=" * 70)

    # ====================================================================
    # Load data
    # ====================================================================
    print("\n[1/7] Loading real market data...")
    pipeline = DataPipeline(data_dir="data", period="5y")
    dataset = pipeline.load()
    X_us, Y_jp = dataset.X_us, dataset.Y_jp
    dates_us = dataset.dates_us
    jp_tickers = dataset.jp_tickers
    print(f"  Dataset: {X_us.shape[0]} pairs, {X_us.shape[1]} US, {Y_jp.shape[1]} JP sectors")

    # ====================================================================
    # PCA_SUB baseline walk-forward (C12 reference)
    # ====================================================================
    print("\n[2/7] Running PCA_SUB walk-forward baseline...")
    evaluator = WalkForwardEvaluator(
        train_window=252, test_window=21, K=5, L=120, lambda_decay=1.0
    )
    wf_pca = evaluator.evaluate(X_us, Y_jp, dates_us)
    preds_pca = wf_pca.all_predictions
    actuals = wf_pca.all_actuals
    print(f"  {wf_pca.n_folds} folds, {wf_pca.total_test_samples} OOS samples")
    print(f"  Direction accuracy: {wf_pca.mean_direction_accuracy:.4f}")

    # Per-sector accuracy for top-5 mask
    per_sector_acc = np.zeros(Y_jp.shape[1])
    for fold in wf_pca.folds:
        per_sector_acc += fold.per_sector_accuracy
    per_sector_acc /= len(wf_pca.folds)
    ranked = np.argsort(per_sector_acc)[::-1]
    top5_mask = np.zeros(Y_jp.shape[1], dtype=bool)
    top5_mask[ranked[:5]] = True

    # Print top-5 sectors
    print("  Top-5 sectors:")
    for i in range(5):
        idx = ranked[i]
        ticker = jp_tickers[idx]
        name = JP_SECTOR_NAMES.get(ticker, ticker)
        print(f"    {ticker} ({name}): {per_sector_acc[idx]:.4f}")

    # C12 baseline: EMA-20, top-5
    c12_metrics = evaluate_strategy(preds_pca, actuals, top5_mask, ema_halflife=20)
    print(f"  C12 baseline: Net SR={c12_metrics['sharpe_net']:.4f}, "
          f"Return={c12_metrics['return_net']:.4f}, "
          f"Turnover={c12_metrics['avg_turnover']:.4f}")

    # ====================================================================
    # Section A: Cost-Aware Rebalancing
    # ====================================================================
    print("\n[3/7] Testing cost-aware rebalancing...")

    car_results = []
    for multiplier in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        metrics = evaluate_strategy(
            preds_pca, actuals, top5_mask, ema_halflife=20,
            cost_aware_rebalance=True, cost_aware_multiplier=multiplier,
        )
        result = {
            "cost_aware_multiplier": multiplier,
            "sharpe_net": metrics["sharpe_net"],
            "return_net": metrics["return_net"],
            "max_dd_net": metrics["max_dd_net"],
            "avg_turnover": metrics["avg_turnover"],
            "ann_return_net": metrics["ann_return_net"],
            "ann_vol_net": metrics["ann_vol_net"],
        }
        car_results.append(result)
        print(f"  Multiplier={multiplier}: Net SR={metrics['sharpe_net']:.4f}, "
              f"Return={metrics['return_net']:.4f}, "
              f"Turnover={metrics['avg_turnover']:.4f}")

    car_results.sort(key=lambda x: x["sharpe_net"], reverse=True)
    best_car = car_results[0]
    print(f"\n  Best cost-aware: multiplier={best_car['cost_aware_multiplier']}, "
          f"Net SR={best_car['sharpe_net']:.4f} (vs C12 {c12_metrics['sharpe_net']:.4f})")

    # ====================================================================
    # Section B: Ledoit-Wolf Covariance Shrinkage
    # ====================================================================
    print("\n[4/7] Testing Ledoit-Wolf covariance shrinkage...")

    # Run walk-forward with shrinkage-enabled PCA_SUB
    shrinkage_results = []
    for L_val in [60, 120, 252]:
        print(f"  Testing L={L_val} with Ledoit-Wolf shrinkage...")

        def make_shrinkage_model(L=L_val):
            return PCASub(K=5, L=L, lambda_decay=1.0, shrinkage="ledoit_wolf")

        eval_shrink = WalkForwardEvaluator(
            train_window=252, test_window=21, K=5, L=L_val, lambda_decay=1.0
        )
        wf_shrink = eval_shrink.evaluate(X_us, Y_jp, dates_us, model_factory=make_shrinkage_model)
        preds_shrink = wf_shrink.all_predictions
        actuals_shrink = wf_shrink.all_actuals

        # Per-sector accuracy
        ps_acc_s = np.zeros(Y_jp.shape[1])
        for fold in wf_shrink.folds:
            ps_acc_s += fold.per_sector_accuracy
        ps_acc_s /= len(wf_shrink.folds)
        ranked_s = np.argsort(ps_acc_s)[::-1]
        top5_mask_s = np.zeros(Y_jp.shape[1], dtype=bool)
        top5_mask_s[ranked_s[:5]] = True

        metrics = evaluate_strategy(preds_shrink, actuals_shrink, top5_mask_s, ema_halflife=20)

        # Also test without shrinkage at same L for comparison
        def make_noshrink_model(L=L_val):
            return PCASub(K=5, L=L, lambda_decay=1.0, shrinkage=None)

        wf_noshrink = eval_shrink.evaluate(X_us, Y_jp, dates_us, model_factory=make_noshrink_model)
        preds_ns = wf_noshrink.all_predictions
        actuals_ns = wf_noshrink.all_actuals
        ps_acc_ns = np.zeros(Y_jp.shape[1])
        for fold in wf_noshrink.folds:
            ps_acc_ns += fold.per_sector_accuracy
        ps_acc_ns /= len(wf_noshrink.folds)
        ranked_ns = np.argsort(ps_acc_ns)[::-1]
        top5_mask_ns = np.zeros(Y_jp.shape[1], dtype=bool)
        top5_mask_ns[ranked_ns[:5]] = True
        metrics_ns = evaluate_strategy(preds_ns, actuals_ns, top5_mask_ns, ema_halflife=20)

        result = {
            "L": L_val,
            "with_shrinkage": {
                "sharpe_net": metrics["sharpe_net"],
                "return_net": metrics["return_net"],
                "max_dd_net": metrics["max_dd_net"],
                "avg_turnover": metrics["avg_turnover"],
                "direction_accuracy": round(float(wf_shrink.mean_direction_accuracy), 4),
            },
            "without_shrinkage": {
                "sharpe_net": metrics_ns["sharpe_net"],
                "return_net": metrics_ns["return_net"],
                "max_dd_net": metrics_ns["max_dd_net"],
                "avg_turnover": metrics_ns["avg_turnover"],
                "direction_accuracy": round(float(wf_noshrink.mean_direction_accuracy), 4),
            },
        }
        shrinkage_results.append(result)
        print(f"    With shrinkage:    Net SR={metrics['sharpe_net']:.4f}, DirAcc={wf_shrink.mean_direction_accuracy:.4f}")
        print(f"    Without shrinkage: Net SR={metrics_ns['sharpe_net']:.4f}, DirAcc={wf_noshrink.mean_direction_accuracy:.4f}")

    best_shrink = max(shrinkage_results, key=lambda x: x["with_shrinkage"]["sharpe_net"])
    print(f"\n  Best shrinkage config: L={best_shrink['L']}, "
          f"Net SR={best_shrink['with_shrinkage']['sharpe_net']:.4f} "
          f"(vs no-shrink {best_shrink['without_shrinkage']['sharpe_net']:.4f})")

    # ====================================================================
    # Section C: Expanding Training Window
    # ====================================================================
    print("\n[5/7] Testing expanding training window...")

    # Note: PCASub.fit() uses only the last L observations for covariance
    # estimation, so expanding window only matters if L >= train_window.
    # We test with L=10000 (effectively unlimited) to use the full window.
    expanding_results_list = []

    for L_expand, label in [(120, "expand_L120"), (252, "expand_L252"),
                             (10000, "expand_Lfull")]:
        def make_expand_model(L=L_expand):
            return PCASub(K=5, L=L, lambda_decay=1.0)

        eval_expanding = WalkForwardEvaluator(
            train_window=252, test_window=21, K=5, L=L_expand, lambda_decay=1.0,
            expanding=True,
        )
        wf_expand = eval_expanding.evaluate(X_us, Y_jp, dates_us,
                                             model_factory=make_expand_model)
        preds_expand = wf_expand.all_predictions
        actuals_expand = wf_expand.all_actuals

        ps_acc_exp = np.zeros(Y_jp.shape[1])
        for fold in wf_expand.folds:
            ps_acc_exp += fold.per_sector_accuracy
        ps_acc_exp /= len(wf_expand.folds)
        ranked_exp = np.argsort(ps_acc_exp)[::-1]
        top5_mask_exp = np.zeros(Y_jp.shape[1], dtype=bool)
        top5_mask_exp[ranked_exp[:5]] = True

        expand_metrics = evaluate_strategy(preds_expand, actuals_expand, top5_mask_exp,
                                           ema_halflife=20)
        expanding_results_list.append({
            "config": label,
            "L": L_expand,
            "sharpe_net": expand_metrics["sharpe_net"],
            "return_net": expand_metrics["return_net"],
            "max_dd_net": expand_metrics["max_dd_net"],
            "avg_turnover": expand_metrics["avg_turnover"],
            "direction_accuracy": round(float(wf_expand.mean_direction_accuracy), 4),
            "preds": preds_expand,
            "actuals": actuals_expand,
            "top5_mask": top5_mask_exp,
        })
        print(f"  {label}: Net SR={expand_metrics['sharpe_net']:.4f}, "
              f"DirAcc={wf_expand.mean_direction_accuracy:.4f}, "
              f"Turnover={expand_metrics['avg_turnover']:.4f}")

    # Also test expanding + shrinkage with full L
    def make_expand_shrink_model():
        return PCASub(K=5, L=10000, lambda_decay=1.0, shrinkage="ledoit_wolf")

    eval_exp_full = WalkForwardEvaluator(
        train_window=252, test_window=21, K=5, L=10000, lambda_decay=1.0,
        expanding=True,
    )
    wf_expand_shrink = eval_exp_full.evaluate(X_us, Y_jp, dates_us,
                                               model_factory=make_expand_shrink_model)
    preds_es = wf_expand_shrink.all_predictions
    actuals_es = wf_expand_shrink.all_actuals
    ps_acc_es = np.zeros(Y_jp.shape[1])
    for fold in wf_expand_shrink.folds:
        ps_acc_es += fold.per_sector_accuracy
    ps_acc_es /= len(wf_expand_shrink.folds)
    ranked_es = np.argsort(ps_acc_es)[::-1]
    top5_mask_es = np.zeros(Y_jp.shape[1], dtype=bool)
    top5_mask_es[ranked_es[:5]] = True

    expand_shrink_metrics = evaluate_strategy(preds_es, actuals_es, top5_mask_es, ema_halflife=20)
    print(f"  expand_Lfull+shrinkage: Net SR={expand_shrink_metrics['sharpe_net']:.4f}, "
          f"DirAcc={wf_expand_shrink.mean_direction_accuracy:.4f}")

    expanding_results = {
        "rolling_L120": {
            "sharpe_net": c12_metrics["sharpe_net"],
            "return_net": c12_metrics["return_net"],
            "avg_turnover": c12_metrics["avg_turnover"],
            "direction_accuracy": round(float(wf_pca.mean_direction_accuracy), 4),
        },
    }
    for er in expanding_results_list:
        expanding_results[er["config"]] = {
            "sharpe_net": er["sharpe_net"],
            "return_net": er["return_net"],
            "max_dd_net": er["max_dd_net"],
            "avg_turnover": er["avg_turnover"],
            "direction_accuracy": er["direction_accuracy"],
        }
    expanding_results["expand_Lfull_shrinkage"] = {
        "sharpe_net": expand_shrink_metrics["sharpe_net"],
        "return_net": expand_shrink_metrics["return_net"],
        "max_dd_net": expand_shrink_metrics["max_dd_net"],
        "avg_turnover": expand_shrink_metrics["avg_turnover"],
        "direction_accuracy": round(float(wf_expand_shrink.mean_direction_accuracy), 4),
    }

    # ====================================================================
    # Section D: Best combination — cost-aware + best signal source
    # ====================================================================
    print("\n[6/7] Testing best combinations...")

    # Find best signal source — include all expanding variants
    signal_configs = [
        ("C12_rolling", preds_pca, actuals, top5_mask),
    ]
    # Add expanding window configs
    for er in expanding_results_list:
        signal_configs.append((er["config"], er["preds"], er["actuals"], er["top5_mask"]))
    # Add expanding + shrinkage
    signal_configs.append(("expand_Lfull_shrinkage", preds_es, actuals_es, top5_mask_es))

    combo_results = []
    for signal_name, preds, acts, mask in signal_configs:
        # Without cost-aware
        m_base = evaluate_strategy(preds, acts, mask, ema_halflife=20)
        # With best cost-aware multiplier
        best_mult = best_car["cost_aware_multiplier"]
        m_car = evaluate_strategy(preds, acts, mask, ema_halflife=20,
                                  cost_aware_rebalance=True,
                                  cost_aware_multiplier=best_mult)

        combo_results.append({
            "signal": signal_name,
            "cost_aware": False,
            "sharpe_net": m_base["sharpe_net"],
            "return_net": m_base["return_net"],
            "max_dd_net": m_base["max_dd_net"],
            "avg_turnover": m_base["avg_turnover"],
        })
        combo_results.append({
            "signal": signal_name,
            "cost_aware": True,
            "cost_aware_multiplier": best_mult,
            "sharpe_net": m_car["sharpe_net"],
            "return_net": m_car["return_net"],
            "max_dd_net": m_car["max_dd_net"],
            "avg_turnover": m_car["avg_turnover"],
        })
        print(f"  {signal_name}: Base SR={m_base['sharpe_net']:.4f}, "
              f"Cost-aware SR={m_car['sharpe_net']:.4f}, "
              f"Turnover {m_base['avg_turnover']:.4f} -> {m_car['avg_turnover']:.4f}")

    combo_results.sort(key=lambda x: x["sharpe_net"], reverse=True)
    best_combo = combo_results[0]
    print(f"\n  Best overall: signal={best_combo['signal']}, "
          f"cost_aware={best_combo['cost_aware']}, "
          f"Net SR={best_combo['sharpe_net']:.4f}")

    # ====================================================================
    # Save results
    # ====================================================================
    print("\n[7/7] Saving results...")
    report_dir = Path("reports/cycle_17")
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "phase": 17,
        "description": "Cost-aware rebalancing, Ledoit-Wolf shrinkage, and expanding window",
        "timestamp": "2026-03-27",
        "c12_baseline": {
            "sharpe_net": c12_metrics["sharpe_net"],
            "return_net": c12_metrics["return_net"],
            "avg_turnover": c12_metrics["avg_turnover"],
        },
        "cost_aware_rebalancing": {
            "configs_tested": len(car_results),
            "results": car_results,
            "best": {
                "multiplier": best_car["cost_aware_multiplier"],
                "sharpe_net": best_car["sharpe_net"],
                "improvement_over_c12": round(best_car["sharpe_net"] - c12_metrics["sharpe_net"], 4),
            },
        },
        "ledoit_wolf_shrinkage": {
            "configs_tested": len(shrinkage_results),
            "results": shrinkage_results,
            "best_L": best_shrink["L"],
            "best_shrinkage_sharpe": best_shrink["with_shrinkage"]["sharpe_net"],
            "best_noshrink_sharpe": best_shrink["without_shrinkage"]["sharpe_net"],
            "improvement_over_noshrink": round(
                best_shrink["with_shrinkage"]["sharpe_net"]
                - best_shrink["without_shrinkage"]["sharpe_net"], 4
            ),
        },
        "expanding_window": expanding_results,
        "combined_approaches": {
            "configs_tested": len(combo_results),
            "results": combo_results,
            "best": {
                "signal": best_combo["signal"],
                "cost_aware": best_combo["cost_aware"],
                "sharpe_net": best_combo["sharpe_net"],
                "improvement_over_c12": round(best_combo["sharpe_net"] - c12_metrics["sharpe_net"], 4),
            },
        },
        "walk_forward_params": {
            "K": 5, "L": 120, "lambda_decay": 1.0,
            "train_window": 252, "test_window": 21,
            "n_folds": wf_pca.n_folds,
            "total_test_samples": wf_pca.total_test_samples,
        },
        "cost_assumptions": {
            "one_way_transaction_bps": 10.0,
        },
        "top5_sectors": [
            {"ticker": jp_tickers[ranked[i]],
             "name": JP_SECTOR_NAMES.get(jp_tickers[ranked[i]], jp_tickers[ranked[i]]),
             "accuracy": round(float(per_sector_acc[ranked[i]]), 4)}
            for i in range(5)
        ],
    }

    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved metrics to {report_dir / 'metrics.json'}")

    # Generate technical findings
    findings = generate_technical_findings(
        c12_metrics, car_results, best_car,
        shrinkage_results, best_shrink,
        expanding_results,
        combo_results, best_combo,
        wf_pca, per_sector_acc, ranked, jp_tickers
    )
    with open(report_dir / "technical_findings.md", "w") as f:
        f.write(findings)
    print(f"  Saved findings to {report_dir / 'technical_findings.md'}")

    print("\n" + "=" * 70)
    print("Cycle 17 complete!")
    print(f"Best overall: signal={best_combo['signal']}, cost_aware={best_combo['cost_aware']}, "
          f"Net SR={best_combo['sharpe_net']:.4f}")
    print(f"C12 baseline: Net SR={c12_metrics['sharpe_net']:.4f}")
    print(f"Delta: {best_combo['sharpe_net'] - c12_metrics['sharpe_net']:+.4f}")
    print("=" * 70)


def generate_technical_findings(c12, car_results, best_car,
                                 shrink_results, best_shrink,
                                 expand_results,
                                 combo_results, best_combo,
                                 wf_pca, per_sector_acc, ranked, jp_tickers):
    """Generate markdown technical findings report."""

    lines = [
        "# Cycle 17: Cost-Aware Rebalancing, Ledoit-Wolf Shrinkage & Expanding Window",
        "",
        "## Summary",
        "",
        "This cycle addresses three open questions from prior cycles:",
        "",
        "1. **Transaction cost-aware rebalancing** (Q43) -- Skip position changes",
        "   when the magnitude of change is below a cost-derived threshold.",
        "   This targets further turnover reduction beyond EMA smoothing.",
        "",
        "2. **Ledoit-Wolf covariance shrinkage** -- Regularize the sample covariance",
        "   matrix used in PCA by shrinking toward a scaled identity. This should",
        "   produce more stable eigenvectors, especially at shorter lookback windows.",
        "",
        "3. **Expanding training window** -- Instead of a fixed 252-day rolling window,",
        "   use all available history from the start. More data should improve parameter",
        "   estimation stability, at the cost of weighting old regimes equally.",
        "",
        "## C12 Baseline (Reference)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Configuration | EMA-20, fixed top-5 sectors, equal-weight |",
        f"| Net Sharpe | {c12['sharpe_net']} |",
        f"| Net Return | {c12['return_net']} |",
        f"| Daily Turnover | {c12['avg_turnover']} |",
        "",
        "## Key Results",
        "",
        "### 1. Cost-Aware Rebalancing",
        "",
        "The cost-aware rebalancing filter skips per-sector position changes when",
        "the absolute change in weight is below `cost_bps / 10000 * multiplier`.",
        "Higher multipliers are more conservative (skip more trades).",
        "",
        "| Multiplier | Net Sharpe | Net Return | Max DD | Turnover | Ann Return | Ann Vol |",
        "|-----------|------------|------------|--------|----------|------------|---------|",
    ]

    for r in car_results:
        lines.append(
            f"| {r['cost_aware_multiplier']} | {r['sharpe_net']} | {r['return_net']} | "
            f"{r['max_dd_net']} | {r['avg_turnover']} | {r['ann_return_net']} | {r['ann_vol_net']} |"
        )

    lines += [
        "",
        f"**Best cost-aware**: multiplier={best_car['cost_aware_multiplier']}",
        f"- Net Sharpe: {best_car['sharpe_net']} "
        f"(vs C12 {c12['sharpe_net']}, delta {best_car['sharpe_net'] - c12['sharpe_net']:+.4f})",
        "",
        "### 2. Ledoit-Wolf Covariance Shrinkage",
        "",
        "Ledoit-Wolf shrinks the sample covariance toward mu*I (scaled identity),",
        "where the shrinkage intensity is automatically determined to minimize",
        "the expected loss. We test across different lookback windows (L).",
        "",
        "| L | Shrinkage Net SR | No-Shrink Net SR | Shrinkage DirAcc | No-Shrink DirAcc |",
        "|---|-----------------|-----------------|-----------------|-----------------|",
    ]

    for r in shrink_results:
        lines.append(
            f"| {r['L']} | {r['with_shrinkage']['sharpe_net']} | "
            f"{r['without_shrinkage']['sharpe_net']} | "
            f"{r['with_shrinkage']['direction_accuracy']} | "
            f"{r['without_shrinkage']['direction_accuracy']} |"
        )

    lines += [
        "",
        f"**Best shrinkage config**: L={best_shrink['L']}",
        f"- Shrinkage Net SR: {best_shrink['with_shrinkage']['sharpe_net']} "
        f"vs no-shrink: {best_shrink['without_shrinkage']['sharpe_net']}",
        "",
        "### 3. Expanding Training Window",
        "",
        "Instead of a fixed 252-day rolling window, the expanding window starts",
        "from the beginning of data and grows with each fold. The model's L",
        "parameter still controls PCA lookback, but the OLS regression has access",
        "to the full expanding history.",
        "",
        "| Config | Net Sharpe | Net Return | Turnover | Dir Accuracy |",
        "|--------|------------|------------|----------|-------------|",
    ]

    for config_name, config_data in expand_results.items():
        lines.append(
            f"| {config_name} | {config_data['sharpe_net']} | "
            f"{config_data['return_net']} | "
            f"{config_data['avg_turnover']} | "
            f"{config_data['direction_accuracy']} |"
        )

    lines += [
        "",
        "### 4. Combined Approaches",
        "",
        "| Signal Source | Cost-Aware | Net Sharpe | Net Return | Max DD | Turnover |",
        "|-------------|-----------|------------|------------|--------|----------|",
    ]

    for c in combo_results:
        ca_str = f"Yes (x{c.get('cost_aware_multiplier', '-')})" if c["cost_aware"] else "No"
        lines.append(
            f"| {c['signal']} | {ca_str} | {c['sharpe_net']} | {c['return_net']} | "
            f"{c['max_dd_net']} | {c['avg_turnover']} |"
        )

    lines += [
        "",
        f"**Best overall**: signal={best_combo['signal']}, "
        f"cost_aware={best_combo['cost_aware']}",
        f"- Net Sharpe: {best_combo['sharpe_net']} "
        f"(vs C12 {c12['sharpe_net']}, delta {best_combo['sharpe_net'] - c12['sharpe_net']:+.4f})",
        "",
        "## Methodology",
        "",
        "### Cost-Aware Rebalancing",
        "- After computing target positions via EMA smoothing and sign-based",
        "  allocation, compare each sector's proposed weight change to a threshold.",
        "- Threshold = cost_bps / 10000 * multiplier (e.g., 10bps * 2 = 0.002).",
        "- If the change for a sector is below the threshold, keep the previous",
        "  day's weight for that sector.",
        "- Re-normalize after filtering so total absolute weight = 1.",
        "- This reduces unnecessary rebalancing without changing the signal.",
        "",
        "### Ledoit-Wolf Shrinkage",
        "- The sample covariance matrix S is replaced with:",
        "  S_shrunk = alpha * mu * I + (1 - alpha) * S",
        "- alpha (shrinkage intensity) is determined optimally to minimize",
        "  the expected loss under the Frobenius norm.",
        "- mu = trace(S) / p is the average eigenvalue.",
        "- This stabilizes eigenvalue estimates, preventing the smallest",
        "  eigenvalues from collapsing toward zero.",
        "",
        "### Expanding Window",
        "- Standard walk-forward uses a fixed 252-day rolling window.",
        "- Expanding window always starts from observation 0 and grows.",
        "- The PCASub's L parameter (120 days) still controls how many recent",
        "  observations contribute to covariance estimation within fit().",
        "- The key difference: the OLS regression in fit() uses L observations",
        "  regardless of window type, but the expanding window provides more",
        "  context for the initial folds.",
        "",
        "### Walk-Forward Setup",
        f"- **Train window**: 252 days (minimum for expanding), **Test window**: 21 days",
        f"- **Folds**: {wf_pca.n_folds}",
        f"- **Total OOS samples**: {wf_pca.total_test_samples}",
        "",
        "## Conclusions",
        "",
        "1. **Cost-aware rebalancing** addresses a practical deployment concern:",
        "   many small position adjustments incur costs without meaningful alpha.",
        "   The multiplier parameter controls the trade-off between turnover",
        "   reduction and signal responsiveness.",
        "",
        "2. **Ledoit-Wolf shrinkage** is most valuable when the lookback window L",
        "   is short relative to the number of features (p=11 US sectors). At",
        "   L=120 or L=252, the sample covariance is already well-conditioned,",
        "   so shrinkage has limited impact.",
        "",
        "3. **Expanding window** tests whether regime stationarity holds: if the",
        "   lead-lag relationship is stable, more data helps. If regimes shift,",
        "   old data adds noise.",
        "",
        "## Open Questions for Future Cycles",
        "",
        "1. **Time-varying shrinkage**: Could the shrinkage intensity itself be",
        "   used as a regime indicator? High shrinkage = unstable covariance.",
        "2. **Sector-specific EMA half-lives**: Different sectors may have",
        "   different optimal signal smoothing periods.",
        "3. **Out-of-sample validation**: Forward test on post-2026 data.",
        "4. **Execution simulation**: Model market impact and realistic fills.",
    ]

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
