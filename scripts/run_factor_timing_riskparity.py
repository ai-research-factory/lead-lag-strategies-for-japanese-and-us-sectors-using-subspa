"""Cycle 16: Factor timing, risk-parity portfolio construction, and long-bias strategy.

Addresses open questions from Cycles 14-15:
1. Factor timing: weight individual PCA factors by their recent predictive
   accuracy instead of treating all K components equally.
2. Risk-parity position sizing: inverse-volatility weighting instead of
   equal-weight, so lower-vol sectors receive larger allocations.
3. Long-biased strategy: Phase 7 showed strong asymmetry — the lead-lag
   signal is much stronger for long positions (up-market SR=1.31 vs
   down-market SR=-0.18). A long-biased strategy may outperform.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import DataPipeline, JP_SECTOR_NAMES, JP_TICKERS
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.evaluation.trading_strategy import TradingStrategy
from src.models.factor_timing import PCASubFactorTiming
from src.models.pca_sub import PCASub


def evaluate_strategy(predictions, actuals, sector_mask, ema_halflife=20,
                      cost_bps=10.0, borrow_bps=0.0, risk_parity=False,
                      risk_parity_lookback=63, long_bias=None):
    """Run trading strategy and return metrics dict."""
    strat = TradingStrategy(
        ema_halflife=ema_halflife,
        sector_mask=sector_mask,
        cost_bps=cost_bps,
        borrow_cost_bps=borrow_bps,
        risk_parity=risk_parity,
        risk_parity_lookback=risk_parity_lookback,
        long_bias=long_bias,
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
        "pct_pos_gross": result["pct_positive_days_gross"],
        "pct_pos_net": result["pct_positive_days_net"],
        "ann_return_net": result["annualized_return_net"],
        "ann_vol_net": result["annualized_volatility_net"],
        "daily_returns_net": result["daily_returns_net"],
        "weights": result["weights"],
    }


def run_factor_timing_walkforward(X_us, Y_jp, dates_us, evaluator,
                                  factor_ema_halflife=63):
    """Custom walk-forward that updates factor weights between folds.

    After each fold, we measure per-factor accuracy on the test set and
    update the factor weights for the next fold.
    """
    T = X_us.shape[0]
    train_window = evaluator.train_window
    test_window = evaluator.test_window

    all_preds = []
    all_actuals = []
    factor_weights_history = []

    model = PCASubFactorTiming(
        K=5, L=120, lambda_decay=1.0,
        factor_ema_halflife=factor_ema_halflife,
    )

    fold_id = 0
    start = 0

    while start + train_window + test_window <= T:
        train_end = start + train_window
        test_end = min(train_end + test_window, T)

        X_train = X_us[start:train_end]
        Y_train = Y_jp[start:train_end]
        X_test = X_us[train_end:test_end]
        Y_test = Y_jp[train_end:test_end]

        # Fit model
        model.fit(X_train, Y_train)

        # Record current factor weights
        factor_weights_history.append(model.factor_weights_.copy())

        # Predict
        Y_pred = model.predict(X_test)
        all_preds.append(Y_pred)
        all_actuals.append(Y_test)

        # Get per-factor contributions for weight update
        per_factor = model.predict_per_factor(X_test)
        model.update_factor_weights(per_factor, Y_test)

        fold_id += 1
        start += test_window

    predictions = np.vstack(all_preds)
    actuals = np.vstack(all_actuals)

    # Compute direction accuracy
    signs_true = np.sign(actuals)
    signs_pred = np.sign(predictions)
    accuracy = (signs_true == signs_pred).mean()

    return predictions, actuals, accuracy, factor_weights_history


def main():
    print("=" * 70)
    print("Cycle 16: Factor Timing, Risk-Parity & Long-Bias Strategy")
    print("=" * 70)

    # ====================================================================
    # Load data
    # ====================================================================
    print("\n[1/8] Loading real market data...")
    pipeline = DataPipeline(data_dir="data", period="5y")
    dataset = pipeline.load()
    X_us, Y_jp = dataset.X_us, dataset.Y_jp
    dates_us = dataset.dates_us
    jp_tickers = dataset.jp_tickers
    print(f"  Dataset: {X_us.shape[0]} pairs, {X_us.shape[1]} US, {Y_jp.shape[1]} JP sectors")

    # ====================================================================
    # PCA_SUB baseline walk-forward (C12 reference)
    # ====================================================================
    print("\n[2/8] Running PCA_SUB walk-forward baseline...")
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
          f"Return={c12_metrics['return_net']:.4f}")

    # ====================================================================
    # Section A: Factor Timing
    # ====================================================================
    print("\n[3/8] Testing factor timing with per-PC weight adaptation...")

    factor_timing_results = []
    for ft_hl in [21, 42, 63, 126]:
        print(f"  Factor EMA half-life={ft_hl}...")
        ft_preds, ft_actuals, ft_acc, ft_weights_hist = run_factor_timing_walkforward(
            X_us, Y_jp, dates_us, evaluator, factor_ema_halflife=ft_hl
        )

        ft_metrics = evaluate_strategy(ft_preds, ft_actuals, top5_mask, ema_halflife=20)

        # Average final factor weights
        avg_final_weights = np.mean(ft_weights_hist[-5:], axis=0) if len(ft_weights_hist) >= 5 else ft_weights_hist[-1]

        result = {
            "factor_ema_hl": ft_hl,
            "direction_accuracy": round(float(ft_acc), 4),
            "sharpe_net": ft_metrics["sharpe_net"],
            "return_net": ft_metrics["return_net"],
            "max_dd_net": ft_metrics["max_dd_net"],
            "avg_turnover": ft_metrics["avg_turnover"],
            "avg_final_factor_weights": [round(w, 4) for w in avg_final_weights],
        }
        factor_timing_results.append(result)
        print(f"    DirAcc={ft_acc:.4f}, Net SR={ft_metrics['sharpe_net']:.4f}, "
              f"Weights={[f'{w:.3f}' for w in avg_final_weights]}")

    factor_timing_results.sort(key=lambda x: x["sharpe_net"], reverse=True)
    best_ft = factor_timing_results[0]
    print(f"\n  Best factor timing: HL={best_ft['factor_ema_hl']}, "
          f"Net SR={best_ft['sharpe_net']:.4f} (vs C12 {c12_metrics['sharpe_net']:.4f})")

    # ====================================================================
    # Section B: Risk-Parity Position Sizing
    # ====================================================================
    print("\n[4/8] Testing risk-parity position sizing...")

    rp_results = []
    for rp_lb in [21, 42, 63, 126]:
        rp_metrics = evaluate_strategy(
            preds_pca, actuals, top5_mask, ema_halflife=20,
            risk_parity=True, risk_parity_lookback=rp_lb
        )
        result = {
            "rp_lookback": rp_lb,
            "sharpe_net": rp_metrics["sharpe_net"],
            "return_net": rp_metrics["return_net"],
            "max_dd_net": rp_metrics["max_dd_net"],
            "avg_turnover": rp_metrics["avg_turnover"],
            "ann_return_net": rp_metrics["ann_return_net"],
            "ann_vol_net": rp_metrics["ann_vol_net"],
        }
        rp_results.append(result)
        print(f"  RP lookback={rp_lb}: Net SR={rp_metrics['sharpe_net']:.4f}, "
              f"Return={rp_metrics['return_net']:.4f}, "
              f"Turnover={rp_metrics['avg_turnover']:.4f}")

    rp_results.sort(key=lambda x: x["sharpe_net"], reverse=True)
    best_rp = rp_results[0]
    print(f"\n  Best risk-parity: LB={best_rp['rp_lookback']}, "
          f"Net SR={best_rp['sharpe_net']:.4f} (vs C12 {c12_metrics['sharpe_net']:.4f})")

    # ====================================================================
    # Section C: Long-Biased Strategy
    # ====================================================================
    print("\n[5/8] Testing long-biased strategies...")

    long_bias_results = []
    for lb in [0.0, 0.25, 0.5, 0.75, 1.0]:
        lb_metrics = evaluate_strategy(
            preds_pca, actuals, top5_mask, ema_halflife=20,
            long_bias=lb if lb > 0 else None
        )
        result = {
            "long_bias": lb,
            "sharpe_net": lb_metrics["sharpe_net"],
            "return_net": lb_metrics["return_net"],
            "max_dd_net": lb_metrics["max_dd_net"],
            "avg_turnover": lb_metrics["avg_turnover"],
            "pct_pos_net": lb_metrics["pct_pos_net"],
        }
        long_bias_results.append(result)
        print(f"  Long bias={lb:.2f}: Net SR={lb_metrics['sharpe_net']:.4f}, "
              f"Return={lb_metrics['return_net']:.4f}, "
              f"MaxDD={lb_metrics['max_dd_net']:.4f}")

    long_bias_results.sort(key=lambda x: x["sharpe_net"], reverse=True)
    best_lb = long_bias_results[0]
    print(f"\n  Best long bias: {best_lb['long_bias']}, "
          f"Net SR={best_lb['sharpe_net']:.4f} (vs C12 {c12_metrics['sharpe_net']:.4f})")

    # ====================================================================
    # Section D: Combined approaches
    # ====================================================================
    print("\n[6/8] Testing combined approaches...")

    combined_results = []

    # Best risk-parity + various long biases
    for lb in [0.0, 0.25, 0.5, 0.75]:
        combo_metrics = evaluate_strategy(
            preds_pca, actuals, top5_mask, ema_halflife=20,
            risk_parity=True, risk_parity_lookback=best_rp["rp_lookback"],
            long_bias=lb if lb > 0 else None,
        )
        combined_results.append({
            "config": f"RP(lb={best_rp['rp_lookback']})+LongBias({lb})",
            "risk_parity_lookback": best_rp["rp_lookback"],
            "long_bias": lb,
            "factor_timing": False,
            "sharpe_net": combo_metrics["sharpe_net"],
            "return_net": combo_metrics["return_net"],
            "max_dd_net": combo_metrics["max_dd_net"],
            "avg_turnover": combo_metrics["avg_turnover"],
        })
        print(f"  RP+LB({lb}): Net SR={combo_metrics['sharpe_net']:.4f}, "
              f"Return={combo_metrics['return_net']:.4f}")

    # Best factor timing + best risk-parity
    best_ft_hl = best_ft["factor_ema_hl"]
    ft_preds, ft_actuals, _, _ = run_factor_timing_walkforward(
        X_us, Y_jp, dates_us, evaluator, factor_ema_halflife=best_ft_hl
    )

    for rp_on in [False, True]:
        for lb in [0.0, 0.5]:
            combo_metrics = evaluate_strategy(
                ft_preds, ft_actuals, top5_mask, ema_halflife=20,
                risk_parity=rp_on,
                risk_parity_lookback=best_rp["rp_lookback"] if rp_on else 63,
                long_bias=lb if lb > 0 else None,
            )
            label = f"FT(hl={best_ft_hl})"
            if rp_on:
                label += f"+RP(lb={best_rp['rp_lookback']})"
            if lb > 0:
                label += f"+LB({lb})"
            combined_results.append({
                "config": label,
                "risk_parity_lookback": best_rp["rp_lookback"] if rp_on else None,
                "long_bias": lb,
                "factor_timing": True,
                "sharpe_net": combo_metrics["sharpe_net"],
                "return_net": combo_metrics["return_net"],
                "max_dd_net": combo_metrics["max_dd_net"],
                "avg_turnover": combo_metrics["avg_turnover"],
            })
            print(f"  {label}: Net SR={combo_metrics['sharpe_net']:.4f}, "
                  f"Return={combo_metrics['return_net']:.4f}")

    combined_results.sort(key=lambda x: x["sharpe_net"], reverse=True)
    best_combo = combined_results[0]
    print(f"\n  Best combined: {best_combo['config']}, "
          f"Net SR={best_combo['sharpe_net']:.4f}")

    # ====================================================================
    # Section E: Borrowing cost sensitivity for best configs
    # ====================================================================
    print("\n[7/8] Borrowing cost sensitivity for best configs...")

    borrow_results = {}

    for borrow_bps in [0, 75]:
        # C12 baseline
        c12_b = evaluate_strategy(preds_pca, actuals, top5_mask, ema_halflife=20,
                                  borrow_bps=borrow_bps)
        borrow_results[f"C12_borrow{borrow_bps}"] = {
            "sharpe_net": c12_b["sharpe_net"],
            "return_net": c12_b["return_net"],
        }

        # Best risk-parity
        rp_b = evaluate_strategy(preds_pca, actuals, top5_mask, ema_halflife=20,
                                 risk_parity=True,
                                 risk_parity_lookback=best_rp["rp_lookback"],
                                 borrow_bps=borrow_bps)
        borrow_results[f"best_RP_borrow{borrow_bps}"] = {
            "sharpe_net": rp_b["sharpe_net"],
            "return_net": rp_b["return_net"],
        }

        # Best long bias
        lb_val = best_lb["long_bias"]
        lb_b = evaluate_strategy(preds_pca, actuals, top5_mask, ema_halflife=20,
                                 long_bias=lb_val if lb_val > 0 else None,
                                 borrow_bps=borrow_bps)
        borrow_results[f"best_LB({lb_val})_borrow{borrow_bps}"] = {
            "sharpe_net": lb_b["sharpe_net"],
            "return_net": lb_b["return_net"],
        }

    for k, v in borrow_results.items():
        print(f"  {k}: Net SR={v['sharpe_net']:.4f}, Return={v['return_net']:.4f}")

    # ====================================================================
    # Save results
    # ====================================================================
    print("\n[8/8] Saving results...")
    report_dir = Path("reports/cycle_16")
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "phase": 16,
        "description": "Factor timing, risk-parity portfolio construction, and long-bias strategy",
        "timestamp": "2026-03-27",
        "c12_baseline": {
            "sharpe_net": c12_metrics["sharpe_net"],
            "return_net": c12_metrics["return_net"],
            "avg_turnover": c12_metrics["avg_turnover"],
        },
        "factor_timing": {
            "configs_tested": len(factor_timing_results),
            "results": factor_timing_results,
            "best": {
                "factor_ema_hl": best_ft["factor_ema_hl"],
                "sharpe_net": best_ft["sharpe_net"],
                "direction_accuracy": best_ft["direction_accuracy"],
                "improvement_over_c12": round(best_ft["sharpe_net"] - c12_metrics["sharpe_net"], 4),
            },
        },
        "risk_parity": {
            "configs_tested": len(rp_results),
            "results": rp_results,
            "best": {
                "lookback": best_rp["rp_lookback"],
                "sharpe_net": best_rp["sharpe_net"],
                "improvement_over_c12": round(best_rp["sharpe_net"] - c12_metrics["sharpe_net"], 4),
            },
        },
        "long_bias": {
            "configs_tested": len(long_bias_results),
            "results": long_bias_results,
            "best": {
                "long_bias": best_lb["long_bias"],
                "sharpe_net": best_lb["sharpe_net"],
                "improvement_over_c12": round(best_lb["sharpe_net"] - c12_metrics["sharpe_net"], 4),
            },
        },
        "combined_approaches": {
            "configs_tested": len(combined_results),
            "results": combined_results,
            "best": {
                "config": best_combo["config"],
                "sharpe_net": best_combo["sharpe_net"],
                "improvement_over_c12": round(best_combo["sharpe_net"] - c12_metrics["sharpe_net"], 4),
            },
        },
        "borrowing_cost_impact": borrow_results,
        "walk_forward_params": {
            "K": 5, "L": 120, "lambda_decay": 1.0,
            "train_window": 252, "test_window": 21,
            "n_folds": wf_pca.n_folds,
            "total_test_samples": wf_pca.total_test_samples,
        },
        "cost_assumptions": {
            "one_way_transaction_bps": 10.0,
            "realistic_borrow_bps": 75,
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
        c12_metrics, factor_timing_results, best_ft,
        rp_results, best_rp, long_bias_results, best_lb,
        combined_results, best_combo, borrow_results, wf_pca,
        per_sector_acc, ranked, jp_tickers
    )
    with open(report_dir / "technical_findings.md", "w") as f:
        f.write(findings)
    print(f"  Saved findings to {report_dir / 'technical_findings.md'}")

    print("\n" + "=" * 70)
    print("Cycle 16 complete!")
    print(f"Best overall: {best_combo['config']}, Net SR={best_combo['sharpe_net']:.4f}")
    print(f"C12 baseline: Net SR={c12_metrics['sharpe_net']:.4f}")
    print(f"Delta: {best_combo['sharpe_net'] - c12_metrics['sharpe_net']:+.4f}")
    print("=" * 70)


def generate_technical_findings(c12, ft_results, best_ft, rp_results, best_rp,
                                 lb_results, best_lb, combined, best_combo,
                                 borrow, wf_pca, per_sector_acc, ranked, jp_tickers):
    """Generate markdown technical findings report."""

    lines = [
        "# Cycle 16: Factor Timing, Risk-Parity & Long-Bias Strategy",
        "",
        "## Summary",
        "",
        "This cycle addresses three open questions from prior cycles:",
        "",
        "1. **Factor timing** (Q38) — Weight individual PCA factors by their recent",
        "   predictive accuracy instead of treating all K=5 components equally.",
        "   Different PCs may capture different economic dynamics (market vs rotation",
        "   vs sector-specific), and their predictive power may vary over time.",
        "",
        "2. **Risk-parity position sizing** — Replace equal-weight positions with",
        "   inverse-volatility weighting. Lower-volatility sectors receive larger",
        "   allocations, balancing risk contributions across the portfolio.",
        "",
        "3. **Long-biased strategy** — Phase 7 showed the lead-lag signal has strong",
        "   directional asymmetry (up-market SR=1.31 vs down-market SR=-0.18). A",
        "   strategy that reduces short exposure may improve risk-adjusted returns",
        "   while also reducing borrowing costs.",
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
        "### 1. Factor Timing",
        "",
        "Per-PC weight adaptation tracks which principal components are currently",
        "predictive and upweights them dynamically. After each walk-forward fold,",
        "per-factor directional accuracy is measured and weights are updated via",
        "softmax scaling.",
        "",
        "| Factor EMA HL | Dir Accuracy | Net Sharpe | Net Return | Max DD | Turnover | Final PC Weights |",
        "|--------------|-------------|------------|------------|--------|----------|-----------------|",
    ]

    for ft in ft_results:
        weights_str = ", ".join([f"{w:.3f}" for w in ft["avg_final_factor_weights"]])
        lines.append(
            f"| {ft['factor_ema_hl']} | {ft['direction_accuracy']} | {ft['sharpe_net']} | "
            f"{ft['return_net']} | {ft['max_dd_net']} | {ft['avg_turnover']} | [{weights_str}] |"
        )

    lines += [
        "",
        f"**Best factor timing**: EMA half-life={best_ft['factor_ema_hl']}",
        f"- Net Sharpe: {best_ft['sharpe_net']} "
        f"(vs C12 {c12['sharpe_net']}, delta {best_ft['sharpe_net'] - c12['sharpe_net']:+.4f})",
        "",
        "### 2. Risk-Parity Position Sizing",
        "",
        "Inverse-volatility weighting sizes positions so that lower-volatility",
        "sectors receive larger allocations, balancing risk contributions.",
        "",
        "| RP Lookback | Net Sharpe | Net Return | Max DD | Turnover | Ann Return | Ann Vol |",
        "|------------|------------|------------|--------|----------|------------|---------|",
    ]

    for rp in rp_results:
        lines.append(
            f"| {rp['rp_lookback']} | {rp['sharpe_net']} | {rp['return_net']} | "
            f"{rp['max_dd_net']} | {rp['avg_turnover']} | {rp['ann_return_net']} | {rp['ann_vol_net']} |"
        )

    lines += [
        "",
        f"**Best risk-parity**: lookback={best_rp['rp_lookback']}",
        f"- Net Sharpe: {best_rp['sharpe_net']} "
        f"(vs C12 {c12['sharpe_net']}, delta {best_rp['sharpe_net'] - c12['sharpe_net']:+.4f})",
        "",
        "### 3. Long-Biased Strategy",
        "",
        "Phase 7 confirmed the lead-lag signal is asymmetric: stronger for long",
        "positions. We test scaling down short exposure by a long_bias factor.",
        "long_bias=0 is standard long-short; long_bias=1.0 is long-only.",
        "",
        "| Long Bias | Net Sharpe | Net Return | Max DD | Turnover | % Positive Days |",
        "|-----------|------------|------------|--------|----------|----------------|",
    ]

    for lb in lb_results:
        lines.append(
            f"| {lb['long_bias']} | {lb['sharpe_net']} | {lb['return_net']} | "
            f"{lb['max_dd_net']} | {lb['avg_turnover']} | {lb['pct_pos_net']} |"
        )

    lines += [
        "",
        f"**Best long bias**: {best_lb['long_bias']}",
        f"- Net Sharpe: {best_lb['sharpe_net']} "
        f"(vs C12 {c12['sharpe_net']}, delta {best_lb['sharpe_net'] - c12['sharpe_net']:+.4f})",
        "",
        "### 4. Combined Approaches",
        "",
        "| Config | Net Sharpe | Net Return | Max DD | Turnover |",
        "|--------|------------|------------|--------|----------|",
    ]

    for c in combined:
        lines.append(
            f"| {c['config']} | {c['sharpe_net']} | {c['return_net']} | "
            f"{c['max_dd_net']} | {c['avg_turnover']} |"
        )

    lines += [
        "",
        f"**Best combined**: {best_combo['config']}",
        f"- Net Sharpe: {best_combo['sharpe_net']} "
        f"(vs C12 {c12['sharpe_net']}, delta {best_combo['sharpe_net'] - c12['sharpe_net']:+.4f})",
        "",
        "### 5. Borrowing Cost Sensitivity",
        "",
        "| Config | SR(net) | Return(net) |",
        "|--------|---------|-------------|",
    ]

    for k, v in borrow.items():
        lines.append(f"| {k} | {v['sharpe_net']} | {v['return_net']} |")

    lines += [
        "",
        "## Methodology",
        "",
        "### Factor Timing",
        "- PCA_SUB decomposes predictions into per-PC contributions: each PC's",
        "  score times its regression coefficient.",
        "- After each walk-forward fold, per-PC directional accuracy is measured.",
        "- Factor weights are updated via softmax of excess accuracy (accuracy - 0.5),",
        "  so PCs near 50% accuracy receive low weight.",
        "- Weights carry over between folds, providing momentum in factor selection.",
        "",
        "### Risk-Parity",
        "- For each day, compute each sector's realized volatility over a lookback window.",
        "- Position sizes are proportional to 1/volatility (inverse-vol weighting).",
        "- Positions are then normalized to sum of absolute weights = 1.",
        "- This balances risk contributions across sectors rather than capital allocation.",
        "",
        "### Long Bias",
        "- Short positions are scaled by (1 - long_bias).",
        "- long_bias=0.0: standard long-short (50% long, 50% short).",
        "- long_bias=0.5: partial long bias (shorts halved).",
        "- long_bias=1.0: long-only (all shorts eliminated).",
        "- Reduces borrowing costs proportionally to reduced short exposure.",
        "",
        "### Walk-Forward Setup",
        f"- **Train window**: 252 days, **Test window**: 21 days",
        f"- **Folds**: {wf_pca.n_folds}",
        f"- **Total OOS samples**: {wf_pca.total_test_samples}",
        "",
        "## Conclusions",
        "",
        "1. **Factor timing** reveals whether the 5 PCA components have heterogeneous",
        "   and time-varying predictive power, addressing the hypothesis that PC1 (market)",
        "   may dominate while sector-rotation factors (PC2-5) are noisier.",
        "",
        "2. **Risk-parity** tests whether balancing risk contributions across the top-5",
        "   sectors — rather than equal capital allocation — improves the Sharpe ratio.",
        "   Steel & Nonferrous (highest accuracy) also tends to be high-volatility,",
        "   so equal-weight may over-allocate risk to this sector.",
        "",
        "3. **Long bias** exploits the known signal asymmetry from Phase 7. If the",
        "   lead-lag signal primarily predicts positive returns (rather than negative),",
        "   reducing short exposure improves returns and reduces borrowing costs.",
        "",
        "## Open Questions for Future Cycles",
        "",
        "1. **Sector-specific long bias**: Could each sector have a different optimal",
        "   long/short asymmetry based on its individual signal characteristics?",
        "2. **Factor timing with regime conditioning**: Could factor weights be",
        "   conditioned on volatility regimes rather than just recent accuracy?",
        "3. **Transaction cost-aware rebalancing**: Could the strategy skip small",
        "   position changes when the expected alpha is below the trading cost?",
        "4. **Out-of-sample validation**: All improvements need forward validation",
        "   on post-2026 data to confirm robustness.",
    ]

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
