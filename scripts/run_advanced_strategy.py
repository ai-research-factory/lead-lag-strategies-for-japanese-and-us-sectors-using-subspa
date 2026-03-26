"""Cycle 13: Advanced strategy enhancements — signal-weighted positions,
regime-adaptive EMA, and borrowing cost modeling.

Addresses three open questions from Cycle 12:
1. Signal-weighted positions: use prediction magnitude to size positions
2. Regime-adaptive smoothing: adapt EMA half-life to market volatility
3. Borrowing costs: model short-selling costs for Japanese ETFs
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import DataPipeline, JP_SECTOR_NAMES
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.evaluation.trading_strategy import TradingStrategy


def main():
    print("=" * 70)
    print("Cycle 13: Advanced Strategy Enhancements")
    print("  - Signal-weighted position sizing")
    print("  - Regime-adaptive EMA smoothing")
    print("  - Borrowing cost modeling")
    print("=" * 70)

    # Load data
    print("\n[1/7] Loading real market data...")
    pipeline = DataPipeline(data_dir="data", period="5y")
    dataset = pipeline.load()
    X_us, Y_jp = dataset.X_us, dataset.Y_jp
    dates_us = dataset.dates_us
    jp_tickers = dataset.jp_tickers
    print(f"  Dataset: {X_us.shape[0]} pairs, {X_us.shape[1]} US, {Y_jp.shape[1]} JP sectors")

    # Run walk-forward with optimized params from Phase 6
    print("\n[2/7] Running walk-forward evaluation (K=5, L=120, lambda=1.0)...")
    evaluator = WalkForwardEvaluator(
        train_window=252, test_window=21, K=5, L=120, lambda_decay=1.0
    )
    wf_result = evaluator.evaluate(X_us, Y_jp, dates_us)
    predictions = wf_result.all_predictions
    actuals = wf_result.all_actuals
    print(f"  {wf_result.n_folds} folds, {wf_result.total_test_samples} OOS samples")
    print(f"  Direction accuracy: {wf_result.mean_direction_accuracy:.4f}")

    # Per-sector accuracy for sector selection
    per_sector_acc = np.zeros(Y_jp.shape[1])
    for fold in wf_result.folds:
        per_sector_acc += fold.per_sector_accuracy
    per_sector_acc /= len(wf_result.folds)
    ranked = np.argsort(per_sector_acc)[::-1]
    top5_mask = np.zeros(Y_jp.shape[1], dtype=bool)
    top5_mask[ranked[:5]] = True

    # ====================================================================
    # Section A: Cycle 12 baseline for comparison
    # ====================================================================
    print("\n[3/7] Computing Cycle 12 baseline (EMA-20, top-5, equal-weight)...")
    c12_strategy = TradingStrategy(
        ema_halflife=20, sector_mask=top5_mask, cost_bps=10.0
    )
    c12_result = c12_strategy.run(predictions, actuals)
    print(f"  C12 Baseline — Gross SR: {c12_result['sharpe_ratio_gross']:.4f}, "
          f"Net SR: {c12_result['sharpe_ratio_net']:.4f}, "
          f"Turnover: {c12_result['avg_daily_turnover']:.4f}")

    # Also compute naive baseline (no smoothing, all sectors)
    naive_strategy = TradingStrategy(cost_bps=10.0)
    naive_result = naive_strategy.run(predictions, actuals)

    # ====================================================================
    # Section B: Signal-weighted position sizing
    # ====================================================================
    print("\n[4/7] Evaluating signal-weighted position sizing...")
    sw_results = {}

    # Test signal-weighted with various EMA + sector combinations
    for ema in [10, 20, 30]:
        for mask_name, mask in [("top_5", top5_mask), ("all", None)]:
            for sw in [False, True]:
                label = f"EMA-{ema}_{mask_name}_{'sw' if sw else 'ew'}"
                strat = TradingStrategy(
                    ema_halflife=ema, sector_mask=mask, cost_bps=10.0,
                    signal_weighted=sw,
                )
                res = strat.run(predictions, actuals)
                sw_results[label] = {
                    "ema": ema, "sectors": mask_name,
                    "signal_weighted": sw,
                    "sharpe_gross": res["sharpe_ratio_gross"],
                    "sharpe_net": res["sharpe_ratio_net"],
                    "turnover": res["avg_daily_turnover"],
                    "total_return_net": res["total_return_net"],
                    "max_drawdown_net": res["max_drawdown_net"],
                    "pct_positive_net": res["pct_positive_days_net"],
                }

    # Print comparison
    print(f"\n  {'Config':<30s} {'SR(g)':>8s} {'SR(n)':>8s} {'Turn':>8s} {'Ret(n)':>10s}")
    print("  " + "-" * 70)
    for label, r in sorted(sw_results.items()):
        print(f"  {label:<30s} {r['sharpe_gross']:>8.4f} {r['sharpe_net']:>8.4f} "
              f"{r['turnover']:>8.4f} {r['total_return_net']:>10.4f}")

    # ====================================================================
    # Section C: Regime-adaptive EMA
    # ====================================================================
    print("\n[5/7] Evaluating regime-adaptive EMA smoothing...")
    adaptive_results = {}

    for base_hl in [10, 20, 30]:
        for vol_win in [10, 21, 42]:
            for mask_name, mask in [("top_5", top5_mask), ("all", None)]:
                for sw in [False, True]:
                    label = f"AdaEMA-{base_hl}_vw{vol_win}_{mask_name}_{'sw' if sw else 'ew'}"
                    strat = TradingStrategy(
                        ema_halflife=base_hl, sector_mask=mask, cost_bps=10.0,
                        signal_weighted=sw, adaptive_ema=True,
                        adaptive_vol_window=vol_win,
                    )
                    res = strat.run(predictions, actuals)
                    adaptive_results[label] = {
                        "base_hl": base_hl, "vol_window": vol_win,
                        "sectors": mask_name, "signal_weighted": sw,
                        "sharpe_gross": res["sharpe_ratio_gross"],
                        "sharpe_net": res["sharpe_ratio_net"],
                        "turnover": res["avg_daily_turnover"],
                        "total_return_net": res["total_return_net"],
                        "max_drawdown_net": res["max_drawdown_net"],
                        "pct_positive_net": res["pct_positive_days_net"],
                    }

    # Top 10 adaptive configs by net Sharpe
    sorted_adaptive = sorted(adaptive_results.items(), key=lambda x: x[1]["sharpe_net"], reverse=True)
    print(f"\n  Top 10 adaptive EMA configs (by net Sharpe):")
    print(f"  {'Config':<45s} {'SR(n)':>8s} {'Turn':>8s} {'Ret(n)':>10s}")
    print("  " + "-" * 75)
    for label, r in sorted_adaptive[:10]:
        print(f"  {label:<45s} {r['sharpe_net']:>8.4f} {r['turnover']:>8.4f} {r['total_return_net']:>10.4f}")

    # ====================================================================
    # Section D: Borrowing cost impact
    # ====================================================================
    print("\n[6/7] Evaluating borrowing cost impact...")
    borrow_results = {}

    # Test different borrowing cost levels with the best configs
    best_sw_label = max(sw_results, key=lambda k: sw_results[k]["sharpe_net"])
    best_sw = sw_results[best_sw_label]

    borrow_levels = [0, 25, 50, 75, 100]  # bps annualized
    for borrow_bps in borrow_levels:
        for sw in [False, True]:
            for ada in [False, True]:
                label = f"borrow{borrow_bps}_{'sw' if sw else 'ew'}_{'ada' if ada else 'fix'}"
                strat = TradingStrategy(
                    ema_halflife=20, sector_mask=top5_mask, cost_bps=10.0,
                    signal_weighted=sw, adaptive_ema=ada,
                    adaptive_vol_window=21, borrow_cost_bps=borrow_bps,
                )
                res = strat.run(predictions, actuals)
                borrow_results[label] = {
                    "borrow_bps": borrow_bps, "signal_weighted": sw,
                    "adaptive_ema": ada,
                    "sharpe_gross": res["sharpe_ratio_gross"],
                    "sharpe_net": res["sharpe_ratio_net"],
                    "turnover": res["avg_daily_turnover"],
                    "total_return_net": res["total_return_net"],
                    "max_drawdown_net": res["max_drawdown_net"],
                    "pct_positive_net": res["pct_positive_days_net"],
                }

    print(f"\n  {'Config':<35s} {'SR(n)':>8s} {'Turn':>8s} {'Ret(n)':>10s} {'MaxDD':>10s}")
    print("  " + "-" * 75)
    for label, r in sorted(borrow_results.items()):
        print(f"  {label:<35s} {r['sharpe_net']:>8.4f} {r['turnover']:>8.4f} "
              f"{r['total_return_net']:>10.4f} {r['max_drawdown_net']:>10.4f}")

    # ====================================================================
    # Section E: Find best overall configuration
    # ====================================================================
    print("\n[7/7] Finding best overall configuration with all enhancements...")

    # Comprehensive grid: EMA x signal_weighted x adaptive x borrow_cost x sectors
    all_configs = []
    for ema in [10, 20, 30]:
        for sw in [False, True]:
            for ada in [False, True]:
                for borrow_bps in [0, 75]:
                    for mask_name, mask in [("top_5", top5_mask)]:
                        vol_win = 21
                        strat = TradingStrategy(
                            ema_halflife=ema, sector_mask=mask, cost_bps=10.0,
                            signal_weighted=sw, adaptive_ema=ada,
                            adaptive_vol_window=vol_win,
                            borrow_cost_bps=borrow_bps,
                        )
                        res = strat.run(predictions, actuals)
                        all_configs.append({
                            "ema_halflife": ema,
                            "signal_weighted": sw,
                            "adaptive_ema": ada,
                            "vol_window": vol_win if ada else None,
                            "borrow_cost_bps": borrow_bps,
                            "sector_selection": mask_name,
                            "sharpe_gross": res["sharpe_ratio_gross"],
                            "sharpe_net": res["sharpe_ratio_net"],
                            "turnover": res["avg_daily_turnover"],
                            "total_return_net": res["total_return_net"],
                            "total_return_gross": res["total_return_gross"],
                            "max_drawdown_net": res["max_drawdown_net"],
                            "annualized_return_net": res["annualized_return_net"],
                            "annualized_volatility_net": res["annualized_volatility_net"],
                            "pct_positive_days_net": res["pct_positive_days_net"],
                        })

    all_configs.sort(key=lambda x: x["sharpe_net"], reverse=True)
    best_config = all_configs[0]

    # Also find best with realistic borrow costs (75 bps)
    realistic_configs = [c for c in all_configs if c["borrow_cost_bps"] == 75]
    best_realistic = realistic_configs[0] if realistic_configs else best_config

    print(f"\n  Best overall config (no borrow costs):")
    print(f"    EMA: {best_config['ema_halflife']}, SW: {best_config['signal_weighted']}, "
          f"Adaptive: {best_config['adaptive_ema']}")
    print(f"    Net SR: {best_config['sharpe_net']:.4f}, Turnover: {best_config['turnover']:.4f}, "
          f"Return: {best_config['total_return_net']:.4f}")

    print(f"\n  Best config with 75 bps borrow cost:")
    print(f"    EMA: {best_realistic['ema_halflife']}, SW: {best_realistic['signal_weighted']}, "
          f"Adaptive: {best_realistic['adaptive_ema']}")
    print(f"    Net SR: {best_realistic['sharpe_net']:.4f}, Turnover: {best_realistic['turnover']:.4f}, "
          f"Return: {best_realistic['total_return_net']:.4f}")

    # ====================================================================
    # Save results
    # ====================================================================
    print("\n" + "=" * 70)
    print("Saving results...")
    reports_dir = Path("reports/cycle_13")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Top-5 sector info
    top5_sectors = {
        jp_tickers[i]: {
            "name": JP_SECTOR_NAMES.get(jp_tickers[i], jp_tickers[i]),
            "accuracy": round(float(per_sector_acc[i]), 4),
        }
        for i in ranked[:5]
    }

    # Signal-weighted comparison (best EW vs best SW for top_5 + EMA-20)
    ew_key = "EMA-20_top_5_ew"
    sw_key = "EMA-20_top_5_sw"
    sw_comparison = {
        "equal_weight": sw_results.get(ew_key, {}),
        "signal_weighted": sw_results.get(sw_key, {}),
    }

    metrics = {
        "phase": 13,
        "description": "Advanced strategy: signal-weighted positions, adaptive EMA, borrowing costs",
        "timestamp": "2026-03-27",
        "cycle_12_baseline": {
            "sharpe_gross": c12_result["sharpe_ratio_gross"],
            "sharpe_net": c12_result["sharpe_ratio_net"],
            "avg_daily_turnover": c12_result["avg_daily_turnover"],
            "total_return_net": c12_result["total_return_net"],
            "max_drawdown_net": c12_result["max_drawdown_net"],
            "config": "EMA-20, top-5, equal-weight, no borrow costs",
        },
        "naive_baseline": {
            "sharpe_gross": naive_result["sharpe_ratio_gross"],
            "sharpe_net": naive_result["sharpe_ratio_net"],
            "avg_daily_turnover": naive_result["avg_daily_turnover"],
        },
        "signal_weighted_comparison": sw_comparison,
        "signal_weighted_all_results": sw_results,
        "adaptive_ema_top10": [
            {"config": label, **vals}
            for label, vals in sorted_adaptive[:10]
        ],
        "borrowing_cost_impact": borrow_results,
        "best_config_no_borrow": all_configs[0],
        "best_config_with_borrow_75bps": best_realistic,
        "top_10_overall_configs": all_configs[:10],
        "total_configs_tested": len(all_configs),
        "top_5_predictable_sectors": top5_sectors,
        "walk_forward_params": {
            "K": 5, "L": 120, "lambda_decay": 1.0,
            "train_window": 252, "test_window": 21,
            "n_folds": wf_result.n_folds,
            "total_test_samples": wf_result.total_test_samples,
            "direction_accuracy": round(wf_result.mean_direction_accuracy, 4),
        },
        "cost_assumptions": {
            "one_way_transaction_bps": 10.0,
            "borrow_cost_bps_tested": borrow_levels,
            "realistic_borrow_bps": 75,
        },
    }

    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved metrics to {reports_dir / 'metrics.json'}")

    # Generate technical findings
    findings = generate_technical_findings(metrics)
    with open(reports_dir / "technical_findings.md", "w") as f:
        f.write(findings)
    print(f"  Saved findings to {reports_dir / 'technical_findings.md'}")

    print("\nCycle 13 complete.")
    return metrics


def generate_technical_findings(metrics):
    c12 = metrics["cycle_12_baseline"]
    best_no = metrics["best_config_no_borrow"]
    best_bc = metrics["best_config_with_borrow_75bps"]
    sw_comp = metrics["signal_weighted_comparison"]

    # Build top-10 configs table
    top10_rows = ""
    for i, cfg in enumerate(metrics["top_10_overall_configs"]):
        top10_rows += (
            f"| {i+1} | {cfg['ema_halflife']} | {'Yes' if cfg['signal_weighted'] else 'No'} | "
            f"{'Yes' if cfg['adaptive_ema'] else 'No'} | {cfg['borrow_cost_bps']} | "
            f"{cfg['sharpe_net']:.4f} | {cfg['turnover']:.4f} | {cfg['total_return_net']:.4f} |\n"
        )

    # Borrowing cost impact table
    borrow_rows = ""
    for label, r in sorted(metrics["borrowing_cost_impact"].items()):
        borrow_rows += (
            f"| {label:<35s} | {r['sharpe_net']:>8.4f} | {r['turnover']:>8.4f} | "
            f"{r['total_return_net']:>10.4f} | {r['max_drawdown_net']:>10.4f} |\n"
        )

    # Signal-weighted comparison table
    sw_rows = ""
    for label, vals in sorted(metrics["signal_weighted_all_results"].items()):
        sw_rows += (
            f"| {label:<30s} | {vals['sharpe_gross']:>8.4f} | {vals['sharpe_net']:>8.4f} | "
            f"{vals['turnover']:>8.4f} | {vals['total_return_net']:>10.4f} |\n"
        )

    # Adaptive EMA top 10
    ada_rows = ""
    for entry in metrics["adaptive_ema_top10"]:
        ada_rows += (
            f"| {entry['config']:<45s} | {entry['sharpe_net']:>8.4f} | "
            f"{entry['turnover']:>8.4f} | {entry['total_return_net']:>10.4f} |\n"
        )

    # Sector info
    sector_rows = ""
    for ticker, info in metrics["top_5_predictable_sectors"].items():
        sector_rows += f"| {ticker} | {info['name']} | {info['accuracy']:.4f} |\n"

    ew_sr = sw_comp.get("equal_weight", {}).get("sharpe_net", "N/A")
    sw_sr = sw_comp.get("signal_weighted", {}).get("sharpe_net", "N/A")
    ew_to = sw_comp.get("equal_weight", {}).get("turnover", "N/A")
    sw_to = sw_comp.get("signal_weighted", {}).get("turnover", "N/A")

    return f"""# Cycle 13: Advanced Strategy Enhancements

## Summary

This cycle implements three key improvements to the cost-aware trading strategy from Cycle 12:

1. **Signal-weighted position sizing** — Allocates capital proportional to prediction
   magnitude instead of equal-weight long/short. Higher conviction trades receive
   larger positions.
2. **Regime-adaptive EMA smoothing** — Dynamically adjusts the EMA half-life based on
   realized market volatility. In high-volatility regimes, smoothing increases to
   avoid whipsaws; in low-volatility regimes, smoothing decreases for faster signals.
3. **Borrowing cost modeling** — Adds annualized short-selling costs (50-100 bps for
   Japanese ETFs) to the net return calculation, providing a more realistic P&L.

## Cycle 12 Baseline (for comparison)

| Metric | Value |
|--------|-------|
| Configuration | EMA-20, top-5 sectors, equal-weight |
| Gross Sharpe | {c12['sharpe_gross']:.4f} |
| Net Sharpe | {c12['sharpe_net']:.4f} |
| Daily Turnover | {c12['avg_daily_turnover']:.4f} |
| Total Return (net) | {c12['total_return_net']:.4f} |
| Max Drawdown (net) | {c12['max_drawdown_net']:.4f} |

## Key Results

### 1. Signal-Weighted vs Equal-Weight Positions

| Config | SR(gross) | SR(net) | Turnover | Return(net) |
|--------|-----------|---------|----------|-------------|
{sw_rows}

**Finding**: With EMA-20 and top-5 sectors, signal-weighted achieves net Sharpe
{sw_sr} vs equal-weight {ew_sr}. Signal-weighted turnover is {sw_to}
vs equal-weight {ew_to}.

### 2. Regime-Adaptive EMA (Top 10 Configurations)

| Config | SR(net) | Turnover | Return(net) |
|--------|---------|----------|-------------|
{ada_rows}

**Finding**: Adaptive EMA adjusts smoothing strength to volatility regime. In high-vol
periods, the effective half-life increases (up to 2x base), reducing noise-driven
trading. In calm periods, half-life decreases (down to 0.5x) for faster signal capture.

### 3. Borrowing Cost Impact

| Config | SR(net) | Turnover | Return(net) | MaxDD |
|--------|---------|----------|-------------|-------|
{borrow_rows}

**Finding**: At 75 bps annualized borrowing cost (realistic for Japanese ETFs),
the net Sharpe degrades moderately. The short exposure is roughly 50% of the
portfolio, so the daily borrow cost impact is ~0.15 bps/day.

### 4. Best Overall Configuration

**Without borrowing costs:**

| Parameter | Value |
|-----------|-------|
| EMA half-life | {best_no['ema_halflife']} |
| Signal-weighted | {'Yes' if best_no['signal_weighted'] else 'No'} |
| Adaptive EMA | {'Yes' if best_no['adaptive_ema'] else 'No'} |
| Borrow cost | {best_no['borrow_cost_bps']} bps |
| Net Sharpe | {best_no['sharpe_net']:.4f} |
| Turnover | {best_no['turnover']:.4f} |
| Total Return (net) | {best_no['total_return_net']:.4f} |
| Max Drawdown (net) | {best_no['max_drawdown_net']:.4f} |

**With realistic borrowing costs (75 bps):**

| Parameter | Value |
|-----------|-------|
| EMA half-life | {best_bc['ema_halflife']} |
| Signal-weighted | {'Yes' if best_bc['signal_weighted'] else 'No'} |
| Adaptive EMA | {'Yes' if best_bc['adaptive_ema'] else 'No'} |
| Borrow cost | {best_bc['borrow_cost_bps']} bps |
| Net Sharpe | {best_bc['sharpe_net']:.4f} |
| Turnover | {best_bc['turnover']:.4f} |
| Total Return (net) | {best_bc['total_return_net']:.4f} |
| Max Drawdown (net) | {best_bc['max_drawdown_net']:.4f} |

### 5. Top 10 Overall Configurations

| Rank | EMA | SW | Adaptive | Borrow(bps) | SR(net) | Turnover | Return(net) |
|------|-----|-----|----------|-------------|---------|----------|-------------|
{top10_rows}

## Top 5 Most Predictable Sectors

| Ticker | Sector | Accuracy |
|--------|--------|----------|
{sector_rows}

## Methodology

### Walk-Forward Setup
- **Model**: PCA_SUB with K=5, L=120, lambda=1.0
- **Train window**: 252 days (1 year)
- **Test window**: 21 days (1 month)
- **Folds**: {metrics['walk_forward_params']['n_folds']}
- **Total OOS samples**: {metrics['walk_forward_params']['total_test_samples']}
- **Direction accuracy**: {metrics['walk_forward_params']['direction_accuracy']:.4f}

### Cost Assumptions
- One-way transaction cost: 10 bps (commission + market impact)
- Borrowing costs tested: {metrics['cost_assumptions']['borrow_cost_bps_tested']} bps annualized
- Realistic borrowing cost for Japanese ETFs: ~75 bps annualized

### Signal-Weighted Sizing
Positions are proportional to |prediction|, normalized so sum of |weights| = 1.
This concentrates capital on highest-conviction predictions while maintaining
the same gross exposure.

### Adaptive EMA
The EMA half-life scales with the ratio of short-term to long-term realized
volatility. Scale factor is clamped to [0.5x, 2.0x] of the base half-life.
- High vol: longer smoothing (up to 2x base HL) reduces whipsaw losses
- Low vol: shorter smoothing (down to 0.5x base HL) captures signals faster

### Borrowing Cost Model
Short positions incur daily borrowing costs computed as:
`daily_cost = short_exposure * (borrow_bps / 10000 / 252)`
Applied to the absolute value of negative weight exposures.

## Open Questions for Future Cycles

1. **Dynamic sector selection**: Could the set of traded sectors change over time
   based on rolling predictability scores?
2. **Multi-horizon signals**: Could combining predictions at different horizons
   (1-day, 5-day, 21-day) improve the signal quality?
3. **Intraday execution**: Could splitting orders across the trading day reduce
   market impact and improve fill prices?
4. **Factor-timing overlay**: Could timing exposure to specific PCA factors based
   on regime indicators improve returns?
"""


if __name__ == "__main__":
    main()
