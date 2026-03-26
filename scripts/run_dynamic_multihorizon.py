"""Cycle 14: Dynamic sector selection and multi-horizon signal ensemble.

Addresses two key open questions from Cycles 12-13:
1. Dynamic sector selection: re-evaluate which sectors to trade based on
   rolling predictability scores instead of a fixed top-5 set.
2. Multi-horizon signals: combine prediction moving averages at different
   windows (1-day, 5-day, 10-day) for a more stable ensemble signal.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import DataPipeline, JP_SECTOR_NAMES, JP_TICKERS
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.evaluation.trading_strategy import TradingStrategy
from src.evaluation.dynamic_sectors import DynamicSectorSelector, DynamicTradingStrategy
from src.evaluation.multi_horizon import MultiHorizonEnsemble


def main():
    print("=" * 70)
    print("Cycle 14: Dynamic Sector Selection & Multi-Horizon Signals")
    print("=" * 70)

    # ====================================================================
    # Load data and run walk-forward
    # ====================================================================
    print("\n[1/8] Loading real market data...")
    pipeline = DataPipeline(data_dir="data", period="5y")
    dataset = pipeline.load()
    X_us, Y_jp = dataset.X_us, dataset.Y_jp
    dates_us = dataset.dates_us
    jp_tickers = dataset.jp_tickers
    print(f"  Dataset: {X_us.shape[0]} pairs, {X_us.shape[1]} US, {Y_jp.shape[1]} JP sectors")

    print("\n[2/8] Running walk-forward evaluation (K=5, L=120, lambda=1.0)...")
    evaluator = WalkForwardEvaluator(
        train_window=252, test_window=21, K=5, L=120, lambda_decay=1.0
    )
    wf_result = evaluator.evaluate(X_us, Y_jp, dates_us)
    predictions = wf_result.all_predictions
    actuals = wf_result.all_actuals
    print(f"  {wf_result.n_folds} folds, {wf_result.total_test_samples} OOS samples")
    print(f"  Direction accuracy: {wf_result.mean_direction_accuracy:.4f}")

    # Per-sector accuracy for fixed top-5 baseline
    per_sector_acc = np.zeros(Y_jp.shape[1])
    for fold in wf_result.folds:
        per_sector_acc += fold.per_sector_accuracy
    per_sector_acc /= len(wf_result.folds)
    ranked = np.argsort(per_sector_acc)[::-1]
    top5_mask = np.zeros(Y_jp.shape[1], dtype=bool)
    top5_mask[ranked[:5]] = True

    # ====================================================================
    # Section A: Cycle 12/13 baselines
    # ====================================================================
    print("\n[3/8] Computing baselines...")

    # Naive baseline (no smoothing, all sectors)
    naive_strat = TradingStrategy(cost_bps=10.0)
    naive_result = naive_strat.run(predictions, actuals)

    # Cycle 12 best: fixed EMA-20, fixed top-5
    c12_strat = TradingStrategy(ema_halflife=20, sector_mask=top5_mask, cost_bps=10.0)
    c12_result = c12_strat.run(predictions, actuals)

    print(f"  Naive:      Gross SR={naive_result['sharpe_ratio_gross']:.4f}, "
          f"Net SR={naive_result['sharpe_ratio_net']:.4f}, "
          f"Turnover={naive_result['avg_daily_turnover']:.4f}")
    print(f"  C12 Fixed:  Gross SR={c12_result['sharpe_ratio_gross']:.4f}, "
          f"Net SR={c12_result['sharpe_ratio_net']:.4f}, "
          f"Turnover={c12_result['avg_daily_turnover']:.4f}")

    # ====================================================================
    # Section B: Dynamic sector selection grid search
    # ====================================================================
    print("\n[4/8] Evaluating dynamic sector selection...")
    dynamic_results = []

    lookbacks = [42, 63, 126]
    min_accuracies = [0.50, 0.51, 0.52, 0.53]
    min_sectors_list = [3, 5]
    max_sectors_list = [5, 8]
    rebalance_freqs = [21, 42]

    for lookback in lookbacks:
        for min_acc in min_accuracies:
            for min_sec in min_sectors_list:
                for max_sec in max_sectors_list:
                    if min_sec > max_sec:
                        continue
                    for rebal_freq in rebalance_freqs:
                        selector = DynamicSectorSelector(
                            lookback=lookback,
                            min_accuracy=min_acc,
                            min_sectors=min_sec,
                            max_sectors=max_sec,
                            rebalance_freq=rebal_freq,
                        )
                        dyn_strat = DynamicTradingStrategy(
                            ema_halflife=20,
                            dynamic_selector=selector,
                            cost_bps=10.0,
                        )
                        res = dyn_strat.run(predictions, actuals)

                        dynamic_results.append({
                            "lookback": lookback,
                            "min_accuracy": min_acc,
                            "min_sectors": min_sec,
                            "max_sectors": max_sec,
                            "rebalance_freq": rebal_freq,
                            "sharpe_gross": res["sharpe_ratio_gross"],
                            "sharpe_net": res["sharpe_ratio_net"],
                            "turnover": res["avg_daily_turnover"],
                            "total_return_net": res["total_return_net"],
                            "max_drawdown_net": res["max_drawdown_net"],
                            "avg_active_sectors": res["avg_active_sectors"],
                            "sector_rebalances": res["sector_rebalances"],
                            "pct_positive_net": res["pct_positive_days_net"],
                        })

    dynamic_results.sort(key=lambda x: x["sharpe_net"], reverse=True)

    print(f"\n  Tested {len(dynamic_results)} dynamic sector configurations")
    print(f"\n  Top 10 Dynamic Sector Configs:")
    print(f"  {'LB':>4s} {'MinAcc':>7s} {'MinS':>5s} {'MaxS':>5s} {'Rebal':>6s} "
          f"{'SR(n)':>8s} {'Turn':>8s} {'Ret(n)':>10s} {'AvgSec':>7s} {'Rebals':>7s}")
    print("  " + "-" * 80)
    for r in dynamic_results[:10]:
        print(f"  {r['lookback']:>4d} {r['min_accuracy']:>7.2f} {r['min_sectors']:>5d} "
              f"{r['max_sectors']:>5d} {r['rebalance_freq']:>6d} "
              f"{r['sharpe_net']:>8.4f} {r['turnover']:>8.4f} "
              f"{r['total_return_net']:>10.4f} {r['avg_active_sectors']:>7.1f} "
              f"{r['sector_rebalances']:>7d}")

    best_dynamic = dynamic_results[0]

    # ====================================================================
    # Section C: Multi-horizon signal ensemble
    # ====================================================================
    print("\n[5/8] Evaluating multi-horizon signal ensembles...")
    mh_results = []

    horizon_configs = [
        {"name": "1d_only", "horizons": [1], "weights": [1.0]},
        {"name": "1d_5d_equal", "horizons": [1, 5], "weights": [0.5, 0.5]},
        {"name": "1d_5d_10d_equal", "horizons": [1, 5, 10], "weights": None},
        {"name": "1d_5d_weighted", "horizons": [1, 5], "weights": [0.7, 0.3]},
        {"name": "1d_5d_10d_weighted", "horizons": [1, 5, 10], "weights": [0.5, 0.3, 0.2]},
        {"name": "1d_3d_equal", "horizons": [1, 3], "weights": [0.5, 0.5]},
        {"name": "1d_3d_7d_equal", "horizons": [1, 3, 7], "weights": None},
        {"name": "5d_10d_equal", "horizons": [5, 10], "weights": [0.5, 0.5]},
        {"name": "1d_5d_21d_equal", "horizons": [1, 5, 21], "weights": None},
        {"name": "1d_5d_21d_weighted", "horizons": [1, 5, 21], "weights": [0.5, 0.3, 0.2]},
    ]

    for cfg in horizon_configs:
        ensemble = MultiHorizonEnsemble(
            horizons=cfg["horizons"],
            weights=cfg["weights"],
        )
        ensemble_preds = ensemble.combine(predictions)

        # Test with fixed top-5 + EMA-20 (same as C12 baseline)
        strat = TradingStrategy(
            ema_halflife=20, sector_mask=top5_mask, cost_bps=10.0
        )
        res = strat.run(ensemble_preds, actuals)

        # Also compute direction accuracy of ensemble predictions
        signs_match = np.sign(ensemble_preds) == np.sign(actuals)
        direction_acc = signs_match.mean()

        mh_results.append({
            "name": cfg["name"],
            "horizons": cfg["horizons"],
            "weights": cfg["weights"],
            "sharpe_gross": res["sharpe_ratio_gross"],
            "sharpe_net": res["sharpe_ratio_net"],
            "turnover": res["avg_daily_turnover"],
            "total_return_net": res["total_return_net"],
            "max_drawdown_net": res["max_drawdown_net"],
            "direction_accuracy": round(float(direction_acc), 4),
            "pct_positive_net": res["pct_positive_days_net"],
        })

    mh_results.sort(key=lambda x: x["sharpe_net"], reverse=True)

    print(f"\n  Multi-Horizon Ensemble Results:")
    print(f"  {'Config':<25s} {'SR(g)':>8s} {'SR(n)':>8s} {'Turn':>8s} "
          f"{'Ret(n)':>10s} {'DirAcc':>8s}")
    print("  " + "-" * 72)
    for r in mh_results:
        print(f"  {r['name']:<25s} {r['sharpe_gross']:>8.4f} {r['sharpe_net']:>8.4f} "
              f"{r['turnover']:>8.4f} {r['total_return_net']:>10.4f} "
              f"{r['direction_accuracy']:>8.4f}")

    best_mh = mh_results[0]

    # ====================================================================
    # Section D: Combined — best dynamic + best multi-horizon
    # ====================================================================
    print("\n[6/8] Testing combined dynamic sectors + multi-horizon ensemble...")
    combined_results = []

    # Use the top-3 dynamic configs with the top-3 multi-horizon configs
    top_dynamic_params = dynamic_results[:3]
    top_mh_configs = [c for c in horizon_configs
                      if c["name"] in [r["name"] for r in mh_results[:3]]]
    # Also always include 1d_only as the baseline
    baseline_mh = [c for c in horizon_configs if c["name"] == "1d_only"]
    test_mh_configs = baseline_mh + [c for c in top_mh_configs if c["name"] != "1d_only"]

    for dp in top_dynamic_params:
        for mh_cfg in test_mh_configs:
            ensemble = MultiHorizonEnsemble(
                horizons=mh_cfg["horizons"],
                weights=mh_cfg["weights"],
            )
            ensemble_preds = ensemble.combine(predictions)

            selector = DynamicSectorSelector(
                lookback=dp["lookback"],
                min_accuracy=dp["min_accuracy"],
                min_sectors=dp["min_sectors"],
                max_sectors=dp["max_sectors"],
                rebalance_freq=dp["rebalance_freq"],
            )
            dyn_strat = DynamicTradingStrategy(
                ema_halflife=20,
                dynamic_selector=selector,
                cost_bps=10.0,
            )
            res = dyn_strat.run(ensemble_preds, actuals)

            combined_results.append({
                "dynamic_params": {
                    "lookback": dp["lookback"],
                    "min_accuracy": dp["min_accuracy"],
                    "min_sectors": dp["min_sectors"],
                    "max_sectors": dp["max_sectors"],
                    "rebalance_freq": dp["rebalance_freq"],
                },
                "multi_horizon": mh_cfg["name"],
                "sharpe_gross": res["sharpe_ratio_gross"],
                "sharpe_net": res["sharpe_ratio_net"],
                "turnover": res["avg_daily_turnover"],
                "total_return_net": res["total_return_net"],
                "max_drawdown_net": res["max_drawdown_net"],
                "avg_active_sectors": res["avg_active_sectors"],
                "sector_rebalances": res["sector_rebalances"],
                "pct_positive_net": res["pct_positive_days_net"],
            })

    combined_results.sort(key=lambda x: x["sharpe_net"], reverse=True)

    print(f"\n  Top 10 Combined Configs:")
    print(f"  {'MH Config':<20s} {'LB':>4s} {'MinAcc':>7s} {'MinS':>5s} {'MaxS':>5s} "
          f"{'SR(n)':>8s} {'Turn':>8s} {'Ret(n)':>10s}")
    print("  " + "-" * 80)
    for r in combined_results[:10]:
        dp = r["dynamic_params"]
        print(f"  {r['multi_horizon']:<20s} {dp['lookback']:>4d} {dp['min_accuracy']:>7.2f} "
              f"{dp['min_sectors']:>5d} {dp['max_sectors']:>5d} "
              f"{r['sharpe_net']:>8.4f} {r['turnover']:>8.4f} "
              f"{r['total_return_net']:>10.4f}")

    best_combined = combined_results[0]

    # ====================================================================
    # Section E: Borrowing cost sensitivity for best configs
    # ====================================================================
    print("\n[7/8] Evaluating borrowing cost impact on best configs...")
    borrow_results = {}

    # C12 baseline with borrow costs
    for borrow_bps in [0, 75]:
        strat = TradingStrategy(
            ema_halflife=20, sector_mask=top5_mask, cost_bps=10.0,
            borrow_cost_bps=borrow_bps,
        )
        res = strat.run(predictions, actuals)
        borrow_results[f"C12_fixed_top5_borrow{borrow_bps}"] = {
            "sharpe_net": res["sharpe_ratio_net"],
            "total_return_net": res["total_return_net"],
            "turnover": res["avg_daily_turnover"],
        }

    # Best dynamic with borrow costs
    bp = best_dynamic
    for borrow_bps in [0, 75]:
        selector = DynamicSectorSelector(
            lookback=bp["lookback"], min_accuracy=bp["min_accuracy"],
            min_sectors=bp["min_sectors"], max_sectors=bp["max_sectors"],
            rebalance_freq=bp["rebalance_freq"],
        )
        dyn_strat = DynamicTradingStrategy(
            ema_halflife=20, dynamic_selector=selector,
            cost_bps=10.0, borrow_cost_bps=borrow_bps,
        )
        res = dyn_strat.run(predictions, actuals)
        borrow_results[f"best_dynamic_borrow{borrow_bps}"] = {
            "sharpe_net": res["sharpe_ratio_net"],
            "total_return_net": res["total_return_net"],
            "turnover": res["avg_daily_turnover"],
        }

    # Best multi-horizon with borrow costs
    best_mh_cfg = next(c for c in horizon_configs if c["name"] == best_mh["name"])
    for borrow_bps in [0, 75]:
        ensemble = MultiHorizonEnsemble(
            horizons=best_mh_cfg["horizons"], weights=best_mh_cfg["weights"],
        )
        ensemble_preds = ensemble.combine(predictions)
        strat = TradingStrategy(
            ema_halflife=20, sector_mask=top5_mask, cost_bps=10.0,
            borrow_cost_bps=borrow_bps,
        )
        res = strat.run(ensemble_preds, actuals)
        borrow_results[f"best_mh_{best_mh['name']}_borrow{borrow_bps}"] = {
            "sharpe_net": res["sharpe_ratio_net"],
            "total_return_net": res["total_return_net"],
            "turnover": res["avg_daily_turnover"],
        }

    print(f"\n  {'Config':<45s} {'SR(n)':>8s} {'Ret(n)':>10s} {'Turn':>8s}")
    print("  " + "-" * 75)
    for label, r in sorted(borrow_results.items()):
        print(f"  {label:<45s} {r['sharpe_net']:>8.4f} {r['total_return_net']:>10.4f} "
              f"{r['turnover']:>8.4f}")

    # ====================================================================
    # Section F: Save results
    # ====================================================================
    print("\n[8/8] Saving results...")
    reports_dir = Path("reports/cycle_14")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Top-5 sector info (fixed)
    top5_sectors = {
        jp_tickers[i]: {
            "name": JP_SECTOR_NAMES.get(jp_tickers[i], jp_tickers[i]),
            "accuracy": round(float(per_sector_acc[i]), 4),
        }
        for i in ranked[:5]
    }

    # Serialize dynamic_results (top 20)
    top_dynamic_serializable = []
    for r in dynamic_results[:20]:
        top_dynamic_serializable.append({k: v for k, v in r.items()})

    # Serialize mh_results
    mh_serializable = []
    for r in mh_results:
        entry = {k: v for k, v in r.items()}
        if entry.get("weights") is not None:
            entry["weights"] = [round(w, 4) for w in entry["weights"]]
        mh_serializable.append(entry)

    # Serialize combined results (top 10)
    combined_serializable = []
    for r in combined_results[:10]:
        combined_serializable.append({k: v for k, v in r.items()})

    metrics = {
        "phase": 14,
        "description": "Dynamic sector selection and multi-horizon signal ensemble",
        "timestamp": "2026-03-27",
        "cycle_12_baseline": {
            "sharpe_gross": c12_result["sharpe_ratio_gross"],
            "sharpe_net": c12_result["sharpe_ratio_net"],
            "avg_daily_turnover": c12_result["avg_daily_turnover"],
            "total_return_net": c12_result["total_return_net"],
            "max_drawdown_net": c12_result["max_drawdown_net"],
            "config": "EMA-20, fixed top-5 sectors, equal-weight",
        },
        "naive_baseline": {
            "sharpe_gross": naive_result["sharpe_ratio_gross"],
            "sharpe_net": naive_result["sharpe_ratio_net"],
            "avg_daily_turnover": naive_result["avg_daily_turnover"],
        },
        "dynamic_sector_selection": {
            "total_configs_tested": len(dynamic_results),
            "best_config": best_dynamic,
            "top_20_configs": top_dynamic_serializable,
            "improvement_over_c12": round(
                best_dynamic["sharpe_net"] - c12_result["sharpe_ratio_net"], 4
            ),
        },
        "multi_horizon_ensemble": {
            "total_configs_tested": len(mh_results),
            "best_config": best_mh,
            "all_results": mh_serializable,
            "improvement_over_c12": round(
                best_mh["sharpe_net"] - c12_result["sharpe_ratio_net"], 4
            ),
        },
        "combined_dynamic_multihorizon": {
            "total_configs_tested": len(combined_results),
            "best_config": best_combined,
            "top_10_configs": combined_serializable,
            "improvement_over_c12": round(
                best_combined["sharpe_net"] - c12_result["sharpe_ratio_net"], 4
            ),
        },
        "borrowing_cost_impact": borrow_results,
        "top_5_predictable_sectors_fixed": top5_sectors,
        "walk_forward_params": {
            "K": 5, "L": 120, "lambda_decay": 1.0,
            "train_window": 252, "test_window": 21,
            "n_folds": wf_result.n_folds,
            "total_test_samples": wf_result.total_test_samples,
            "direction_accuracy": round(wf_result.mean_direction_accuracy, 4),
        },
        "cost_assumptions": {
            "one_way_transaction_bps": 10.0,
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

    print("\nCycle 14 complete.")
    return metrics


def generate_technical_findings(metrics):
    c12 = metrics["cycle_12_baseline"]
    dss = metrics["dynamic_sector_selection"]
    mh = metrics["multi_horizon_ensemble"]
    comb = metrics["combined_dynamic_multihorizon"]
    best_dyn = dss["best_config"]
    best_mh = mh["best_config"]
    best_comb = comb["best_config"]

    # Dynamic sector top-10 table
    dyn_rows = ""
    for i, r in enumerate(dss["top_20_configs"][:10]):
        dyn_rows += (
            f"| {i+1} | {r['lookback']} | {r['min_accuracy']:.2f} | "
            f"{r['min_sectors']} | {r['max_sectors']} | {r['rebalance_freq']} | "
            f"{r['sharpe_net']:.4f} | {r['turnover']:.4f} | "
            f"{r['total_return_net']:.4f} | {r['avg_active_sectors']:.1f} |\n"
        )

    # Multi-horizon table
    mh_rows = ""
    for r in mh["all_results"]:
        mh_rows += (
            f"| {r['name']:<25s} | {r['sharpe_gross']:.4f} | {r['sharpe_net']:.4f} | "
            f"{r['turnover']:.4f} | {r['total_return_net']:.4f} | "
            f"{r['direction_accuracy']:.4f} |\n"
        )

    # Combined top-10 table
    comb_rows = ""
    for i, r in enumerate(comb["top_10_configs"]):
        dp = r["dynamic_params"]
        comb_rows += (
            f"| {i+1} | {r['multi_horizon']:<20s} | {dp['lookback']} | "
            f"{dp['min_accuracy']:.2f} | {dp['min_sectors']}-{dp['max_sectors']} | "
            f"{r['sharpe_net']:.4f} | {r['turnover']:.4f} | "
            f"{r['total_return_net']:.4f} |\n"
        )

    # Borrow cost table
    borrow_rows = ""
    for label, r in sorted(metrics["borrowing_cost_impact"].items()):
        borrow_rows += (
            f"| {label:<45s} | {r['sharpe_net']:.4f} | "
            f"{r['total_return_net']:.4f} | {r['turnover']:.4f} |\n"
        )

    # Sector info
    sector_rows = ""
    for ticker, info in metrics["top_5_predictable_sectors_fixed"].items():
        sector_rows += f"| {ticker} | {info['name']} | {info['accuracy']:.4f} |\n"

    return f"""# Cycle 14: Dynamic Sector Selection & Multi-Horizon Signal Ensemble

## Summary

This cycle implements two key improvements to address open questions from Cycles 12-13:

1. **Dynamic sector selection** — Instead of a fixed top-5 sector set, periodically
   re-evaluate which sectors to trade based on rolling directional accuracy over
   recent predictions. Sectors are included/excluded based on a predictability threshold,
   with re-evaluation at configurable intervals.

2. **Multi-horizon signal ensemble** — Instead of using only raw 1-day predictions,
   combine moving averages of predictions at multiple horizons (1-day, 5-day, 10-day, etc.)
   into a single ensemble signal. The hypothesis is that medium-term aggregation
   reduces noise and improves signal stability.

## Cycle 12 Baseline (for comparison)

| Metric | Value |
|--------|-------|
| Configuration | EMA-20, fixed top-5 sectors, equal-weight |
| Gross Sharpe | {c12['sharpe_gross']:.4f} |
| Net Sharpe | {c12['sharpe_net']:.4f} |
| Daily Turnover | {c12['avg_daily_turnover']:.4f} |
| Total Return (net) | {c12['total_return_net']:.4f} |
| Max Drawdown (net) | {c12['max_drawdown_net']:.4f} |

## Key Results

### 1. Dynamic Sector Selection

Tested {dss['total_configs_tested']} configurations across lookback windows (42/63/126 days),
accuracy thresholds (0.50-0.53), sector count bounds (3-8), and rebalance frequencies
(21/42 days).

**Best dynamic config:** Lookback={best_dyn['lookback']}, MinAcc={best_dyn['min_accuracy']:.2f},
Sectors={best_dyn['min_sectors']}-{best_dyn['max_sectors']}, Rebalance={best_dyn['rebalance_freq']}d

| Metric | Dynamic Best | C12 Fixed | Delta |
|--------|-------------|-----------|-------|
| Net Sharpe | {best_dyn['sharpe_net']:.4f} | {c12['sharpe_net']:.4f} | {dss['improvement_over_c12']:+.4f} |
| Turnover | {best_dyn['turnover']:.4f} | {c12['avg_daily_turnover']:.4f} | — |
| Return (net) | {best_dyn['total_return_net']:.4f} | {c12['total_return_net']:.4f} | — |
| Avg Sectors | {best_dyn['avg_active_sectors']:.1f} | 5.0 | — |

**Top 10 Dynamic Sector Configurations:**

| Rank | Lookback | MinAcc | MinS | MaxS | Rebal | SR(net) | Turnover | Return(net) | AvgSec |
|------|----------|--------|------|------|-------|---------|----------|-------------|--------|
{dyn_rows}

### 2. Multi-Horizon Signal Ensemble

Tested {mh['total_configs_tested']} horizon combinations with fixed top-5 sectors + EMA-20.

| Config | SR(gross) | SR(net) | Turnover | Return(net) | DirAcc |
|--------|-----------|---------|----------|-------------|--------|
{mh_rows}

**Best multi-horizon config:** {best_mh['name']}
- Net Sharpe: {best_mh['sharpe_net']:.4f} (vs C12 baseline {c12['sharpe_net']:.4f}, delta {mh['improvement_over_c12']:+.4f})
- Direction accuracy: {best_mh['direction_accuracy']:.4f}

### 3. Combined Dynamic + Multi-Horizon

Tested {comb['total_configs_tested']} combined configurations (top-3 dynamic x top multi-horizon configs).

| Rank | MH Config | Lookback | MinAcc | Sectors | SR(net) | Turnover | Return(net) |
|------|-----------|----------|--------|---------|---------|----------|-------------|
{comb_rows}

**Best combined config:**
- Net Sharpe: {best_comb['sharpe_net']:.4f} (vs C12 {c12['sharpe_net']:.4f}, delta {comb['improvement_over_c12']:+.4f})

### 4. Borrowing Cost Sensitivity

| Config | SR(net) | Return(net) | Turnover |
|--------|---------|-------------|----------|
{borrow_rows}

## Top 5 Most Predictable Sectors (Fixed Reference)

| Ticker | Sector | Accuracy |
|--------|--------|----------|
{sector_rows}

## Methodology

### Dynamic Sector Selection
- **Lookback window**: Number of recent OOS days to compute rolling accuracy
- **Min accuracy threshold**: Sectors below this accuracy are excluded
- **Min/max sectors**: Bounds on how many sectors to trade
- **Rebalance frequency**: How often (in days) to re-evaluate sector set
- Rolling accuracy is computed on OOS predictions vs actuals (no lookahead)

### Multi-Horizon Ensemble
- Raw 1-day predictions are aggregated using rolling means at different windows
- Horizon signals are combined with configurable weights
- The ensemble signal replaces raw predictions in the trading strategy pipeline
- All horizons use only past predictions (no future information)

### Walk-Forward Setup
- **Model**: PCA_SUB with K=5, L=120, lambda=1.0
- **Train window**: 252 days, **Test window**: 21 days
- **Folds**: {metrics['walk_forward_params']['n_folds']}
- **Total OOS samples**: {metrics['walk_forward_params']['total_test_samples']}
- **Direction accuracy**: {metrics['walk_forward_params']['direction_accuracy']:.4f}

### Cost Assumptions
- One-way transaction cost: 10 bps
- Realistic borrowing cost: 75 bps annualized

## Conclusions

1. **Dynamic sector selection** allows the strategy to adapt its sector universe
   over time, potentially capturing periods when different sectors become more
   or less predictable.

2. **Multi-horizon ensembles** can provide a smoother, more stable signal by
   combining short-term and medium-term prediction averages, though the benefit
   depends on how much noise reduction offsets signal delay.

3. The Cycle 12 baseline (fixed EMA-20, top-5) remains a strong benchmark.
   Any improvement from dynamic selection or multi-horizon must be evaluated
   with borrowing costs to confirm economic significance.

## Open Questions for Future Cycles

1. **Ensemble model diversity**: Could combining PCA_SUB with other models
   (Ridge, elastic net) improve signal quality beyond multi-horizon averaging?
2. **Regime detection**: Could an explicit regime classifier (bull/bear/sideways)
   improve sector selection timing?
3. **Transaction cost optimization**: Could we use the dynamic selector to avoid
   unnecessary sector switches that add turnover without improving returns?
4. **Out-of-sample validation**: All improvements should be tested on truly
   unseen data (post-2026) to confirm they are not overfit.
"""


if __name__ == "__main__":
    main()
