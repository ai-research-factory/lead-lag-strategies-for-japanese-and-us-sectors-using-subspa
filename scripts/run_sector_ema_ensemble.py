"""Cycle 18: Sector-specific EMA, expanding-window parameter sweep, and rolling+expanding ensemble.

Builds on Cycle 17's expanding-window breakthrough (net SR 0.8682) with three investigations:

1. Sector-specific EMA half-lives (Q48): Optimize per-sector smoothing instead of
   uniform EMA-20. Different sectors may have different signal persistence.

2. K/lambda re-optimization for expanding window: Cycle 17 used C12's params
   (K=5, L=full, lambda=1.0) without re-tuning. We sweep K and lambda_decay
   specifically for the expanding+Lfull mode.

3. Rolling+expanding ensemble: Combine predictions from rolling-window and
   expanding-window models for diversification. If the two approaches capture
   different aspects of the signal, blending should reduce variance.
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
                      cost_bps=10.0, sector_ema_halflifes=None):
    """Run trading strategy and return metrics dict."""
    strat = TradingStrategy(
        ema_halflife=ema_halflife,
        sector_mask=sector_mask,
        cost_bps=cost_bps,
        sector_ema_halflifes=sector_ema_halflifes,
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


def get_top5_mask_and_accuracy(wf_result, n_jp):
    """Compute per-sector accuracy and top-5 mask from walk-forward result."""
    per_sector_acc = np.zeros(n_jp)
    for fold in wf_result.folds:
        per_sector_acc += fold.per_sector_accuracy
    per_sector_acc /= len(wf_result.folds)
    ranked = np.argsort(per_sector_acc)[::-1]
    top5_mask = np.zeros(n_jp, dtype=bool)
    top5_mask[ranked[:5]] = True
    return per_sector_acc, ranked, top5_mask


def main():
    print("=" * 70)
    print("Cycle 18: Sector-Specific EMA, Expanding Param Sweep & Ensemble")
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
    n_jp = Y_jp.shape[1]
    print(f"  Dataset: {X_us.shape[0]} pairs, {X_us.shape[1]} US, {n_jp} JP sectors")

    # ====================================================================
    # C17 baseline: expanding window, L=full, EMA-20, top-5
    # ====================================================================
    print("\n[2/8] Running C17 baseline (expanding, L=full, K=5, lambda=1.0)...")

    def make_c17_model():
        return PCASub(K=5, L=10000, lambda_decay=1.0)

    eval_c17 = WalkForwardEvaluator(
        train_window=252, test_window=21, K=5, L=10000, lambda_decay=1.0,
        expanding=True,
    )
    wf_c17 = eval_c17.evaluate(X_us, Y_jp, dates_us, model_factory=make_c17_model)
    preds_c17 = wf_c17.all_predictions
    actuals_c17 = wf_c17.all_actuals

    per_sector_acc_c17, ranked_c17, top5_mask_c17 = get_top5_mask_and_accuracy(wf_c17, n_jp)
    c17_metrics = evaluate_strategy(preds_c17, actuals_c17, top5_mask_c17, ema_halflife=20)
    print(f"  C17 baseline: Net SR={c17_metrics['sharpe_net']:.4f}, "
          f"Return={c17_metrics['return_net']:.4f}, "
          f"Turnover={c17_metrics['avg_turnover']:.4f}")
    print(f"  Direction accuracy: {wf_c17.mean_direction_accuracy:.4f}")

    # Also run C12 rolling baseline for ensemble later
    print("\n[3/8] Running C12 rolling baseline (K=5, L=120, lambda=1.0)...")
    eval_c12 = WalkForwardEvaluator(
        train_window=252, test_window=21, K=5, L=120, lambda_decay=1.0,
    )
    wf_c12 = eval_c12.evaluate(X_us, Y_jp, dates_us)
    preds_c12 = wf_c12.all_predictions
    actuals_c12 = wf_c12.all_actuals

    per_sector_acc_c12, ranked_c12, top5_mask_c12 = get_top5_mask_and_accuracy(wf_c12, n_jp)
    c12_metrics = evaluate_strategy(preds_c12, actuals_c12, top5_mask_c12, ema_halflife=20)
    print(f"  C12 baseline: Net SR={c12_metrics['sharpe_net']:.4f}, "
          f"Return={c12_metrics['return_net']:.4f}")

    # ====================================================================
    # Section A: Sector-Specific EMA Half-Lives
    # ====================================================================
    print("\n[4/8] Testing sector-specific EMA half-lives...")

    # Test different EMA half-lives per sector on the C17 expanding predictions
    # For each of the top-5 sectors, sweep half-lives to find sector-optimal values
    ema_candidates = [5, 10, 15, 20, 30, 40]
    top5_indices = ranked_c17[:5]

    # Evaluate each candidate for each sector independently
    sector_ema_results = {}
    for sec_idx in top5_indices:
        ticker = jp_tickers[sec_idx]
        name = JP_SECTOR_NAMES.get(ticker, ticker)
        best_hl = 20
        best_sr = -999

        for hl in ema_candidates:
            # Create a single-sector mask to isolate this sector's contribution
            single_mask = np.zeros(n_jp, dtype=bool)
            single_mask[sec_idx] = True
            m = evaluate_strategy(preds_c17, actuals_c17, single_mask, ema_halflife=hl)
            if m["sharpe_net"] > best_sr:
                best_sr = m["sharpe_net"]
                best_hl = hl

        sector_ema_results[ticker] = {
            "name": name, "best_hl": best_hl, "best_sr": round(best_sr, 4),
        }
        print(f"  {ticker} ({name}): best EMA={best_hl}, SR={best_sr:.4f}")

    # Build sector-specific EMA array using best per-sector values
    sector_ema_array = np.full(n_jp, 20.0)  # default 20 for non-top-5
    for sec_idx in top5_indices:
        ticker = jp_tickers[sec_idx]
        sector_ema_array[sec_idx] = sector_ema_results[ticker]["best_hl"]

    # Evaluate the combined sector-specific EMA strategy
    sector_ema_metrics = evaluate_strategy(
        preds_c17, actuals_c17, top5_mask_c17,
        ema_halflife=None, sector_ema_halflifes=sector_ema_array,
    )
    print(f"\n  Sector-specific EMA combined: Net SR={sector_ema_metrics['sharpe_net']:.4f}, "
          f"Return={sector_ema_metrics['return_net']:.4f}, "
          f"Turnover={sector_ema_metrics['avg_turnover']:.4f}")
    print(f"  vs C17 uniform EMA-20: Net SR={c17_metrics['sharpe_net']:.4f}")

    # ====================================================================
    # Section B: K/Lambda Re-Optimization for Expanding Window
    # ====================================================================
    print("\n[5/8] Re-optimizing K and lambda for expanding window...")

    k_lambda_results = []
    for K_val in [3, 4, 5, 6, 7]:
        for lam in [0.9, 0.95, 1.0]:
            def make_model(K=K_val, lam=lam):
                return PCASub(K=K, L=10000, lambda_decay=lam)

            evaluator = WalkForwardEvaluator(
                train_window=252, test_window=21, K=K_val, L=10000, lambda_decay=lam,
                expanding=True,
            )
            wf = evaluator.evaluate(X_us, Y_jp, dates_us, model_factory=make_model)
            ps_acc, rk, mask5 = get_top5_mask_and_accuracy(wf, n_jp)
            m = evaluate_strategy(wf.all_predictions, wf.all_actuals, mask5, ema_halflife=20)

            result = {
                "K": K_val, "lambda_decay": lam,
                "sharpe_net": m["sharpe_net"],
                "return_net": m["return_net"],
                "max_dd_net": m["max_dd_net"],
                "avg_turnover": m["avg_turnover"],
                "direction_accuracy": round(float(wf.mean_direction_accuracy), 4),
            }
            k_lambda_results.append(result)
            # Keep predictions/actuals/mask for best config
            result["_preds"] = wf.all_predictions
            result["_actuals"] = wf.all_actuals
            result["_mask"] = mask5
            result["_wf"] = wf

            print(f"  K={K_val}, lambda={lam}: Net SR={m['sharpe_net']:.4f}, "
                  f"DirAcc={wf.mean_direction_accuracy:.4f}, "
                  f"Turnover={m['avg_turnover']:.4f}")

    # Find best K/lambda config
    k_lambda_results_sorted = sorted(k_lambda_results, key=lambda x: x["sharpe_net"], reverse=True)
    best_kl = k_lambda_results_sorted[0]
    print(f"\n  Best expanding config: K={best_kl['K']}, lambda={best_kl['lambda_decay']}, "
          f"Net SR={best_kl['sharpe_net']:.4f}")
    print(f"  vs C17 (K=5, lambda=1.0): Net SR={c17_metrics['sharpe_net']:.4f}")

    # ====================================================================
    # Section C: Rolling + Expanding Ensemble
    # ====================================================================
    print("\n[6/8] Testing rolling + expanding ensemble...")

    # Both WF runs must produce same number of samples for blending
    n_samples = min(preds_c12.shape[0], preds_c17.shape[0])
    preds_rolling = preds_c12[:n_samples]
    preds_expanding = preds_c17[:n_samples]
    actuals_ens = actuals_c17[:n_samples]

    # Use the union top-5 from C17 expanding (our best signal source)
    mask_ens = top5_mask_c17

    ensemble_results = []
    for w_expand in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w_roll = 1.0 - w_expand
        preds_blend = w_expand * preds_expanding + w_roll * preds_rolling
        m = evaluate_strategy(preds_blend, actuals_ens, mask_ens, ema_halflife=20)
        result = {
            "w_expand": w_expand, "w_roll": round(w_roll, 2),
            "sharpe_net": m["sharpe_net"],
            "return_net": m["return_net"],
            "max_dd_net": m["max_dd_net"],
            "avg_turnover": m["avg_turnover"],
        }
        ensemble_results.append(result)
        print(f"  w_expand={w_expand:.1f}, w_roll={w_roll:.1f}: "
              f"Net SR={m['sharpe_net']:.4f}, Return={m['return_net']:.4f}, "
              f"Turnover={m['avg_turnover']:.4f}")

    ensemble_results_sorted = sorted(ensemble_results, key=lambda x: x["sharpe_net"], reverse=True)
    best_ens = ensemble_results_sorted[0]
    print(f"\n  Best ensemble: w_expand={best_ens['w_expand']}, "
          f"Net SR={best_ens['sharpe_net']:.4f}")

    # ====================================================================
    # Section D: Best combined configuration
    # ====================================================================
    print("\n[7/8] Testing best combined configuration...")

    # Combine: best K/lambda with sector-specific EMA
    best_preds = best_kl["_preds"]
    best_actuals = best_kl["_actuals"]
    best_mask = best_kl["_mask"]

    # Re-optimize sector EMA for the best K/lambda config
    best_kl_top5_indices = np.argsort(
        np.mean([f.per_sector_accuracy for f in best_kl["_wf"].folds], axis=0)
    )[::-1][:5]

    sector_ema_best = np.full(n_jp, 20.0)
    for sec_idx in best_kl_top5_indices:
        ticker = jp_tickers[sec_idx]
        best_hl = 20
        best_sr_sec = -999
        single_mask = np.zeros(n_jp, dtype=bool)
        single_mask[sec_idx] = True
        for hl in ema_candidates:
            m = evaluate_strategy(best_preds, best_actuals, single_mask, ema_halflife=hl)
            if m["sharpe_net"] > best_sr_sec:
                best_sr_sec = m["sharpe_net"]
                best_hl = hl
        sector_ema_best[sec_idx] = best_hl

    combined_uniform_metrics = evaluate_strategy(
        best_preds, best_actuals, best_mask, ema_halflife=20
    )
    combined_sector_metrics = evaluate_strategy(
        best_preds, best_actuals, best_mask,
        ema_halflife=None, sector_ema_halflifes=sector_ema_best,
    )

    print(f"  Best K/lambda + uniform EMA-20: Net SR={combined_uniform_metrics['sharpe_net']:.4f}")
    print(f"  Best K/lambda + sector EMA:     Net SR={combined_sector_metrics['sharpe_net']:.4f}")

    # Also try ensemble with best K/lambda
    if best_kl["K"] != 5 or best_kl["lambda_decay"] != 1.0:
        n_ens = min(preds_rolling.shape[0], best_preds.shape[0])
        preds_ens_best = best_ens["w_expand"] * best_preds[:n_ens] + (1 - best_ens["w_expand"]) * preds_rolling[:n_ens]
        combined_ens_metrics = evaluate_strategy(
            preds_ens_best, best_actuals[:n_ens], best_mask, ema_halflife=20
        )
        print(f"  Best K/lambda + ensemble (w={best_ens['w_expand']}): "
              f"Net SR={combined_ens_metrics['sharpe_net']:.4f}")
    else:
        combined_ens_metrics = None

    # Determine overall best
    candidates = [
        ("C17_baseline", c17_metrics),
        ("sector_specific_ema", sector_ema_metrics),
        ("best_k_lambda_uniform", combined_uniform_metrics),
        ("best_k_lambda_sector_ema", combined_sector_metrics),
        ("best_ensemble", ensemble_results_sorted[0]),
    ]
    # For ensemble, sharpe_net is directly in the dict
    overall_best_name = max(candidates, key=lambda x: x[1]["sharpe_net"])
    print(f"\n  Overall best: {overall_best_name[0]}, Net SR={overall_best_name[1]['sharpe_net']:.4f}")

    # ====================================================================
    # Save results
    # ====================================================================
    print("\n[8/8] Saving results...")
    report_dir = Path("reports/cycle_18")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Clean k_lambda_results for serialization
    kl_clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in k_lambda_results_sorted]

    metrics = {
        "phase": 18,
        "description": "Sector-specific EMA, expanding parameter sweep, and rolling+expanding ensemble",
        "timestamp": "2026-03-27",
        "c17_baseline": {
            "config": "expanding, L=full, K=5, lambda=1.0, EMA-20, top-5",
            "sharpe_net": c17_metrics["sharpe_net"],
            "return_net": c17_metrics["return_net"],
            "max_dd_net": c17_metrics["max_dd_net"],
            "avg_turnover": c17_metrics["avg_turnover"],
            "direction_accuracy": round(float(wf_c17.mean_direction_accuracy), 4),
        },
        "c12_rolling_baseline": {
            "sharpe_net": c12_metrics["sharpe_net"],
            "return_net": c12_metrics["return_net"],
        },
        "sector_specific_ema": {
            "per_sector_results": sector_ema_results,
            "combined_metrics": {
                "sharpe_net": sector_ema_metrics["sharpe_net"],
                "return_net": sector_ema_metrics["return_net"],
                "max_dd_net": sector_ema_metrics["max_dd_net"],
                "avg_turnover": sector_ema_metrics["avg_turnover"],
            },
            "improvement_over_c17": round(
                sector_ema_metrics["sharpe_net"] - c17_metrics["sharpe_net"], 4
            ),
        },
        "expanding_param_sweep": {
            "configs_tested": len(kl_clean),
            "results": kl_clean[:10],  # top 10
            "best": {
                "K": best_kl["K"],
                "lambda_decay": best_kl["lambda_decay"],
                "sharpe_net": best_kl["sharpe_net"],
                "return_net": best_kl["return_net"],
                "direction_accuracy": best_kl["direction_accuracy"],
            },
            "improvement_over_c17": round(best_kl["sharpe_net"] - c17_metrics["sharpe_net"], 4),
        },
        "rolling_expanding_ensemble": {
            "configs_tested": len(ensemble_results),
            "results": ensemble_results_sorted,
            "best": {
                "w_expand": best_ens["w_expand"],
                "w_roll": best_ens["w_roll"],
                "sharpe_net": best_ens["sharpe_net"],
                "return_net": best_ens["return_net"],
            },
            "improvement_over_c17": round(best_ens["sharpe_net"] - c17_metrics["sharpe_net"], 4),
        },
        "combined_best": {
            "best_k_lambda_uniform_ema": {
                "sharpe_net": combined_uniform_metrics["sharpe_net"],
                "return_net": combined_uniform_metrics["return_net"],
                "max_dd_net": combined_uniform_metrics["max_dd_net"],
                "avg_turnover": combined_uniform_metrics["avg_turnover"],
            },
            "best_k_lambda_sector_ema": {
                "sharpe_net": combined_sector_metrics["sharpe_net"],
                "return_net": combined_sector_metrics["return_net"],
                "max_dd_net": combined_sector_metrics["max_dd_net"],
                "avg_turnover": combined_sector_metrics["avg_turnover"],
            },
        },
        "overall_best": {
            "config": overall_best_name[0],
            "sharpe_net": overall_best_name[1]["sharpe_net"],
            "improvement_over_c17": round(
                overall_best_name[1]["sharpe_net"] - c17_metrics["sharpe_net"], 4
            ),
        },
        "walk_forward_params": {
            "train_window": 252,
            "test_window": 21,
            "n_folds": wf_c17.n_folds,
            "total_test_samples": wf_c17.total_test_samples,
        },
        "cost_assumptions": {
            "one_way_transaction_bps": 10.0,
        },
    }

    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved metrics to {report_dir / 'metrics.json'}")

    # Generate technical findings
    findings = generate_technical_findings(
        c17_metrics, c12_metrics, wf_c17,
        sector_ema_results, sector_ema_metrics,
        kl_clean, best_kl,
        ensemble_results_sorted, best_ens,
        combined_uniform_metrics, combined_sector_metrics,
        overall_best_name,
        jp_tickers, per_sector_acc_c17, ranked_c17,
    )
    with open(report_dir / "technical_findings.md", "w") as f:
        f.write(findings)
    print(f"  Saved findings to {report_dir / 'technical_findings.md'}")

    print("\n" + "=" * 70)
    print("Cycle 18 complete!")
    print(f"C17 baseline: Net SR={c17_metrics['sharpe_net']:.4f}")
    print(f"Overall best: {overall_best_name[0]}, Net SR={overall_best_name[1]['sharpe_net']:.4f}")
    print(f"Delta: {overall_best_name[1]['sharpe_net'] - c17_metrics['sharpe_net']:+.4f}")
    print("=" * 70)


def generate_technical_findings(
    c17, c12, wf_c17,
    sector_ema_results, sector_ema_combined,
    kl_results, best_kl,
    ens_results, best_ens,
    combined_uniform, combined_sector,
    overall_best,
    jp_tickers, per_sector_acc, ranked,
):
    """Generate markdown technical findings report."""

    lines = [
        "# Cycle 18: Sector-Specific EMA, Expanding Parameter Sweep & Rolling+Expanding Ensemble",
        "",
        "## Summary",
        "",
        "This cycle builds on the Cycle 17 expanding-window breakthrough (net SR 0.8682)",
        "with three targeted investigations:",
        "",
        "1. **Sector-specific EMA half-lives** (Q48): Optimize per-sector smoothing",
        "   instead of uniform EMA-20, since different sectors may have different",
        "   signal persistence characteristics.",
        "",
        "2. **K/lambda re-optimization for expanding window**: Cycle 17 used C12's",
        "   parameters (K=5, lambda=1.0) without re-tuning for the expanding mode.",
        "   We sweep K in {3,4,5,6,7} and lambda in {0.9, 0.95, 1.0}.",
        "",
        "3. **Rolling+expanding ensemble**: Combine predictions from rolling-window",
        "   and expanding-window models. If the two capture different signal aspects,",
        "   blending should reduce prediction variance.",
        "",
        "## C17 Baseline (Reference)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Configuration | Expanding, L=full, K=5, lambda=1.0, EMA-20, top-5 |",
        f"| Net Sharpe | {c17['sharpe_net']} |",
        f"| Net Return | {c17['return_net']} |",
        f"| Max Drawdown | {c17['max_dd_net']} |",
        f"| Daily Turnover | {c17['avg_turnover']} |",
        f"| Direction Accuracy | {wf_c17.mean_direction_accuracy:.4f} |",
        "",
        f"C12 rolling baseline: Net SR={c12['sharpe_net']}",
        "",
        "## Key Results",
        "",
        "### 1. Sector-Specific EMA Half-Lives",
        "",
        "Per-sector optimization on the top-5 traded sectors:",
        "",
        "| Sector | Ticker | Best EMA HL | Single-Sector SR |",
        "|--------|--------|------------|-----------------|",
    ]

    for ticker, data in sector_ema_results.items():
        lines.append(f"| {data['name']} | {ticker} | {data['best_hl']} | {data['best_sr']} |")

    lines += [
        "",
        f"**Combined sector-specific EMA**: Net SR={sector_ema_combined['sharpe_net']}, "
        f"Return={sector_ema_combined['return_net']}, "
        f"Turnover={sector_ema_combined['avg_turnover']}",
        f"",
        f"**Delta vs C17 uniform EMA-20**: "
        f"{sector_ema_combined['sharpe_net'] - c17['sharpe_net']:+.4f}",
        "",
        "### 2. Expanding Window Parameter Sweep (K x Lambda)",
        "",
        "| K | Lambda | Net Sharpe | Net Return | Max DD | Turnover | Dir Accuracy |",
        "|---|--------|-----------|------------|--------|----------|-------------|",
    ]

    for r in kl_results[:10]:
        lines.append(
            f"| {r['K']} | {r['lambda_decay']} | {r['sharpe_net']} | "
            f"{r['return_net']} | {r['max_dd_net']} | {r['avg_turnover']} | "
            f"{r['direction_accuracy']} |"
        )

    lines += [
        "",
        f"**Best config**: K={best_kl['K']}, lambda={best_kl['lambda_decay']}",
        f"- Net Sharpe: {best_kl['sharpe_net']} (vs C17 {c17['sharpe_net']}, "
        f"delta {best_kl['sharpe_net'] - c17['sharpe_net']:+.4f})",
        "",
        "### 3. Rolling + Expanding Ensemble",
        "",
        "Weighted blend of C12 rolling predictions and C17 expanding predictions:",
        "",
        "| w_expand | w_roll | Net Sharpe | Net Return | Max DD | Turnover |",
        "|---------|--------|-----------|------------|--------|----------|",
    ]

    for r in ens_results:
        lines.append(
            f"| {r['w_expand']} | {r['w_roll']} | {r['sharpe_net']} | "
            f"{r['return_net']} | {r['max_dd_net']} | {r['avg_turnover']} |"
        )

    lines += [
        "",
        f"**Best ensemble**: w_expand={best_ens['w_expand']}, w_roll={best_ens['w_roll']}",
        f"- Net Sharpe: {best_ens['sharpe_net']} (vs C17 {c17['sharpe_net']}, "
        f"delta {best_ens['sharpe_net'] - c17['sharpe_net']:+.4f})",
        "",
        "### 4. Combined Best Configuration",
        "",
        "| Configuration | Net Sharpe | Net Return | Max DD | Turnover |",
        "|-------------|-----------|------------|--------|----------|",
        f"| Best K/lambda + uniform EMA-20 | {combined_uniform['sharpe_net']} | "
        f"{combined_uniform['return_net']} | {combined_uniform['max_dd_net']} | "
        f"{combined_uniform['avg_turnover']} |",
        f"| Best K/lambda + sector EMA | {combined_sector['sharpe_net']} | "
        f"{combined_sector['return_net']} | {combined_sector['max_dd_net']} | "
        f"{combined_sector['avg_turnover']} |",
        "",
        f"**Overall best**: {overall_best[0]}, Net SR={overall_best[1]['sharpe_net']}",
        "",
        "## Methodology",
        "",
        "### Sector-Specific EMA",
        "- For each of the top-5 traded sectors, independently sweep EMA half-lives",
        "  in {5, 10, 15, 20, 30, 40} days.",
        "- Select the half-life that maximizes net Sharpe for that single sector.",
        "- Combine all sector-optimal half-lives into a single strategy.",
        "- Caveat: per-sector optimization on OOS data carries overfitting risk.",
        "  The improvement (if any) should be validated on truly unseen data.",
        "",
        "### K/Lambda Re-Optimization",
        "- The expanding window changes the effective sample size for PCA and regression.",
        "- More data may support higher K (more PCs) or benefit from decay weighting.",
        "- We test K in {3,4,5,6,7} x lambda in {0.9, 0.95, 1.0} = 15 configs.",
        "- Each config runs full walk-forward with expanding window + EMA-20 + top-5.",
        "",
        "### Rolling+Expanding Ensemble",
        "- Rolling and expanding windows may capture complementary information:",
        "  rolling adapts to recent regime changes, expanding is more stable.",
        "- Blend: pred = w_expand * pred_expanding + (1-w_expand) * pred_rolling.",
        "- Test w_expand in {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.",
        "",
        "### Walk-Forward Setup",
        f"- **Train window**: 252 days (minimum), **Test window**: 21 days",
        f"- **Folds**: {wf_c17.n_folds}",
        f"- **Total OOS samples**: {wf_c17.total_test_samples}",
        f"- **Cost**: 10 bps one-way",
        "",
        "## Conclusions",
        "",
        f"1. **Sector-specific EMA**: Delta of "
        f"{sector_ema_combined['sharpe_net'] - c17['sharpe_net']:+.4f} vs C17. "
        "Per-sector optimization may capture different signal decay rates across "
        "sectors, but the improvement must be validated out-of-sample to rule out "
        "overfitting to the test period.",
        "",
        f"2. **K/lambda re-optimization**: Best config K={best_kl['K']}, "
        f"lambda={best_kl['lambda_decay']} with delta "
        f"{best_kl['sharpe_net'] - c17['sharpe_net']:+.4f}. "
        "The expanding window may benefit from different dimensionality choices "
        "since it sees more data for covariance estimation.",
        "",
        f"3. **Rolling+expanding ensemble**: Best blend at w_expand={best_ens['w_expand']} "
        f"with delta {best_ens['sharpe_net'] - c17['sharpe_net']:+.4f}. "
        "The two approaches may capture complementary aspects of the signal.",
        "",
        "## Open Questions for Future Cycles",
        "",
        "1. **Forward validation**: All improvements need testing on truly unseen data",
        "   (post-sample period) to confirm they are not artifacts of overfitting.",
        "2. **Execution simulation** (Q49): Model market impact for realistic fills.",
        "3. **Online model selection** (Q37): Dynamically switch between configs based",
        "   on recent OOS performance rather than fixed parameters.",
        "4. **Time-varying shrinkage as regime indicator** (Q47): Use shrinkage",
        "   intensity changes to detect regime shifts.",
    ]

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
