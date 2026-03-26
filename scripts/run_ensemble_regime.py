"""Cycle 15: Ensemble model diversity and regime-aware positioning.

Addresses open questions from Cycle 14:
1. Ensemble model diversity: combine PCA_SUB with Ridge and ElasticNet
   to improve prediction quality through model diversity.
2. Regime-aware positioning: use volatility-based regime detection to
   scale exposure — reducing positions in high-volatility regimes where
   the lead-lag signal is weaker (confirmed in Phase 7).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import DataPipeline, JP_SECTOR_NAMES, JP_TICKERS
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.evaluation.trading_strategy import TradingStrategy
from src.evaluation.regime_detector import VolatilityRegimeDetector
from src.models.ensemble import EnsembleModel


def run_walk_forward_with_model(evaluator, X_us, Y_jp, dates_us, model_factory):
    """Run walk-forward using a custom model factory."""
    return evaluator.evaluate(X_us, Y_jp, dates_us, model_factory=model_factory)


def evaluate_strategy(predictions, actuals, sector_mask, ema_halflife=20,
                      cost_bps=10.0, borrow_bps=0.0, exposure_scales=None):
    """Run trading strategy and return metrics dict."""
    strat = TradingStrategy(
        ema_halflife=ema_halflife,
        sector_mask=sector_mask,
        cost_bps=cost_bps,
        borrow_cost_bps=borrow_bps,
    )
    result = strat.run(predictions, actuals)

    # Apply regime-based exposure scaling if provided
    if exposure_scales is not None:
        T = len(result["daily_returns_gross"])
        scales = exposure_scales[:T]
        # Re-compute returns with scaled positions
        weights = result["weights"] * scales[:, None]
        daily_gross = (weights * actuals[:T]).sum(axis=1)

        # Recompute turnover and costs
        pos_changes = np.abs(np.diff(weights, axis=0))
        daily_turnover = pos_changes.sum(axis=1)
        avg_turnover = float(daily_turnover.mean())
        cost_rate = cost_bps / 10000.0
        daily_costs = np.concatenate([[0.0], daily_turnover * cost_rate])

        daily_borrow = np.zeros(T)
        if borrow_bps > 0:
            daily_borrow_rate = borrow_bps / 10000.0 / 252.0
            short_exp = np.abs(np.minimum(weights, 0)).sum(axis=1)
            daily_borrow = short_exp * daily_borrow_rate

        daily_net = daily_gross - daily_costs - daily_borrow

        def _metrics(rets, label):
            mu = rets.mean()
            sigma = rets.std()
            sr = (mu / sigma * np.sqrt(252)) if sigma > 1e-12 else 0.0
            cum = np.cumprod(1 + rets)
            dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
            tr = float(cum[-1] - 1) if len(cum) > 0 else 0.0
            return {
                f"sharpe_{label}": round(sr, 4),
                f"return_{label}": round(tr, 6),
                f"max_dd_{label}": round(dd, 6),
                f"pct_pos_{label}": round(float(np.mean(rets > 0) * 100), 2),
            }

        g = _metrics(daily_gross, "gross")
        n = _metrics(daily_net, "net")
        return {**g, **n, "avg_turnover": round(avg_turnover, 4)}
    else:
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
            "daily_returns_net": result["daily_returns_net"],
            "weights": result["weights"],
        }


def main():
    print("=" * 70)
    print("Cycle 15: Ensemble Model Diversity & Regime-Aware Positioning")
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

    # C12 baseline: EMA-20, top-5
    c12_metrics = evaluate_strategy(preds_pca, actuals, top5_mask, ema_halflife=20)
    print(f"  C12 baseline: Net SR={c12_metrics['sharpe_net']:.4f}, "
          f"Return={c12_metrics['return_net']:.4f}")

    # ====================================================================
    # Section A: Ensemble model walk-forward
    # ====================================================================
    print("\n[3/7] Running ensemble model walk-forward evaluations...")

    ensemble_configs = [
        {
            "name": "equal_weight_avg",
            "weights": {"pca_sub": 1/3, "ridge": 1/3, "enet": 1/3},
            "combine": "weighted_avg",
            "ridge_alpha": 1.0,
            "enet_alpha": 0.01,
            "enet_l1": 0.5,
        },
        {
            "name": "pca_heavy_avg",
            "weights": {"pca_sub": 0.6, "ridge": 0.2, "enet": 0.2},
            "combine": "weighted_avg",
            "ridge_alpha": 1.0,
            "enet_alpha": 0.01,
            "enet_l1": 0.5,
        },
        {
            "name": "pca_dominant_avg",
            "weights": {"pca_sub": 0.8, "ridge": 0.1, "enet": 0.1},
            "combine": "weighted_avg",
            "ridge_alpha": 1.0,
            "enet_alpha": 0.01,
            "enet_l1": 0.5,
        },
        {
            "name": "equal_weight_vote",
            "weights": {"pca_sub": 1/3, "ridge": 1/3, "enet": 1/3},
            "combine": "sign_vote",
            "ridge_alpha": 1.0,
            "enet_alpha": 0.01,
            "enet_l1": 0.5,
        },
        {
            "name": "pca_heavy_vote",
            "weights": {"pca_sub": 0.6, "ridge": 0.2, "enet": 0.2},
            "combine": "sign_vote",
            "ridge_alpha": 1.0,
            "enet_alpha": 0.01,
            "enet_l1": 0.5,
        },
        {
            "name": "ridge_strong",
            "weights": {"pca_sub": 0.4, "ridge": 0.4, "enet": 0.2},
            "combine": "weighted_avg",
            "ridge_alpha": 0.1,
            "enet_alpha": 0.01,
            "enet_l1": 0.5,
        },
        {
            "name": "enet_strong",
            "weights": {"pca_sub": 0.4, "ridge": 0.2, "enet": 0.4},
            "combine": "weighted_avg",
            "ridge_alpha": 1.0,
            "enet_alpha": 0.005,
            "enet_l1": 0.3,
        },
        {
            "name": "pca_ridge_only",
            "weights": {"pca_sub": 0.5, "ridge": 0.5, "enet": 0.0},
            "combine": "weighted_avg",
            "ridge_alpha": 1.0,
            "enet_alpha": 0.01,
            "enet_l1": 0.5,
        },
    ]

    ensemble_results = []
    for cfg in ensemble_configs:
        print(f"  Testing: {cfg['name']}...")

        def make_model(c=cfg):
            return EnsembleModel(
                pca_sub_params={"K": 5, "L": 120, "lambda_decay": 1.0},
                ridge_alpha=c["ridge_alpha"],
                enet_alpha=c["enet_alpha"],
                enet_l1_ratio=c["enet_l1"],
                weights=c["weights"],
                combine_method=c["combine"],
            )

        wf = evaluator.evaluate(X_us, Y_jp, dates_us, model_factory=make_model)
        preds = wf.all_predictions

        # Per-sector accuracy for this ensemble
        ens_per_sector_acc = np.zeros(Y_jp.shape[1])
        for fold in wf.folds:
            ens_per_sector_acc += fold.per_sector_accuracy
        ens_per_sector_acc /= len(wf.folds)
        ens_ranked = np.argsort(ens_per_sector_acc)[::-1]
        ens_top5 = np.zeros(Y_jp.shape[1], dtype=bool)
        ens_top5[ens_ranked[:5]] = True

        # Strategy: EMA-20, top-5 (same as C12 setup)
        metrics = evaluate_strategy(preds, actuals, ens_top5, ema_halflife=20)

        # Also test with PCA_SUB's top-5 (fixed reference)
        metrics_fixed = evaluate_strategy(preds, actuals, top5_mask, ema_halflife=20)

        result_entry = {
            "name": cfg["name"],
            "combine_method": cfg["combine"],
            "weights": cfg["weights"],
            "direction_accuracy": round(wf.mean_direction_accuracy, 4),
            "sharpe_net_own_top5": metrics["sharpe_net"],
            "return_net_own_top5": metrics["return_net"],
            "turnover_own_top5": metrics["avg_turnover"],
            "sharpe_net_fixed_top5": metrics_fixed["sharpe_net"],
            "return_net_fixed_top5": metrics_fixed["return_net"],
            "turnover_fixed_top5": metrics_fixed["avg_turnover"],
        }
        ensemble_results.append(result_entry)

        print(f"    DirAcc={wf.mean_direction_accuracy:.4f}, "
              f"Net SR(own)={metrics['sharpe_net']:.4f}, "
              f"Net SR(fixed)={metrics_fixed['sharpe_net']:.4f}")

    # Sort by net Sharpe (fixed top5 for fair comparison)
    ensemble_results.sort(key=lambda x: x["sharpe_net_fixed_top5"], reverse=True)
    best_ensemble = ensemble_results[0]

    print(f"\n  Best ensemble: {best_ensemble['name']}, "
          f"Net SR={best_ensemble['sharpe_net_fixed_top5']:.4f} "
          f"(vs C12 {c12_metrics['sharpe_net']:.4f})")

    # ====================================================================
    # Section B: Individual model walk-forward comparison
    # ====================================================================
    print("\n[4/7] Running individual model comparisons...")

    from src.models.baselines import DirectRidge
    from src.models.ensemble import ElasticNetMultiOutput

    individual_results = {}

    # PCA_SUB alone (already have)
    pca_strat = evaluate_strategy(preds_pca, actuals, top5_mask, ema_halflife=20)
    individual_results["PCA_SUB"] = {
        "direction_accuracy": round(wf_pca.mean_direction_accuracy, 4),
        "sharpe_net": pca_strat["sharpe_net"],
        "return_net": pca_strat["return_net"],
        "turnover": pca_strat["avg_turnover"],
    }

    # Ridge alone
    print("  Testing Ridge alone...")
    def make_ridge():
        return DirectRidge(alpha=1.0)
    wf_ridge = evaluator.evaluate(X_us, Y_jp, dates_us, model_factory=make_ridge)
    ridge_metrics = evaluate_strategy(wf_ridge.all_predictions, actuals, top5_mask, ema_halflife=20)
    individual_results["Ridge"] = {
        "direction_accuracy": round(wf_ridge.mean_direction_accuracy, 4),
        "sharpe_net": ridge_metrics["sharpe_net"],
        "return_net": ridge_metrics["return_net"],
        "turnover": ridge_metrics["avg_turnover"],
    }
    print(f"    Ridge: DirAcc={wf_ridge.mean_direction_accuracy:.4f}, "
          f"Net SR={ridge_metrics['sharpe_net']:.4f}")

    # ElasticNet alone
    print("  Testing ElasticNet alone...")
    def make_enet():
        return ElasticNetMultiOutput(alpha=0.01, l1_ratio=0.5)
    wf_enet = evaluator.evaluate(X_us, Y_jp, dates_us, model_factory=make_enet)
    enet_metrics = evaluate_strategy(wf_enet.all_predictions, actuals, top5_mask, ema_halflife=20)
    individual_results["ElasticNet"] = {
        "direction_accuracy": round(wf_enet.mean_direction_accuracy, 4),
        "sharpe_net": enet_metrics["sharpe_net"],
        "return_net": enet_metrics["return_net"],
        "turnover": enet_metrics["avg_turnover"],
    }
    print(f"    ElasticNet: DirAcc={wf_enet.mean_direction_accuracy:.4f}, "
          f"Net SR={enet_metrics['sharpe_net']:.4f}")

    # ====================================================================
    # Section C: Regime-aware positioning
    # ====================================================================
    print("\n[5/7] Testing regime-aware positioning...")

    # Use the best available predictions (PCA_SUB baseline or best ensemble)
    # Compute portfolio-level returns for regime detection
    c12_full = evaluate_strategy(preds_pca, actuals, top5_mask, ema_halflife=20)
    baseline_daily_net = c12_full.get("daily_returns_net")
    if baseline_daily_net is None:
        # Rerun to get daily returns
        strat = TradingStrategy(ema_halflife=20, sector_mask=top5_mask, cost_bps=10.0)
        strat_result = strat.run(preds_pca, actuals)
        baseline_daily_net = strat_result["daily_returns_net"]

    # Also compute US market proxy returns for regime detection
    # Use equal-weight US returns from actuals as proxy
    us_proxy = X_us[-len(actuals):]  # aligned with OOS predictions
    us_market_returns = us_proxy.mean(axis=1) if len(us_proxy) == len(actuals) else actuals.mean(axis=1)

    regime_configs = []

    for vol_lb in [10, 21, 42, 63]:
        for n_reg in [2, 3]:
            for scale_set_name, scale_set in [
                ("conservative", {0: 1.0, 1: 0.3} if n_reg == 2 else {0: 1.0, 1: 0.6, 2: 0.3}),
                ("moderate", {0: 1.0, 1: 0.5} if n_reg == 2 else {0: 1.0, 1: 0.75, 2: 0.5}),
                ("mild", {0: 1.0, 1: 0.7} if n_reg == 2 else {0: 1.0, 1: 0.85, 2: 0.7}),
            ]:
                detector = VolatilityRegimeDetector(
                    vol_lookback=vol_lb, n_regimes=n_reg
                )
                # Detect regimes from US market returns (no lookahead)
                scales = detector.compute_regime_exposure(us_market_returns, scale_set)
                regimes = detector.classify_regimes(us_market_returns)

                regime_metrics = evaluate_strategy(
                    preds_pca, actuals, top5_mask, ema_halflife=20,
                    exposure_scales=scales
                )

                regime_dist = {i: int(np.sum(regimes == i)) for i in range(n_reg)}

                regime_configs.append({
                    "vol_lookback": vol_lb,
                    "n_regimes": n_reg,
                    "scale_set": scale_set_name,
                    "scales": scale_set,
                    "sharpe_net": regime_metrics["sharpe_net"],
                    "return_net": regime_metrics["return_net"],
                    "max_dd_net": regime_metrics["max_dd_net"],
                    "avg_turnover": regime_metrics["avg_turnover"],
                    "regime_distribution": regime_dist,
                })

    regime_configs.sort(key=lambda x: x["sharpe_net"], reverse=True)

    print(f"  Tested {len(regime_configs)} regime configurations")
    print(f"  Best regime config: vol_lb={regime_configs[0]['vol_lookback']}, "
          f"n_reg={regime_configs[0]['n_regimes']}, "
          f"scale={regime_configs[0]['scale_set']}")
    print(f"    Net SR={regime_configs[0]['sharpe_net']:.4f} "
          f"(vs C12 {c12_metrics['sharpe_net']:.4f})")

    # ====================================================================
    # Section D: Combined ensemble + regime
    # ====================================================================
    print("\n[6/7] Testing combined ensemble + regime approach...")

    # Use best ensemble predictions
    best_ens_name = best_ensemble["name"]
    best_ens_cfg = next(c for c in ensemble_configs if c["name"] == best_ens_name)

    def make_best_ensemble():
        return EnsembleModel(
            pca_sub_params={"K": 5, "L": 120, "lambda_decay": 1.0},
            ridge_alpha=best_ens_cfg["ridge_alpha"],
            enet_alpha=best_ens_cfg["enet_alpha"],
            enet_l1_ratio=best_ens_cfg["enet_l1"],
            weights=best_ens_cfg["weights"],
            combine_method=best_ens_cfg["combine"],
        )

    wf_best_ens = evaluator.evaluate(X_us, Y_jp, dates_us, model_factory=make_best_ensemble)
    preds_best_ens = wf_best_ens.all_predictions

    # Top-3 regime configs applied to best ensemble
    combined_results = []
    for rc in regime_configs[:5]:
        detector = VolatilityRegimeDetector(
            vol_lookback=rc["vol_lookback"], n_regimes=rc["n_regimes"]
        )
        scales = detector.compute_regime_exposure(us_market_returns, rc["scales"])

        # With PCA_SUB predictions + regime
        pca_regime = evaluate_strategy(
            preds_pca, actuals, top5_mask, ema_halflife=20,
            exposure_scales=scales
        )

        # With ensemble predictions + regime
        ens_regime = evaluate_strategy(
            preds_best_ens, actuals, top5_mask, ema_halflife=20,
            exposure_scales=scales
        )

        combined_results.append({
            "regime_config": f"vol{rc['vol_lookback']}_reg{rc['n_regimes']}_{rc['scale_set']}",
            "pca_sub_sharpe_net": pca_regime["sharpe_net"],
            "pca_sub_return_net": pca_regime["return_net"],
            "ensemble_sharpe_net": ens_regime["sharpe_net"],
            "ensemble_return_net": ens_regime["return_net"],
            "pca_sub_turnover": pca_regime["avg_turnover"],
            "ensemble_turnover": ens_regime["avg_turnover"],
        })

    combined_results.sort(key=lambda x: max(x["pca_sub_sharpe_net"], x["ensemble_sharpe_net"]), reverse=True)

    for cr in combined_results[:3]:
        print(f"  {cr['regime_config']}: "
              f"PCA_SUB SR={cr['pca_sub_sharpe_net']:.4f}, "
              f"Ensemble SR={cr['ensemble_sharpe_net']:.4f}")

    # ====================================================================
    # Section E: Borrowing cost sensitivity
    # ====================================================================
    print("\n[6.5/7] Borrowing cost sensitivity...")

    borrow_results = {}
    for borrow_bps in [0, 75]:
        # C12 baseline
        c12_b = evaluate_strategy(preds_pca, actuals, top5_mask, ema_halflife=20,
                                  borrow_bps=borrow_bps)
        borrow_results[f"C12_borrow{borrow_bps}"] = {
            "sharpe_net": c12_b["sharpe_net"],
            "return_net": c12_b["return_net"],
        }

        # Best ensemble with fixed top5
        ens_b = evaluate_strategy(preds_best_ens, actuals, top5_mask, ema_halflife=20,
                                  borrow_bps=borrow_bps)
        borrow_results[f"best_ensemble_borrow{borrow_bps}"] = {
            "sharpe_net": ens_b["sharpe_net"],
            "return_net": ens_b["return_net"],
        }

        # Best regime config
        best_rc = regime_configs[0]
        detector = VolatilityRegimeDetector(
            vol_lookback=best_rc["vol_lookback"], n_regimes=best_rc["n_regimes"]
        )
        scales = detector.compute_regime_exposure(us_market_returns, best_rc["scales"])
        reg_b = evaluate_strategy(preds_pca, actuals, top5_mask, ema_halflife=20,
                                  borrow_bps=borrow_bps, exposure_scales=scales)
        borrow_results[f"best_regime_borrow{borrow_bps}"] = {
            "sharpe_net": reg_b["sharpe_net"],
            "return_net": reg_b["return_net"],
        }

    for k, v in borrow_results.items():
        print(f"  {k}: Net SR={v['sharpe_net']:.4f}, Return={v['return_net']:.4f}")

    # ====================================================================
    # Save results
    # ====================================================================
    print("\n[7/7] Saving results...")
    report_dir = Path("reports/cycle_15")
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "phase": 15,
        "description": "Ensemble model diversity and regime-aware positioning",
        "timestamp": "2026-03-27",
        "c12_baseline": {
            "sharpe_net": c12_metrics["sharpe_net"],
            "return_net": c12_metrics["return_net"],
            "avg_turnover": c12_metrics["avg_turnover"],
        },
        "individual_models": individual_results,
        "ensemble_results": ensemble_results,
        "best_ensemble": {
            "name": best_ensemble["name"],
            "sharpe_net_fixed_top5": best_ensemble["sharpe_net_fixed_top5"],
            "return_net_fixed_top5": best_ensemble["return_net_fixed_top5"],
            "direction_accuracy": best_ensemble["direction_accuracy"],
            "improvement_over_c12": round(
                best_ensemble["sharpe_net_fixed_top5"] - c12_metrics["sharpe_net"], 4
            ),
        },
        "regime_detection": {
            "total_configs_tested": len(regime_configs),
            "best_config": {
                "vol_lookback": regime_configs[0]["vol_lookback"],
                "n_regimes": regime_configs[0]["n_regimes"],
                "scale_set": regime_configs[0]["scale_set"],
                "sharpe_net": regime_configs[0]["sharpe_net"],
                "return_net": regime_configs[0]["return_net"],
                "improvement_over_c12": round(
                    regime_configs[0]["sharpe_net"] - c12_metrics["sharpe_net"], 4
                ),
            },
            "top_10_configs": [
                {k: v for k, v in rc.items() if k != "scales"}
                for rc in regime_configs[:10]
            ],
        },
        "combined_ensemble_regime": combined_results,
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
    }

    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved metrics to {report_dir / 'metrics.json'}")

    # Generate technical findings
    findings = generate_technical_findings(
        c12_metrics, individual_results, ensemble_results,
        best_ensemble, regime_configs, combined_results, borrow_results,
        wf_pca
    )
    with open(report_dir / "technical_findings.md", "w") as f:
        f.write(findings)
    print(f"  Saved findings to {report_dir / 'technical_findings.md'}")

    print("\n" + "=" * 70)
    print("Cycle 15 complete!")
    print("=" * 70)


def generate_technical_findings(c12, individual, ensemble_results, best_ens,
                                 regime_configs, combined, borrow, wf_pca):
    """Generate markdown technical findings report."""

    lines = [
        "# Cycle 15: Ensemble Model Diversity & Regime-Aware Positioning",
        "",
        "## Summary",
        "",
        "This cycle addresses two open questions from Cycle 14:",
        "",
        "1. **Ensemble model diversity** — Combine PCA_SUB with Ridge and ElasticNet",
        "   regression to improve prediction quality through model diversity, rather",
        "   than multi-horizon averaging of a single model's predictions.",
        "",
        "2. **Regime-aware positioning** — Use volatility-based regime detection to",
        "   scale exposure, reducing positions in high-volatility regimes where the",
        "   lead-lag signal is weaker (confirmed in Phase 7).",
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
        "### 1. Individual Model Comparison",
        "",
        "| Model | Direction Accuracy | Net Sharpe | Net Return | Turnover |",
        "|-------|-------------------|------------|------------|----------|",
    ]

    for name, m in individual.items():
        lines.append(f"| {name} | {m['direction_accuracy']} | {m['sharpe_net']} | "
                     f"{m['return_net']} | {m['turnover']} |")

    lines += [
        "",
        "### 2. Ensemble Model Results",
        "",
        "Tested 8 ensemble configurations with different weighting schemes",
        "and combination methods (weighted average vs sign vote).",
        "",
        "| Rank | Config | Method | Dir Acc | SR(net, fixed top5) | Return(net) | Turnover |",
        "|------|--------|--------|---------|---------------------|-------------|----------|",
    ]

    for i, er in enumerate(ensemble_results[:8]):
        lines.append(f"| {i+1} | {er['name']} | {er['combine_method']} | "
                     f"{er['direction_accuracy']} | {er['sharpe_net_fixed_top5']} | "
                     f"{er['return_net_fixed_top5']} | {er['turnover_fixed_top5']} |")

    lines += [
        "",
        f"**Best ensemble**: {best_ens['name']}",
        f"- Net Sharpe: {best_ens['sharpe_net_fixed_top5']} "
        f"(vs C12 {c12['sharpe_net']}, "
        f"delta {best_ens['sharpe_net_fixed_top5'] - c12['sharpe_net']:+.4f})",
        f"- Direction accuracy: {best_ens['direction_accuracy']}",
        "",
        "### 3. Regime-Aware Positioning",
        "",
        f"Tested {len(regime_configs)} regime configurations across volatility lookbacks",
        "(10/21/42/63 days), regime counts (2/3), and exposure scaling levels",
        "(conservative/moderate/mild).",
        "",
        "| Rank | Vol LB | Regimes | Scale | SR(net) | Return(net) | Max DD | Turnover |",
        "|------|--------|---------|-------|---------|-------------|--------|----------|",
    ]

    for i, rc in enumerate(regime_configs[:10]):
        lines.append(f"| {i+1} | {rc['vol_lookback']} | {rc['n_regimes']} | "
                     f"{rc['scale_set']} | {rc['sharpe_net']} | {rc['return_net']} | "
                     f"{rc['max_dd_net']} | {rc['avg_turnover']} |")

    best_rc = regime_configs[0]
    lines += [
        "",
        f"**Best regime config**: vol_lookback={best_rc['vol_lookback']}, "
        f"n_regimes={best_rc['n_regimes']}, scale={best_rc['scale_set']}",
        f"- Net Sharpe: {best_rc['sharpe_net']} "
        f"(vs C12 {c12['sharpe_net']}, "
        f"delta {best_rc['sharpe_net'] - c12['sharpe_net']:+.4f})",
        f"- Regime distribution: {best_rc['regime_distribution']}",
        "",
        "### 4. Combined Ensemble + Regime",
        "",
        "| Config | PCA_SUB SR(net) | Ensemble SR(net) | PCA_SUB Return | Ensemble Return |",
        "|--------|----------------|------------------|----------------|-----------------|",
    ]

    for cr in combined[:5]:
        lines.append(f"| {cr['regime_config']} | {cr['pca_sub_sharpe_net']} | "
                     f"{cr['ensemble_sharpe_net']} | {cr['pca_sub_return_net']} | "
                     f"{cr['ensemble_return_net']} |")

    lines += [
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
        "### Ensemble Model",
        "- **PCA_SUB**: K=5, L=120, lambda=1.0 (optimized params from Phase 6)",
        "- **Ridge**: L2-regularized direct regression (US->JP sectors)",
        "- **ElasticNet**: L1+L2 regularized regression (per-sector)",
        "- **Combination methods**: Weighted average and sign vote",
        "- All models fit independently on each walk-forward training window",
        "",
        "### Regime Detection",
        "- Rolling realized volatility of US equal-weight sector returns",
        "- Expanding-window quantile classification (no lookahead)",
        "- Exposure scaled down in high-vol regimes (confirmed weaker signal in Phase 7)",
        "",
        "### Walk-Forward Setup",
        f"- **Train window**: 252 days, **Test window**: 21 days",
        f"- **Folds**: {wf_pca.n_folds}",
        f"- **Total OOS samples**: {wf_pca.total_test_samples}",
        f"- **Direction accuracy (PCA_SUB)**: {wf_pca.mean_direction_accuracy:.4f}",
        "",
        "### Cost Assumptions",
        "- One-way transaction cost: 10 bps",
        "- Realistic borrowing cost: 75 bps annualized",
        "",
        "## Conclusions",
        "",
        "1. **Model diversity** through ensemble combining PCA_SUB, Ridge, and ElasticNet",
        "   provides an alternative prediction approach. The comparison reveals whether",
        "   combining fundamentally different regression approaches (subspace vs direct)",
        "   improves the cross-market lead-lag signal.",
        "",
        "2. **Regime-aware positioning** scales exposure based on market volatility.",
        "   Phase 7 confirmed the model performs better in low-vol regimes (SR=0.72 vs",
        "   0.47 in high-vol), so reducing exposure in high-vol periods should improve",
        "   risk-adjusted returns.",
        "",
        "3. The Cycle 12 baseline (EMA-20, top-5, net Sharpe 0.63) remains the",
        "   benchmark to beat. Any improvement must be evaluated with borrowing costs.",
        "",
        "## Open Questions for Future Cycles",
        "",
        "1. **Online model selection**: Could we dynamically choose between PCA_SUB and",
        "   ensemble based on recent performance, rather than fixed weights?",
        "2. **Factor timing**: Could exposure to specific PCA factors be timed based on",
        "   regime indicators rather than blanket exposure scaling?",
        "3. **Transaction cost optimization in regime transitions**: Could position",
        "   changes during regime transitions be smoothed to avoid unnecessary turnover?",
        "4. **Out-of-sample validation**: All improvements should be tested on post-2026",
        "   data to confirm they are not overfit.",
    ]

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
