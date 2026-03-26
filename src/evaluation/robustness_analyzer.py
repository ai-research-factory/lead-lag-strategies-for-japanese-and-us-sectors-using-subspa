"""Robustness verification and sensitivity analysis for PCA_SUB model.

Evaluates model stability across:
1. Parameter sensitivity: one-at-a-time variation of K, L, lambda_decay
2. Market regime analysis: high-vol vs low-vol, trending vs ranging
3. Temporal stability: rolling Sharpe and accuracy over time
4. Sector-level robustness: consistency of sector predictability
5. Sub-period analysis: yearly performance breakdown
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.walk_forward import WalkForwardEvaluator, WalkForwardResult


@dataclass
class SensitivityPoint:
    """Result for a single parameter configuration."""
    param_name: str
    param_value: float
    direction_accuracy: float
    sharpe_ratio_gross: float
    annualized_return: float
    max_drawdown: float
    n_folds: int


@dataclass
class RegimeResult:
    """Performance metrics for a specific market regime."""
    regime_name: str
    n_days: int
    direction_accuracy: float
    sharpe_ratio_gross: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    pct_positive_days: float


@dataclass
class SubPeriodResult:
    """Performance metrics for a specific time sub-period."""
    period_label: str
    start_date: str
    end_date: str
    n_days: int
    direction_accuracy: float
    sharpe_ratio_gross: float
    annualized_return: float
    max_drawdown: float


@dataclass
class RobustnessResult:
    """Full robustness analysis results."""
    sensitivity: dict = field(default_factory=dict)
    regime_analysis: list = field(default_factory=list)
    temporal_stability: dict = field(default_factory=dict)
    sector_robustness: dict = field(default_factory=dict)
    sub_period_analysis: list = field(default_factory=list)
    bootstrap_confidence: dict = field(default_factory=dict)


class RobustnessAnalyzer:
    """Robustness and sensitivity analyzer for PCA_SUB.

    Parameters
    ----------
    base_K : int
        Baseline number of principal components.
    base_L : int
        Baseline lookback window.
    base_lambda : float
        Baseline decay rate.
    train_window : int
        Walk-forward training window size.
    test_window : int
        Walk-forward test window size.
    """

    def __init__(
        self,
        base_K: int = 3,
        base_L: int = 60,
        base_lambda: float = 0.9,
        train_window: int = 252,
        test_window: int = 21,
    ):
        self.base_K = base_K
        self.base_L = base_L
        self.base_lambda = base_lambda
        self.train_window = train_window
        self.test_window = test_window

    def _run_wf(
        self, X_us, Y_jp, dates_us, K, L, lam
    ) -> tuple[WalkForwardResult, dict]:
        """Run walk-forward and compute strategy metrics."""
        evaluator = WalkForwardEvaluator(
            train_window=self.train_window,
            test_window=self.test_window,
            K=K, L=L, lambda_decay=lam,
        )
        result = evaluator.evaluate(X_us, Y_jp, dates_us)
        strategy = evaluator.compute_oos_sharpe(result)
        return result, strategy

    def run_parameter_sensitivity(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
    ) -> dict[str, list[SensitivityPoint]]:
        """One-at-a-time parameter sensitivity analysis.

        Varies each parameter while holding others at baseline values.
        """
        sensitivity = {}

        # K sensitivity
        k_values = [1, 2, 3, 4, 5, 6, 7]
        print("Parameter sensitivity: K")
        points = []
        for k in k_values:
            print(f"  K={k}...", end="", flush=True)
            result, strategy = self._run_wf(
                X_us, Y_jp, dates_us, K=k, L=self.base_L, lam=self.base_lambda
            )
            sp = SensitivityPoint(
                param_name="K", param_value=k,
                direction_accuracy=result.mean_direction_accuracy,
                sharpe_ratio_gross=strategy.get("sharpe_ratio_gross", 0.0),
                annualized_return=strategy.get("annualized_return_gross", 0.0),
                max_drawdown=strategy.get("max_drawdown", 0.0),
                n_folds=result.n_folds,
            )
            points.append(sp)
            print(f" acc={sp.direction_accuracy:.4f}, SR={sp.sharpe_ratio_gross:.4f}")
        sensitivity["K"] = points

        # L sensitivity
        l_values = [10, 20, 40, 60, 80, 120, 160, 200]
        print("Parameter sensitivity: L")
        points = []
        for l_val in l_values:
            print(f"  L={l_val}...", end="", flush=True)
            result, strategy = self._run_wf(
                X_us, Y_jp, dates_us, K=self.base_K, L=l_val, lam=self.base_lambda
            )
            sp = SensitivityPoint(
                param_name="L", param_value=l_val,
                direction_accuracy=result.mean_direction_accuracy,
                sharpe_ratio_gross=strategy.get("sharpe_ratio_gross", 0.0),
                annualized_return=strategy.get("annualized_return_gross", 0.0),
                max_drawdown=strategy.get("max_drawdown", 0.0),
                n_folds=result.n_folds,
            )
            points.append(sp)
            print(f" acc={sp.direction_accuracy:.4f}, SR={sp.sharpe_ratio_gross:.4f}")
        sensitivity["L"] = points

        # lambda_decay sensitivity
        lam_values = [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 1.0]
        print("Parameter sensitivity: lambda_decay")
        points = []
        for lam in lam_values:
            print(f"  λ={lam}...", end="", flush=True)
            result, strategy = self._run_wf(
                X_us, Y_jp, dates_us, K=self.base_K, L=self.base_L, lam=lam
            )
            sp = SensitivityPoint(
                param_name="lambda_decay", param_value=lam,
                direction_accuracy=result.mean_direction_accuracy,
                sharpe_ratio_gross=strategy.get("sharpe_ratio_gross", 0.0),
                annualized_return=strategy.get("annualized_return_gross", 0.0),
                max_drawdown=strategy.get("max_drawdown", 0.0),
                n_folds=result.n_folds,
            )
            points.append(sp)
            print(f" acc={sp.direction_accuracy:.4f}, SR={sp.sharpe_ratio_gross:.4f}")
        sensitivity["lambda_decay"] = points

        return sensitivity

    def run_regime_analysis(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
        wf_result: WalkForwardResult,
        strategy: dict,
    ) -> list[RegimeResult]:
        """Analyze model performance under different market regimes.

        Splits out-of-sample predictions by market volatility and trend regimes.
        """
        Y_pred = wf_result.all_predictions
        Y_actual = wf_result.all_actuals
        daily_returns = strategy.get("daily_returns")

        if Y_pred is None or Y_actual is None or daily_returns is None:
            return []

        # Compute regime indicators from actual JP market returns
        jp_daily_vol = np.std(Y_actual, axis=1)  # cross-sector vol per day
        jp_market_return = np.mean(Y_actual, axis=1)  # equal-weight JP market return

        # Regime 1: High vs Low volatility (median split)
        vol_median = np.median(jp_daily_vol)
        high_vol_mask = jp_daily_vol >= vol_median
        low_vol_mask = ~high_vol_mask

        # Regime 2: Positive vs Negative JP market days
        up_market_mask = jp_market_return > 0
        down_market_mask = jp_market_return <= 0

        # Compute US market volatility using 21-day rolling window on OOS returns
        us_market = np.mean(np.abs(Y_actual), axis=1)  # proxy for activity
        us_vol_median = np.median(us_market)
        high_us_activity = us_market >= us_vol_median
        low_us_activity = ~high_us_activity

        regimes = [
            ("High JP Volatility", high_vol_mask),
            ("Low JP Volatility", low_vol_mask),
            ("JP Up Market", up_market_mask),
            ("JP Down Market", down_market_mask),
            ("High Cross-Sector Dispersion", high_us_activity),
            ("Low Cross-Sector Dispersion", low_us_activity),
        ]

        results = []
        for name, mask in regimes:
            n = mask.sum()
            if n < 5:
                continue

            pred_sub = Y_pred[mask]
            actual_sub = Y_actual[mask]
            returns_sub = daily_returns[mask]

            # Direction accuracy
            correct = np.sign(pred_sub) == np.sign(actual_sub)
            acc = float(correct.mean())

            # Strategy metrics
            mean_d = returns_sub.mean()
            std_d = returns_sub.std()
            sr = (mean_d / std_d * np.sqrt(252)) if std_d > 1e-12 else 0.0
            ann_ret = float(mean_d * 252)
            ann_vol = float(std_d * np.sqrt(252))

            cum = np.cumprod(1 + returns_sub)
            max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
            pct_pos = float(np.mean(returns_sub > 0) * 100)

            results.append(RegimeResult(
                regime_name=name, n_days=int(n),
                direction_accuracy=round(acc, 6),
                sharpe_ratio_gross=round(sr, 4),
                annualized_return=round(ann_ret, 6),
                annualized_volatility=round(ann_vol, 6),
                max_drawdown=round(max_dd, 6),
                pct_positive_days=round(pct_pos, 2),
            ))

        return results

    def run_temporal_stability(
        self,
        wf_result: WalkForwardResult,
    ) -> dict:
        """Analyze temporal stability via rolling fold-level metrics."""
        folds = wf_result.folds
        if len(folds) < 3:
            return {}

        accs = [f.direction_accuracy for f in folds]
        corrs = [f.mean_correlation for f in folds]

        # Rolling 6-fold (approx 6 months) metrics
        window = min(6, len(folds))
        rolling_acc = []
        rolling_corr = []
        for i in range(len(folds) - window + 1):
            rolling_acc.append(float(np.mean(accs[i:i+window])))
            rolling_corr.append(float(np.mean(corrs[i:i+window])))

        # Compute trend in accuracy (simple linear regression)
        x = np.arange(len(accs))
        acc_trend_slope = float(np.polyfit(x, accs, 1)[0])

        # Consecutive positive/negative streaks
        above_50 = [1 if a > 0.5 else 0 for a in accs]
        max_pos_streak = _max_consecutive(above_50, 1)
        max_neg_streak = _max_consecutive(above_50, 0)

        return {
            "fold_accuracies": [round(a, 6) for a in accs],
            "fold_correlations": [round(c, 6) for c in corrs],
            "fold_dates": [f.test_start_date for f in folds],
            "rolling_6fold_accuracy": [round(a, 6) for a in rolling_acc],
            "rolling_6fold_correlation": [round(c, 6) for c in rolling_corr],
            "accuracy_trend_slope_per_fold": round(acc_trend_slope, 8),
            "max_consecutive_above_50pct": max_pos_streak,
            "max_consecutive_below_50pct": max_neg_streak,
            "accuracy_range": round(max(accs) - min(accs), 6),
            "accuracy_iqr": round(float(np.percentile(accs, 75) - np.percentile(accs, 25)), 6),
        }

    def run_sector_robustness(
        self,
        wf_result: WalkForwardResult,
        jp_tickers: list[str],
        jp_sector_names: dict[str, str],
    ) -> dict:
        """Analyze robustness of sector-level predictability.

        For each JP sector, compute:
        - Overall direction accuracy
        - Fraction of folds where accuracy > 50%
        - Stability (std of per-fold accuracy)
        - Per-sector long-short Sharpe
        """
        folds = wf_result.folds
        if len(folds) == 0:
            return {}

        n_jp = folds[0].Y_true.shape[1]
        sector_results = {}

        for j in range(n_jp):
            ticker = jp_tickers[j] if j < len(jp_tickers) else f"sector_{j}"
            name = jp_sector_names.get(ticker, ticker)

            # Per-fold accuracy for this sector
            fold_accs = [float(f.per_sector_accuracy[j]) for f in folds]
            fold_corrs = [float(f.per_sector_correlation[j]) for f in folds]

            # Overall accuracy from concatenated predictions
            all_true = np.concatenate([f.Y_true[:, j] for f in folds])
            all_pred = np.concatenate([f.Y_pred[:, j] for f in folds])
            overall_acc = float(np.mean(np.sign(all_true) == np.sign(all_pred)))

            # Per-sector long-short return
            positions = np.sign(all_pred)
            sector_returns = positions * all_true
            mean_d = sector_returns.mean()
            std_d = sector_returns.std()
            sector_sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 1e-12 else 0.0

            # Consistency: fraction of folds with > 50% accuracy
            folds_above_50 = float(np.mean([1 if a > 0.5 else 0 for a in fold_accs]) * 100)

            sector_results[ticker] = {
                "name": name,
                "overall_accuracy": round(overall_acc, 6),
                "accuracy_std": round(float(np.std(fold_accs)), 6),
                "accuracy_min": round(min(fold_accs), 6),
                "accuracy_max": round(max(fold_accs), 6),
                "folds_above_50pct": round(folds_above_50, 2),
                "mean_correlation": round(float(np.mean(fold_corrs)), 6),
                "correlation_std": round(float(np.std(fold_corrs)), 6),
                "sector_sharpe": round(sector_sharpe, 4),
                "pct_positive_days": round(float(np.mean(sector_returns > 0) * 100), 2),
            }

        return sector_results

    def run_sub_period_analysis(
        self,
        wf_result: WalkForwardResult,
        strategy: dict,
    ) -> list[SubPeriodResult]:
        """Break performance into yearly sub-periods."""
        Y_pred = wf_result.all_predictions
        Y_actual = wf_result.all_actuals
        dates = wf_result.all_dates_us
        daily_returns = strategy.get("daily_returns")

        if Y_pred is None or dates is None or daily_returns is None:
            return []

        dates_arr = pd.DatetimeIndex(dates)
        years = sorted(set(d.year for d in dates_arr))

        results = []
        for year in years:
            mask = np.array([d.year == year for d in dates_arr])
            n = mask.sum()
            if n < 5:
                continue

            pred_sub = Y_pred[mask]
            actual_sub = Y_actual[mask]
            returns_sub = daily_returns[mask]

            correct = np.sign(pred_sub) == np.sign(actual_sub)
            acc = float(correct.mean())

            mean_d = returns_sub.mean()
            std_d = returns_sub.std()
            sr = (mean_d / std_d * np.sqrt(252)) if std_d > 1e-12 else 0.0

            cum = np.cumprod(1 + returns_sub)
            max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))

            year_dates = dates_arr[mask]
            results.append(SubPeriodResult(
                period_label=str(year),
                start_date=str(year_dates[0].date()),
                end_date=str(year_dates[-1].date()),
                n_days=int(n),
                direction_accuracy=round(acc, 6),
                sharpe_ratio_gross=round(sr, 4),
                annualized_return=round(float(mean_d * 252), 6),
                max_drawdown=round(max_dd, 6),
            ))

        return results

    def run_bootstrap_confidence(
        self,
        wf_result: WalkForwardResult,
        strategy: dict,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> dict:
        """Bootstrap confidence intervals for key metrics.

        Resamples daily strategy returns (block bootstrap using fold-level blocks)
        to estimate confidence intervals for Sharpe ratio and direction accuracy.
        """
        folds = wf_result.folds
        daily_returns = strategy.get("daily_returns")
        if daily_returns is None or len(folds) == 0:
            return {}

        rng = np.random.default_rng(seed)

        # Block bootstrap: resample folds (preserves temporal structure within folds)
        fold_returns = []
        fold_accuracies_per_day = []
        for f in folds:
            positions = np.sign(f.Y_pred)
            n_active = np.abs(positions).sum(axis=1, keepdims=True)
            n_active = np.where(n_active == 0, 1, n_active)
            weights = positions / n_active
            f_returns = (weights * f.Y_true).sum(axis=1)
            fold_returns.append(f_returns)

            correct = (np.sign(f.Y_pred) == np.sign(f.Y_true)).mean(axis=1)
            fold_accuracies_per_day.append(correct)

        n_folds = len(folds)
        boot_sharpes = []
        boot_accs = []
        boot_returns = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_folds, size=n_folds)
            sampled_rets = np.concatenate([fold_returns[i] for i in idx])
            sampled_accs = np.concatenate([fold_accuracies_per_day[i] for i in idx])

            mean_d = sampled_rets.mean()
            std_d = sampled_rets.std()
            sr = (mean_d / std_d * np.sqrt(252)) if std_d > 1e-12 else 0.0
            boot_sharpes.append(sr)
            boot_accs.append(float(sampled_accs.mean()))
            boot_returns.append(float(mean_d * 252))

        boot_sharpes = np.array(boot_sharpes)
        boot_accs = np.array(boot_accs)
        boot_returns = np.array(boot_returns)

        return {
            "n_bootstrap": n_bootstrap,
            "sharpe_ratio": {
                "mean": round(float(boot_sharpes.mean()), 4),
                "std": round(float(boot_sharpes.std()), 4),
                "ci_5pct": round(float(np.percentile(boot_sharpes, 5)), 4),
                "ci_95pct": round(float(np.percentile(boot_sharpes, 95)), 4),
                "pct_positive": round(float(np.mean(boot_sharpes > 0) * 100), 2),
            },
            "direction_accuracy": {
                "mean": round(float(boot_accs.mean()), 6),
                "std": round(float(boot_accs.std()), 6),
                "ci_5pct": round(float(np.percentile(boot_accs, 5)), 6),
                "ci_95pct": round(float(np.percentile(boot_accs, 95)), 6),
                "pct_above_50": round(float(np.mean(boot_accs > 0.5) * 100), 2),
            },
            "annualized_return": {
                "mean": round(float(boot_returns.mean()), 6),
                "std": round(float(boot_returns.std()), 6),
                "ci_5pct": round(float(np.percentile(boot_returns, 5)), 6),
                "ci_95pct": round(float(np.percentile(boot_returns, 95)), 6),
                "pct_positive": round(float(np.mean(boot_returns > 0) * 100), 2),
            },
        }

    def run_full_analysis(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
        jp_tickers: list[str],
        jp_sector_names: dict[str, str],
    ) -> RobustnessResult:
        """Run all robustness analyses."""
        result = RobustnessResult()

        # 1. Baseline walk-forward (used by several analyses)
        print("=" * 60)
        print("Running baseline walk-forward...")
        wf_result, strategy = self._run_wf(
            X_us, Y_jp, dates_us,
            K=self.base_K, L=self.base_L, lam=self.base_lambda,
        )
        print(f"  Baseline: acc={wf_result.mean_direction_accuracy:.4f}, "
              f"SR={strategy.get('sharpe_ratio_gross', 0):.4f}")
        print()

        # 2. Parameter sensitivity
        print("=" * 60)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 60)
        result.sensitivity = self.run_parameter_sensitivity(X_us, Y_jp, dates_us)
        print()

        # 3. Market regime analysis
        print("=" * 60)
        print("MARKET REGIME ANALYSIS")
        print("=" * 60)
        result.regime_analysis = self.run_regime_analysis(
            X_us, Y_jp, dates_us, wf_result, strategy
        )
        for r in result.regime_analysis:
            print(f"  {r.regime_name}: acc={r.direction_accuracy:.4f}, SR={r.sharpe_ratio_gross:.4f}")
        print()

        # 4. Temporal stability
        print("=" * 60)
        print("TEMPORAL STABILITY ANALYSIS")
        print("=" * 60)
        result.temporal_stability = self.run_temporal_stability(wf_result)
        print(f"  Accuracy trend slope: {result.temporal_stability.get('accuracy_trend_slope_per_fold', 0):.6f}")
        print(f"  Max consecutive >50%: {result.temporal_stability.get('max_consecutive_above_50pct', 0)}")
        print(f"  Max consecutive <50%: {result.temporal_stability.get('max_consecutive_below_50pct', 0)}")
        print()

        # 5. Sector robustness
        print("=" * 60)
        print("SECTOR-LEVEL ROBUSTNESS")
        print("=" * 60)
        result.sector_robustness = self.run_sector_robustness(
            wf_result, jp_tickers, jp_sector_names
        )
        for ticker, sr in result.sector_robustness.items():
            print(f"  {sr['name']:30s} acc={sr['overall_accuracy']:.4f}  "
                  f"SR={sr['sector_sharpe']:.4f}  consistency={sr['folds_above_50pct']:.0f}%")
        print()

        # 6. Sub-period analysis
        print("=" * 60)
        print("SUB-PERIOD (YEARLY) ANALYSIS")
        print("=" * 60)
        result.sub_period_analysis = self.run_sub_period_analysis(wf_result, strategy)
        for sp in result.sub_period_analysis:
            print(f"  {sp.period_label}: n={sp.n_days:3d}, acc={sp.direction_accuracy:.4f}, "
                  f"SR={sp.sharpe_ratio_gross:.4f}, DD={sp.max_drawdown:.4f}")
        print()

        # 7. Bootstrap confidence intervals
        print("=" * 60)
        print("BOOTSTRAP CONFIDENCE INTERVALS")
        print("=" * 60)
        result.bootstrap_confidence = self.run_bootstrap_confidence(
            wf_result, strategy, n_bootstrap=2000
        )
        bc = result.bootstrap_confidence
        if bc:
            print(f"  Sharpe 90% CI: [{bc['sharpe_ratio']['ci_5pct']:.4f}, "
                  f"{bc['sharpe_ratio']['ci_95pct']:.4f}]")
            print(f"  Accuracy 90% CI: [{bc['direction_accuracy']['ci_5pct']:.6f}, "
                  f"{bc['direction_accuracy']['ci_95pct']:.6f}]")
            print(f"  P(Sharpe > 0): {bc['sharpe_ratio']['pct_positive']:.1f}%")
            print(f"  P(Accuracy > 50%): {bc['direction_accuracy']['pct_above_50']:.1f}%")
        print()

        return result


def _max_consecutive(arr: list, value) -> int:
    """Find maximum number of consecutive occurrences of value in arr."""
    max_streak = 0
    current = 0
    for v in arr:
        if v == value:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak
