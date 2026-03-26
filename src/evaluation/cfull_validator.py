"""Cfull covariance estimation period validation.

Compares fixed-period covariance estimation (as proposed in the paper)
against rolling covariance estimation, using walk-forward evaluation
to measure the impact on out-of-sample performance.

The paper uses Cfull=2010-2014 for a fixed covariance matrix. Since our
data spans 2021-2026, we adapt by using an early portion of available data
as the fixed Cfull period and testing multiple Cfull window sizes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.pca_sub import PCASub
from src.evaluation.walk_forward import WalkForwardEvaluator, WalkForwardResult


@dataclass
class CfullComparisonResult:
    """Results from comparing fixed vs rolling covariance estimation."""
    method: str  # "rolling" or "cfull_<N>d"
    cfull_window_days: int | None  # None for rolling
    wf_result: WalkForwardResult
    strategy_metrics: dict
    description: str


class CfullValidator:
    """Validates fixed-period (Cfull) vs rolling covariance estimation.

    The paper proposes computing the PCA eigenvectors once from a fixed
    historical period (Cfull) and keeping them constant throughout the
    walk-forward. This contrasts with the rolling approach where eigenvectors
    are re-estimated at each fold from the current training window.

    Parameters
    ----------
    train_window : int
        Walk-forward training window size.
    test_window : int
        Walk-forward test window size.
    K : int
        Number of principal components.
    L : int
        Lookback window for regression (used in both modes).
    lambda_decay : float
        Exponential decay rate.
    cfull_windows : list[int]
        List of fixed Cfull window sizes (in trading days) to test.
        E.g., [126, 252, 504] tests ~6mo, ~1yr, ~2yr fixed periods.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 21,
        K: int = 3,
        L: int = 60,
        lambda_decay: float = 0.9,
        cfull_windows: list[int] | None = None,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.K = K
        self.L = L
        self.lambda_decay = lambda_decay
        self.cfull_windows = cfull_windows or [126, 252, 504]

    def run_rolling_baseline(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
    ) -> CfullComparisonResult:
        """Run walk-forward with standard rolling covariance (baseline)."""
        evaluator = WalkForwardEvaluator(
            train_window=self.train_window,
            test_window=self.test_window,
            K=self.K,
            L=self.L,
            lambda_decay=self.lambda_decay,
        )
        wf_result = evaluator.evaluate(X_us, Y_jp, dates_us)
        strategy = evaluator.compute_oos_sharpe(wf_result)

        return CfullComparisonResult(
            method="rolling",
            cfull_window_days=None,
            wf_result=wf_result,
            strategy_metrics=strategy,
            description="Rolling covariance: PCA re-estimated at each fold from training window",
        )

    def run_cfull_fixed(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
        cfull_window: int,
    ) -> CfullComparisonResult:
        """Run walk-forward with fixed Cfull covariance estimation.

        Uses the first `cfull_window` observations to compute eigenvectors
        once, then keeps them fixed throughout the entire walk-forward.
        The walk-forward itself starts after the Cfull period.
        """
        if cfull_window >= X_us.shape[0] - self.train_window - self.test_window:
            raise ValueError(
                f"Cfull window ({cfull_window}) too large for available data "
                f"({X_us.shape[0]} obs). Need room for walk-forward after Cfull."
            )

        # Compute fixed eigenvectors from the Cfull period
        X_cfull = X_us[:cfull_window]
        fixed_eigvecs = PCASub.compute_cfull_eigvecs(
            X_cfull, K=self.K, lambda_decay=self.lambda_decay
        )

        # Run walk-forward on data AFTER the Cfull period
        # This ensures no overlap between Cfull estimation and test data
        X_post = X_us[cfull_window:]
        Y_post = Y_jp[cfull_window:]
        dates_post = dates_us[cfull_window:]

        # Custom walk-forward with fixed eigenvectors
        wf_result = self._walk_forward_fixed_eigvecs(
            X_post, Y_post, dates_post, fixed_eigvecs
        )

        # Also run walk-forward on the SAME post-Cfull data with rolling
        # for a fair comparison (same test period)
        strategy = self._compute_strategy_metrics(wf_result)

        return CfullComparisonResult(
            method=f"cfull_{cfull_window}d",
            cfull_window_days=cfull_window,
            wf_result=wf_result,
            strategy_metrics=strategy,
            description=(
                f"Fixed Cfull ({cfull_window}d): PCA from first {cfull_window} obs, "
                f"walk-forward on remaining {X_post.shape[0]} obs"
            ),
        )

    def run_rolling_matched(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
        cfull_window: int,
    ) -> CfullComparisonResult:
        """Run rolling walk-forward on the SAME post-Cfull data for fair comparison."""
        X_post = X_us[cfull_window:]
        Y_post = Y_jp[cfull_window:]
        dates_post = dates_us[cfull_window:]

        evaluator = WalkForwardEvaluator(
            train_window=self.train_window,
            test_window=self.test_window,
            K=self.K,
            L=self.L,
            lambda_decay=self.lambda_decay,
        )
        wf_result = evaluator.evaluate(X_post, Y_post, dates_post)
        strategy = evaluator.compute_oos_sharpe(wf_result)

        return CfullComparisonResult(
            method=f"rolling_matched_{cfull_window}d",
            cfull_window_days=None,
            wf_result=wf_result,
            strategy_metrics=strategy,
            description=(
                f"Rolling (matched period): same post-Cfull data ({X_post.shape[0]} obs) "
                f"with rolling PCA re-estimation"
            ),
        )

    def _walk_forward_fixed_eigvecs(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
        fixed_eigvecs: np.ndarray,
    ) -> WalkForwardResult:
        """Walk-forward evaluation using fixed eigenvectors (Cfull mode)."""
        evaluator = WalkForwardEvaluator(
            train_window=self.train_window,
            test_window=self.test_window,
            K=self.K,
            L=self.L,
            lambda_decay=self.lambda_decay,
        )

        T = X_us.shape[0]
        folds = []
        all_preds = []
        all_actuals = []
        all_dates = []

        fold_id = 0
        start = 0

        while start + self.train_window + self.test_window <= T:
            train_end = start + self.train_window
            test_end = min(train_end + self.test_window, T)

            X_train = X_us[start:train_end]
            Y_train = Y_jp[start:train_end]
            X_test = X_us[train_end:test_end]
            Y_test = Y_jp[train_end:test_end]

            # Fit model with FIXED eigenvectors — only regression is re-estimated
            model = PCASub(
                K=self.K, L=self.L, lambda_decay=self.lambda_decay,
                fixed_eigvecs=fixed_eigvecs,
            )
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            metrics = evaluator._compute_fold_metrics(Y_test, Y_pred)

            def _date_str(idx):
                if dates_us is not None and idx < len(dates_us):
                    return str(dates_us[idx].date())
                return str(idx)

            from src.evaluation.walk_forward import FoldResult
            fold_result = FoldResult(
                fold_id=fold_id,
                train_start=start,
                train_end=train_end - 1,
                test_start=train_end,
                test_end=test_end - 1,
                train_start_date=_date_str(start),
                train_end_date=_date_str(train_end - 1),
                test_start_date=_date_str(train_end),
                test_end_date=_date_str(test_end - 1),
                Y_true=Y_test,
                Y_pred=Y_pred,
                **metrics,
            )
            folds.append(fold_result)
            all_preds.append(Y_pred)
            all_actuals.append(Y_test)
            if dates_us is not None:
                all_dates.extend(dates_us[train_end:test_end].tolist())

            fold_id += 1
            start += self.test_window

        # Aggregate
        result = WalkForwardResult()
        result.folds = folds
        result.n_folds = len(folds)

        if len(folds) == 0:
            return result

        accs = [f.direction_accuracy for f in folds]
        corrs = [f.mean_correlation for f in folds]
        rmses = [f.rmse for f in folds]

        result.total_test_samples = sum(f.Y_true.shape[0] for f in folds)
        result.mean_direction_accuracy = float(np.mean(accs))
        result.std_direction_accuracy = float(np.std(accs))
        result.mean_correlation = float(np.mean(corrs))
        result.std_correlation = float(np.std(corrs))
        result.mean_rmse = float(np.mean(rmses))
        result.std_rmse = float(np.std(rmses))
        result.positive_accuracy_folds_pct = float(
            np.mean([1.0 if a > 0.5 else 0.0 for a in accs]) * 100
        )
        result.all_predictions = np.vstack(all_preds)
        result.all_actuals = np.vstack(all_actuals)
        result.all_dates_us = all_dates if dates_us is not None else None

        return result

    def _compute_strategy_metrics(self, result: WalkForwardResult) -> dict:
        """Compute strategy metrics from walk-forward results."""
        evaluator = WalkForwardEvaluator(
            train_window=self.train_window,
            test_window=self.test_window,
            K=self.K,
            L=self.L,
            lambda_decay=self.lambda_decay,
        )
        return evaluator.compute_oos_sharpe(result)

    def run_full_comparison(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
    ) -> list[CfullComparisonResult]:
        """Run the full Cfull validation: rolling baseline + multiple fixed windows.

        For each Cfull window, runs both fixed-Cfull and rolling on the same
        post-Cfull data for a fair comparison.
        """
        results = []

        # Full rolling baseline (uses all data)
        print("=" * 60)
        print("Running rolling baseline (full data)...")
        rolling_full = self.run_rolling_baseline(X_us, Y_jp, dates_us)
        results.append(rolling_full)
        print(f"  Folds: {rolling_full.wf_result.n_folds}")
        print(f"  Direction accuracy: {rolling_full.wf_result.mean_direction_accuracy:.4f}")
        sr = rolling_full.strategy_metrics.get("sharpe_ratio_gross", 0)
        print(f"  Sharpe (gross): {sr:.4f}")

        # For each Cfull window size
        for cw in self.cfull_windows:
            available = X_us.shape[0]
            needed = cw + self.train_window + self.test_window
            if needed > available:
                print(f"\nSkipping Cfull={cw}d: need {needed} obs, have {available}")
                continue

            print(f"\n{'=' * 60}")
            print(f"Testing Cfull={cw}d ({cw/252:.1f}yr fixed period)...")

            # Fixed Cfull
            cfull_result = self.run_cfull_fixed(X_us, Y_jp, dates_us, cw)
            results.append(cfull_result)
            print(f"  [Fixed]   Folds: {cfull_result.wf_result.n_folds}, "
                  f"Acc: {cfull_result.wf_result.mean_direction_accuracy:.4f}, "
                  f"Sharpe: {cfull_result.strategy_metrics.get('sharpe_ratio_gross', 0):.4f}")

            # Matched rolling (same post-Cfull period)
            rolling_matched = self.run_rolling_matched(X_us, Y_jp, dates_us, cw)
            results.append(rolling_matched)
            print(f"  [Rolling] Folds: {rolling_matched.wf_result.n_folds}, "
                  f"Acc: {rolling_matched.wf_result.mean_direction_accuracy:.4f}, "
                  f"Sharpe: {rolling_matched.strategy_metrics.get('sharpe_ratio_gross', 0):.4f}")

        return results
