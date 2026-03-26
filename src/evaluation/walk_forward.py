"""Walk-forward evaluation framework for PCA_SUB model.

Implements rolling-window walk-forward analysis where the model is
repeatedly trained on a fixed-length window and tested on the subsequent
period. This avoids lookahead bias and provides realistic out-of-sample
performance estimates.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.pca_sub import PCASub


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    Y_true: np.ndarray       # (test_len, n_jp)
    Y_pred: np.ndarray       # (test_len, n_jp)
    direction_accuracy: float
    per_sector_accuracy: np.ndarray  # (n_jp,)
    mean_correlation: float
    per_sector_correlation: np.ndarray  # (n_jp,)
    rmse: float
    per_sector_rmse: np.ndarray  # (n_jp,)


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward evaluation."""
    folds: list[FoldResult] = field(default_factory=list)
    n_folds: int = 0
    total_test_samples: int = 0
    mean_direction_accuracy: float = 0.0
    std_direction_accuracy: float = 0.0
    mean_correlation: float = 0.0
    std_correlation: float = 0.0
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    positive_accuracy_folds_pct: float = 0.0
    all_predictions: np.ndarray | None = None
    all_actuals: np.ndarray | None = None
    all_dates_us: list | None = None


class WalkForwardEvaluator:
    """Walk-forward evaluation for the PCA_SUB model.

    Parameters
    ----------
    train_window : int
        Number of observations in each training window.
    test_window : int
        Number of observations in each test window (step size).
    K : int
        Number of principal components for PCASub.
    L : int
        Lookback window for PCASub (uses min of L and train_window).
    lambda_decay : float
        Exponential decay rate for PCASub.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 21,
        K: int = 3,
        L: int = 60,
        lambda_decay: float = 0.9,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.K = K
        self.L = L
        self.lambda_decay = lambda_decay

    def _compute_fold_metrics(
        self, Y_true: np.ndarray, Y_pred: np.ndarray
    ) -> dict:
        """Compute evaluation metrics for a single fold."""
        # Direction accuracy: fraction of correct sign predictions
        signs_true = np.sign(Y_true)
        signs_pred = np.sign(Y_pred)
        correct = (signs_true == signs_pred)
        direction_accuracy = correct.mean()
        per_sector_accuracy = correct.mean(axis=0)

        # Per-sector correlation
        n_jp = Y_true.shape[1]
        per_sector_corr = np.zeros(n_jp)
        for j in range(n_jp):
            if np.std(Y_true[:, j]) > 1e-12 and np.std(Y_pred[:, j]) > 1e-12:
                per_sector_corr[j] = np.corrcoef(Y_true[:, j], Y_pred[:, j])[0, 1]
        mean_corr = np.nanmean(per_sector_corr)

        # RMSE
        per_sector_rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))
        rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2))

        return {
            "direction_accuracy": float(direction_accuracy),
            "per_sector_accuracy": per_sector_accuracy,
            "mean_correlation": float(mean_corr),
            "per_sector_correlation": per_sector_corr,
            "rmse": float(rmse),
            "per_sector_rmse": per_sector_rmse,
        }

    def evaluate(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex | None = None,
        model_factory=None,
    ) -> WalkForwardResult:
        """Run walk-forward evaluation.

        Parameters
        ----------
        X_us : ndarray of shape (T, n_us)
            Full array of U.S. sector returns.
        Y_jp : ndarray of shape (T, n_jp)
            Full array of Japanese sector returns (aligned).
        dates_us : DatetimeIndex, optional
            Dates corresponding to rows of X_us for reporting.
        model_factory : callable, optional
            A callable that returns a fresh model instance with fit/predict
            interface. If None, uses PCASub with the evaluator's parameters.

        Returns
        -------
        WalkForwardResult
        """
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

            # Fit model on training window
            if model_factory is not None:
                model = model_factory()
            else:
                model = PCASub(K=self.K, L=self.L, lambda_decay=self.lambda_decay)
            model.fit(X_train, Y_train)

            # Predict on test window
            Y_pred = model.predict(X_test)

            # Compute metrics
            metrics = self._compute_fold_metrics(Y_test, Y_pred)

            # Date info
            def _date_str(idx):
                if dates_us is not None and idx < len(dates_us):
                    return str(dates_us[idx].date())
                return str(idx)

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
            start += self.test_window  # slide forward by test_window

        # Aggregate results
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

    def compute_oos_sharpe(self, result: WalkForwardResult) -> dict:
        """Compute out-of-sample Sharpe-like metrics from walk-forward results.

        Uses a simple equal-weight long-short strategy based on predicted signs:
        go long sectors with positive predicted return, short sectors with
        negative predicted return.

        Returns dict with gross strategy metrics (no transaction costs yet).
        """
        Y_pred = result.all_predictions
        Y_actual = result.all_actuals

        if Y_pred is None or Y_actual is None:
            return {}

        # Equal-weight long-short: +1 if pred > 0, -1 if pred < 0
        positions = np.sign(Y_pred)
        # Normalize to equal weight across sectors
        n_active = np.abs(positions).sum(axis=1, keepdims=True)
        n_active = np.where(n_active == 0, 1, n_active)
        weights = positions / n_active

        # Daily strategy return = sum of (weight * actual return) across sectors
        daily_returns = (weights * Y_actual).sum(axis=1)

        # Annualize (approx 252 trading days)
        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        sharpe_ratio = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 1e-12 else 0.0

        cumulative = np.cumprod(1 + daily_returns)
        max_drawdown = float(np.min(cumulative / np.maximum.accumulate(cumulative) - 1))

        total_return = float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0
        annualized_return = float(mean_daily * 252)
        annualized_vol = float(std_daily * np.sqrt(252))

        return {
            "sharpe_ratio_gross": round(sharpe_ratio, 4),
            "annualized_return_gross": round(annualized_return, 6),
            "annualized_volatility": round(annualized_vol, 6),
            "max_drawdown": round(max_drawdown, 6),
            "total_return": round(total_return, 6),
            "n_days": len(daily_returns),
            "pct_positive_days": round(float(np.mean(daily_returns > 0) * 100), 2),
            "mean_daily_return": round(float(mean_daily), 8),
            "daily_returns": daily_returns,
        }
