"""Baseline comparison framework.

Evaluates multiple models using the same walk-forward methodology,
enabling fair comparison of PCA_SUB against baseline models.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from src.evaluation.walk_forward import FoldResult, WalkForwardResult


@dataclass
class ModelComparisonResult:
    """Results from comparing a single model via walk-forward evaluation."""
    model_name: str
    wf_result: WalkForwardResult
    strategy_metrics: dict


class BaselineComparator:
    """Runs walk-forward evaluation across multiple models for comparison.

    Parameters
    ----------
    train_window : int
        Number of observations in each training window.
    test_window : int
        Number of observations in each test window (step size).
    """

    def __init__(self, train_window: int = 252, test_window: int = 21):
        self.train_window = train_window
        self.test_window = test_window

    def _compute_fold_metrics(
        self, Y_true: np.ndarray, Y_pred: np.ndarray
    ) -> dict:
        """Compute evaluation metrics for a single fold."""
        signs_true = np.sign(Y_true)
        signs_pred = np.sign(Y_pred)
        correct = (signs_true == signs_pred)
        direction_accuracy = correct.mean()
        per_sector_accuracy = correct.mean(axis=0)

        n_jp = Y_true.shape[1]
        per_sector_corr = np.zeros(n_jp)
        for j in range(n_jp):
            if np.std(Y_true[:, j]) > 1e-12 and np.std(Y_pred[:, j]) > 1e-12:
                per_sector_corr[j] = np.corrcoef(Y_true[:, j], Y_pred[:, j])[0, 1]
        mean_corr = np.nanmean(per_sector_corr)

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

    def _compute_strategy_metrics(self, result: WalkForwardResult) -> dict:
        """Compute equal-weight long-short strategy metrics."""
        Y_pred = result.all_predictions
        Y_actual = result.all_actuals

        if Y_pred is None or Y_actual is None:
            return {}

        positions = np.sign(Y_pred)
        n_active = np.abs(positions).sum(axis=1, keepdims=True)
        n_active = np.where(n_active == 0, 1, n_active)
        weights = positions / n_active

        daily_returns = (weights * Y_actual).sum(axis=1)

        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        sharpe_ratio = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 1e-12 else 0.0

        cumulative = np.cumprod(1 + daily_returns)
        max_drawdown = float(np.min(cumulative / np.maximum.accumulate(cumulative) - 1))

        total_return = float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0
        annualized_return = float(mean_daily * 252)
        annualized_vol = float(std_daily * np.sqrt(252))

        # Transaction cost estimate: turnover-based
        n_days = len(daily_returns)
        if n_days > 1:
            position_changes = np.diff(positions, axis=0)
            daily_turnover = np.abs(position_changes).sum(axis=1).mean() / positions.shape[1]
        else:
            daily_turnover = 0.0

        # Net metrics with 10bps one-way cost
        cost_per_trade = 0.0010  # 10 bps one-way
        daily_cost = daily_turnover * cost_per_trade
        net_daily_returns = daily_returns.copy()
        if n_days > 1:
            # Apply costs from day 1 onwards (day 0 is initial position)
            turnover_daily = np.zeros(n_days)
            if n_days > 1:
                tc = np.abs(np.diff(positions, axis=0)).sum(axis=1) / positions.shape[1]
                turnover_daily[1:] = tc
            net_daily_returns = daily_returns - turnover_daily * cost_per_trade

        net_mean = net_daily_returns.mean()
        net_std = net_daily_returns.std()
        net_sharpe = (net_mean / net_std * np.sqrt(252)) if net_std > 1e-12 else 0.0
        net_cumulative = np.cumprod(1 + net_daily_returns)
        net_max_dd = float(np.min(net_cumulative / np.maximum.accumulate(net_cumulative) - 1))

        return {
            "sharpe_ratio_gross": round(sharpe_ratio, 4),
            "sharpe_ratio_net": round(net_sharpe, 4),
            "annualized_return_gross": round(annualized_return, 6),
            "annualized_return_net": round(float(net_mean * 252), 6),
            "annualized_volatility": round(annualized_vol, 6),
            "max_drawdown_gross": round(max_drawdown, 6),
            "max_drawdown_net": round(net_max_dd, 6),
            "total_return_gross": round(total_return, 6),
            "total_return_net": round(float(net_cumulative[-1] - 1) if len(net_cumulative) > 0 else 0.0, 6),
            "n_days": n_days,
            "pct_positive_days": round(float(np.mean(daily_returns > 0) * 100), 2),
            "mean_daily_turnover": round(float(daily_turnover), 6),
            "mean_daily_return": round(float(mean_daily), 8),
        }

    def evaluate_model(
        self,
        model,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex | None = None,
    ) -> ModelComparisonResult:
        """Run walk-forward evaluation for a single model.

        The model must implement fit(X_us, Y_jp) and predict(X_us_new).
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

            # Fresh model instance for each fold
            model_instance = model.__class__.__new__(model.__class__)
            model_instance.__dict__.update(model.__dict__)
            model_instance.fit(X_train, Y_train)
            Y_pred = model_instance.predict(X_test)

            metrics = self._compute_fold_metrics(Y_test, Y_pred)

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
            start += self.test_window

        # Aggregate
        result = WalkForwardResult()
        result.folds = folds
        result.n_folds = len(folds)

        if len(folds) == 0:
            return ModelComparisonResult(
                model_name=getattr(model, "name", model.__class__.__name__),
                wf_result=result,
                strategy_metrics={},
            )

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

        strategy = self._compute_strategy_metrics(result)

        return ModelComparisonResult(
            model_name=getattr(model, "name", model.__class__.__name__),
            wf_result=result,
            strategy_metrics=strategy,
        )

    @staticmethod
    def paired_sharpe_test(
        daily_returns_a: np.ndarray,
        daily_returns_b: np.ndarray,
    ) -> dict:
        """Test if strategy A has significantly higher Sharpe than strategy B.

        Uses the Ledoit-Wolf (2008) approach approximated by a paired t-test
        on the return differences.
        """
        diff = daily_returns_a - daily_returns_b
        n = len(diff)
        if n < 2:
            return {"t_stat": 0.0, "p_value": 1.0, "significant_5pct": False}

        mean_diff = diff.mean()
        se_diff = diff.std(ddof=1) / np.sqrt(n)

        if se_diff < 1e-15:
            return {"t_stat": 0.0, "p_value": 1.0, "significant_5pct": False}

        t_stat = mean_diff / se_diff

        # Two-sided p-value using normal approximation (n is large)
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

        return {
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant_5pct": bool(p_value < 0.05),
        }
