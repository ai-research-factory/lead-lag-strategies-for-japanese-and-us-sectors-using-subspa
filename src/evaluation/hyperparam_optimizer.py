"""Nested walk-forward hyperparameter optimization for PCA_SUB model.

Implements a two-level walk-forward scheme:
- Outer loop: standard walk-forward splits for unbiased OOS evaluation.
- Inner loop: within each outer training window, runs a nested walk-forward
  to select the best hyperparameter combination (K, L, lambda_decay).

This avoids any information leakage from test data into parameter selection.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from src.models.pca_sub import PCASub


@dataclass
class ParamResult:
    """Result for a single hyperparameter combination."""
    K: int
    L: int
    lambda_decay: float
    inner_direction_accuracy: float
    inner_sharpe: float
    inner_n_folds: int


@dataclass
class OuterFoldResult:
    """Result for a single outer fold with optimized parameters."""
    fold_id: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    best_K: int
    best_L: int
    best_lambda_decay: float
    best_inner_sharpe: float
    test_direction_accuracy: float
    test_mean_correlation: float
    test_rmse: float
    Y_true: np.ndarray
    Y_pred: np.ndarray
    all_param_results: list[ParamResult] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Full optimization results."""
    outer_folds: list[OuterFoldResult] = field(default_factory=list)
    param_grid: dict = field(default_factory=dict)
    n_outer_folds: int = 0
    total_test_samples: int = 0
    mean_direction_accuracy: float = 0.0
    std_direction_accuracy: float = 0.0
    mean_correlation: float = 0.0
    mean_rmse: float = 0.0
    strategy_metrics: dict = field(default_factory=dict)
    param_selection_summary: dict = field(default_factory=dict)


class HyperparamOptimizer:
    """Nested walk-forward hyperparameter optimizer.

    Parameters
    ----------
    param_grid : dict
        Grid of parameters to search. Keys: 'K', 'L', 'lambda_decay'.
        Values: lists of values to try.
    outer_train_window : int
        Training window for outer walk-forward loop.
    outer_test_window : int
        Test window (step size) for outer loop.
    inner_train_window : int
        Training window for inner (nested) walk-forward.
    inner_test_window : int
        Test window for inner walk-forward.
    selection_metric : str
        Metric to maximize in inner loop. One of 'sharpe', 'direction_accuracy'.
    """

    def __init__(
        self,
        param_grid: dict | None = None,
        outer_train_window: int = 504,
        outer_test_window: int = 21,
        inner_train_window: int = 252,
        inner_test_window: int = 21,
        selection_metric: str = "sharpe",
    ):
        self.param_grid = param_grid or {
            "K": [1, 2, 3, 4, 5],
            "L": [20, 40, 60, 80, 120],
            "lambda_decay": [0.8, 0.85, 0.9, 0.95, 1.0],
        }
        self.outer_train_window = outer_train_window
        self.outer_test_window = outer_test_window
        self.inner_train_window = inner_train_window
        self.inner_test_window = inner_test_window
        self.selection_metric = selection_metric

    def _run_inner_walkforward(
        self, X_us: np.ndarray, Y_jp: np.ndarray, K: int, L: int, lambda_decay: float
    ) -> dict:
        """Run walk-forward on inner training data for one param combination."""
        T = X_us.shape[0]
        all_preds = []
        all_actuals = []
        n_folds = 0

        start = 0
        while start + self.inner_train_window + self.inner_test_window <= T:
            tr_end = start + self.inner_train_window
            te_end = min(tr_end + self.inner_test_window, T)

            X_tr = X_us[start:tr_end]
            Y_tr = Y_jp[start:tr_end]
            X_te = X_us[tr_end:te_end]
            Y_te = Y_jp[tr_end:te_end]

            model = PCASub(K=K, L=L, lambda_decay=lambda_decay)
            model.fit(X_tr, Y_tr)
            Y_pred = model.predict(X_te)

            all_preds.append(Y_pred)
            all_actuals.append(Y_te)
            n_folds += 1
            start += self.inner_test_window

        if n_folds == 0:
            return {"direction_accuracy": 0.0, "sharpe": -999.0, "n_folds": 0}

        Y_pred_all = np.vstack(all_preds)
        Y_actual_all = np.vstack(all_actuals)

        # Direction accuracy
        correct = np.sign(Y_pred_all) == np.sign(Y_actual_all)
        dir_acc = float(correct.mean())

        # Simple long-short Sharpe
        positions = np.sign(Y_pred_all)
        n_active = np.abs(positions).sum(axis=1, keepdims=True)
        n_active = np.where(n_active == 0, 1, n_active)
        weights = positions / n_active
        daily_returns = (weights * Y_actual_all).sum(axis=1)

        mean_d = daily_returns.mean()
        std_d = daily_returns.std()
        sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 1e-12 else 0.0

        return {
            "direction_accuracy": dir_acc,
            "sharpe": float(sharpe),
            "n_folds": n_folds,
        }

    def _compute_test_metrics(
        self, Y_true: np.ndarray, Y_pred: np.ndarray
    ) -> dict:
        """Compute metrics on outer test fold."""
        correct = np.sign(Y_true) == np.sign(Y_pred)
        dir_acc = float(correct.mean())

        n_jp = Y_true.shape[1]
        per_sector_corr = np.zeros(n_jp)
        for j in range(n_jp):
            if np.std(Y_true[:, j]) > 1e-12 and np.std(Y_pred[:, j]) > 1e-12:
                per_sector_corr[j] = np.corrcoef(Y_true[:, j], Y_pred[:, j])[0, 1]
        mean_corr = float(np.nanmean(per_sector_corr))

        rmse = float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))

        return {
            "direction_accuracy": dir_acc,
            "mean_correlation": mean_corr,
            "rmse": rmse,
        }

    def optimize(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex | None = None,
    ) -> OptimizationResult:
        """Run nested walk-forward hyperparameter optimization.

        Parameters
        ----------
        X_us : ndarray of shape (T, 11)
        Y_jp : ndarray of shape (T, 17)
        dates_us : DatetimeIndex, optional

        Returns
        -------
        OptimizationResult
        """
        T = X_us.shape[0]
        param_combos = list(product(
            self.param_grid["K"],
            self.param_grid["L"],
            self.param_grid["lambda_decay"],
        ))
        n_combos = len(param_combos)

        outer_folds = []
        fold_id = 0
        start = 0

        print(f"Nested walk-forward optimization: {n_combos} param combos")
        print(f"Outer: train={self.outer_train_window}, test={self.outer_test_window}")
        print(f"Inner: train={self.inner_train_window}, test={self.inner_test_window}")
        print(f"Selection metric: {self.selection_metric}")
        print(f"Total observations: {T}")
        print()

        while start + self.outer_train_window + self.outer_test_window <= T:
            outer_tr_end = start + self.outer_train_window
            outer_te_end = min(outer_tr_end + self.outer_test_window, T)

            X_outer_train = X_us[start:outer_tr_end]
            Y_outer_train = Y_jp[start:outer_tr_end]
            X_outer_test = X_us[outer_tr_end:outer_te_end]
            Y_outer_test = Y_jp[outer_tr_end:outer_te_end]

            # --- Inner loop: evaluate all param combos on outer training data ---
            param_results = []
            best_score = -np.inf
            best_params = None

            for K, L, lam in param_combos:
                inner_res = self._run_inner_walkforward(
                    X_outer_train, Y_outer_train, K, L, lam
                )
                score = inner_res[self.selection_metric]
                pr = ParamResult(
                    K=K, L=L, lambda_decay=lam,
                    inner_direction_accuracy=inner_res["direction_accuracy"],
                    inner_sharpe=inner_res["sharpe"],
                    inner_n_folds=inner_res["n_folds"],
                )
                param_results.append(pr)

                if score > best_score:
                    best_score = score
                    best_params = (K, L, lam)

            # --- Outer test: use best params to predict ---
            best_K, best_L, best_lam = best_params
            model = PCASub(K=best_K, L=best_L, lambda_decay=best_lam)
            model.fit(X_outer_train, Y_outer_train)
            Y_pred = model.predict(X_outer_test)

            test_metrics = self._compute_test_metrics(Y_outer_test, Y_pred)

            def _date_str(idx):
                if dates_us is not None and idx < len(dates_us):
                    return str(dates_us[idx].date())
                return str(idx)

            fold_result = OuterFoldResult(
                fold_id=fold_id,
                train_start_date=_date_str(start),
                train_end_date=_date_str(outer_tr_end - 1),
                test_start_date=_date_str(outer_tr_end),
                test_end_date=_date_str(outer_te_end - 1),
                best_K=best_K,
                best_L=best_L,
                best_lambda_decay=best_lam,
                best_inner_sharpe=best_score,
                test_direction_accuracy=test_metrics["direction_accuracy"],
                test_mean_correlation=test_metrics["mean_correlation"],
                test_rmse=test_metrics["rmse"],
                Y_true=Y_outer_test,
                Y_pred=Y_pred,
                all_param_results=param_results,
            )
            outer_folds.append(fold_result)

            print(
                f"  Fold {fold_id}: test={_date_str(outer_tr_end)}→{_date_str(outer_te_end-1)} | "
                f"best K={best_K}, L={best_L}, λ={best_lam:.2f} | "
                f"inner {self.selection_metric}={best_score:.4f} | "
                f"OOS acc={test_metrics['direction_accuracy']:.4f}"
            )

            fold_id += 1
            start += self.outer_test_window

        # --- Aggregate results ---
        result = OptimizationResult()
        result.outer_folds = outer_folds
        result.param_grid = self.param_grid
        result.n_outer_folds = len(outer_folds)

        if len(outer_folds) == 0:
            return result

        accs = [f.test_direction_accuracy for f in outer_folds]
        corrs = [f.test_mean_correlation for f in outer_folds]
        rmses = [f.test_rmse for f in outer_folds]

        result.total_test_samples = sum(f.Y_true.shape[0] for f in outer_folds)
        result.mean_direction_accuracy = float(np.mean(accs))
        result.std_direction_accuracy = float(np.std(accs))
        result.mean_correlation = float(np.mean(corrs))
        result.mean_rmse = float(np.mean(rmses))

        # Strategy metrics from concatenated OOS predictions
        Y_pred_all = np.vstack([f.Y_pred for f in outer_folds])
        Y_actual_all = np.vstack([f.Y_true for f in outer_folds])

        positions = np.sign(Y_pred_all)
        n_active = np.abs(positions).sum(axis=1, keepdims=True)
        n_active = np.where(n_active == 0, 1, n_active)
        weights = positions / n_active
        daily_returns = (weights * Y_actual_all).sum(axis=1)

        mean_d = daily_returns.mean()
        std_d = daily_returns.std()
        sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 1e-12 else 0.0
        cumulative = np.cumprod(1 + daily_returns)
        max_dd = float(np.min(cumulative / np.maximum.accumulate(cumulative) - 1))

        result.strategy_metrics = {
            "sharpe_ratio_gross": round(float(sharpe), 4),
            "annualized_return_gross": round(float(mean_d * 252), 6),
            "annualized_volatility": round(float(std_d * np.sqrt(252)), 6),
            "max_drawdown": round(max_dd, 6),
            "total_return": round(float(cumulative[-1] - 1), 6),
            "n_days": len(daily_returns),
            "pct_positive_days": round(float(np.mean(daily_returns > 0) * 100), 2),
        }

        # Param selection summary: how often each value was selected
        k_counts = {}
        l_counts = {}
        lam_counts = {}
        for f in outer_folds:
            k_counts[f.best_K] = k_counts.get(f.best_K, 0) + 1
            l_counts[f.best_L] = l_counts.get(f.best_L, 0) + 1
            lam_counts[f.best_lambda_decay] = lam_counts.get(f.best_lambda_decay, 0) + 1

        result.param_selection_summary = {
            "K_selection_counts": dict(sorted(k_counts.items())),
            "L_selection_counts": dict(sorted(l_counts.items())),
            "lambda_decay_selection_counts": dict(sorted(lam_counts.items())),
            "most_frequent_K": max(k_counts, key=k_counts.get),
            "most_frequent_L": max(l_counts, key=l_counts.get),
            "most_frequent_lambda_decay": max(lam_counts, key=lam_counts.get),
        }

        return result
