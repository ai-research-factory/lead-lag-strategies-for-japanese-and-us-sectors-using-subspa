"""Tests for the walk-forward evaluation framework."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.walk_forward import WalkForwardEvaluator, WalkForwardResult, FoldResult


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def data_with_dates(rng):
    T, n_us, n_jp = 400, 11, 17
    X_us = rng.standard_normal((T, n_us)) * 0.01
    Y_jp = rng.standard_normal((T, n_jp)) * 0.01
    dates = pd.date_range("2022-01-01", periods=T, freq="B")
    return X_us, Y_jp, dates


class TestWalkForwardEvaluator:
    def test_default_params(self):
        ev = WalkForwardEvaluator()
        assert ev.train_window == 252
        assert ev.test_window == 21
        assert ev.K == 3

    def test_evaluate_produces_folds(self, data_with_dates):
        X, Y, dates = data_with_dates
        ev = WalkForwardEvaluator(train_window=100, test_window=21, K=3, L=60)
        result = ev.evaluate(X, Y, dates)
        assert result.n_folds > 0
        assert len(result.folds) == result.n_folds

    def test_no_overlapping_test_periods(self, data_with_dates):
        X, Y, dates = data_with_dates
        ev = WalkForwardEvaluator(train_window=100, test_window=21)
        result = ev.evaluate(X, Y, dates)
        for i in range(1, len(result.folds)):
            assert result.folds[i].test_start > result.folds[i - 1].test_end

    def test_train_before_test(self, data_with_dates):
        X, Y, dates = data_with_dates
        ev = WalkForwardEvaluator(train_window=100, test_window=21)
        result = ev.evaluate(X, Y, dates)
        for fold in result.folds:
            assert fold.train_end < fold.test_start

    def test_fold_metrics_are_valid(self, data_with_dates):
        X, Y, dates = data_with_dates
        ev = WalkForwardEvaluator(train_window=100, test_window=21)
        result = ev.evaluate(X, Y, dates)
        for fold in result.folds:
            assert 0.0 <= fold.direction_accuracy <= 1.0
            assert fold.rmse >= 0.0
            assert fold.Y_true.shape == fold.Y_pred.shape

    def test_aggregated_metrics(self, data_with_dates):
        X, Y, dates = data_with_dates
        ev = WalkForwardEvaluator(train_window=100, test_window=21)
        result = ev.evaluate(X, Y, dates)
        assert 0.0 <= result.mean_direction_accuracy <= 1.0
        assert result.std_direction_accuracy >= 0.0
        assert result.total_test_samples > 0

    def test_all_predictions_shape(self, data_with_dates):
        X, Y, dates = data_with_dates
        ev = WalkForwardEvaluator(train_window=100, test_window=21)
        result = ev.evaluate(X, Y, dates)
        assert result.all_predictions.shape[1] == 17
        assert result.all_actuals.shape == result.all_predictions.shape
        assert result.all_predictions.shape[0] == result.total_test_samples

    def test_empty_result_when_insufficient_data(self, rng):
        X = rng.standard_normal((50, 11))
        Y = rng.standard_normal((50, 17))
        ev = WalkForwardEvaluator(train_window=252, test_window=21)
        result = ev.evaluate(X, Y)
        assert result.n_folds == 0

    def test_without_dates(self, rng):
        X = rng.standard_normal((400, 11)) * 0.01
        Y = rng.standard_normal((400, 17)) * 0.01
        ev = WalkForwardEvaluator(train_window=100, test_window=21)
        result = ev.evaluate(X, Y)
        assert result.n_folds > 0


class TestComputeOOSSharpe:
    def test_sharpe_dict_keys(self, data_with_dates):
        X, Y, dates = data_with_dates
        ev = WalkForwardEvaluator(train_window=100, test_window=21)
        result = ev.evaluate(X, Y, dates)
        sharpe = ev.compute_oos_sharpe(result)
        assert "sharpe_ratio_gross" in sharpe
        assert "annualized_return_gross" in sharpe
        assert "max_drawdown" in sharpe
        assert "total_return" in sharpe
        assert "n_days" in sharpe

    def test_sharpe_is_finite(self, data_with_dates):
        X, Y, dates = data_with_dates
        ev = WalkForwardEvaluator(train_window=100, test_window=21)
        result = ev.evaluate(X, Y, dates)
        sharpe = ev.compute_oos_sharpe(result)
        assert np.isfinite(sharpe["sharpe_ratio_gross"])
        assert np.isfinite(sharpe["max_drawdown"])

    def test_max_drawdown_is_nonpositive(self, data_with_dates):
        X, Y, dates = data_with_dates
        ev = WalkForwardEvaluator(train_window=100, test_window=21)
        result = ev.evaluate(X, Y, dates)
        sharpe = ev.compute_oos_sharpe(result)
        assert sharpe["max_drawdown"] <= 0.0

    def test_empty_result_returns_empty_dict(self):
        ev = WalkForwardEvaluator()
        result = WalkForwardResult()
        sharpe = ev.compute_oos_sharpe(result)
        assert sharpe == {}
