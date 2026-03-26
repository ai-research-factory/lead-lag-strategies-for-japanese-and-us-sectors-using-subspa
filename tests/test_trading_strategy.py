"""Tests for the cost-aware trading strategy module."""

import numpy as np
import pytest

from src.evaluation.trading_strategy import TradingStrategy, find_optimal_strategy


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_data(rng):
    """Generate sample predictions and actuals for testing."""
    T, n_jp = 100, 5
    predictions = rng.randn(T, n_jp) * 0.01
    actuals = rng.randn(T, n_jp) * 0.01
    return predictions, actuals


class TestTradingStrategy:
    def test_baseline_runs(self, sample_data):
        predictions, actuals = sample_data
        strategy = TradingStrategy(cost_bps=10.0)
        result = strategy.run(predictions, actuals)
        assert "sharpe_ratio_gross" in result
        assert "sharpe_ratio_net" in result
        assert "avg_daily_turnover" in result
        assert result["n_days"] == 100

    def test_ema_reduces_turnover(self, sample_data):
        predictions, actuals = sample_data
        baseline = TradingStrategy(cost_bps=10.0).run(predictions, actuals)
        smoothed = TradingStrategy(ema_halflife=10, cost_bps=10.0).run(predictions, actuals)
        assert smoothed["avg_daily_turnover"] < baseline["avg_daily_turnover"]

    def test_ema_smoothing_shape(self, sample_data):
        predictions, actuals = sample_data
        strategy = TradingStrategy(ema_halflife=5)
        smoothed = strategy._ema_smooth(predictions)
        assert smoothed.shape == predictions.shape

    def test_ema_none_returns_copy(self, sample_data):
        predictions, actuals = sample_data
        strategy = TradingStrategy(ema_halflife=None)
        smoothed = strategy._ema_smooth(predictions)
        np.testing.assert_array_equal(smoothed, predictions)
        # Verify it's a copy not a reference
        smoothed[0, 0] = 999
        assert predictions[0, 0] != 999

    def test_threshold_zeros_small_signals(self, rng):
        signals = np.array([[0.001, -0.0001, 0.0005]])
        strategy = TradingStrategy(signal_threshold=0.0003)
        filtered = strategy._apply_threshold(signals)
        assert filtered[0, 0] == 0.001  # above threshold
        assert filtered[0, 1] == 0.0  # below threshold
        assert filtered[0, 2] == 0.0005  # above threshold

    def test_threshold_zero_passes_all(self, sample_data):
        predictions, actuals = sample_data
        strategy = TradingStrategy(signal_threshold=0.0)
        filtered = strategy._apply_threshold(predictions)
        np.testing.assert_array_equal(filtered, predictions)

    def test_sector_mask(self, sample_data):
        predictions, actuals = sample_data
        mask = np.array([True, False, True, False, True])
        strategy = TradingStrategy(sector_mask=mask, cost_bps=10.0)
        result = strategy.run(predictions, actuals)
        # Check that masked sectors have zero weight
        weights = result["weights"]
        assert np.all(weights[:, 1] == 0)
        assert np.all(weights[:, 3] == 0)

    def test_sector_mask_none_trades_all(self, sample_data):
        predictions, actuals = sample_data
        strategy = TradingStrategy(sector_mask=None, cost_bps=10.0)
        result = strategy.run(predictions, actuals)
        # At least some non-zero weights in all columns
        weights = result["weights"]
        for j in range(5):
            assert np.any(weights[:, j] != 0)

    def test_position_weights_normalized(self, sample_data):
        predictions, actuals = sample_data
        strategy = TradingStrategy(cost_bps=10.0)
        result = strategy.run(predictions, actuals)
        weights = result["weights"]
        # Sum of absolute weights should be ~1 for each day with active positions
        active_days = np.abs(weights).sum(axis=1) > 1e-12
        abs_sums = np.abs(weights[active_days]).sum(axis=1)
        np.testing.assert_allclose(abs_sums, 1.0, atol=1e-10)

    def test_max_position_change_limits_turnover(self, sample_data):
        predictions, actuals = sample_data
        unlimited = TradingStrategy(cost_bps=10.0).run(predictions, actuals)
        limited = TradingStrategy(max_position_change=0.05, cost_bps=10.0).run(predictions, actuals)
        assert limited["avg_daily_turnover"] <= unlimited["avg_daily_turnover"]

    def test_cost_applied_correctly(self, sample_data):
        predictions, actuals = sample_data
        no_cost = TradingStrategy(cost_bps=0.0).run(predictions, actuals)
        with_cost = TradingStrategy(cost_bps=50.0).run(predictions, actuals)
        # Net return should be lower with higher costs
        assert with_cost["total_return_net"] <= no_cost["total_return_net"]

    def test_zero_cost_gross_equals_net(self, sample_data):
        predictions, actuals = sample_data
        result = TradingStrategy(cost_bps=0.0).run(predictions, actuals)
        np.testing.assert_allclose(
            result["daily_returns_gross"],
            result["daily_returns_net"],
            atol=1e-15,
        )

    def test_daily_returns_shape(self, sample_data):
        predictions, actuals = sample_data
        result = TradingStrategy(cost_bps=10.0).run(predictions, actuals)
        assert result["daily_returns_gross"].shape == (100,)
        assert result["daily_returns_net"].shape == (100,)

    def test_all_metrics_finite(self, sample_data):
        predictions, actuals = sample_data
        result = TradingStrategy(ema_halflife=5, signal_threshold=0.001, cost_bps=10.0).run(
            predictions, actuals
        )
        for key in ["sharpe_ratio_gross", "sharpe_ratio_net", "avg_daily_turnover",
                     "total_return_gross", "total_return_net", "max_drawdown_gross", "max_drawdown_net"]:
            assert np.isfinite(result[key]), f"{key} is not finite"

    def test_combined_strategies(self, sample_data):
        predictions, actuals = sample_data
        mask = np.array([True, True, True, False, False])
        strategy = TradingStrategy(
            ema_halflife=10,
            signal_threshold=0.001,
            sector_mask=mask,
            max_position_change=0.1,
            cost_bps=10.0,
        )
        result = strategy.run(predictions, actuals)
        assert result["n_days"] == 100
        assert np.isfinite(result["sharpe_ratio_net"])


class TestFindOptimalStrategy:
    def test_returns_expected_keys(self, sample_data):
        predictions, actuals = sample_data
        result = find_optimal_strategy(predictions, actuals, cost_bps=10.0)
        assert "best_params" in result
        assert "best_result" in result
        assert "top_10_configs" in result
        assert "total_configs_tested" in result
        assert "baseline_result" in result

    def test_best_has_highest_net_sharpe(self, sample_data):
        predictions, actuals = sample_data
        result = find_optimal_strategy(predictions, actuals, cost_bps=10.0)
        best_sharpe = result["best_result"]["sharpe_ratio_net"]
        for cfg in result["top_10_configs"]:
            assert cfg["sharpe_net"] <= best_sharpe + 1e-10

    def test_with_sector_accuracy(self, sample_data):
        predictions, actuals = sample_data
        per_sector_acc = np.array([0.52, 0.48, 0.55, 0.50, 0.51])
        result = find_optimal_strategy(
            predictions, actuals, per_sector_acc, cost_bps=10.0
        )
        assert result["total_configs_tested"] >= 100  # at least as many as without masks

    def test_baseline_included_in_results(self, sample_data):
        predictions, actuals = sample_data
        result = find_optimal_strategy(predictions, actuals, cost_bps=10.0)
        baseline = result["baseline_result"]
        assert baseline["ema_halflife"] is None
        assert baseline["signal_threshold"] == 0.0
