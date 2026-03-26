"""Tests for baseline comparison models."""

import numpy as np
import pytest

from src.models.baselines import (
    ZeroPredictor,
    HistoricalMeanPredictor,
    DirectOLS,
    DirectRidge,
    SimplePCA,
    SectorMomentum,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def data(rng):
    T, n_us, n_jp = 100, 11, 17
    X_us = rng.standard_normal((T, n_us)) * 0.01
    Y_jp = rng.standard_normal((T, n_jp)) * 0.01
    return X_us, Y_jp


ALL_MODELS = [
    ZeroPredictor,
    HistoricalMeanPredictor,
    DirectOLS,
    DirectRidge,
    SimplePCA,
    SectorMomentum,
]


class TestBaselineInterface:
    """All baselines must follow the same fit/predict interface."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_fit_predict_interface(self, ModelClass, data):
        X_us, Y_jp = data
        model = ModelClass()
        model.fit(X_us[:80], Y_jp[:80])
        Y_pred = model.predict(X_us[80:])
        assert Y_pred.shape == (20, 17)

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_predictions_are_finite(self, ModelClass, data):
        X_us, Y_jp = data
        model = ModelClass()
        model.fit(X_us[:80], Y_jp[:80])
        Y_pred = model.predict(X_us[80:])
        assert np.all(np.isfinite(Y_pred))

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_has_name_attribute(self, ModelClass):
        model = ModelClass()
        assert hasattr(model, "name") or hasattr(model.__class__, "name")

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_fit_returns_self(self, ModelClass, data):
        X_us, Y_jp = data
        model = ModelClass()
        result = model.fit(X_us, Y_jp)
        assert result is model


class TestZeroPredictor:
    def test_always_predicts_zero(self, data):
        X_us, Y_jp = data
        model = ZeroPredictor()
        model.fit(X_us, Y_jp)
        Y_pred = model.predict(X_us[:10])
        np.testing.assert_array_equal(Y_pred, 0.0)


class TestHistoricalMeanPredictor:
    def test_predicts_training_mean(self, data):
        X_us, Y_jp = data
        model = HistoricalMeanPredictor()
        model.fit(X_us, Y_jp)
        Y_pred = model.predict(X_us[:5])
        expected = np.tile(Y_jp.mean(axis=0), (5, 1))
        np.testing.assert_allclose(Y_pred, expected, atol=1e-10)


class TestDirectRidge:
    def test_different_alpha_values(self, data):
        X_us, Y_jp = data
        for alpha in [0.01, 1.0, 100.0]:
            model = DirectRidge(alpha=alpha)
            model.fit(X_us[:80], Y_jp[:80])
            Y_pred = model.predict(X_us[80:])
            assert Y_pred.shape == (20, 17)


class TestSimplePCA:
    def test_default_K(self, data):
        X_us, Y_jp = data
        model = SimplePCA()
        assert model.K == 3
        assert model.L == 60

    def test_custom_K(self, data):
        X_us, Y_jp = data
        model = SimplePCA(K=5, L=40)
        model.fit(X_us, Y_jp)
        assert model.eigvecs_.shape == (11, 5)

    def test_eigvecs_set_after_fit(self, data):
        X_us, Y_jp = data
        model = SimplePCA(K=3)
        model.fit(X_us, Y_jp)
        assert model.eigvecs_ is not None
        assert model.beta_ is not None
        assert model.intercept_ is not None
