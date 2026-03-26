"""Tests for the PCA_SUB core model."""

import numpy as np
import pytest

from src.models.pca_sub import PCASub


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_data(rng):
    """Generate correlated synthetic US/JP returns."""
    T, n_us, n_jp = 200, 11, 17
    # US returns with some correlation structure
    cov_us = np.eye(n_us) + 0.3 * np.ones((n_us, n_us))
    X_us = rng.multivariate_normal(np.zeros(n_us), cov_us, size=T) * 0.01
    # JP returns with linear dependence on US + noise
    B = rng.standard_normal((n_us, n_jp)) * 0.5
    Y_jp = X_us @ B + rng.standard_normal((T, n_jp)) * 0.005
    return X_us, Y_jp


class TestPCASubInit:
    def test_default_parameters(self):
        model = PCASub()
        assert model.K == 3
        assert model.L == 60
        assert model.lambda_decay == 0.9
        assert model.fixed_eigvecs is None

    def test_custom_parameters(self):
        model = PCASub(K=5, L=120, lambda_decay=1.0)
        assert model.K == 5
        assert model.L == 120
        assert model.lambda_decay == 1.0

    def test_unfitted_attributes_are_none(self):
        model = PCASub()
        assert model.eigvecs_ is None
        assert model.beta_ is None
        assert model.intercept_ is None


class TestDecayWeights:
    def test_weights_sum_to_one(self):
        model = PCASub(lambda_decay=0.9)
        weights = model._compute_decay_weights(100)
        assert pytest.approx(weights.sum(), abs=1e-10) == 1.0

    def test_weights_shape(self):
        model = PCASub()
        weights = model._compute_decay_weights(50)
        assert weights.shape == (50,)

    def test_most_recent_has_highest_weight(self):
        model = PCASub(lambda_decay=0.9)
        weights = model._compute_decay_weights(10)
        assert weights[-1] > weights[0]

    def test_equal_weights_when_lambda_is_one(self):
        model = PCASub(lambda_decay=1.0)
        weights = model._compute_decay_weights(10)
        np.testing.assert_allclose(weights, np.ones(10) / 10, atol=1e-10)

    def test_single_observation(self):
        model = PCASub(lambda_decay=0.9)
        weights = model._compute_decay_weights(1)
        assert weights.shape == (1,)
        assert pytest.approx(weights[0]) == 1.0


class TestWeightedCovariance:
    def test_shape(self, rng):
        model = PCASub()
        X = rng.standard_normal((50, 11))
        weights = np.ones(50) / 50
        cov = model._weighted_covariance(X, weights)
        assert cov.shape == (11, 11)

    def test_symmetric(self, rng):
        model = PCASub()
        X = rng.standard_normal((50, 5))
        weights = model._compute_decay_weights(50)
        cov = model._weighted_covariance(X, weights)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_positive_semidefinite(self, rng):
        model = PCASub()
        X = rng.standard_normal((100, 11))
        weights = model._compute_decay_weights(100)
        cov = model._weighted_covariance(X, weights)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)


class TestFit:
    def test_fit_sets_attributes(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60, lambda_decay=0.9)
        model.fit(X_us, Y_jp)
        assert model.eigvecs_ is not None
        assert model.beta_ is not None
        assert model.intercept_ is not None

    def test_eigvecs_shape(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60)
        model.fit(X_us, Y_jp)
        assert model.eigvecs_.shape == (11, 3)

    def test_beta_shape(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60)
        model.fit(X_us, Y_jp)
        assert model.beta_.shape == (3, 17)

    def test_intercept_shape(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60)
        model.fit(X_us, Y_jp)
        assert model.intercept_.shape == (17,)

    def test_fit_returns_self(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60)
        result = model.fit(X_us, Y_jp)
        assert result is model

    def test_fit_with_data_shorter_than_L(self, rng):
        X_us = rng.standard_normal((30, 11)) * 0.01
        Y_jp = rng.standard_normal((30, 17)) * 0.01
        model = PCASub(K=3, L=60)
        model.fit(X_us, Y_jp)  # L=60 > T=30, should use all data
        assert model.eigvecs_.shape == (11, 3)

    def test_different_K_values(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        for K in [1, 2, 5, 7, 11]:
            model = PCASub(K=K, L=60)
            model.fit(X_us, Y_jp)
            assert model.eigvecs_.shape == (11, K)
            assert model.beta_.shape == (K, 17)

    def test_eigvecs_are_orthonormal(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60)
        model.fit(X_us, Y_jp)
        # Columns should be approximately orthonormal
        gram = model.eigvecs_.T @ model.eigvecs_
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)


class TestPredict:
    def test_predict_shape(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60)
        model.fit(X_us[:150], Y_jp[:150])
        Y_pred = model.predict(X_us[150:])
        assert Y_pred.shape == (50, 17)

    def test_predict_single_sample(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60)
        model.fit(X_us, Y_jp)
        Y_pred = model.predict(X_us[:1])
        assert Y_pred.shape == (1, 17)

    def test_predict_before_fit_raises(self):
        model = PCASub()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.zeros((10, 11)))

    def test_predictions_are_finite(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60)
        model.fit(X_us, Y_jp)
        Y_pred = model.predict(X_us)
        assert np.all(np.isfinite(Y_pred))

    def test_predictions_vary_with_input(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        model = PCASub(K=3, L=60)
        model.fit(X_us, Y_jp)
        pred1 = model.predict(X_us[:10])
        pred2 = model.predict(X_us[10:20])
        assert not np.allclose(pred1, pred2)


class TestCfullMode:
    def test_compute_cfull_eigvecs_shape(self, synthetic_data):
        X_us, _ = synthetic_data
        eigvecs = PCASub.compute_cfull_eigvecs(X_us[:100], K=3)
        assert eigvecs.shape == (11, 3)

    def test_cfull_eigvecs_orthonormal(self, synthetic_data):
        X_us, _ = synthetic_data
        eigvecs = PCASub.compute_cfull_eigvecs(X_us[:100], K=5)
        gram = eigvecs.T @ eigvecs
        np.testing.assert_allclose(gram, np.eye(5), atol=1e-10)

    def test_fixed_eigvecs_mode(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        fixed = PCASub.compute_cfull_eigvecs(X_us[:100], K=3)
        model = PCASub(K=3, L=60, fixed_eigvecs=fixed)
        model.fit(X_us[100:], Y_jp[100:])
        # Model should use provided eigvecs, not compute its own
        np.testing.assert_array_equal(model.eigvecs_, fixed)

    def test_cfull_predict_works(self, synthetic_data):
        X_us, Y_jp = synthetic_data
        fixed = PCASub.compute_cfull_eigvecs(X_us[:100], K=3)
        model = PCASub(K=3, L=60, fixed_eigvecs=fixed)
        model.fit(X_us[100:150], Y_jp[100:150])
        Y_pred = model.predict(X_us[150:])
        assert Y_pred.shape == (50, 17)
        assert np.all(np.isfinite(Y_pred))
