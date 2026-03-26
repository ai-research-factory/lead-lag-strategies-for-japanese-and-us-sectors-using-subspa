"""Baseline models for comparison against PCA_SUB.

All baselines follow the same fit(X_us, Y_jp) / predict(X_us_new) interface
as PCASub, enabling drop-in evaluation with the walk-forward framework.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


class ZeroPredictor:
    """Naive baseline that always predicts zero return (random walk hypothesis).

    Under efficient markets, the best forecast for next-period return is zero.
    This is the most important baseline to beat.
    """

    name = "Zero Predictor (Random Walk)"

    def __init__(self):
        self.n_jp = None

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "ZeroPredictor":
        self.n_jp = Y_jp.shape[1]
        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        return np.zeros((X_us_new.shape[0], self.n_jp))


class HistoricalMeanPredictor:
    """Predicts the historical mean return for each Japanese sector.

    Uses the training window mean as the constant forecast.
    """

    name = "Historical Mean"

    def __init__(self):
        self.mean_ = None

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "HistoricalMeanPredictor":
        self.mean_ = Y_jp.mean(axis=0)  # (n_jp,)
        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        T = X_us_new.shape[0]
        return np.tile(self.mean_, (T, 1))


class DirectOLS:
    """Direct OLS regression from U.S. returns to Japanese returns (no PCA).

    Maps 11 U.S. sector returns directly to 17 Japanese sector returns.
    Tests whether the PCA dimensionality reduction in PCA_SUB adds value.
    """

    name = "Direct OLS"

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "DirectOLS":
        self.model.fit(X_us, Y_jp)
        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        return self.model.predict(X_us_new)


class DirectRidge:
    """Ridge regression from U.S. returns to Japanese returns.

    Adds L2 regularization to the direct regression, which may help
    when U.S. sectors are collinear. Tests whether PCA_SUB's subspace
    approach is better than simple regularization.
    """

    name = "Ridge Regression"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "DirectRidge":
        self.model.fit(X_us, Y_jp)
        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        return self.model.predict(X_us_new)


class SimplePCA:
    """Standard PCA + OLS without exponential decay weighting.

    Uses sklearn's PCA with equal-weighted covariance, then OLS regression.
    Tests whether the decay weighting in PCA_SUB adds value.
    """

    name = "Simple PCA (no decay)"

    def __init__(self, K: int = 3, L: int = 60):
        self.K = K
        self.L = L
        self.eigvecs_ = None
        self.beta_ = None
        self.intercept_ = None

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "SimplePCA":
        T = X_us.shape[0]
        L_eff = min(self.L, T)
        X = X_us[-L_eff:]
        Y = Y_jp[-L_eff:]

        # Equal-weighted covariance (no decay)
        X_centered = X - X.mean(axis=0)
        cov = X_centered.T @ X_centered / (L_eff - 1)

        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:self.K]
        self.eigvecs_ = eigenvectors[:, idx]

        # Project and regress (equal-weighted OLS)
        scores = X @ self.eigvecs_
        design = np.hstack([scores, np.ones((L_eff, 1))])
        params, _, _, _ = np.linalg.lstsq(design, Y, rcond=None)
        self.beta_ = params[:self.K]
        self.intercept_ = params[self.K]

        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        scores = X_us_new @ self.eigvecs_
        return scores @ self.beta_ + self.intercept_


class SectorMomentum:
    """Cross-market momentum: use previous-day JP returns as the forecast.

    Tests whether simple momentum in JP sectors is a better signal than
    the cross-market lead-lag captured by PCA_SUB.

    Since the walk-forward framework provides aligned X_us/Y_jp pairs,
    we approximate this by learning a regression from US returns
    (as a proxy for the market momentum signal) using a simple
    equal-weighted average approach.
    """

    name = "Equal-Weight Market Signal"

    def __init__(self):
        self.mean_beta_ = None

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "SectorMomentum":
        # Simple signal: average US return as a scalar predictor for each JP sector
        us_mean = X_us.mean(axis=1, keepdims=True)  # (T, 1)
        # Regress mean US return against each JP sector
        design = np.hstack([us_mean, np.ones((X_us.shape[0], 1))])
        self.mean_beta_, _, _, _ = np.linalg.lstsq(design, Y_jp, rcond=None)
        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        us_mean = X_us_new.mean(axis=1, keepdims=True)
        design = np.hstack([us_mean, np.ones((X_us_new.shape[0], 1))])
        return design @ self.mean_beta_
