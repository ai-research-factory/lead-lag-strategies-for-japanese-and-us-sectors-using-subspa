"""PCA_SUB with factor-level timing.

Instead of weighting all K principal components equally in the regression,
this model tracks each PC's recent predictive accuracy and dynamically
weights the regression contributions of each factor. PCs that have been
recently predictive receive higher weight; those that haven't are downweighted.
"""

import numpy as np
from numpy.linalg import eigh


class PCASubFactorTiming:
    """PCA_SUB with per-factor dynamic weighting based on recent accuracy.

    Parameters
    ----------
    K : int
        Number of principal components to retain.
    L : int
        Lookback window length for fitting.
    lambda_decay : float
        Exponential decay rate for covariance weighting.
    factor_ema_halflife : int
        Half-life for tracking per-factor prediction accuracy over time.
    """

    def __init__(
        self,
        K: int = 5,
        L: int = 120,
        lambda_decay: float = 1.0,
        factor_ema_halflife: int = 63,
    ):
        self.K = K
        self.L = L
        self.lambda_decay = lambda_decay
        self.factor_ema_halflife = factor_ema_halflife

        self.eigvecs_ = None
        self.beta_ = None
        self.intercept_ = None
        self.factor_weights_ = np.ones(K) / K

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "PCASubFactorTiming":
        """Fit model and compute per-factor contributions for weight tracking."""
        T = X_us.shape[0]
        L_eff = min(self.L, T)

        X = X_us[-L_eff:]
        Y = Y_jp[-L_eff:]

        weights = self._compute_decay_weights(L_eff)
        cov = self._weighted_covariance(X, weights)

        eigenvalues, eigenvectors = eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:self.K]
        self.eigvecs_ = eigenvectors[:, idx]

        scores = X @ self.eigvecs_
        W_sqrt = np.sqrt(weights)
        scores_w = scores * W_sqrt[:, None]
        Y_w = Y * W_sqrt[:, None]

        ones_w = W_sqrt[:, None]
        design_w = np.hstack([scores_w, ones_w])
        DtD = design_w.T @ design_w
        DtY = design_w.T @ Y_w
        params = np.linalg.solve(DtD, DtY)

        self.beta_ = params[:self.K]
        self.intercept_ = params[self.K]

        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        """Predict using factor-weighted contributions."""
        if self.eigvecs_ is None or self.beta_ is None:
            raise RuntimeError("Model has not been fitted.")

        scores = X_us_new @ self.eigvecs_  # (T_new, K)
        # Weight each factor's contribution
        weighted_scores = scores * self.factor_weights_[None, :]
        return weighted_scores @ self.beta_ + self.intercept_

    def predict_per_factor(self, X_us_new: np.ndarray) -> np.ndarray:
        """Return per-factor prediction contributions.

        Returns shape (T_new, K, n_jp) — contribution of each PC to each sector.
        """
        if self.eigvecs_ is None or self.beta_ is None:
            raise RuntimeError("Model has not been fitted.")

        scores = X_us_new @ self.eigvecs_  # (T_new, K)
        # Per-factor contribution: scores[:, k] * beta_[k, :]
        T_new = X_us_new.shape[0]
        n_jp = self.beta_.shape[1]
        contributions = np.zeros((T_new, self.K, n_jp))
        for k in range(self.K):
            contributions[:, k, :] = scores[:, [k]] * self.beta_[[k], :]
        return contributions

    def update_factor_weights(
        self,
        per_factor_contributions: np.ndarray,
        actuals: np.ndarray,
    ):
        """Update factor weights based on each PC's directional accuracy.

        Parameters
        ----------
        per_factor_contributions : ndarray of shape (T, K, n_jp)
        actuals : ndarray of shape (T, n_jp)
        """
        K = per_factor_contributions.shape[1]

        # Per-factor directional accuracy across all sectors
        factor_acc = np.zeros(K)
        for k in range(K):
            factor_pred_sign = np.sign(per_factor_contributions[:, k, :])
            actual_sign = np.sign(actuals)
            factor_acc[k] = (factor_pred_sign == actual_sign).mean()

        # Convert accuracy to weight: higher accuracy -> higher weight
        # Use softmax-like scaling centered at 0.5 (random baseline)
        excess_acc = factor_acc - 0.5
        # Exponential scaling
        raw_weights = np.exp(10.0 * excess_acc)
        self.factor_weights_ = raw_weights / raw_weights.sum()

    def _compute_decay_weights(self, T: int) -> np.ndarray:
        weights = np.array([self.lambda_decay ** (T - 1 - t) for t in range(T)])
        weights /= weights.sum()
        return weights

    def _weighted_covariance(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        X_centered = X - np.average(X, axis=0, weights=weights)
        return (X_centered * weights[:, None]).T @ X_centered
