"""PCA_SUB (Subspace Regularization PCA) model.

Captures lead-lag spillover effects from U.S. sector returns to Japanese
sector returns using PCA-based dimensionality reduction and regression.
"""

import numpy as np
from numpy.linalg import eigh


class PCASub:
    """Subspace regularization PCA for cross-market return prediction.

    Parameters
    ----------
    K : int
        Number of principal components to retain.
    L : int
        Lookback window length for fitting.
    lambda_decay : float
        Exponential decay rate for weighting recent observations.
    """

    def __init__(self, K: int = 3, L: int = 60, lambda_decay: float = 0.9):
        self.K = K
        self.L = L
        self.lambda_decay = lambda_decay

        # Fitted attributes
        self.eigvecs_ = None  # (n_us, K) principal component loadings
        self.beta_ = None     # (K, n_jp) regression coefficients
        self.intercept_ = None  # (n_jp,) intercept term

    def _compute_decay_weights(self, T: int) -> np.ndarray:
        """Compute exponential decay weights for T observations.

        Most recent observation gets highest weight.
        """
        weights = np.array([self.lambda_decay ** (T - 1 - t) for t in range(T)])
        weights /= weights.sum()
        return weights

    def _weighted_covariance(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute weighted covariance matrix of X.

        Parameters
        ----------
        X : ndarray of shape (T, n_features)
        weights : ndarray of shape (T,)

        Returns
        -------
        cov : ndarray of shape (n_features, n_features)
        """
        X_centered = X - np.average(X, axis=0, weights=weights)
        return (X_centered * weights[:, None]).T @ X_centered

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "PCASub":
        """Fit the PCA_SUB model.

        Steps:
        1. Use the last L observations (or all if fewer).
        2. Compute decay-weighted covariance of U.S. returns.
        3. Extract top-K eigenvectors (principal components).
        4. Project U.S. returns onto the K-dimensional subspace.
        5. Regress projected scores against Japanese returns (OLS).

        Parameters
        ----------
        X_us : ndarray of shape (T, n_us)
            U.S. sector returns.
        Y_jp : ndarray of shape (T, n_jp)
            Japanese sector returns (same T as X_us).

        Returns
        -------
        self
        """
        T = X_us.shape[0]
        L_eff = min(self.L, T)

        X = X_us[-L_eff:]
        Y = Y_jp[-L_eff:]

        weights = self._compute_decay_weights(L_eff)

        # Step 1: Weighted covariance of U.S. returns
        cov = self._weighted_covariance(X, weights)

        # Step 2: Eigen-decomposition, take top-K components
        eigenvalues, eigenvectors = eigh(cov)
        # eigh returns ascending order; take last K columns
        idx = np.argsort(eigenvalues)[::-1][:self.K]
        self.eigvecs_ = eigenvectors[:, idx]  # (n_us, K)

        # Step 3: Project U.S. returns onto principal component subspace
        scores = X @ self.eigvecs_  # (L_eff, K)

        # Step 4: Weighted OLS regression — scores -> Japanese returns
        W_sqrt = np.sqrt(weights)
        scores_w = scores * W_sqrt[:, None]
        Y_w = Y * W_sqrt[:, None]

        # Add intercept via augmented design matrix
        ones_w = W_sqrt[:, None]
        design_w = np.hstack([scores_w, ones_w])  # (L_eff, K+1)

        # Solve normal equations: (D^T D) beta = D^T Y
        DtD = design_w.T @ design_w
        DtY = design_w.T @ Y_w
        params = np.linalg.solve(DtD, DtY)  # (K+1, n_jp)

        self.beta_ = params[:self.K]      # (K, n_jp)
        self.intercept_ = params[self.K]   # (n_jp,)

        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        """Predict Japanese sector returns from new U.S. returns.

        Parameters
        ----------
        X_us_new : ndarray of shape (T_new, n_us)
            New U.S. sector returns.

        Returns
        -------
        Y_pred : ndarray of shape (T_new, n_jp)
            Predicted Japanese sector returns.
        """
        if self.eigvecs_ is None or self.beta_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        scores = X_us_new @ self.eigvecs_  # (T_new, K)
        return scores @ self.beta_ + self.intercept_  # (T_new, n_jp)
