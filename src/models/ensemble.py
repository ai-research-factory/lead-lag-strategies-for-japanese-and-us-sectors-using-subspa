"""Ensemble model combining PCA_SUB with Ridge and ElasticNet.

Generates predictions from multiple model types and combines them
using configurable weighting schemes. Each sub-model follows the
same fit(X_us, Y_jp) / predict(X_us_new) interface.
"""

import numpy as np
from sklearn.linear_model import Ridge, ElasticNet

from src.models.pca_sub import PCASub


class ElasticNetMultiOutput:
    """ElasticNet regression from U.S. returns to Japanese returns.

    sklearn's ElasticNet only supports single-output, so we fit one
    per Japanese sector and stack predictions.
    """

    name = "ElasticNet"

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.models = None

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "ElasticNetMultiOutput":
        n_jp = Y_jp.shape[1]
        self.models = []
        for j in range(n_jp):
            m = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)
            m.fit(X_us, Y_jp[:, j])
            self.models.append(m)
        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        preds = np.column_stack([m.predict(X_us_new) for m in self.models])
        return preds


class EnsembleModel:
    """Ensemble that combines predictions from multiple models.

    Parameters
    ----------
    pca_sub_params : dict
        Parameters for PCA_SUB model (K, L, lambda_decay).
    ridge_alpha : float
        Regularization strength for Ridge regression.
    enet_alpha : float
        Regularization strength for ElasticNet.
    enet_l1_ratio : float
        L1/L2 mixing for ElasticNet.
    weights : dict or None
        Model weights like {"pca_sub": 0.5, "ridge": 0.3, "enet": 0.2}.
        None uses equal weights.
    combine_method : str
        "weighted_avg" or "sign_vote" (majority vote on direction).
    """

    name = "Ensemble (PCA_SUB + Ridge + ElasticNet)"

    def __init__(
        self,
        pca_sub_params: dict | None = None,
        ridge_alpha: float = 1.0,
        enet_alpha: float = 0.01,
        enet_l1_ratio: float = 0.5,
        weights: dict | None = None,
        combine_method: str = "weighted_avg",
    ):
        pca_params = pca_sub_params or {"K": 5, "L": 120, "lambda_decay": 1.0}
        self.pca_sub = PCASub(**pca_params)
        self.ridge = Ridge(alpha=ridge_alpha)
        self.enet = ElasticNetMultiOutput(alpha=enet_alpha, l1_ratio=enet_l1_ratio)

        if weights is None:
            self.weights = {"pca_sub": 1 / 3, "ridge": 1 / 3, "enet": 1 / 3}
        else:
            self.weights = weights

        self.combine_method = combine_method

    def fit(self, X_us: np.ndarray, Y_jp: np.ndarray) -> "EnsembleModel":
        self.pca_sub.fit(X_us, Y_jp)
        self.ridge.fit(X_us, Y_jp)
        self.enet.fit(X_us, Y_jp)
        return self

    def predict(self, X_us_new: np.ndarray) -> np.ndarray:
        pred_pca = self.pca_sub.predict(X_us_new)
        pred_ridge = self.ridge.predict(X_us_new)
        pred_enet = self.enet.predict(X_us_new)

        if self.combine_method == "weighted_avg":
            return (
                self.weights["pca_sub"] * pred_pca
                + self.weights["ridge"] * pred_ridge
                + self.weights["enet"] * pred_enet
            )
        elif self.combine_method == "sign_vote":
            # Majority vote on direction, magnitude from weighted avg
            signs = (
                self.weights["pca_sub"] * np.sign(pred_pca)
                + self.weights["ridge"] * np.sign(pred_ridge)
                + self.weights["enet"] * np.sign(pred_enet)
            )
            magnitudes = (
                self.weights["pca_sub"] * np.abs(pred_pca)
                + self.weights["ridge"] * np.abs(pred_ridge)
                + self.weights["enet"] * np.abs(pred_enet)
            )
            return np.sign(signs) * magnitudes
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")

    def predict_individual(self, X_us_new: np.ndarray) -> dict:
        """Return individual model predictions for analysis."""
        return {
            "pca_sub": self.pca_sub.predict(X_us_new),
            "ridge": self.ridge.predict(X_us_new),
            "enet": self.enet.predict(X_us_new),
        }
