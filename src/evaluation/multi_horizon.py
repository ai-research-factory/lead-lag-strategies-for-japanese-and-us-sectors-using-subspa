"""Multi-horizon signal ensemble for PCA_SUB predictions.

Instead of using only 1-day-ahead predictions, this module generates
predictions at multiple aggregation horizons and combines them into
a single ensemble signal. The hypothesis is that medium-term signal
averages are more stable and reduce noise.
"""

import numpy as np

from src.models.pca_sub import PCASub


class MultiHorizonEnsemble:
    """Combines PCA_SUB predictions at multiple aggregation windows.

    For each walk-forward fold, the raw 1-day predictions are aggregated
    over different horizons (e.g., 1-day, 5-day, 10-day moving averages)
    and combined with configurable weights.

    Parameters
    ----------
    horizons : list of int
        Aggregation windows (in days) for prediction smoothing.
        Each horizon h produces a signal that is the h-day moving average
        of the raw predictions.
    weights : list of float or None
        Weights for combining horizon signals. If None, uses equal weights.
        Must match length of horizons.
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        weights: list[float] | None = None,
    ):
        self.horizons = horizons or [1, 5, 10]
        if weights is not None:
            if len(weights) != len(self.horizons):
                raise ValueError("weights must match length of horizons")
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            n = len(self.horizons)
            self.weights = [1.0 / n] * n

    def combine(self, predictions: np.ndarray) -> np.ndarray:
        """Combine raw predictions across multiple horizons.

        For each horizon h, computes a rolling mean of the last h predictions.
        Then takes a weighted average across horizons.

        Parameters
        ----------
        predictions : ndarray of shape (T, n_jp)
            Raw 1-day-ahead predictions from walk-forward.

        Returns
        -------
        ensemble : ndarray of shape (T, n_jp)
            Combined multi-horizon signal.
        """
        T, n_jp = predictions.shape
        ensemble = np.zeros_like(predictions)

        for h, w in zip(self.horizons, self.weights):
            if h == 1:
                horizon_signal = predictions.copy()
            else:
                horizon_signal = np.zeros_like(predictions)
                for t in range(T):
                    start = max(0, t - h + 1)
                    horizon_signal[t] = predictions[start:t + 1].mean(axis=0)
            ensemble += w * horizon_signal

        return ensemble


def generate_multi_horizon_predictions(
    X_us: np.ndarray,
    Y_jp: np.ndarray,
    train_window: int = 252,
    test_window: int = 21,
    K: int = 5,
    L: int = 120,
    lambda_decay: float = 1.0,
    horizons: list[int] | None = None,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run walk-forward and return multi-horizon ensemble predictions.

    This is a convenience function that runs the standard walk-forward
    evaluation and applies multi-horizon ensemble combination to the
    concatenated out-of-sample predictions.

    Parameters
    ----------
    X_us, Y_jp : ndarrays
        Full dataset arrays.
    train_window, test_window : int
        Walk-forward window sizes.
    K, L, lambda_decay : model params
    horizons : list of int
        Aggregation windows.
    weights : list of float or None
        Horizon combination weights.

    Returns
    -------
    ensemble_predictions : ndarray of shape (T_oos, n_jp)
    actuals : ndarray of shape (T_oos, n_jp)
    """
    from src.evaluation.walk_forward import WalkForwardEvaluator

    evaluator = WalkForwardEvaluator(
        train_window=train_window,
        test_window=test_window,
        K=K, L=L, lambda_decay=lambda_decay,
    )
    wf_result = evaluator.evaluate(X_us, Y_jp)

    raw_predictions = wf_result.all_predictions
    actuals = wf_result.all_actuals

    ensemble = MultiHorizonEnsemble(horizons=horizons, weights=weights)
    ensemble_predictions = ensemble.combine(raw_predictions)

    return ensemble_predictions, actuals, wf_result
