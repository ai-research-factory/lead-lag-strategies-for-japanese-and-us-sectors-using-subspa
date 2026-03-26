"""Regime detection based on realized volatility of U.S. market returns.

Classifies each trading day into a volatility regime (low/medium/high)
using rolling realized volatility. The regime label can be used to
modulate strategy exposure — e.g., reduce position sizes in high-vol
regimes where lead-lag signal is weaker.
"""

import numpy as np


class VolatilityRegimeDetector:
    """Detects volatility regimes from portfolio-level returns.

    Parameters
    ----------
    vol_lookback : int
        Lookback window for realized volatility estimation.
    n_regimes : int
        Number of regimes (2 or 3). With 2: low/high. With 3: low/med/high.
    quantile_thresholds : list of float or None
        Quantile breakpoints for regime classification. For 3 regimes,
        default is [0.33, 0.67]. For 2 regimes, default is [0.5].
        Computed on an expanding window to avoid lookahead.
    """

    def __init__(
        self,
        vol_lookback: int = 21,
        n_regimes: int = 2,
        quantile_thresholds: list | None = None,
    ):
        self.vol_lookback = vol_lookback
        self.n_regimes = n_regimes
        if quantile_thresholds is None:
            if n_regimes == 2:
                self.quantile_thresholds = [0.5]
            else:
                self.quantile_thresholds = [0.33, 0.67]
        else:
            self.quantile_thresholds = quantile_thresholds

    def compute_rolling_vol(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling realized volatility (annualized std).

        Parameters
        ----------
        returns : ndarray of shape (T,)
            Daily portfolio returns.

        Returns
        -------
        vol : ndarray of shape (T,)
            Rolling annualized volatility. First vol_lookback values
            use expanding window.
        """
        T = len(returns)
        vol = np.full(T, np.nan)
        for t in range(1, T):
            start = max(0, t - self.vol_lookback)
            window = returns[start:t]
            if len(window) >= 2:
                vol[t] = np.std(window) * np.sqrt(252)
            else:
                vol[t] = 0.0
        vol[0] = 0.0
        return vol

    def classify_regimes(self, returns: np.ndarray) -> np.ndarray:
        """Classify each day into a volatility regime.

        Uses expanding-window quantiles (only past data) to avoid lookahead.

        Parameters
        ----------
        returns : ndarray of shape (T,)
            Daily portfolio returns.

        Returns
        -------
        regimes : ndarray of shape (T,) with int values 0..n_regimes-1
            0 = lowest volatility regime, n_regimes-1 = highest.
        """
        vol = self.compute_rolling_vol(returns)
        T = len(vol)
        regimes = np.zeros(T, dtype=int)

        min_history = max(self.vol_lookback, 30)

        for t in range(T):
            if t < min_history:
                regimes[t] = 0  # default to low-vol when insufficient history
                continue

            # Expanding window: use all vol values up to t
            past_vol = vol[1:t + 1]
            past_vol = past_vol[np.isfinite(past_vol)]
            if len(past_vol) < 10:
                regimes[t] = 0
                continue

            current_vol = vol[t]
            for i, q in enumerate(self.quantile_thresholds):
                threshold = np.quantile(past_vol, q)
                if current_vol <= threshold:
                    regimes[t] = i
                    break
            else:
                regimes[t] = self.n_regimes - 1

        return regimes

    def compute_regime_exposure(
        self,
        returns: np.ndarray,
        regime_scales: dict | None = None,
    ) -> np.ndarray:
        """Compute per-day exposure scaling based on regime.

        Parameters
        ----------
        returns : ndarray of shape (T,)
            Daily portfolio returns for regime detection.
        regime_scales : dict mapping regime_id -> scale factor, or None.
            Default for 2 regimes: {0: 1.0, 1: 0.5} (full exposure in
            low-vol, half in high-vol).

        Returns
        -------
        scales : ndarray of shape (T,)
            Per-day exposure multiplier.
        """
        regimes = self.classify_regimes(returns)

        if regime_scales is None:
            if self.n_regimes == 2:
                regime_scales = {0: 1.0, 1: 0.5}
            else:
                regime_scales = {0: 1.0, 1: 0.75, 2: 0.5}

        scales = np.array([regime_scales.get(r, 1.0) for r in regimes])
        return scales
