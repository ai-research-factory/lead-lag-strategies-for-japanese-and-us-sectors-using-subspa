"""Cost-aware trading strategy with turnover reduction.

Implements signal smoothing, threshold-based positioning, selective
sector trading, signal-weighted positions, regime-adaptive EMA,
borrowing cost modeling, risk-parity sizing, and long-bias options.
"""

import numpy as np


class TradingStrategy:
    """Cost-aware long-short strategy with turnover management.

    Parameters
    ----------
    ema_halflife : int or None
        Half-life (in days) for EMA smoothing of raw predictions.
        None disables smoothing (raw sign-based strategy).
    signal_threshold : float
        Minimum absolute prediction magnitude to take a position.
        Predictions below this threshold result in zero position.
    sector_mask : ndarray of bool or None
        Boolean mask of shape (n_jp,) indicating which sectors to trade.
        None trades all sectors.
    cost_bps : float
        One-way transaction cost in basis points (applied to turnover).
    max_position_change : float or None
        Maximum allowed position change per sector per day (0 to 1 scale).
        None means no limit.
    signal_weighted : bool
        If True, use prediction magnitude to size positions (higher conviction
        = larger allocation). If False, use equal-weight sign-based positions.
    adaptive_ema : bool
        If True, adapt EMA half-life based on realized volatility of actuals.
        Requires ema_halflife to be set as the base half-life.
    adaptive_vol_window : int
        Lookback window for realized volatility estimation (used when
        adaptive_ema=True).
    borrow_cost_bps : float
        Annualized borrowing cost for short positions in basis points.
        Applied daily as borrow_cost_bps / 252 / 10000 to short exposure.
    risk_parity : bool
        If True, size positions inversely proportional to each sector's
        rolling realized volatility (risk-parity weighting).
    risk_parity_lookback : int
        Lookback window for estimating per-sector volatility (used when
        risk_parity=True).
    long_bias : float or None
        If set, shifts position weights toward long exposure. A value of
        0.0 means no bias (standard long-short). A value of 1.0 means
        long-only. Values between scale the short positions down:
        short_weight *= (1 - long_bias).
    cost_aware_rebalance : bool
        If True, skip position changes for individual sectors when the
        magnitude of the change is below a threshold derived from
        transaction costs. Only rebalance when the expected alpha from
        the position change exceeds the round-trip cost.
    cost_aware_multiplier : float
        Multiplier for the cost-aware threshold. The threshold is
        cost_bps / 10000 * multiplier. Higher values are more
        conservative (skip more trades). Default 2.0 means we require
        the signal change to be at least 2x the one-way cost.
    sector_ema_halflifes : ndarray or None
        Per-sector EMA half-lives of shape (n_jp,). When set, each sector
        uses its own smoothing period instead of the uniform ema_halflife.
        Overrides ema_halflife for the sectors where values are provided.
    """

    def __init__(
        self,
        ema_halflife: int | None = None,
        signal_threshold: float = 0.0,
        sector_mask: np.ndarray | None = None,
        cost_bps: float = 10.0,
        max_position_change: float | None = None,
        signal_weighted: bool = False,
        adaptive_ema: bool = False,
        adaptive_vol_window: int = 21,
        borrow_cost_bps: float = 0.0,
        risk_parity: bool = False,
        risk_parity_lookback: int = 63,
        long_bias: float | None = None,
        cost_aware_rebalance: bool = False,
        cost_aware_multiplier: float = 2.0,
        sector_ema_halflifes: np.ndarray | None = None,
    ):
        self.ema_halflife = ema_halflife
        self.signal_threshold = signal_threshold
        self.sector_mask = sector_mask
        self.cost_bps = cost_bps
        self.max_position_change = max_position_change
        self.signal_weighted = signal_weighted
        self.adaptive_ema = adaptive_ema
        self.adaptive_vol_window = adaptive_vol_window
        self.borrow_cost_bps = borrow_cost_bps
        self.risk_parity = risk_parity
        self.risk_parity_lookback = risk_parity_lookback
        self.long_bias = long_bias
        self.cost_aware_rebalance = cost_aware_rebalance
        self.cost_aware_multiplier = cost_aware_multiplier
        self.sector_ema_halflifes = sector_ema_halflifes

    def _ema_smooth(self, predictions: np.ndarray, actuals: np.ndarray | None = None) -> np.ndarray:
        """Apply exponential moving average smoothing to predictions.

        When sector_ema_halflifes is set, each sector uses its own half-life.
        When adaptive_ema=True, the half-life scales with realized volatility:
        higher volatility -> longer half-life (smoother signals to avoid
        whipsaws), lower volatility -> shorter half-life (faster reaction).

        Parameters
        ----------
        predictions : ndarray of shape (T, n_jp)
        actuals : ndarray of shape (T, n_jp), optional
            Required when adaptive_ema=True, used for volatility estimation.

        Returns
        -------
        smoothed : ndarray of shape (T, n_jp)
        """
        if self.ema_halflife is None and self.sector_ema_halflifes is None:
            return predictions.copy()

        T, n_jp = predictions.shape
        smoothed = np.zeros_like(predictions)
        smoothed[0] = predictions[0]

        if self.sector_ema_halflifes is not None:
            # Per-sector EMA: each sector has its own half-life
            alphas = np.array([
                1 - np.exp(-np.log(2) / max(hl, 1.0))
                for hl in self.sector_ema_halflifes
            ])
            for t in range(1, T):
                smoothed[t] = alphas * predictions[t] + (1 - alphas) * smoothed[t - 1]
        elif self.adaptive_ema and actuals is not None:
            # Compute rolling realized volatility of portfolio returns
            port_returns = actuals.mean(axis=1)  # equal-weight proxy
            vol_window = self.adaptive_vol_window

            for t in range(1, T):
                # Realized vol over lookback window
                start_idx = max(0, t - vol_window)
                window_rets = port_returns[start_idx:t]
                if len(window_rets) >= 2:
                    realized_vol = np.std(window_rets)
                else:
                    realized_vol = np.std(port_returns[:t]) if t > 0 else 0.01

                # Scale half-life: vol_ratio > 1 means higher vol -> longer HL
                long_term_vol = np.std(port_returns[:t]) if t > 1 else realized_vol
                if long_term_vol > 1e-12:
                    vol_ratio = realized_vol / long_term_vol
                else:
                    vol_ratio = 1.0

                # Clamp scaling factor between 0.5x and 2x base half-life
                scale = np.clip(vol_ratio, 0.5, 2.0)
                adaptive_hl = self.ema_halflife * scale
                alpha = 1 - np.exp(-np.log(2) / max(adaptive_hl, 1.0))
                smoothed[t] = alpha * predictions[t] + (1 - alpha) * smoothed[t - 1]
        else:
            alpha = 1 - np.exp(-np.log(2) / self.ema_halflife)
            for t in range(1, T):
                smoothed[t] = alpha * predictions[t] + (1 - alpha) * smoothed[t - 1]

        return smoothed

    def _apply_threshold(self, signals: np.ndarray) -> np.ndarray:
        """Zero out signals below the threshold magnitude.

        Parameters
        ----------
        signals : ndarray of shape (T, n_jp)

        Returns
        -------
        filtered : ndarray of shape (T, n_jp)
        """
        filtered = signals.copy()
        filtered[np.abs(filtered) < self.signal_threshold] = 0.0
        return filtered

    def _apply_sector_mask(self, signals: np.ndarray) -> np.ndarray:
        """Zero out signals for excluded sectors."""
        if self.sector_mask is None:
            return signals
        masked = signals.copy()
        masked[:, ~self.sector_mask] = 0.0
        return masked

    def _compute_positions(self, signals: np.ndarray, actuals: np.ndarray | None = None) -> np.ndarray:
        """Convert signals to normalized positions.

        When signal_weighted=True, position sizes are proportional to the
        absolute prediction magnitude (higher conviction = larger allocation).
        Otherwise, uses equal-weight sign-based positions.

        When risk_parity=True, positions are scaled inversely by each sector's
        rolling realized volatility, so lower-vol sectors get larger weight.

        When long_bias is set, short positions are scaled down.

        Parameters
        ----------
        signals : ndarray of shape (T, n_jp)
        actuals : ndarray of shape (T, n_jp), optional
            Required for risk_parity mode.

        Returns
        -------
        weights : ndarray of shape (T, n_jp)
            Position weights normalized to sum of abs weights = 1.
        """
        if self.signal_weighted:
            positions = signals.copy()
        else:
            positions = np.sign(signals)

        # Risk-parity: inverse-volatility weighting
        if self.risk_parity and actuals is not None:
            T, n_jp = positions.shape
            inv_vol = np.ones((T, n_jp))
            for t in range(1, T):
                start = max(0, t - self.risk_parity_lookback)
                window = actuals[start:t]
                if len(window) >= 5:
                    sector_vol = np.std(window, axis=0)
                    sector_vol = np.where(sector_vol < 1e-8, 1e-8, sector_vol)
                    inv_vol[t] = 1.0 / sector_vol
            positions = positions * inv_vol

        # Long bias: scale down short positions
        if self.long_bias is not None and self.long_bias > 0:
            short_mask = positions < 0
            positions[short_mask] *= (1.0 - self.long_bias)

        total_abs = np.abs(positions).sum(axis=1, keepdims=True)
        total_abs = np.where(total_abs < 1e-12, 1.0, total_abs)
        return positions / total_abs

    def _apply_position_limits(self, weights: np.ndarray) -> np.ndarray:
        """Limit position changes between consecutive days."""
        if self.max_position_change is None:
            return weights

        limited = weights.copy()
        for t in range(1, len(limited)):
            delta = limited[t] - limited[t - 1]
            clipped = np.clip(delta, -self.max_position_change, self.max_position_change)
            limited[t] = limited[t - 1] + clipped
            # Re-normalize
            total = np.abs(limited[t]).sum()
            if total > 1e-12:
                limited[t] /= total

        return limited

    def _apply_cost_aware_rebalance(self, weights: np.ndarray) -> np.ndarray:
        """Skip per-sector position changes when they are too small to justify the cost.

        For each sector on each day, if the absolute change in weight is below
        the cost threshold, keep the previous day's weight for that sector.
        Then re-normalize so total absolute weight = 1.
        """
        if not self.cost_aware_rebalance:
            return weights

        threshold = (self.cost_bps / 10000.0) * self.cost_aware_multiplier
        filtered = weights.copy()

        for t in range(1, len(filtered)):
            delta = filtered[t] - filtered[t - 1]
            # Keep previous weight for sectors where change is below threshold
            small_change = np.abs(delta) < threshold
            filtered[t] = np.where(small_change, filtered[t - 1], filtered[t])
            # Re-normalize
            total = np.abs(filtered[t]).sum()
            if total > 1e-12:
                filtered[t] /= total

        return filtered

    def _apply_signal_deadband(self, signals: np.ndarray) -> np.ndarray:
        """Apply deadband filter to smoothed signals before position computation.

        When cost_aware_rebalance is enabled, keep the previous signal value for
        sectors where the change in smoothed signal magnitude is small. This
        prevents unnecessary sign flips when the signal is near zero or oscillating.

        The deadband threshold is the median absolute signal value times the
        cost_aware_multiplier fraction, ensuring it scales with signal magnitude.
        """
        if not self.cost_aware_rebalance:
            return signals

        filtered = signals.copy()
        # Use a fraction of typical signal magnitude as deadband
        median_abs = np.median(np.abs(signals[signals != 0])) if np.any(signals != 0) else 1e-6
        deadband = median_abs * self.cost_aware_multiplier * 0.1

        for t in range(1, len(filtered)):
            for j in range(filtered.shape[1]):
                # If signal change is small AND a sign flip would occur, keep old signal
                old_sign = np.sign(filtered[t - 1, j])
                new_sign = np.sign(filtered[t, j])
                change = abs(filtered[t, j] - filtered[t - 1, j])

                if change < deadband and old_sign != 0:
                    filtered[t, j] = filtered[t - 1, j]

        return filtered

    def run(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> dict:
        """Execute the trading strategy and compute performance metrics.

        Parameters
        ----------
        predictions : ndarray of shape (T, n_jp)
            Model predictions of Japanese sector returns.
        actuals : ndarray of shape (T, n_jp)
            Actual Japanese sector returns.

        Returns
        -------
        dict with strategy metrics including gross/net Sharpe, turnover, etc.
        """
        T, n_jp = predictions.shape

        # Step 1: EMA smoothing (pass actuals for adaptive mode)
        smoothed = self._ema_smooth(predictions, actuals)

        # Step 2: Apply sector mask
        smoothed = self._apply_sector_mask(smoothed)

        # Step 3: Apply signal threshold
        filtered = self._apply_threshold(smoothed)

        # Step 3b: Apply signal deadband (cost-aware at signal level)
        filtered = self._apply_signal_deadband(filtered)

        # Step 4: Compute positions
        weights = self._compute_positions(filtered, actuals)

        # Step 5: Apply position change limits
        weights = self._apply_position_limits(weights)

        # Step 6: Cost-aware rebalancing (skip small trades)
        weights = self._apply_cost_aware_rebalance(weights)

        # Compute daily returns
        daily_returns_gross = (weights * actuals).sum(axis=1)

        # Compute turnover: average daily absolute change in positions
        position_changes = np.abs(np.diff(weights, axis=0))
        daily_turnover = position_changes.sum(axis=1)
        avg_turnover = float(daily_turnover.mean()) if len(daily_turnover) > 0 else 0.0

        # Transaction costs
        cost_per_trade = self.cost_bps / 10000.0
        daily_costs = daily_turnover * cost_per_trade
        # Pad costs to match returns length (no cost on first day)
        daily_costs_full = np.concatenate([[0.0], daily_costs])

        # Borrowing costs for short positions
        if self.borrow_cost_bps > 0:
            daily_borrow_rate = self.borrow_cost_bps / 10000.0 / 252.0
            short_exposure = np.abs(np.minimum(weights, 0)).sum(axis=1)
            daily_borrow_cost = short_exposure * daily_borrow_rate
        else:
            daily_borrow_cost = np.zeros(T)

        daily_returns_net = daily_returns_gross - daily_costs_full - daily_borrow_cost

        # Metrics
        def _compute_metrics(returns, label):
            mean_daily = returns.mean()
            std_daily = returns.std()
            sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 1e-12 else 0.0
            cumulative = np.cumprod(1 + returns)
            max_dd = float(np.min(cumulative / np.maximum.accumulate(cumulative) - 1))
            total_ret = float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0
            ann_ret = float(mean_daily * 252)
            ann_vol = float(std_daily * np.sqrt(252))
            pct_pos = float(np.mean(returns > 0) * 100)
            return {
                f"sharpe_ratio_{label}": round(sharpe, 4),
                f"annualized_return_{label}": round(ann_ret, 6),
                f"total_return_{label}": round(total_ret, 6),
                f"max_drawdown_{label}": round(max_dd, 6),
                f"pct_positive_days_{label}": round(pct_pos, 2),
                f"annualized_volatility_{label}": round(ann_vol, 6),
            }

        gross_metrics = _compute_metrics(daily_returns_gross, "gross")
        net_metrics = _compute_metrics(daily_returns_net, "net")

        # Active sectors
        active_sectors = int(np.mean(np.abs(weights).sum(axis=1) > 1e-12) * n_jp) if n_jp > 0 else 0
        n_active_per_day = np.abs(np.sign(weights)).sum(axis=1).mean()

        return {
            **gross_metrics,
            **net_metrics,
            "avg_daily_turnover": round(avg_turnover, 4),
            "avg_active_sectors": round(float(n_active_per_day), 1),
            "n_days": T,
            "cost_bps": self.cost_bps,
            "borrow_cost_bps": self.borrow_cost_bps,
            "ema_halflife": self.ema_halflife,
            "signal_threshold": self.signal_threshold,
            "max_position_change": self.max_position_change,
            "signal_weighted": self.signal_weighted,
            "adaptive_ema": self.adaptive_ema,
            "risk_parity": self.risk_parity,
            "long_bias": self.long_bias,
            "cost_aware_rebalance": self.cost_aware_rebalance,
            "cost_aware_multiplier": self.cost_aware_multiplier,
            "daily_returns_gross": daily_returns_gross,
            "daily_returns_net": daily_returns_net,
            "weights": weights,
        }


def find_optimal_strategy(
    predictions: np.ndarray,
    actuals: np.ndarray,
    per_sector_accuracy: np.ndarray | None = None,
    cost_bps: float = 10.0,
) -> dict:
    """Grid search over strategy parameters to maximize net Sharpe ratio.

    Parameters
    ----------
    predictions : ndarray of shape (T, n_jp)
    actuals : ndarray of shape (T, n_jp)
    per_sector_accuracy : ndarray of shape (n_jp,), optional
        Per-sector direction accuracy for sector selection.
    cost_bps : float
        Transaction cost in basis points.

    Returns
    -------
    dict with best strategy results and parameter comparison table.
    """
    n_jp = predictions.shape[1]

    # Build sector masks based on accuracy ranking
    sector_masks = {"all_sectors": None}
    if per_sector_accuracy is not None:
        ranked = np.argsort(per_sector_accuracy)[::-1]
        for n_top in [5, 8, 10]:
            if n_top < n_jp:
                mask = np.zeros(n_jp, dtype=bool)
                mask[ranked[:n_top]] = True
                sector_masks[f"top_{n_top}_sectors"] = mask

    # Parameter grid
    ema_halflifes = [None, 3, 5, 10, 20]
    thresholds = [0.0, 0.0001, 0.0003, 0.0005, 0.001]
    max_changes = [None, 0.05, 0.1, 0.2]

    results = []

    for mask_name, mask in sector_masks.items():
        for ema in ema_halflifes:
            for thresh in thresholds:
                for max_chg in max_changes:
                    strategy = TradingStrategy(
                        ema_halflife=ema,
                        signal_threshold=thresh,
                        sector_mask=mask,
                        cost_bps=cost_bps,
                        max_position_change=max_chg,
                    )
                    result = strategy.run(predictions, actuals)
                    results.append({
                        "ema_halflife": ema,
                        "signal_threshold": thresh,
                        "sector_selection": mask_name,
                        "max_position_change": max_chg,
                        "sharpe_gross": result["sharpe_ratio_gross"],
                        "sharpe_net": result["sharpe_ratio_net"],
                        "turnover": result["avg_daily_turnover"],
                        "total_return_net": result["total_return_net"],
                        "max_drawdown_net": result["max_drawdown_net"],
                        "active_sectors": result["avg_active_sectors"],
                        "full_result": result,
                    })

    # Sort by net Sharpe
    results.sort(key=lambda x: x["sharpe_net"], reverse=True)

    best = results[0]
    best_result = best["full_result"]

    # Top 10 configurations summary
    top_configs = []
    for r in results[:10]:
        top_configs.append({
            "ema_halflife": r["ema_halflife"],
            "signal_threshold": r["signal_threshold"],
            "sector_selection": r["sector_selection"],
            "max_position_change": r["max_position_change"],
            "sharpe_gross": r["sharpe_gross"],
            "sharpe_net": r["sharpe_net"],
            "turnover": r["turnover"],
            "total_return_net": r["total_return_net"],
        })

    return {
        "best_params": {
            "ema_halflife": best["ema_halflife"],
            "signal_threshold": best["signal_threshold"],
            "sector_selection": best["sector_selection"],
            "max_position_change": best["max_position_change"],
        },
        "best_result": best_result,
        "top_10_configs": top_configs,
        "total_configs_tested": len(results),
        "baseline_result": next(
            r["full_result"] for r in results
            if r["ema_halflife"] is None
            and r["signal_threshold"] == 0.0
            and r["sector_selection"] == "all_sectors"
            and r["max_position_change"] is None
        ),
    }
