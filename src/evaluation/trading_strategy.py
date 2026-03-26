"""Cost-aware trading strategy with turnover reduction.

Implements signal smoothing, threshold-based positioning, and selective
sector trading to address the critical turnover problem identified in
Phases 1-11 (76% daily turnover destroying all gross alpha).
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
    """

    def __init__(
        self,
        ema_halflife: int | None = None,
        signal_threshold: float = 0.0,
        sector_mask: np.ndarray | None = None,
        cost_bps: float = 10.0,
        max_position_change: float | None = None,
    ):
        self.ema_halflife = ema_halflife
        self.signal_threshold = signal_threshold
        self.sector_mask = sector_mask
        self.cost_bps = cost_bps
        self.max_position_change = max_position_change

    def _ema_smooth(self, predictions: np.ndarray) -> np.ndarray:
        """Apply exponential moving average smoothing to predictions.

        Parameters
        ----------
        predictions : ndarray of shape (T, n_jp)

        Returns
        -------
        smoothed : ndarray of shape (T, n_jp)
        """
        if self.ema_halflife is None:
            return predictions.copy()

        alpha = 1 - np.exp(-np.log(2) / self.ema_halflife)
        smoothed = np.zeros_like(predictions)
        smoothed[0] = predictions[0]

        for t in range(1, len(predictions)):
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

    def _compute_positions(self, signals: np.ndarray) -> np.ndarray:
        """Convert signals to normalized equal-weight positions.

        Parameters
        ----------
        signals : ndarray of shape (T, n_jp)

        Returns
        -------
        weights : ndarray of shape (T, n_jp)
            Position weights normalized to sum of abs weights = 1.
        """
        positions = np.sign(signals)
        n_active = np.abs(positions).sum(axis=1, keepdims=True)
        n_active = np.where(n_active == 0, 1, n_active)
        return positions / n_active

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

        # Step 1: EMA smoothing
        smoothed = self._ema_smooth(predictions)

        # Step 2: Apply sector mask
        smoothed = self._apply_sector_mask(smoothed)

        # Step 3: Apply signal threshold
        filtered = self._apply_threshold(smoothed)

        # Step 4: Compute positions
        weights = self._compute_positions(filtered)

        # Step 5: Apply position change limits
        weights = self._apply_position_limits(weights)

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
        daily_returns_net = daily_returns_gross - daily_costs_full

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
            "ema_halflife": self.ema_halflife,
            "signal_threshold": self.signal_threshold,
            "max_position_change": self.max_position_change,
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
