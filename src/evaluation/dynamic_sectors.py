"""Dynamic sector selection based on rolling predictability scores.

Instead of using a fixed set of top-N sectors (determined once from the
full walk-forward evaluation), this module re-evaluates which sectors
to trade at regular intervals using a rolling window of recent prediction
accuracy. Sectors whose recent directional accuracy exceeds a threshold
are included; others are excluded.
"""

import numpy as np


class DynamicSectorSelector:
    """Selects tradeable sectors based on rolling directional accuracy.

    Parameters
    ----------
    lookback : int
        Number of recent days to evaluate sector accuracy over.
    min_accuracy : float
        Minimum directional accuracy (0-1) to include a sector.
    min_sectors : int
        Always include at least this many sectors (top-N by accuracy)
        even if none meets min_accuracy.
    max_sectors : int
        Maximum number of sectors to include.
    rebalance_freq : int
        How often (in days) to re-evaluate the sector set.
    """

    def __init__(
        self,
        lookback: int = 63,
        min_accuracy: float = 0.52,
        min_sectors: int = 3,
        max_sectors: int = 8,
        rebalance_freq: int = 21,
    ):
        self.lookback = lookback
        self.min_accuracy = min_accuracy
        self.min_sectors = min_sectors
        self.max_sectors = max_sectors
        self.rebalance_freq = rebalance_freq

    def compute_rolling_masks(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> np.ndarray:
        """Compute time-varying sector masks based on rolling accuracy.

        Parameters
        ----------
        predictions : ndarray of shape (T, n_jp)
        actuals : ndarray of shape (T, n_jp)

        Returns
        -------
        masks : ndarray of bool, shape (T, n_jp)
            Per-day, per-sector trading mask.
        """
        T, n_jp = predictions.shape
        masks = np.ones((T, n_jp), dtype=bool)

        # Need at least lookback days before we can make a selection
        current_mask = np.ones(n_jp, dtype=bool)

        for t in range(T):
            if t >= self.lookback and t % self.rebalance_freq == 0:
                # Compute rolling accuracy over the lookback window
                start = t - self.lookback
                pred_window = predictions[start:t]
                actual_window = actuals[start:t]

                signs_match = np.sign(pred_window) == np.sign(actual_window)
                sector_accuracy = signs_match.mean(axis=0)

                # Select sectors meeting the accuracy threshold
                above_threshold = sector_accuracy >= self.min_accuracy
                n_above = above_threshold.sum()

                if n_above >= self.min_sectors:
                    # Use all sectors above threshold, capped at max_sectors
                    if n_above > self.max_sectors:
                        # Take top max_sectors by accuracy
                        ranked = np.argsort(sector_accuracy)[::-1]
                        current_mask = np.zeros(n_jp, dtype=bool)
                        current_mask[ranked[:self.max_sectors]] = True
                    else:
                        current_mask = above_threshold
                else:
                    # Fall back to top min_sectors by accuracy
                    ranked = np.argsort(sector_accuracy)[::-1]
                    current_mask = np.zeros(n_jp, dtype=bool)
                    current_mask[ranked[:self.min_sectors]] = True

            masks[t] = current_mask

        return masks


class DynamicTradingStrategy:
    """Trading strategy with time-varying dynamic sector selection.

    Combines EMA smoothing with dynamic sector masks that change over time
    based on rolling predictability.

    Parameters
    ----------
    ema_halflife : int
        Half-life for EMA smoothing of predictions.
    dynamic_selector : DynamicSectorSelector
        Selector that provides time-varying sector masks.
    cost_bps : float
        One-way transaction cost in basis points.
    borrow_cost_bps : float
        Annualized borrowing cost for short positions in basis points.
    """

    def __init__(
        self,
        ema_halflife: int = 20,
        dynamic_selector: DynamicSectorSelector | None = None,
        cost_bps: float = 10.0,
        borrow_cost_bps: float = 0.0,
    ):
        self.ema_halflife = ema_halflife
        self.dynamic_selector = dynamic_selector
        self.cost_bps = cost_bps
        self.borrow_cost_bps = borrow_cost_bps

    def run(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> dict:
        """Execute the dynamic strategy.

        Parameters
        ----------
        predictions : ndarray of shape (T, n_jp)
        actuals : ndarray of shape (T, n_jp)

        Returns
        -------
        dict with strategy metrics.
        """
        T, n_jp = predictions.shape

        # Step 1: EMA smoothing
        alpha = 1 - np.exp(-np.log(2) / self.ema_halflife)
        smoothed = np.zeros_like(predictions)
        smoothed[0] = predictions[0]
        for t in range(1, T):
            smoothed[t] = alpha * predictions[t] + (1 - alpha) * smoothed[t - 1]

        # Step 2: Compute dynamic sector masks
        if self.dynamic_selector is not None:
            masks = self.dynamic_selector.compute_rolling_masks(predictions, actuals)
        else:
            masks = np.ones((T, n_jp), dtype=bool)

        # Step 3: Apply masks and compute positions
        masked_signals = smoothed.copy()
        masked_signals[~masks] = 0.0

        positions = np.sign(masked_signals)
        total_abs = np.abs(positions).sum(axis=1, keepdims=True)
        total_abs = np.where(total_abs < 1e-12, 1.0, total_abs)
        weights = positions / total_abs

        # Compute daily returns
        daily_returns_gross = (weights * actuals).sum(axis=1)

        # Turnover
        position_changes = np.abs(np.diff(weights, axis=0))
        daily_turnover = position_changes.sum(axis=1)
        avg_turnover = float(daily_turnover.mean()) if len(daily_turnover) > 0 else 0.0

        # Transaction costs
        cost_per_trade = self.cost_bps / 10000.0
        daily_costs = daily_turnover * cost_per_trade
        daily_costs_full = np.concatenate([[0.0], daily_costs])

        # Borrowing costs
        if self.borrow_cost_bps > 0:
            daily_borrow_rate = self.borrow_cost_bps / 10000.0 / 252.0
            short_exposure = np.abs(np.minimum(weights, 0)).sum(axis=1)
            daily_borrow_cost = short_exposure * daily_borrow_rate
        else:
            daily_borrow_cost = np.zeros(T)

        daily_returns_net = daily_returns_gross - daily_costs_full - daily_borrow_cost

        # Metrics
        def _metrics(returns, label):
            mean_d = returns.mean()
            std_d = returns.std()
            sr = (mean_d / std_d * np.sqrt(252)) if std_d > 1e-12 else 0.0
            cum = np.cumprod(1 + returns)
            max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
            total_ret = float(cum[-1] - 1) if len(cum) > 0 else 0.0
            ann_ret = float(mean_d * 252)
            ann_vol = float(std_d * np.sqrt(252))
            pct_pos = float(np.mean(returns > 0) * 100)
            return {
                f"sharpe_ratio_{label}": round(sr, 4),
                f"annualized_return_{label}": round(ann_ret, 6),
                f"total_return_{label}": round(total_ret, 6),
                f"max_drawdown_{label}": round(max_dd, 6),
                f"pct_positive_days_{label}": round(pct_pos, 2),
                f"annualized_volatility_{label}": round(ann_vol, 6),
            }

        gross_m = _metrics(daily_returns_gross, "gross")
        net_m = _metrics(daily_returns_net, "net")

        # Track how many sectors are active over time
        active_per_day = masks.sum(axis=1).astype(float)
        avg_active = float(active_per_day.mean())

        # Count how many times the sector set changed
        mask_changes = np.any(masks[1:] != masks[:-1], axis=1).sum()

        return {
            **gross_m,
            **net_m,
            "avg_daily_turnover": round(avg_turnover, 4),
            "avg_active_sectors": round(avg_active, 1),
            "sector_rebalances": int(mask_changes),
            "n_days": T,
            "cost_bps": self.cost_bps,
            "borrow_cost_bps": self.borrow_cost_bps,
            "ema_halflife": self.ema_halflife,
            "daily_returns_gross": daily_returns_gross,
            "daily_returns_net": daily_returns_net,
            "weights": weights,
            "sector_masks": masks,
        }
