"""Data pipeline for U.S. and Japanese sector ETF returns.

Fetches OHLCV data from the ARF Data API, computes appropriate returns
(close-to-close for U.S., open-to-close for Japan as specified in the paper),
and aligns trading days with lead-lag structure.
"""

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"

US_TICKERS = [
    "XLE", "XLF", "XLU", "XLI", "XLK", "XLV", "XLY", "XLP", "XLB", "XLRE", "XLC",
]

JP_TICKERS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
    "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
    "1631.T", "1632.T", "1633.T",
]

# Human-readable sector names for TOPIX-17
JP_SECTOR_NAMES = {
    "1617.T": "Foods", "1618.T": "Energy Resources", "1619.T": "Construction & Materials",
    "1620.T": "Raw Materials & Chemicals", "1621.T": "Pharmaceuticals",
    "1622.T": "Automobiles & Transport", "1623.T": "Steel & Nonferrous",
    "1624.T": "Machinery", "1625.T": "Electric & Precision",
    "1626.T": "IT & Services", "1627.T": "Electric Power & Gas",
    "1628.T": "Transportation & Logistics", "1629.T": "Trading Companies",
    "1630.T": "Finance (ex Banks)", "1631.T": "Real Estate",
    "1632.T": "Banks", "1633.T": "Retail",
}

US_SECTOR_NAMES = {
    "XLE": "Energy", "XLF": "Financials", "XLU": "Utilities",
    "XLI": "Industrials", "XLK": "Technology", "XLV": "Health Care",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication Services",
}


class AlignedDataset(NamedTuple):
    """Aligned dataset for lead-lag modeling.

    Attributes
    ----------
    X_us : np.ndarray, shape (T, 11)
        U.S. close-to-close returns on day t.
    Y_jp : np.ndarray, shape (T, 17)
        Japanese open-to-close returns on day t+1.
    dates_us : pd.DatetimeIndex
        Trading dates for U.S. returns.
    dates_jp : pd.DatetimeIndex
        Trading dates for Japanese returns (one day after dates_us).
    us_tickers : list[str]
        U.S. ticker symbols (column order).
    jp_tickers : list[str]
        Japanese ticker symbols (column order).
    us_returns_df : pd.DataFrame
        Full U.S. returns DataFrame before alignment.
    jp_returns_df : pd.DataFrame
        Full Japanese returns DataFrame before alignment.
    """
    X_us: np.ndarray
    Y_jp: np.ndarray
    dates_us: pd.DatetimeIndex
    dates_jp: pd.DatetimeIndex
    us_tickers: list
    jp_tickers: list
    us_returns_df: pd.DataFrame
    jp_returns_df: pd.DataFrame


class DataPipeline:
    """Fetches and preprocesses U.S./Japan sector ETF data for lead-lag modeling.

    Parameters
    ----------
    data_dir : str or Path
        Directory for caching downloaded OHLCV data.
    interval : str
        Data interval (default "1d").
    period : str
        Data period (default "5y").
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        interval: str = "1d",
        period: str = "5y",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.interval = interval
        self.period = period

    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Fetch OHLCV data from ARF Data API with local caching.

        Returns DataFrame indexed by timestamp with columns: open, high, low, close, volume.
        """
        safe_name = ticker.replace("/", "_").replace(".", "_")
        cache_path = self.data_dir / f"{safe_name}_{self.interval}_{self.period}.csv"

        if cache_path.exists():
            df = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")
            return df

        url = f"{API_BASE}?ticker={ticker}&interval={self.interval}&period={self.period}"
        print(f"  Fetching {ticker} from API...")
        df = pd.read_csv(url, parse_dates=["timestamp"], index_col="timestamp")
        df.to_csv(cache_path)
        return df

    def compute_us_returns(self) -> pd.DataFrame:
        """Compute close-to-close returns for U.S. sector ETFs.

        Returns DataFrame of shape (T, 11) indexed by date.
        """
        closes = {}
        for ticker in US_TICKERS:
            ohlcv = self.fetch_ohlcv(ticker)
            closes[ticker] = ohlcv["close"]

        close_df = pd.DataFrame(closes).sort_index()
        # Drop rows where any ticker has missing data
        close_df = close_df.dropna()
        returns = close_df.pct_change().iloc[1:]  # drop first NaN row
        return returns

    def compute_jp_returns(self) -> pd.DataFrame:
        """Compute open-to-close returns for Japanese sector ETFs.

        The paper specifies open-to-close returns for the Japanese market,
        capturing intraday movement on the day following U.S. trading.

        Returns DataFrame of shape (T, 17) indexed by date.
        """
        otc_returns = {}
        for ticker in JP_TICKERS:
            ohlcv = self.fetch_ohlcv(ticker)
            # Open-to-close return: (close - open) / open
            otc = (ohlcv["close"] - ohlcv["open"]) / ohlcv["open"]
            otc_returns[ticker] = otc

        returns_df = pd.DataFrame(otc_returns).sort_index()
        # Drop rows with any NaN (missing open or close)
        returns_df = returns_df.dropna()
        return returns_df

    def align_lead_lag(
        self,
        us_returns: pd.DataFrame,
        jp_returns: pd.DataFrame,
    ) -> AlignedDataset:
        """Align U.S. and Japanese returns with lead-lag structure.

        The paper's lead-lag setup: U.S. close-to-close return on day t
        predicts Japanese open-to-close return on the NEXT Japanese trading day.

        Due to timezone differences (U.S. market closes after Japan opens for
        the next day), we pair each U.S. trading day with the next available
        Japanese trading day.

        Parameters
        ----------
        us_returns : DataFrame, shape (T_us, 11)
        jp_returns : DataFrame, shape (T_jp, 17)

        Returns
        -------
        AlignedDataset
        """
        us_dates = us_returns.index.normalize()
        jp_dates = jp_returns.index.normalize()

        us_returns = us_returns.copy()
        jp_returns = jp_returns.copy()
        us_returns.index = us_dates
        jp_returns.index = jp_dates

        # For each U.S. trading day, find the next Japanese trading day
        jp_dates_sorted = jp_dates.sort_values().unique()

        aligned_us_rows = []
        aligned_jp_rows = []
        aligned_us_dates = []
        aligned_jp_dates = []

        for us_date in us_dates.unique():
            # Find next JP trading day strictly after us_date
            future_jp = jp_dates_sorted[jp_dates_sorted > us_date]
            if len(future_jp) == 0:
                continue
            next_jp_date = future_jp[0]

            if us_date in us_returns.index and next_jp_date in jp_returns.index:
                us_row = us_returns.loc[us_date]
                jp_row = jp_returns.loc[next_jp_date]

                # Handle duplicate dates (take first if multiple)
                if isinstance(us_row, pd.DataFrame):
                    us_row = us_row.iloc[0]
                if isinstance(jp_row, pd.DataFrame):
                    jp_row = jp_row.iloc[0]

                aligned_us_rows.append(us_row.values)
                aligned_jp_rows.append(jp_row.values)
                aligned_us_dates.append(us_date)
                aligned_jp_dates.append(next_jp_date)

        X_us = np.array(aligned_us_rows)
        Y_jp = np.array(aligned_jp_rows)

        # Remove any rows with NaN/Inf
        valid = np.all(np.isfinite(X_us), axis=1) & np.all(np.isfinite(Y_jp), axis=1)
        X_us = X_us[valid]
        Y_jp = Y_jp[valid]
        dates_us = pd.DatetimeIndex([d for d, v in zip(aligned_us_dates, valid) if v])
        dates_jp = pd.DatetimeIndex([d for d, v in zip(aligned_jp_dates, valid) if v])

        return AlignedDataset(
            X_us=X_us,
            Y_jp=Y_jp,
            dates_us=dates_us,
            dates_jp=dates_jp,
            us_tickers=list(us_returns.columns),
            jp_tickers=list(jp_returns.columns),
            us_returns_df=us_returns,
            jp_returns_df=jp_returns,
        )

    def load(self) -> AlignedDataset:
        """Full pipeline: fetch data, compute returns, align with lead-lag.

        Returns
        -------
        AlignedDataset
        """
        print("Loading U.S. sector ETF data...")
        us_returns = self.compute_us_returns()
        print(f"  U.S. returns: {us_returns.shape[0]} days x {us_returns.shape[1]} sectors")

        print("Loading Japanese sector ETF data...")
        jp_returns = self.compute_jp_returns()
        print(f"  JP returns: {jp_returns.shape[0]} days x {jp_returns.shape[1]} sectors")

        print("Aligning with lead-lag structure...")
        dataset = self.align_lead_lag(us_returns, jp_returns)
        print(f"  Aligned pairs: {dataset.X_us.shape[0]}")
        print(f"  Date range: {dataset.dates_us[0].date()} to {dataset.dates_us[-1].date()}")

        return dataset
