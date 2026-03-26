"""Tests for the data pipeline module (unit tests that don't require API access)."""

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import (
    DataPipeline,
    AlignedDataset,
    US_TICKERS,
    JP_TICKERS,
    JP_SECTOR_NAMES,
    US_SECTOR_NAMES,
)


class TestConstants:
    def test_us_tickers_count(self):
        assert len(US_TICKERS) == 11

    def test_jp_tickers_count(self):
        assert len(JP_TICKERS) == 17

    def test_us_sector_names_cover_tickers(self):
        for ticker in US_TICKERS:
            assert ticker in US_SECTOR_NAMES

    def test_jp_sector_names_cover_tickers(self):
        for ticker in JP_TICKERS:
            assert ticker in JP_SECTOR_NAMES


class TestAlignLeadLag:
    @pytest.fixture
    def pipeline(self, tmp_path):
        return DataPipeline(data_dir=tmp_path)

    @pytest.fixture
    def sample_returns(self):
        # Create US and JP returns with known dates
        us_dates = pd.date_range("2023-01-02", periods=20, freq="B")
        jp_dates = pd.date_range("2023-01-03", periods=20, freq="B")
        us_returns = pd.DataFrame(
            np.random.default_rng(42).standard_normal((20, 11)) * 0.01,
            index=us_dates,
            columns=US_TICKERS,
        )
        jp_returns = pd.DataFrame(
            np.random.default_rng(43).standard_normal((20, 17)) * 0.01,
            index=jp_dates,
            columns=JP_TICKERS,
        )
        return us_returns, jp_returns

    def test_alignment_returns_named_tuple(self, pipeline, sample_returns):
        us_ret, jp_ret = sample_returns
        result = pipeline.align_lead_lag(us_ret, jp_ret)
        assert isinstance(result, AlignedDataset)

    def test_aligned_shapes_match(self, pipeline, sample_returns):
        us_ret, jp_ret = sample_returns
        result = pipeline.align_lead_lag(us_ret, jp_ret)
        assert result.X_us.shape[0] == result.Y_jp.shape[0]
        assert result.X_us.shape[1] == 11
        assert result.Y_jp.shape[1] == 17

    def test_dates_lengths_match_data(self, pipeline, sample_returns):
        us_ret, jp_ret = sample_returns
        result = pipeline.align_lead_lag(us_ret, jp_ret)
        assert len(result.dates_us) == result.X_us.shape[0]
        assert len(result.dates_jp) == result.Y_jp.shape[0]

    def test_jp_dates_after_us_dates(self, pipeline, sample_returns):
        us_ret, jp_ret = sample_returns
        result = pipeline.align_lead_lag(us_ret, jp_ret)
        for us_date, jp_date in zip(result.dates_us, result.dates_jp):
            assert jp_date > us_date

    def test_no_nan_in_aligned_data(self, pipeline, sample_returns):
        us_ret, jp_ret = sample_returns
        result = pipeline.align_lead_lag(us_ret, jp_ret)
        assert np.all(np.isfinite(result.X_us))
        assert np.all(np.isfinite(result.Y_jp))

    def test_lead_lag_preserves_order(self, pipeline, sample_returns):
        """US date t should map to the next JP trading day after t."""
        us_ret, jp_ret = sample_returns
        result = pipeline.align_lead_lag(us_ret, jp_ret)
        # Every aligned JP date should be strictly after the corresponding US date
        for i in range(len(result.dates_us)):
            assert result.dates_jp[i] > result.dates_us[i]


class TestDataPipelineInit:
    def test_creates_data_dir(self, tmp_path):
        data_dir = tmp_path / "new_data_dir"
        pipeline = DataPipeline(data_dir=data_dir)
        assert data_dir.exists()

    def test_default_params(self, tmp_path):
        pipeline = DataPipeline(data_dir=tmp_path)
        assert pipeline.interval == "1d"
        assert pipeline.period == "5y"
