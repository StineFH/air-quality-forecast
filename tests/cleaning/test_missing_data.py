from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning.missing_data import (
    ConditionalMeanStrategy,
    ExternalSourceStrategy,
    LinearInterpolationStrategy,
    MissingDataHandler,
)


def _daily_frame(values: list, start: str = "2020-01-01") -> pd.DataFrame:
    """Build a minimal daily DataFrame matching the cleaner's output format."""
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "pm25_mean": values,
            "n_hourly_obs": [0 if np.isnan(v) else 24 for v in values],
            "is_interpolated": False,
            "is_conditional_mean": False,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# LinearInterpolationStrategy
# ---------------------------------------------------------------------------

class TestLinearInterpolationStrategy:
    def test_fills_single_gap(self):
        df = _daily_frame([10.0, np.nan, 20.0])
        result = LinearInterpolationStrategy(max_days=7).fill(df)
        assert not np.isnan(result["pm25_mean"].iloc[1])

    def test_interpolated_value_is_midpoint(self):
        df = _daily_frame([10.0, np.nan, 20.0])
        result = LinearInterpolationStrategy(max_days=7).fill(df)
        assert result["pm25_mean"].iloc[1] == pytest.approx(15.0)

    def test_sets_is_interpolated_flag(self):
        df = _daily_frame([10.0, np.nan, 20.0])
        result = LinearInterpolationStrategy(max_days=7).fill(df)
        assert result["is_interpolated"].iloc[1]

    def test_sets_n_hourly_obs_to_zero(self):
        df = _daily_frame([10.0, np.nan, 20.0])
        result = LinearInterpolationStrategy(max_days=7).fill(df)
        assert result["n_hourly_obs"].iloc[1] == 0

    def test_gap_exceeding_limit_stays_nan(self):
        df = _daily_frame([10.0, np.nan, np.nan, np.nan, np.nan, 20.0])
        result = LinearInterpolationStrategy(max_days=2).fill(df)
        assert result["pm25_mean"].isna().sum() > 0

    def test_does_not_modify_original(self):
        df = _daily_frame([10.0, np.nan, 20.0])
        LinearInterpolationStrategy(max_days=7).fill(df)
        assert np.isnan(df["pm25_mean"].iloc[1])

    def test_no_gap_leaves_data_unchanged(self):
        df = _daily_frame([5.0, 10.0, 15.0])
        result = LinearInterpolationStrategy(max_days=7).fill(df)
        assert not result["is_interpolated"].any()


# ---------------------------------------------------------------------------
# ExternalSourceStrategy
# ---------------------------------------------------------------------------

class TestExternalSourceStrategy:
    def _source(self, values: list, start: str = "2020-01-01") -> pd.Series:
        idx = pd.date_range(start, periods=len(values), freq="D")
        return pd.Series(values, index=idx)

    def test_fills_gap_from_source(self):
        df = _daily_frame([10.0, np.nan, 15.0])
        source = self._source([9.0, 12.0, 14.0])
        result = ExternalSourceStrategy(source=source, bias=0.0).fill(df)
        assert not np.isnan(result["pm25_mean"].iloc[1])

    def test_bias_subtracted_from_source(self):
        df = _daily_frame([np.nan])
        source = self._source([15.0])
        result = ExternalSourceStrategy(source=source, bias=3.0).fill(df)
        assert result["pm25_mean"].iloc[0] == pytest.approx(12.0)

    def test_sets_n_hourly_obs_to_zero(self):
        df = _daily_frame([np.nan])
        source = self._source([10.0])
        result = ExternalSourceStrategy(source=source).fill(df)
        assert result["n_hourly_obs"].iloc[0] == 0

    def test_does_not_overwrite_existing_values(self):
        df = _daily_frame([10.0, np.nan])
        source = self._source([99.0, 12.0])
        result = ExternalSourceStrategy(source=source).fill(df)
        assert result["pm25_mean"].iloc[0] == pytest.approx(10.0)

    def test_missing_source_date_leaves_nan(self):
        df = _daily_frame([np.nan, np.nan])
        source = self._source([10.0], start="2020-01-01")
        result = ExternalSourceStrategy(source=source).fill(df)
        assert np.isnan(result["pm25_mean"].iloc[1])

    def test_does_not_modify_original(self):
        df = _daily_frame([np.nan])
        source = self._source([10.0])
        ExternalSourceStrategy(source=source).fill(df)
        assert np.isnan(df["pm25_mean"].iloc[0])


# ---------------------------------------------------------------------------
# ConditionalMeanStrategy
# ---------------------------------------------------------------------------

class TestConditionalMeanStrategy:
    def test_fills_from_climatology(self):
        # Jan 1 observed in 2018 and 2019; missing in 2020
        idx = pd.date_range("2018-01-01", periods=3, freq="YS")
        values = [20.0, 30.0, np.nan]
        df = pd.DataFrame(
            {
                "pm25_mean": values,
                "n_hourly_obs": [24, 24, 0],
                "is_interpolated": False,
                "is_conditional_mean": False,
            },
            index=idx,
        )
        result = ConditionalMeanStrategy().fill(df)
        assert result["pm25_mean"].iloc[2] == pytest.approx(25.0)

    def test_sets_is_conditional_mean_flag(self):
        idx = pd.date_range("2018-01-01", periods=3, freq="YS")
        df = pd.DataFrame(
            {
                "pm25_mean": [20.0, 30.0, np.nan],
                "n_hourly_obs": [24, 24, 0],
                "is_interpolated": False,
                "is_conditional_mean": False,
            },
            index=idx,
        )
        result = ConditionalMeanStrategy().fill(df)
        assert result["is_conditional_mean"].iloc[2]

    def test_sets_n_hourly_obs_to_zero(self):
        idx = pd.date_range("2018-01-01", periods=3, freq="YS")
        df = pd.DataFrame(
            {
                "pm25_mean": [20.0, 30.0, np.nan],
                "n_hourly_obs": [24, 24, 0],
                "is_interpolated": False,
                "is_conditional_mean": False,
            },
            index=idx,
        )
        result = ConditionalMeanStrategy().fill(df)
        assert result["n_hourly_obs"].iloc[2] == 0

    def test_no_climatology_leaves_nan(self):
        # Feb 29 only appears in leap years — missing in a non-leap gap year
        df = _daily_frame([np.nan], start="2021-02-28")
        result = ConditionalMeanStrategy().fill(df)
        assert np.isnan(result["pm25_mean"].iloc[0])

    def test_does_not_modify_original(self):
        idx = pd.date_range("2018-01-01", periods=3, freq="YS")
        df = pd.DataFrame(
            {
                "pm25_mean": [20.0, 30.0, np.nan],
                "n_hourly_obs": [24, 24, 0],
                "is_interpolated": False,
                "is_conditional_mean": False,
            },
            index=idx,
        )
        ConditionalMeanStrategy().fill(df)
        assert np.isnan(df["pm25_mean"].iloc[2])


# ---------------------------------------------------------------------------
# MissingDataHandler
# ---------------------------------------------------------------------------

class TestMissingDataHandler:
    def test_applies_strategies_in_order(self):
        df = _daily_frame([10.0, np.nan, 20.0])
        handler = MissingDataHandler([LinearInterpolationStrategy(max_days=7)])
        result = handler.fill(df)
        assert not result["pm25_mean"].isna().any()

    def test_stops_early_when_no_missing(self):
        df = _daily_frame([10.0, 15.0, 20.0])
        handler = MissingDataHandler([LinearInterpolationStrategy(max_days=7)])
        result = handler.fill(df)
        assert not result["pm25_mean"].isna().any()

    def test_multiple_strategies_chain(self):
        # Gap too long for interpolation — external source fills it
        df = _daily_frame([10.0] + [np.nan] * 10 + [20.0])
        idx = df.index
        external = pd.Series([12.0] * 10, index=idx[1:11])
        handler = MissingDataHandler([
            LinearInterpolationStrategy(max_days=3),
            ExternalSourceStrategy(source=external, bias=0.0),
        ])
        result = handler.fill(df)
        assert not result["pm25_mean"].isna().any()
