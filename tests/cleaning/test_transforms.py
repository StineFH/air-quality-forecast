from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning.transforms import (
    combine_to_hourly_series,
    filter_date_range,
    remove_sensor_errors,
    resample_to_daily,
)


def _hourly_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h")


# ---------------------------------------------------------------------------
# filter_date_range
# ---------------------------------------------------------------------------

class TestFilterDateRange:
    def test_keeps_rows_in_range(self):
        idx = pd.date_range("2014-12-31", periods=3, freq="YS")
        df = pd.DataFrame({"value": [1, 2, 3]}, index=idx)
        result = filter_date_range(df, start_year=2015)
        assert all(result.index.year >= 2015)

    def test_drops_earlier_years(self):
        idx = pd.date_range("2013-01-01", periods=4, freq="YS")
        df = pd.DataFrame({"value": range(4)}, index=idx)
        result = filter_date_range(df, start_year=2015)
        assert len(result) == 2

    def test_returns_copy(self):
        idx = pd.date_range("2015-01-01", periods=2, freq="YS")
        df = pd.DataFrame({"value": [1, 2]}, index=idx)
        result = filter_date_range(df, start_year=2015)
        result["value"] = 99
        assert df["value"].iloc[0] != 99

    def test_empty_result_when_all_before_start(self):
        idx = pd.date_range("2010-01-01", periods=3, freq="YS")
        df = pd.DataFrame({"value": [1, 2, 3]}, index=idx)
        result = filter_date_range(df, start_year=2020)
        assert result.empty


# ---------------------------------------------------------------------------
# remove_sensor_errors
# ---------------------------------------------------------------------------

class TestRemoveSensorErrors:
    def test_negative_values_become_nan(self):
        s = pd.Series([-1.0, 0.0, 5.0])
        result = remove_sensor_errors(s)
        assert np.isnan(result.iloc[0])

    def test_zero_is_kept(self):
        s = pd.Series([0.0, 1.0])
        result = remove_sensor_errors(s)
        assert result.iloc[0] == 0.0

    def test_at_threshold_becomes_nan(self):
        s = pd.Series([299.9, 300.0, 301.0])
        result = remove_sensor_errors(s, threshold=300.0)
        assert not np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])

    def test_valid_values_unchanged(self):
        s = pd.Series([10.5, 25.0, 50.0])
        result = remove_sensor_errors(s)
        pd.testing.assert_series_equal(result, s)

    def test_returns_copy(self):
        s = pd.Series([-1.0, 5.0])
        result = remove_sensor_errors(s)
        assert s.iloc[0] == -1.0

    def test_custom_threshold(self):
        s = pd.Series([50.0, 100.0, 150.0])
        result = remove_sensor_errors(s, threshold=100.0)
        assert not np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])


# ---------------------------------------------------------------------------
# combine_to_hourly_series
# ---------------------------------------------------------------------------

class TestCombineToHourlySeries:
    def test_single_station_passes_through(self):
        idx = _hourly_index("2020-01-01", 3)
        df = pd.DataFrame({"value": [10.0, 20.0, 30.0]}, index=idx)
        result = combine_to_hourly_series(df)
        pd.testing.assert_series_equal(result, df["value"], check_names=False)

    def test_two_stations_averaged(self):
        idx = _hourly_index("2020-01-01", 2)
        df = pd.DataFrame(
            {"value": [10.0, 20.0, 30.0, 40.0]},
            index=idx.append(idx),
        )
        result = combine_to_hourly_series(df)
        assert result.iloc[0] == pytest.approx(20.0)
        assert result.iloc[1] == pytest.approx(30.0)

    def test_result_is_series_named_value(self):
        idx = _hourly_index("2020-06-01", 4)
        df = pd.DataFrame({"value": [5.0] * 4}, index=idx)
        result = combine_to_hourly_series(df)
        assert isinstance(result, pd.Series)
        assert result.name == "value"

    def test_nan_in_one_station_reduces_mean(self):
        idx = _hourly_index("2020-01-01", 1)
        df = pd.DataFrame(
            {"value": [10.0, np.nan]},
            index=idx.append(idx),
        )
        result = combine_to_hourly_series(df)
        assert result.iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# resample_to_daily
# ---------------------------------------------------------------------------

class TestResampleToDaily:
    @pytest.fixture()
    def hourly_series(self):
        idx = _hourly_index("2020-03-01", 48)
        values = [10.0] * 24 + [20.0] * 24
        return pd.Series(values, index=idx, name="value")

    def test_output_columns(self, hourly_series):
        result = resample_to_daily(hourly_series)
        assert set(result.columns) == {
            "pm25_mean", "n_hourly_obs", "is_interpolated", "is_conditional_mean"
        }

    def test_index_is_datetime(self, hourly_series):
        result = resample_to_daily(hourly_series)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_daily_mean_is_correct(self, hourly_series):
        result = resample_to_daily(hourly_series)
        assert result["pm25_mean"].iloc[0] == pytest.approx(10.0)
        assert result["pm25_mean"].iloc[1] == pytest.approx(20.0)

    def test_obs_count_is_24_for_complete_day(self, hourly_series):
        result = resample_to_daily(hourly_series)
        assert result["n_hourly_obs"].iloc[0] == 24

    def test_obs_count_excludes_nans(self):
        idx = _hourly_index("2020-01-01", 24)
        values = [5.0] * 20 + [np.nan] * 4
        s = pd.Series(values, index=idx)
        result = resample_to_daily(s)
        assert result["n_hourly_obs"].iloc[0] == 20

    def test_flags_are_false_by_default(self, hourly_series):
        result = resample_to_daily(hourly_series)
        assert not result["is_interpolated"].any()
        assert not result["is_conditional_mean"].any()

    def test_nan_day_has_zero_obs(self):
        idx = _hourly_index("2020-01-01", 24)
        s = pd.Series([np.nan] * 24, index=idx)
        result = resample_to_daily(s)
        assert result["n_hourly_obs"].iloc[0] == 0
        assert np.isnan(result["pm25_mean"].iloc[0])
