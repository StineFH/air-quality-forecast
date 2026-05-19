from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.lag_features import LagFeatures
from src.features.log_transform import LogTransform
from src.features.rolling_features import RollingFeatures
from src.features.time_features import TimeFeatures


def _daily_df(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"pm25": values}, index=idx)


# ---------------------------------------------------------------------------
# LagFeatures
# ---------------------------------------------------------------------------

class TestLagFeatures:
    def test_creates_lag_columns(self):
        df = _daily_df([1.0, 2.0, 3.0])
        result = LagFeatures(lags=[1]).build(df, "pm25")
        assert "pm25_lag_1" in result.columns

    def test_lag_shifts_values(self):
        df = _daily_df([10.0, 20.0, 30.0])
        result = LagFeatures(lags=[1]).build(df, "pm25")
        assert result["pm25_lag_1"].iloc[1] == pytest.approx(10.0)
        assert result["pm25_lag_1"].iloc[2] == pytest.approx(20.0)

    def test_first_row_is_nan(self):
        df = _daily_df([1.0, 2.0, 3.0])
        result = LagFeatures(lags=[1]).build(df, "pm25")
        assert np.isnan(result["pm25_lag_1"].iloc[0])

    def test_multiple_lags_create_multiple_columns(self):
        df = _daily_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = LagFeatures(lags=[1, 3]).build(df, "pm25")
        assert "pm25_lag_1" in result.columns
        assert "pm25_lag_3" in result.columns

    def test_does_not_modify_original(self):
        df = _daily_df([1.0, 2.0, 3.0])
        original_cols = set(df.columns)
        LagFeatures(lags=[1]).build(df, "pm25")
        assert set(df.columns) == original_cols

    def test_preserves_existing_columns(self):
        df = _daily_df([1.0, 2.0, 3.0])
        result = LagFeatures(lags=[1]).build(df, "pm25")
        assert "pm25" in result.columns


# ---------------------------------------------------------------------------
# RollingFeatures
# ---------------------------------------------------------------------------

class TestRollingFeatures:
    def test_creates_rolling_column(self):
        df = _daily_df([1.0] * 10)
        result = RollingFeatures(windows=[3]).build(df, "pm25")
        assert "pm25_rolling_3d" in result.columns

    def test_rolling_mean_is_correct(self):
        df = _daily_df([10.0, 20.0, 30.0, 40.0, 50.0])
        result = RollingFeatures(windows=[3], lag=1).build(df, "pm25")
        # lag-1 series: [NaN, 10, 20, 30, 40]
        # rolling(3) at index 3: mean(10, 20, 30) = 20
        assert result["pm25_rolling_3d"].iloc[3] == pytest.approx(20.0)

    def test_no_leakage_lag_shifts_before_rolling(self):
        df = _daily_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = RollingFeatures(windows=[2], lag=1).build(df, "pm25")
        # lag-1 series: [NaN, 1, 2, 3, 4]
        # rolling(2) at index 2: mean(1, 2) = 1.5
        assert result["pm25_rolling_2d"].iloc[2] == pytest.approx(1.5)

    def test_raises_on_lag_zero(self):
        with pytest.raises(ValueError, match="lag must be >= 1"):
            RollingFeatures(windows=[3], lag=0)

    def test_raises_on_negative_lag(self):
        with pytest.raises(ValueError, match="lag must be >= 1"):
            RollingFeatures(windows=[3], lag=-1)

    def test_does_not_modify_original(self):
        df = _daily_df([1.0] * 5)
        original_cols = set(df.columns)
        RollingFeatures(windows=[3]).build(df, "pm25")
        assert set(df.columns) == original_cols

    def test_preserves_existing_columns(self):
        df = _daily_df([1.0] * 5)
        result = RollingFeatures(windows=[3]).build(df, "pm25")
        assert "pm25" in result.columns


# ---------------------------------------------------------------------------
# LogTransform
# ---------------------------------------------------------------------------

class TestLogTransform:
    def test_creates_log_column(self):
        df = _daily_df([1.0, 4.0, 9.0])
        result = LogTransform().build(df, "pm25")
        assert "log_pm25" in result.columns

    def test_applies_log1p(self):
        df = _daily_df([0.0, 1.0, np.e - 1])
        result = LogTransform().build(df, "pm25")
        assert result["log_pm25"].iloc[0] == pytest.approx(0.0)
        assert result["log_pm25"].iloc[1] == pytest.approx(np.log(2))
        assert result["log_pm25"].iloc[2] == pytest.approx(1.0)

    def test_preserves_original_column(self):
        df = _daily_df([5.0, 10.0])
        result = LogTransform().build(df, "pm25")
        assert "pm25" in result.columns

    def test_does_not_modify_original(self):
        df = _daily_df([1.0, 2.0])
        original_cols = set(df.columns)
        LogTransform().build(df, "pm25")
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# TimeFeatures
# ---------------------------------------------------------------------------

class TestTimeFeatures:
    def test_adds_month_and_dayofweek(self):
        df = _daily_df([1.0])
        result = TimeFeatures().build(df, "pm25")
        assert "month" in result.columns
        assert "dayofweek" in result.columns

    def test_month_values(self):
        idx = pd.DatetimeIndex(["2020-03-10", "2020-07-20"])
        df = pd.DataFrame({"pm25": [1.0, 2.0]}, index=idx)
        result = TimeFeatures().build(df, "pm25")
        assert result["month"].iloc[0] == 3
        assert result["month"].iloc[1] == 7

    def test_dayofweek_values(self):
        # 2020-01-06 is a Monday (0), 2020-01-07 Tuesday (1), 2020-01-08 Wednesday (2)
        idx = pd.date_range("2020-01-06", periods=3, freq="D")
        df = pd.DataFrame({"pm25": [1.0, 2.0, 3.0]}, index=idx)
        result = TimeFeatures().build(df, "pm25")
        assert result["dayofweek"].iloc[0] == 0
        assert result["dayofweek"].iloc[1] == 1
        assert result["dayofweek"].iloc[2] == 2

    def test_does_not_modify_original(self):
        df = _daily_df([1.0])
        original_cols = set(df.columns)
        TimeFeatures().build(df, "pm25")
        assert set(df.columns) == original_cols
