from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from src.schemas.cleaning import CleanedEEADataset, CleanedEEARecord


def _valid_record_kwargs(**overrides) -> dict:
    base = {
        "date": pd.Timestamp("2020-01-01"),
        "pm25_mean": 15.0,
        "n_hourly_obs": 20,
        "is_interpolated": False,
        "is_conditional_mean": False,
    }
    return {**base, **overrides}


def _valid_dataframe(n: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "pm25_mean": [10.0] * n,
            "n_hourly_obs": [24] * n,
            "is_interpolated": [False] * n,
            "is_conditional_mean": [False] * n,
        }
    )


# ---------------------------------------------------------------------------
# CleanedEEARecord
# ---------------------------------------------------------------------------

class TestCleanedEEARecord:
    def test_valid_record(self):
        record = CleanedEEARecord(**_valid_record_kwargs())
        assert record.pm25_mean == 15.0

    def test_negative_pm25_raises(self):
        with pytest.raises(ValidationError):
            CleanedEEARecord(**_valid_record_kwargs(pm25_mean=-1.0))

    def test_pm25_at_300_raises(self):
        with pytest.raises(ValidationError):
            CleanedEEARecord(**_valid_record_kwargs(pm25_mean=300.0))

    def test_pm25_just_below_300_is_valid(self):
        record = CleanedEEARecord(**_valid_record_kwargs(pm25_mean=299.9))
        assert record.pm25_mean == pytest.approx(299.9)

    def test_pm25_zero_is_valid(self):
        record = CleanedEEARecord(**_valid_record_kwargs(pm25_mean=0.0))
        assert record.pm25_mean == 0.0

    def test_n_hourly_obs_above_24_raises(self):
        with pytest.raises(ValidationError):
            CleanedEEARecord(**_valid_record_kwargs(n_hourly_obs=25))

    def test_n_hourly_obs_negative_raises(self):
        with pytest.raises(ValidationError):
            CleanedEEARecord(**_valid_record_kwargs(n_hourly_obs=-1))

    def test_interpolated_with_nonzero_obs_raises(self):
        with pytest.raises(ValidationError):
            CleanedEEARecord(**_valid_record_kwargs(is_interpolated=True, n_hourly_obs=5))

    def test_interpolated_with_zero_obs_is_valid(self):
        record = CleanedEEARecord(
            **_valid_record_kwargs(is_interpolated=True, n_hourly_obs=0)
        )
        assert record.is_interpolated

    def test_conditional_mean_with_zero_obs_is_valid(self):
        record = CleanedEEARecord(
            **_valid_record_kwargs(is_conditional_mean=True, n_hourly_obs=0)
        )
        assert record.is_conditional_mean


# ---------------------------------------------------------------------------
# CleanedEEADataset
# ---------------------------------------------------------------------------

class TestCleanedEEADataset:
    def test_from_dataframe_valid(self):
        df = _valid_dataframe(3)
        dataset = CleanedEEADataset.from_dataframe(df)
        assert len(dataset.records) == 3

    def test_from_dataframe_missing_column_raises(self):
        df = _valid_dataframe(2).drop(columns=["n_hourly_obs"])
        with pytest.raises(ValueError, match="missing columns"):
            CleanedEEADataset.from_dataframe(df)

    def test_from_dataframe_nan_pm25_raises(self):
        import numpy as np
        df = _valid_dataframe(2)
        df.loc[0, "pm25_mean"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            CleanedEEADataset.from_dataframe(df)

    def test_from_dataframe_invalid_record_raises(self):
        df = _valid_dataframe(2)
        df.loc[0, "pm25_mean"] = 350.0
        with pytest.raises((ValueError, Exception)):
            CleanedEEADataset.from_dataframe(df)

    def test_records_have_correct_types(self):
        df = _valid_dataframe(1)
        dataset = CleanedEEADataset.from_dataframe(df)
        record = dataset.records[0]
        assert isinstance(record, CleanedEEARecord)
        assert isinstance(record.is_interpolated, bool)
        assert isinstance(record.n_hourly_obs, int)
