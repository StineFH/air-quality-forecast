from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.cleaning.eea_cleaning import (
    OPENAQ_BIAS,
    SENSOR_ERROR_THRESHOLD,
    START_YEAR,
    EEACleaner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_eea_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal EEA-style CSV to *path*."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _eea_row(station: str, start: str, end: str, value: float) -> dict:
    return {"station_id": station, "start": start, "end": end, "value": value}


def _make_eea_csv(tmp_path: Path) -> Path:
    """Create a small EEA CSV covering two full days from 2016."""
    rows = []
    for hour in range(24):
        ts_start = f"2016-01-01 {hour:02d}:00:00"
        ts_end = f"2016-01-01 {hour:02d}:59:00"
        rows.append(_eea_row("STA", ts_start, ts_end, 10.0))
    for hour in range(24):
        ts_start = f"2016-01-02 {hour:02d}:00:00"
        ts_end = f"2016-01-02 {hour:02d}:59:00"
        rows.append(_eea_row("STA", ts_start, ts_end, 20.0))
    p = tmp_path / "DK_PM25.csv"
    _write_eea_csv(p, rows)
    return p


# ---------------------------------------------------------------------------
# EEACleaner._load_raw
# ---------------------------------------------------------------------------

class TestEEACleanerLoadRaw:
    def test_returns_dataframe_with_datetime_index(self, tmp_path):
        p = _make_eea_csv(tmp_path)
        cleaner = EEACleaner(raw_path=p, openaq_path=tmp_path / "missing.csv")
        data = cleaner._load_raw()
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_index_is_sorted(self, tmp_path):
        p = _make_eea_csv(tmp_path)
        cleaner = EEACleaner(raw_path=p, openaq_path=tmp_path / "missing.csv")
        data = cleaner._load_raw()
        assert data.index.is_monotonic_increasing

    def test_value_column_present(self, tmp_path):
        p = _make_eea_csv(tmp_path)
        cleaner = EEACleaner(raw_path=p, openaq_path=tmp_path / "missing.csv")
        data = cleaner._load_raw()
        assert "value" in data.columns


# ---------------------------------------------------------------------------
# EEACleaner._prepare_series
# ---------------------------------------------------------------------------

class TestEEACleanerPrepareSeries:
    def test_filters_before_start_year(self, tmp_path):
        rows = [_eea_row("STA", "2014-06-01 00:00:00", "2014-06-01 00:59:00", 5.0)]
        rows += [_eea_row("STA", "2016-06-01 00:00:00", "2016-06-01 00:59:00", 8.0)]
        p = tmp_path / "DK_PM25.csv"
        _write_eea_csv(p, rows)
        cleaner = EEACleaner(raw_path=p, openaq_path=tmp_path / "missing.csv")
        data = cleaner._load_raw()
        series = cleaner._prepare_series(data)
        assert all(series.index.year >= START_YEAR)

    def test_removes_negative_values(self, tmp_path):
        rows = [_eea_row("STA", "2016-01-01 00:00:00", "2016-01-01 00:59:00", -5.0)]
        p = tmp_path / "DK_PM25.csv"
        _write_eea_csv(p, rows)
        cleaner = EEACleaner(raw_path=p, openaq_path=tmp_path / "missing.csv")
        data = cleaner._load_raw()
        series = cleaner._prepare_series(data)
        assert series.isna().all()

    def test_removes_values_at_threshold(self, tmp_path):
        rows = [
            _eea_row("STA", "2016-01-01 00:00:00", "2016-01-01 00:59:00",
                     SENSOR_ERROR_THRESHOLD),
        ]
        p = tmp_path / "DK_PM25.csv"
        _write_eea_csv(p, rows)
        cleaner = EEACleaner(raw_path=p, openaq_path=tmp_path / "missing.csv")
        data = cleaner._load_raw()
        series = cleaner._prepare_series(data)
        assert series.isna().all()


# ---------------------------------------------------------------------------
# EEACleaner._load_openaq
# ---------------------------------------------------------------------------

class TestEEACleanerLoadOpenAQ:
    def test_returns_none_when_file_missing(self, tmp_path):
        eea_path = _make_eea_csv(tmp_path)
        cleaner = EEACleaner(raw_path=eea_path, openaq_path=tmp_path / "no_file.csv")
        assert cleaner._load_openaq() is None

    def test_returns_series_when_file_exists(self, tmp_path):
        eea_path = _make_eea_csv(tmp_path)
        oaq_path = tmp_path / "pm25.csv"
        dates = pd.date_range("2016-01-01", periods=3, freq="h")
        pd.DataFrame({"date": dates, "value": [10.0, 12.0, 11.0]}).to_csv(
            oaq_path, index=False
        )
        cleaner = EEACleaner(raw_path=eea_path, openaq_path=oaq_path)
        result = cleaner._load_openaq()
        assert isinstance(result, pd.Series)

    def test_openaq_sensor_errors_removed(self, tmp_path):
        eea_path = _make_eea_csv(tmp_path)
        oaq_path = tmp_path / "pm25.csv"
        dates = pd.date_range("2016-01-01", periods=3, freq="h")
        pd.DataFrame(
            {"date": dates, "value": [SENSOR_ERROR_THRESHOLD, 10.0, 5.0]}
        ).to_csv(oaq_path, index=False)
        cleaner = EEACleaner(raw_path=eea_path, openaq_path=oaq_path)
        result = cleaner._load_openaq()
        assert not (result >= SENSOR_ERROR_THRESHOLD).any()


# ---------------------------------------------------------------------------
# EEACleaner.clean (integration)
# ---------------------------------------------------------------------------

class TestEEACleanerIntegration:
    def test_clean_returns_expected_columns(self, tmp_path):
        p = _make_eea_csv(tmp_path)
        cleaner = EEACleaner(raw_path=p, openaq_path=tmp_path / "missing.csv")
        df = cleaner.clean()
        assert set(df.columns) == {
            "pm25_mean", "n_hourly_obs", "is_interpolated", "is_conditional_mean"
        }

    def test_clean_no_nan_in_pm25(self, tmp_path):
        p = _make_eea_csv(tmp_path)
        cleaner = EEACleaner(raw_path=p, openaq_path=tmp_path / "missing.csv")
        df = cleaner.clean()
        assert not df["pm25_mean"].isna().any()

    def test_clean_daily_means_are_correct(self, tmp_path):
        p = _make_eea_csv(tmp_path)
        cleaner = EEACleaner(raw_path=p, openaq_path=tmp_path / "missing.csv")
        df = cleaner.clean()
        assert df["pm25_mean"].iloc[0] == pytest.approx(10.0)
        assert df["pm25_mean"].iloc[1] == pytest.approx(20.0)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EEACleaner(raw_path=tmp_path / "nonexistent.csv")
