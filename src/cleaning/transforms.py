from __future__ import annotations

import numpy as np
import pandas as pd


def filter_date_range(df: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Drop rows whose index year is before start_year."""
    return df[df.index.year >= start_year].copy()


def remove_sensor_errors(s: pd.Series, threshold: float = 300.0) -> pd.Series:
    """
    Replace physically impossible readings with NaN:
      - Negative values       → sensor errors
      - Values ≥ threshold    → isolated spikes / instrument faults
    """
    s = s.copy()
    s[s < 0]           = np.nan
    s[s >= threshold]  = np.nan
    return s


def combine_to_hourly_series(df: pd.DataFrame, value_col: str = "value") -> pd.Series:
    """
    Collapse a multi-station dataframe into a single hourly series by taking
    the mean across stations for each timestamp. Suitable when stations are
    co-located and should be treated as one continuous sensor.
    """
    s = df.groupby(df.index)[value_col].mean()
    s.name = value_col
    return s


def resample_to_daily(s: pd.Series) -> pd.DataFrame:
    """
    Resample an hourly Series to daily means.

    Returns a DataFrame with columns:
        pm25_mean       - daily mean (NaN where no valid hourly obs exist)
        n_hourly_obs    - number of valid (non-NaN) hourly readings that day
        is_interpolated - False for all rows (set downstream by gap fillers)
        is_conditional_mean - False for all rows (set downstream by gap fillers)
    """
    daily = pd.DataFrame({
        "pm25_mean":          s.resample("D").mean(),
        "n_hourly_obs":       s.resample("D").count(),
        "is_interpolated":    False,
        "is_conditional_mean": False,
    })
    daily.index = pd.to_datetime(daily.index.date)
    daily.rename_axis("date", inplace=True)
    return daily
