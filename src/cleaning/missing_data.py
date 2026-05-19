from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

class FillStrategy(ABC):
    """
    A single gap-filling strategy. Strategies are applied in order by
    MissingDataHandler — each one fills what it can and leaves the rest as NaN
    for the next strategy to handle.
    """

    @abstractmethod
    def fill(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN values in daily['pm25_mean'] where possible.
        Must return the dataframe with is_interpolated / is_conditional_mean
        flags updated for any rows it fills.
        """

class ExternalSourceStrategy(FillStrategy):
    """
    Fill gaps using an external daily Series (e.g. OpenAQ or another station).
    An optional bias correction is subtracted from the source before filling.

    Parameters
    ----------
    source : pd.Series
        Daily Series indexed by date, same units as pm25_mean.
    bias   : float
        Mean bias of the source relative to the primary (source - primary).
        Subtracted before filling. Pass 0.0 to skip correction.
    label  : str
        Used only for print output to identify the source.
    """

    def __init__(self, source: pd.Series, bias: float = 0.0, label: str = "external") -> None:
        self.source = source - bias
        self.label  = label

    def fill(self, daily: pd.DataFrame) -> pd.DataFrame:
        daily = daily.copy()
        missing = daily["pm25_mean"].isna()
        available = self.source.reindex(daily.index)

        filled = missing & available.notna()
        daily.loc[filled, "pm25_mean"]   = available[filled]
        daily.loc[filled, "n_hourly_obs"] = 0

        print(f"[{self.label}] Filled {filled.sum()} days from external source.")
        return daily


class LinearInterpolationStrategy(FillStrategy):
    """
    Fill gaps of at most `max_days` consecutive NaN days using linear
    interpolation. Longer gaps are left for the next strategy.

    Parameters
    ----------
    max_days : int
        Maximum consecutive NaN days to interpolate across (default 7).
    """

    def __init__(self, max_days: int = 7) -> None:
        self.max_days = max_days

    def fill(self, daily: pd.DataFrame) -> pd.DataFrame:
        daily = daily.copy()
        was_missing = daily["pm25_mean"].isna()

        daily["pm25_mean"] = daily["pm25_mean"].interpolate(
            method="linear",
            limit=self.max_days,
            limit_direction="forward",
        )

        newly_filled = was_missing & daily["pm25_mean"].notna()
        daily.loc[newly_filled, "is_interpolated"] = True
        daily.loc[newly_filled, "n_hourly_obs"]    = 0

        n_remaining = daily["pm25_mean"].isna().sum()
        print(
            f"[LinearInterpolation] Filled {newly_filled.sum()} days "
            f"(limit={self.max_days}). {n_remaining} days still missing."
        )
        return daily


class ConditionalMeanStrategy(FillStrategy):
    """
    Fill remaining gaps using conditional mean imputation: each missing day
    is replaced by the mean of that same calendar day (month + day of month)
    across all non-missing years in the series.

    This is the approach recommended by Quinteros et al. (2019) for month-long
    gaps in fixed ambient monitoring stations with extended historical records.
    """

    def fill(self, daily: pd.DataFrame) -> pd.DataFrame:
        daily = daily.copy()

        # Build climatology from all observed (non-NaN) days
        observed = daily.dropna(subset=["pm25_mean"]).copy()
        observed["_month"] = observed.index.month
        observed["_day"]   = observed.index.day
        climatology = observed.groupby(["_month", "_day"])["pm25_mean"].mean()

        still_missing = daily["pm25_mean"].isna()
        n_filled = 0

        for date in daily.index[still_missing]:
            key = (date.month, date.day)
            if key in climatology.index:
                daily.loc[date, "pm25_mean"]            = climatology[key]
                daily.loc[date, "is_conditional_mean"]  = True
                daily.loc[date, "n_hourly_obs"]         = 0
                n_filled += 1

        n_remaining = daily["pm25_mean"].isna().sum()
        print(
            f"[ConditionalMean] Filled {n_filled} days with calendar-day climatology. "
            f"{n_remaining} days still missing."
        )
        return daily


class MissingDataHandler:
    """
    Applies a prioritised list of FillStrategy objects in order.
    Each strategy fills what it can; remaining NaNs are passed to the next.

    Usage
    -----
    handler = MissingDataHandler(strategies=[
        ExternalSourceStrategy(source=openaq_daily, bias=-3.35, label="OpenAQ"),
        LinearInterpolationStrategy(max_days=7),
        ConditionalMeanStrategy(),
    ])
    daily = handler.fill(daily)

    Any cleaner can construct its own handler with the strategies relevant to
    its data source — the EEACleaner is not the only possible consumer.
    """

    def __init__(self, strategies: list[FillStrategy]) -> None:
        self.strategies = strategies

    def fill(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Apply all strategies in priority order and return the filled dataframe."""
        n_missing_start = daily["pm25_mean"].isna().sum()
        print(f"[MissingDataHandler] Starting with {n_missing_start} missing days.")

        for strategy in self.strategies:
            
            if daily["pm25_mean"].isna().sum() == 0:
                break
            daily = strategy.fill(daily)

        n_missing_end = daily["pm25_mean"].isna().sum()
        print(f"[MissingDataHandler] Done. {n_missing_end} days remain missing.")
        return daily
