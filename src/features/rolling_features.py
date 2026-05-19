from __future__ import annotations

import pandas as pd

from src.features.base import BaseFeatureBuilder


class RollingFeatures(BaseFeatureBuilder):
    """
    Creates rolling means over the lag-1 series of target_col.

    Using lag=1 (default) ensures no leakage — the rolling window
    only looks at values available before the prediction day.

    Parameters
    ----------
    windows : list of integers, e.g. [7, 14]
        Each window w creates a column <target_col>_rolling_<w>d.
    lag : int
        How many days to shift before computing the rolling mean.
        Must be >= 1 to avoid data leakage. Defaults to 1.
    """

    def __init__(self, windows: list[int], lag: int = 1) -> None:
        if lag < 1:
            raise ValueError("lag must be >= 1 to avoid leakage.")
        self.windows = windows
        self.lag = lag

    def build(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        lagged = df[target_col].shift(self.lag)
        for window in self.windows:
            df[f"{target_col}_rolling_{window}d"] = lagged.rolling(window).mean()
        return df
