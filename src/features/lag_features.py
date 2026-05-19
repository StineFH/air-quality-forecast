from __future__ import annotations

import pandas as pd

from src.features.base import BaseFeatureBuilder


class LagFeatures(BaseFeatureBuilder):
    """
    Creates lagged versions of target_col.

    Parameters
    ----------
    lags : list of integers, e.g. [1, 7, 14]
        Each lag n creates a column <target_col>_lag_<n> shifted n days back.
    """

    def __init__(self, lags: list[int]) -> None:
        self.lags = lags

    def build(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        for lag in self.lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
        return df
