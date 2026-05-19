from __future__ import annotations

import pandas as pd

from src.features.base import BaseFeatureBuilder


class TimeFeatures(BaseFeatureBuilder):
    """
    Adds calendar features derived from the datetime index.
    These do not depend on target_col.

    Columns added: month (1-12), dayofweek (0=Mon, 6=Sun).
    Both are kept as integers for LightGBM categorical handling.
    """

    def build(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        df["month"] = df.index.month
        df["dayofweek"] = df.index.dayofweek
        return df
