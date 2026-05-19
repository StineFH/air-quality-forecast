from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import BaseFeatureBuilder


class LogTransform(BaseFeatureBuilder):
    """
    Applies log1p to the target column and adds it as log_<target_col>.
    The pipeline detects this builder and updates target_col for all
    subsequent builders so all derived features are on the log scale.
    """

    def build(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        df[f"log_{target_col}"] = np.log1p(df[target_col])
        return df
