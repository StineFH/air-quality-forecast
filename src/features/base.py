from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFeatureBuilder(ABC):
    """
    Abstract base for all feature builders.

    Each builder receives the full dataframe (with all previously built
    features) and returns it with new columns added. Builders must not
    modify or drop existing columns.

    The pipeline passes target_col so builders know which column to derive
    features from — this will be the log-transformed column if LogTransform
    has already run.
    """

    @abstractmethod
    def build(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add new feature columns to df and return it."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
