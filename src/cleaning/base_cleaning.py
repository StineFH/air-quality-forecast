from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseCleaner(ABC):
    """
    Abstract base class for all data source cleaners.

    Subclasses must implement:
        - clean()  : apply all cleaning steps and return a validated dataframe
        - validate(): check that the output conforms to the expected schema
    """

    def __init__(self, raw_path: str | Path) -> None:
        self.raw_path = Path(raw_path)
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_path}")

    @abstractmethod
    def clean(self) -> pd.DataFrame:
        """
        Load raw data, apply all cleaning steps, validate, and return a
        clean dataframe whose structure matches the source-specific schema.
        """

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> None:
        """
        Raise an exception if df does not conform to the expected schema.
        Called at the end of clean() before returning.
        """